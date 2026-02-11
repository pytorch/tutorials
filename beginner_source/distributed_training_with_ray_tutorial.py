"""
Distributed training at scale with PyTorch and Ray Train
=========================================================

**Author:** `Ricardo Decal <https://github.com/crypdick>`__

This tutorial shows how to distribute PyTorch training across multiple GPUs
using Ray Train and Ray Data for scalable, production-ready model training.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` You will learn how to:
       :class-card: card-prerequisites

       * Pre-train a GPT-2 (~124M-parameter) language model using PyTorch
         and Hugging Face Transformers.
       * Distribute training across multiple GPUs with Ray Train with minimal code changes.
       * Stream training data from Hugging Face datasets with Ray Data's distributed workers.
       * Save and load distributed checkpoints.
       * Scale from a single node to a multinode cluster with minimal code changes.
       * Optimize cost and performance with heterogeneous clusters.
       * Monitor training with the Ray dashboard.

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.9+.
       * Ray Train (``ray[train]``) v2.52.1+.
       * ``tiktoken``, ``datasets``, and ``transformers`` (Hugging Face).
       * One or more GPUs are recommended but not required.

`Ray Train <https://docs.ray.io/en/latest/train/train.html>`__ is a
scalable framework for distributed deep learning.
Ray Train builds on top of `Ray <https://docs.ray.io/en/latest/index.html>`_, a
unified framework for scaling AI and Python applications that
simplifies the complexities of distributed computing. Ray is also open source
and part of the PyTorch Foundation.

Ray Train enables you to
scale from a single GPU to hundreds of GPUs without rewriting your training
loop. Combined with `Ray Data <https://docs.ray.io/en/latest/data/data.html>`__
for streaming data ingestion, you get an end-to-end distributed training
pipeline that handles data loading, sharding, gradient synchronization,
checkpointing, and fault tolerance.

Setup
-----

To install the dependencies, run ``pip install "ray[train]" torch tiktoken datasets transformers``.

Then, import the required libraries:
"""

import time

import numpy as np
import ray
import ray.train
import tiktoken
import torch
from datasets import load_dataset
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import GPT2Config, GPT2LMHeadModel

# Enable smoke test to run this tutorial quickly.
SMOKE_TEST = True

# Reduce Ray Data verbosity
ray.data.DataContext.get_current().enable_progress_bars = False
ray.data.DataContext.get_current().print_on_execution_start = False

###############################################################################
# Load the dataset with Ray Data
# ------------------------------
#
# This tutorial uses the `Wikitext-103 <https://huggingface.co/datasets/Salesforce/wikitext>`__
# dataset, a collection of over 100 million tokens from verified Good and
# Featured articles on Wikipedia.
#
# The ``ray.data.from_huggingface()`` function converts a Hugging Face
# dataset into a Ray Dataset, enabling distributed streaming and
# preprocessing across all available nodes.

hf_ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
train_ds = ray.data.from_huggingface(hf_ds["train"])
val_ds = ray.data.from_huggingface(hf_ds["validation"])

# Limit dataset size for fast iteration during smoke tests.
if SMOKE_TEST:
    train_ds = train_ds.limit(2500)
    val_ds = val_ds.limit(2500)

print(f"Dataset schema:\n{train_ds.schema()}")

###############################################################################
# The schema can look like this:
#
# .. code-block:: text
#
#    Column  Type
#    ------  ----
#    text    string
#
# This means that the dataset has one column called ``text`` and it is a string.
#
# Inspect raw data
#
# ~~~~~~~~~~~~~~~~
#
# Use ``take(n)`` to fetch a small number of rows for inspection.
# Each row is a dictionary with the column names as keys.

print("--- Raw data sample ---")
sample = train_ds.take(2)
for i, row in enumerate(sample):
    text_preview = (row["text"][:120] + "...") if len(row["text"]) > 120 else row["text"]
    print(f"  Row {i}: {text_preview!r}")

###############################################################################
# You'll see output like:
#
# .. code-block:: text
#
#    Row 0: ''
#    Row 1: ' = Valkyria Chronicles III = '
#
# Each row in Wikitext-103 is a single line from a Wikipedia article.
# Consecutive rows belong to the same article, with empty rows separating
# paragraphs. New articles begin with a title line like
# ``= Article Title =``. The tokenization step below inserts an
# ``<|endoftext|>`` separator token before each title line so the model
# learns to reset context at article boundaries.
#
# Tokenize and chunk the data
# ----------------------------
#
# Language models consume fixed-length sequences of token IDs. The
# preprocessing step converts raw text into input/target pairs for
# next-token prediction.
#
# This tutorial uses ``tiktoken`` with the GPT-2 encoding (vocabulary size
# 50,257). ``tiktoken`` is a fast, standalone tokenizer that has no
# dependency on the Hugging Face ``transformers`` library.
#
# The ``tokenize_and_chunk`` function does the following:
#
# * Tokenizes each batch of text, concatenating into a single stream.
#   Article title lines (for example, ``= Article Title =``) trigger an
#   ``<|endoftext|>`` separator so the model resets context at article
#   boundaries.
# * Splits the stream into fixed-length blocks of ``block_size + 1``
#   tokens.
# * Returns ``input_ids`` (the first ``block_size`` tokens) and
#   ``labels`` (shifted by one position for next-token prediction).

BLOCK_SIZE = 256
VOCAB_SIZE = 50257

encoding = tiktoken.get_encoding("gpt2")
EOT_TOKEN = encoding.eot_token  # <|endoftext|> token ID (50256)


def _is_article_title(text: str) -> bool:
    """Detect Wikitext article title lines like ' = Some Title = '."""
    stripped = text.strip()
    return stripped.startswith("= ") and stripped.endswith(" =") and not stripped.startswith("= =")


def tokenize_and_chunk(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Tokenize text and split into fixed-length chunks for language modeling."""
    # Reconstruct the original text stream by joining rows with newlines.
    # Article title lines signal new articles, so we insert an
    # <|endoftext|> separator before them.
    all_tokens: list[int] = []
    for text in batch["text"]:
        if _is_article_title(text):
            all_tokens.append(EOT_TOKEN)
        all_tokens.extend(encoding.encode_ordinary(text + "\n"))

    # Split into chunks of block_size + 1 (input + 1 shifted target)
    chunk_len = BLOCK_SIZE + 1
    num_chunks = len(all_tokens) // chunk_len
    all_tokens = all_tokens[: num_chunks * chunk_len]

    if num_chunks == 0:
        return {"input_ids": [], "labels": []}

    tokens_array = np.array(all_tokens, dtype=np.int64).reshape(num_chunks, chunk_len)
    input_ids = tokens_array[:, :-1]
    labels = tokens_array[:, 1:]
    return {
        "input_ids": input_ids,
        "labels": labels,
    }



###############################################################################
# Apply the tokenization with ``map_batches()``. This operation is **lazy**,
# meaning that Ray Data defers execution until a downstream consumer requests the
# results. Lazy execution lets Ray optimize the entire pipeline before any
# work begins.

# These do not trigger execution.
train_ds = train_ds.map_batches(tokenize_and_chunk, batch_format="numpy")
val_ds = val_ds.map_batches(tokenize_and_chunk, batch_format="numpy")

###############################################################################
# Inspect the tokenized output with ``take(2)``:

print("--- After tokenization ---")
tokenized_sample = train_ds.take(2)
for i, row in enumerate(tokenized_sample):
    ids = row["input_ids"]
    print(f"  Row {i}: input_ids shape={ids.shape}, first 10 tokens={ids[:10].tolist()}")
    print(f"          Decoded: {encoding.decode(ids[:30].tolist())!r}...")

###############################################################################
# Each row now contains a fixed-length ``input_ids`` array of 256 tokens and
# a corresponding ``labels`` array shifted by one position. These are the
# input/target pairs for next-token prediction.
#
# Streaming execution
# ~~~~~~~~~~~~~~~~~~~
#
# Internally, Ray divides the data into **blocks** and dispatches them to
# workers. This block-based architecture enables **streaming execution**: as
# soon as a stage outputs a block, the next stage can begin processing it
# immediately without waiting for previous stages to finish the entire
# dataset. This means the ``map_batches`` tokenization above runs in a
# streaming pipeline with the training loop, so the full dataset never needs
# to fit in memory at once.
#
# When training starts, Ray Data logs the execution plan. For this tutorial
# one possible plan is:
#
# .. code-block:: text
#
#    Execution plan: InputDataBuffer[Input]
#        -> TaskPoolMapOperator[MapBatches(tokenize_and_chunk)]
#        -> OutputSplitter[split(8, equal=True)]
#
# This tells you exactly how Ray Data will stream through tokenization
# and split the data across 8 trainer workers.
#
#
# Define the transformer model
# ----------------------------
#
# The model is a decoder-only transformer language model using Hugging Face's
# ``GPT2LMHeadModel``. The hyperparameters below are for the standard GPT-2 "small" architecture.
#


def create_model():
    """Create a GPT-2 small model with random weights."""
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=BLOCK_SIZE,
        n_embd=768,
        n_layer=12,
        n_head=12,
    ))
    model.loss_type = "ForCausalLM"
    return model



###############################################################################
# Verify the model size:

model = create_model()
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params / 1e6:.1f}M")

del model  # Free memory before training

###############################################################################
# You can see approximately 123.8M parameters.

###############################################################################
# Define the distributed training function
# -----------------------------------------
#
# The training function runs on each worker process. Ray Train
# manages the distributed setup: it wraps the model in
# ``DistributedDataParallel``, shards the data across workers, and
# synchronizes gradients automatically.
#
# The key Ray Train integration points are:
#
# - ``ray.train.get_dataset_shard("train")`` retrieves the worker's portion of the
#   dataset, and Ray Data automatically splits the dataset across all workers.
# - ``ray.train.torch.prepare_model(model)`` wraps the model in
#   ``DistributedDataParallel`` and moves it to the correct GPU.
# - ``shard.iter_torch_batches(batch_size=...)`` returns an iterator
#   of ``dict[str, torch.Tensor]`` batches, with tensors automatically placed on the worker's GPU. Setting ``prefetch_batches=2`` opportunistically fetches 2 batches ahead of the current batch.
# - ``ray.train.report(metrics, checkpoint=...)`` reports metrics to the driver and optionally saves a checkpoint.


def train_func_per_worker(config: dict):
    """Training function executed by each distributed worker."""
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    max_grad_norm = config["max_grad_norm"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    max_steps_per_epoch = config.get("max_steps_per_epoch")

    # --- Data -----------------------------------------------------------
    # Each worker gets an automatic shard of the dataset.
    train_data_shard = ray.train.get_dataset_shard("train")
    val_data_shard = ray.train.get_dataset_shard("validation")

    # --- Model ----------------------------------------------------------
    model = create_model()
    # prepare_model wraps the model in DistributedDataParallel and places
    # it on the correct device.
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Training loop --------------------------------------------------
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_tokens = 0
        epoch_start = time.perf_counter()

        # iter_torch_batches returns dicts of tensors already on the GPU.
        for batch in train_data_shard.iter_torch_batches(
            batch_size=batch_size, dtypes=torch.long, prefetch_batches=2
        ):
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            # GPT2LMHeadModel computes cross-entropy loss internally
            # when labels are provided.
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1
            train_tokens += input_ids.numel()

            if max_steps_per_epoch and train_batches >= max_steps_per_epoch:
                break

        train_elapsed = time.perf_counter() - epoch_start
        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # --- Validation -----------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_data_shard.iter_torch_batches(
                batch_size=batch_size, dtypes=torch.long, prefetch_batches=2
            ):
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss
                val_loss_sum += loss.item()
                val_batches += 1

                if max_steps_per_epoch and val_batches >= max_steps_per_epoch:
                    break

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        epoch_elapsed = time.perf_counter() - epoch_start

        # --- Report metrics -------------------------------------------------
        metrics = {
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "epoch": epoch,
            "epoch_time_sec": round(epoch_elapsed, 2),
            "epoch_tokens": train_tokens,
            "tokens_per_sec": round(train_tokens / max(train_elapsed, 1e-6), 2),
        }
        ray.train.report(
            metrics=metrics,
            checkpoint=None,  # If we were checkpointing, we'd pass a Checkpoint here
        )



###############################################################################
# Configure and launch distributed training
# ------------------------------------------
#
# The ``TorchTrainer`` brings everything together. Running ``trainer.fit()`` finally
# triggers the execution of the full data pipeline and training loop. The Trainer accepts:
#
# - ``train_func_per_worker``: the function each worker executes.
# - ``train_loop_config``: a dictionary of hyperparameters forwarded
#   to the training function.
# - ``datasets``: a dictionary of Ray Datasets. Ray Train automatically
#   splits each dataset across workers.
# - ``scaling_config``: specifies the number of workers and whether to
#   use GPUs.
#
# Setting ``num_workers=8`` launches 8 parallel workers, one per GPU. Ray
# Train handles ``torch.distributed`` initialization, NCCL backend setup,
# and ``DistributedDataParallel`` wrapping behind the scenes. In the logs,
# you see each worker assigned a rank and device:
#
# .. code-block:: text
#
#    Started training worker group of size 8:
#
#    * (ip=10.0.176.183, pid=25636) world_rank=0, local_rank=0, node_rank=0
#    * (ip=10.0.176.183, pid=25637) world_rank=1, local_rank=1, node_rank=0
#    ...
#    Moving model to device: cuda:0
#    Wrapping provided model in DistributedDataParallel.
#
# ``batch_size_per_worker`` is the number of sequences each worker
# processes per gradient step. With 8 workers and a per-worker batch size
# of 16, the **effective global batch size** is 8 × 16 = 128 sequences,
# or 128 × 256 = 32,768 tokens per optimizer step.

NUM_WORKERS = 8  # One worker per GPU on this machine
NUM_EPOCHS = 5
BATCH_SIZE_PER_WORKER = 16
LR = 3e-4
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0

trainer = TorchTrainer(
    train_loop_per_worker=train_func_per_worker,
    train_loop_config={
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "epochs": NUM_EPOCHS,
        "batch_size_per_worker": BATCH_SIZE_PER_WORKER,
        "max_steps_per_epoch": 5 if SMOKE_TEST else None,
    },
    # Register the datasets,
    datasets={"train": train_ds, "validation": val_ds},
    scaling_config=ScalingConfig(
        num_workers=NUM_WORKERS,
        use_gpu=True,
    ),
)

result = trainer.fit()

###############################################################################
# Inspect results
# ---------------
#
# After training, the ``Result`` object contains the final metrics and
# checkpoint. ``result.metrics`` comes from the last
# ``ray.train.report()`` call. ``result.checkpoint`` is ``None`` here
# because this tutorial doesn't save checkpoints.

print("\nTraining finished!")

###############################################################################
# ``result.metrics`` contains the metrics dict from the last
# ``ray.train.report()`` call:
#
# .. code-block:: text
#
#    {'train_loss': 7.0646, 'val_loss': 7.6051, 'epoch': 4,
#     'epoch_time_sec': 12.34, 'epoch_tokens': 20480, 'tokens_per_sec': 1759.8}
#
# The per-worker logs show training loss, validation loss, and throughput
# metrics for each epoch. With random weights and only a few steps, expect
# a high loss (~10-11).

###############################################################################
# Checkpointing
# ~~~~~~~~~~~~~
#
# In production training, you can enable checkpointing to make
# your training jobs robust to unexpected failures. Checkpointing
# permits you to take advantage of Ray Train's fault tolerance mechanisms described in the
# `Fault tolerance`_ section.
#
# Ray Train offers several checkpointing optimizations. Asynchronous
# uploading enables you to continue training while checkpoints stream to remote storage
# in the background.
# Distributed checkpointing uploads shards from each worker in parallel, avoiding 
# a gather step into a single worker's memory that risks OOM errors for large models.
#
# For a full guide on checkpointing with Ray Train, see the
# `Ray Train checkpointing guide
# <https://docs.ray.io/en/latest/train/user-guides/checkpoints.html>`__.
#
# Scaling to a multi-node cluster
# -------------------------------
#
# The code above runs on a single 8-GPU machine. Scaling to a multi-node
# cluster requires only two changes:
#
# 1. **Increase ``num_workers``** to match the total number of GPUs in the cluster.
# 2. **Set a shared storage path** so that all nodes can access checkpoints.
#
# For example, to train on a cluster of 4 nodes with 8 GPUs each
# (32 GPUs total):
#
# .. code-block:: python
#
#    trainer = TorchTrainer(
#        train_loop_per_worker=train_func_per_worker,
#        train_loop_config={...},
#        datasets={"train": train_ds, "validation": val_ds},
#        scaling_config=ScalingConfig(
#            num_workers=32,  # 4 nodes x 8 GPUs
#            use_gpu=True,
#        ),
#        run_config=RunConfig(
#            # Shared storage accessible from all nodes
#            storage_path="s3://my-bucket/ray-checkpoints",
#            checkpoint_config=CheckpointConfig(num_to_keep=2),
#        ),
#    )
#
# Ray Train automatically:
#
# * Launches workers across all available nodes, bringing up new nodes if needed in an autoscaling Ray cluster.
# * Shards data across all workers.
#
# No changes to the training function are needed. The same
# ``train_func_per_worker`` runs identically whether on 1 GPU or 256 GPUs.
#
# This tutorial uses ``DistributedDataParallel`` (DDP), which replicates
# the full model on every GPU. For larger models that don't fit on a
# single GPU, you can switch to
# `FullyShardedDataParallel <https://docs.pytorch.org/docs/stable/fsdp.html>`__
# (FSDP) to shard parameters, gradients, and optimizer states across
# workers by setting ``prepare_model(parallel_strategy="fsdp")``.
#
# Heterogeneous clusters: separate data and training resources
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Because Ray Data and Ray Train are separate systems, they don't have to
# share the same machines. By default, Ray Data preprocessing and training
# workers all run on the same nodes. However, you can optionally add
# **CPU-only nodes** to your cluster and Ray Data automatically
# schedules preprocessing tasks on them, keeping your expensive GPU nodes
# free for training.
#
# This is useful when data preprocessing is a bottleneck. If you notice
# low GPU use because workers are waiting on data, you can add
# cheaper CPU-only nodes to the cluster and Ray Data scales out
# preprocessing to them.
#
# For more information, see `Configuring data ingest
# <https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html>`__.

###############################################################################
# .. _Fault tolerance:
#
# Fault tolerance
# ---------------
#
# Long-running distributed training jobs are vulnerable to hardware
# failures. These include hardware failures, network failures, or preemption.
# Without fault tolerance, any of these events can force you to restart
# training from scratch, wasting time and compute.
#
# Ray Train has features that handle these failures automatically. When a worker process
# crashes, Ray Train restarts it in place and resumes training. If an
# entire node goes down, Ray Train provisions a replacement and
# recovers from the most recent checkpoint so that only a small amount
# of work is lost. This makes it practical to interrupt training jobs and resume
# them later.
#
# To enable automatic failure recovery, configure ``FailureConfig`` in
# your ``RunConfig``. The ``max_failures`` parameter controls how many
# consecutive failures Ray Train tolerates before giving up:
#
# .. code-block:: python
#
#    from ray.train import FailureConfig
#
#    run_config = RunConfig(
#        storage_path="s3://my-bucket/ray-checkpoints",
#        failure_config=FailureConfig(max_failures=3),
#        checkpoint_config=CheckpointConfig(num_to_keep=2),
#    )
#
# For more details, see the `Ray Train fault tolerance guide
# <https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html>`__.

###############################################################################
# Monitor your training jobs
# --------------------------
#
# Monitoring is critical when running distributed training.
# The `Ray dashboard <https://docs.ray.io/en/latest/ray-observability/getting-started.html>`__
# displays real-time metrics including:
#
# - Training loss and validation metrics per epoch
# - GPU utilization and memory usage per worker
# - Data loading throughput
# - Worker status and error logs
#
# To view the dashboard, open the link printed in the logs after Ray
# initializes. Typically, this link is ``http://localhost:8265``.
#
# The dashboard lets you:
#
# - Monitor training progress across all workers
# - Inspect logs from individual workers
# - Identify data loading or communication bottlenecks
# - View resource use for CPU, GPU, and memory per worker
# - Debug failures with detailed error messages and stack traces
#
# For more information, see the `Ray Train monitoring
# documentation <https://docs.ray.io/en/latest/train/user-guides/monitoring-logging.html>`__.

###############################################################################
# Conclusion
# ----------
#
# In this tutorial, you:
#
# - Pre-trained a GPT-2 (~124M-parameter) language model using
#   Hugging Face Transformers and PyTorch.
# - Loaded and preprocessed the Wikitext-103 dataset using Ray Data
#   with distributed streaming.
# - Ran distributed training across 8 GPUs using Ray Train's
#   ``TorchTrainer`` with only minimal changes to a standard PyTorch
#   training loop.
# - Learned how to save and load distributed checkpoints for model
#   recovery.
# - Learned how to scale to multi-node clusters by changing
#   ``ScalingConfig`` and ``RunConfig``.
# - Learned how heterogeneous clusters let you run data preprocessing
#   on CPU nodes and training on GPU nodes for cost and performance
#   optimization.
# - Learned about Ray Train's **fault tolerance** mechanisms for
#   production training jobs.
# - Monitored training with the Ray dashboard.
#

###############################################################################
# Further reading
# ---------------
#
# - `Ray Train documentation <https://docs.ray.io/en/latest/train/train.html>`__
# - `Ray Data for training <https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html>`__
# - `Saving and loading checkpoints <https://docs.ray.io/en/latest/train/user-guides/checkpoints.html>`__
# - `PyTorch DistributedDataParallel <https://docs.pytorch.org/docs/stable/notes/ddp.html>`__
# - `Ray Train fault tolerance <https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html>`__
# - `Ray cluster setup <https://docs.ray.io/en/latest/cluster/getting-started.html>`__
