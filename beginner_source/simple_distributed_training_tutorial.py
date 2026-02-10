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
       * Distribute training across multiple GPUs with Ray Train.
       * Stream training data from Hugging Face datasets with Ray Data.
       * Save and load distributed checkpoints.
       * Scale from a single node to a multi-node cluster with minimal code changes.
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

###############################################################################

import math
import os
import tempfile

import numpy as np
import ray
import ray.train
import tiktoken
import torch
import torch.nn as nn
from datasets import load_dataset
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import GPT2Config, GPT2LMHeadModel

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

print(f"Dataset schema:\n{train_ds.schema()}")

###############################################################################
# The schema should look like this:
# ```text`
# Schema: Column  Type
# ------  ----
# text    string
# ```
#
# This means that the dataset has one column called "text" and it is a string.
#
# Inspect raw data
# ~~~~~~~~~~~~~~~~
#
# Use ``take(n)`` to fetch a small number of rows for inspection.
# Each row is a dictionary with the column names as keys.
print("--- Raw data sample (train_ds.take(2)) ---")
sample = train_ds.take(2)
for i, row in enumerate(sample):
    text_preview = row["text"][:120] + "..." if len(row["text"]) > 120 else row["text"]
    print(f"  Row {i}: {text_preview!r}")

###############################################################################
# You'll see output like:
#
# .. code-block:: text
#
#    Row 0: ''
#    Row 1: ' = Valkyria Chronicles III = \n'
#
# The raw dataset evidently contains empty lines and short headers that would
# produce zero tokens after chunking. Use ``filter()`` to keep only rows
# with at least 20 words, which removes noise and avoids wasted work in
# downstream stages.

MIN_WORDS = 20
train_ds = train_ds.filter(lambda row: len(row["text"].split()) >= MIN_WORDS)
val_ds = val_ds.filter(lambda row: len(row["text"].split()) >= MIN_WORDS)

# DEBUG: limit dataset size for fast iteration
train_ds = train_ds.limit(100)
val_ds = val_ds.limit(100)

print("--- After filtering short rows (train_ds.take(2)) ---")
filtered_sample = train_ds.take(2)
for i, row in enumerate(filtered_sample):
    text_preview = row["text"][:120] + "..." if len(row["text"]) > 120 else row["text"]
    print(f"  Row {i}: {text_preview!r}")

###############################################################################
# After filtering, only substantive paragraphs remain:
#
# .. code-block:: text
#
#    Row 0: ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : ...'
#    Row 1: ' The game began development in 2010 , carrying over a large ...'
#
# Tokenize and chunk the data
# ----------------------------
#
# Language models consume fixed-length sequences of token IDs. The
# preprocessing step converts raw text into overlapping input/target pairs
# for next-token prediction.
#
# This tutorial uses ``tiktoken`` with the GPT-2 encoding (vocabulary size
# 50,257). ``tiktoken`` is a fast, standalone tokenizer that has no
# dependency on the Hugging Face ``transformers`` library.
#
# The ``tokenize_and_chunk`` function:
#
# 1. Tokenizes each batch of text.
# 2. Concatenates all tokens into a single stream.
# 3. Splits the stream into fixed-length blocks of ``block_size + 1``
#    tokens.
# 4. Returns ``input_ids`` (the first ``block_size`` tokens) and
#    ``labels`` (shifted by one position for next-token prediction).

BLOCK_SIZE = 256
VOCAB_SIZE = 50257

encoding = tiktoken.get_encoding("gpt2")


def tokenize_and_chunk(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Tokenize text and split into fixed-length chunks for language modeling."""
    # Tokenize all texts in the batch and concatenate
    all_tokens: list[int] = []
    for text in batch["text"]:
        if text.strip():  # skip empty lines
            all_tokens.extend(encoding.encode_ordinary(text))

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

train_ds = train_ds.map_batches(tokenize_and_chunk, batch_format="numpy")
val_ds = val_ds.map_batches(tokenize_and_chunk, batch_format="numpy")

###############################################################################
# Inspect the tokenized output with ``take(2)``:

print("--- Tokenized data sample (train_ds.take(2)) ---")
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
# Under the hood, Ray divides the data into **blocks** and dispatches them to
# workers. This block-based architecture enables **streaming execution**: as
# soon as a stage outputs a block, the next stage can begin processing it
# immediately without waiting for previous stages to finish the entire
# dataset. This means the ``map_batches`` tokenization above runs in a
# streaming pipeline with the training loop, so the full dataset never needs
# to fit in memory at once.
#
# When training starts, Ray Data logs the execution plan. For this tutorial
# it looks like:
#
# .. code-block:: text
#
#    Execution plan: InputDataBuffer[Input]
#        -> TaskPoolMapOperator[Filter]
#        -> TaskPoolMapOperator[MapBatches(tokenize_and_chunk)]
#        -> OutputSplitter[split(8, equal=True)]
#
# This tells you exactly how Ray Data will stream through filter, tokenize,
# and split the data across 8 workers.

###############################################################################
# Define the transformer model
# ----------------------------
#
# The model is a decoder-only transformer language model using Hugging Face's
# ``GPT2LMHeadModel``.
#
# The GPT-2 "small" architecture:
#
# * 12 transformer layers, 12 attention heads, 768 hidden size
# * ~124M parameters
# * Built-in causal attention masking and weight tying

MODEL_CONFIG = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=BLOCK_SIZE,
    n_embd=768,
    n_layer=12,
    n_head=12,
)


def create_model():
    """Create a fresh GPT-2 model from config (random weights)."""
    model = GPT2LMHeadModel(MODEL_CONFIG)
    model.loss_type = "ForCausalLM"
    return model


###############################################################################
# Verify the model size:

model = create_model()
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

###############################################################################
# You should see approximately **123.8M parameters** — the standard GPT-2
# "small" size.
#
# Quick smoke test: run a forward pass on CPU to verify the model produces the
# expected output shape before launching distributed training:

model.eval()
dummy_input = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
with torch.no_grad():
    out = model(dummy_input)
print(f"Smoke test passed — logits shape: {out.logits.shape}")
del model, dummy_input, out  # Free memory before distributed training



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
# 1. **``ray.train.get_dataset_shard("train")``** retrieves the
#    worker's portion of the data. Ray Data automatically splits the
#    dataset across all workers.
# 2. **``ray.train.torch.prepare_model(model)``** wraps the model in
#    ``DistributedDataParallel`` and moves it to the correct GPU.
# 3. **``shard.iter_torch_batches(batch_size=...)``** returns an iterator
#    of ``dict[str, torch.Tensor]`` batches, with tensors automatically
#    placed on the worker's GPU.
# 4. **``ray.train.report(metrics, checkpoint=...)``** reports metrics
#    to the driver and optionally saves a checkpoint.

def train_func_per_worker(config: dict):
    """Training function executed by each distributed worker."""
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    max_steps_per_epoch = config.get("max_steps_per_epoch")  # DEBUG: cap steps

    # --- Data -----------------------------------------------------------
    # Each worker gets an automatic shard of the dataset.
    train_data_shard = ray.train.get_dataset_shard("train")
    val_data_shard = ray.train.get_dataset_shard("validation")

    # --- Model ----------------------------------------------------------
    model = create_model()
    # prepare_model wraps the model in DistributedDataParallel and places
    # it on the correct device.
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # --- Training loop --------------------------------------------------
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        # iter_torch_batches returns dicts of tensors already on the GPU.
        for batch in train_data_shard.iter_torch_batches(
            batch_size=batch_size, dtypes=torch.long
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

            if max_steps_per_epoch and train_batches >= max_steps_per_epoch:
                break  # DEBUG: early stop

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # --- Validation -------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_data_shard.iter_torch_batches(
                batch_size=batch_size, dtypes=torch.long
            ):
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss
                val_loss_sum += loss.item()
                val_batches += 1

                if max_steps_per_epoch and val_batches >= max_steps_per_epoch:
                    break  # DEBUG: early stop

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_perplexity = math.exp(min(avg_val_loss, 20))  # cap to avoid overflow

        # --- Report metrics -----------------------------------------------
        ray.train.report(
            metrics={
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_perplexity": val_perplexity,
                "epoch": epoch,
            },
            checkpoint=None,  # If we were checkpointing, we'd pass checkpoint to Ray Train here
        )

        if ray.train.get_context().get_world_rank() == 0:
            print(
                f"Epoch {epoch}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}, "
                f"val_perplexity={val_perplexity:.2f}"
            )



###############################################################################
# Configure and launch distributed training
# ------------------------------------------
#
# The ``TorchTrainer`` brings everything together. It accepts:
#
# * **``train_func_per_worker``**: the function each worker executes.
# * **``train_loop_config``**: a dictionary of hyperparameters forwarded
#   to the training function.
# * **``datasets``**: a dictionary of Ray Datasets. Ray Train automatically
#   splits each dataset across workers.
# * **``scaling_config``**: specifies the number of workers and whether to
#   use GPUs.
#
# Setting ``num_workers=8`` launches 8 parallel workers, one per GPU. Ray
# Train handles ``torch.distributed`` initialization, NCCL backend setup,
# and ``DistributedDataParallel`` wrapping behind the scenes. In the logs
# you will see each worker being assigned a rank and device:
#
# .. code-block:: text
#
#    Started training worker group of size 8:
#    - (ip=10.0.176.183, pid=25636) world_rank=0, local_rank=0, node_rank=0
#    - (ip=10.0.176.183, pid=25637) world_rank=1, local_rank=1, node_rank=0
#    ...
#    Moving model to device: cuda:0
#    Wrapping provided model in DistributedDataParallel.

NUM_WORKERS = 8  # One worker per GPU on this machine
NUM_EPOCHS = 1  # DEBUG: reduced from 2
BATCH_SIZE_PER_WORKER = 16

trainer = TorchTrainer(
    train_loop_per_worker=train_func_per_worker,
    train_loop_config={
        "lr": 3e-4,
        "epochs": NUM_EPOCHS,
        "batch_size_per_worker": BATCH_SIZE_PER_WORKER,
        "max_steps_per_epoch": 5,  # DEBUG: cap at 5 steps for fast iteration
    },
    datasets={"train": train_ds, "validation": val_ds},
    scaling_config=ScalingConfig(
        num_workers=NUM_WORKERS,
        use_gpu=True,
    ),
    # run_config=RunConfig(),
)

result = trainer.fit()

###############################################################################
# Inspect results
# ---------------
#
# After training, the ``Result`` object contains the final metrics reported
# by the workers.

print(f"\nTraining finished!")
print(f"Final metrics: {result.metrics}")

###############################################################################
# The logs from each worker show training and validation metrics per epoch.
# With random weights and only a few steps, expect a high loss (~10–11)
# and perplexity in the tens of thousands — this is normal.
#
# .. code-block:: text
#
#    Epoch 0: train_loss=10.9492, val_loss=10.0157, val_perplexity=22374.06
#
# In a real training run with more epochs and the full dataset, you would
# see these values steadily decrease.

###############################################################################
# Checkpointing
# ~~~~~~~~~~~~~
#
# In a production training run you would enable checkpointing so that
# training can resume from the last saved state after a failure. This
# requires a **shared storage path** (e.g. an S3 bucket or NFS mount)
# accessible from all nodes:
#
# .. code-block:: python
#
#    trainer = TorchTrainer(
#        ...,
#        run_config=RunConfig(
#            storage_path="s3://my-bucket/ray-checkpoints",
#            checkpoint_config=CheckpointConfig(num_to_keep=2),
#        ),
#    )
#
# Inside the training function, save a checkpoint with
# ``ray.train.report()``:
#
# .. code-block:: python
#
#    with tempfile.TemporaryDirectory() as tmp_dir:
#        model.module.save_pretrained(tmp_dir)  # .module unwraps DDP
#        checkpoint = ray.train.Checkpoint.from_directory(tmp_dir)
#        ray.train.report(metrics={...}, checkpoint=checkpoint)
#
# Scaling to a multi-node cluster
# -------------------------------
#
# The code above runs on a single 8-GPU machine. Scaling to a multi-node
# cluster requires only two changes:
#
# 1. **Increase ``num_workers``** to match the total number of GPUs across
#    all nodes.
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
# * Launches workers across all available nodes.
# * Initializes ``torch.distributed`` with the NCCL backend.
# * Configures ``DistributedDataParallel`` across nodes.
# * Shards data across all workers.
#
# No changes to the training function are needed. The same
# ``train_func_per_worker`` runs identically whether on 1 GPU or 256 GPUs.

###############################################################################
# Fault tolerance
# ---------------
#
# Long-running distributed training jobs are vulnerable to hardware
# failures. Ray Train provides fault tolerance so that training can
# recover from failures without restarting from scratch.
#
# Ray Train's fault tolerance mechanisms include:
#
# * **Worker restart**: If a worker process crashes, Ray Train
#   automatically restarts it and resumes training from the last
#   checkpoint.
# * **Checkpoint recovery**: Ray Train saves checkpoints to persistent
#   storage. When recovering from a failure, training resumes from the
#   latest checkpoint rather than starting over.
# * **Node failure handling**: If an entire node goes down, Ray
#   redistributes work to surviving nodes and replaces the failed node
#   when new resources become available.
#
# To enable automatic failure recovery, configure ``FailureConfig`` in your ``RunConfig``:
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

###############################################################################
# Monitor your training jobs
# --------------------------
#
# Monitoring is critical when running distributed training.
# The `Ray dashboard <https://docs.ray.io/en/latest/ray-observability/getting-started.html>`__
# displays real-time metrics including:
#
# * Training loss and validation metrics per epoch
# * GPU utilization and memory usage per worker
# * Data loading throughput
# * Worker status and error logs
#
# To view the dashboard, open the link printed in the logs after Ray
# initializes. Typically, this link is ``http://localhost:8265``.
#
# The dashboard lets you:
#
# * Monitor training progress across all workers
# * Inspect logs from individual workers
# * Identify data loading or communication bottlenecks
# * View resource utilization for CPU, GPU, and memory per worker
# * Debug failures with detailed error messages and stack traces
#
# For more information, see the `Ray Train monitoring
# documentation <https://docs.ray.io/en/latest/train/user-guides/monitoring-logging.html>`__.

###############################################################################
# Conclusion
# ----------
#
# In this tutorial, you:
#
# * Pre-trained a GPT-2 (~124M-parameter) language model using
#   Hugging Face Transformers and PyTorch.
# * Loaded and preprocessed the Wikitext-103 dataset using Ray Data
#   with distributed streaming.
# * Distributed training across 8 GPUs using Ray Train's
#   ``TorchTrainer`` with only minimal changes to a standard PyTorch
#   training loop.
# * Saved and loaded distributed checkpoints for model recovery.
# * Learned how to scale to multi-node clusters by changing
#   ``ScalingConfig`` and ``RunConfig``.
# * Learned about Ray Train's **fault tolerance** mechanisms for
#   production training jobs.
# * Monitored training with the Ray dashboard.
#
# Ray Train handles the complexity of distributed systems, gradient
# synchronization, and resource allocation so that you can focus on
# your model and data.

###############################################################################
# Further reading
# ---------------
#
# * `Ray Train documentation <https://docs.ray.io/en/latest/train/train.html>`__
# * `Ray Data for training <https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html>`__
# * `PyTorch DistributedDataParallel <https://pytorch.org/docs/stable/notes/ddp.html>`__
# * `Ray Train fault tolerance <https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html>`__
# * `Ray cluster setup <https://docs.ray.io/en/latest/cluster/getting-started.html>`__
