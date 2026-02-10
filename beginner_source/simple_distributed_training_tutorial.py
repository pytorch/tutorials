"""
Distributed training at scale with PyTorch and Ray Train
=========================================================

**Author:** `Ricardo Decal <https://github.com/crypdick>`__

This tutorial shows how to distribute PyTorch training across multiple GPUs
using Ray Train and Ray Data for scalable, production-ready model training.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` You will learn how to:
       :class-card: card-prerequisites

       * Pre-train a ~117M-parameter decoder-only transformer language model
         using PyTorch.
       * Distribute training across multiple GPUs with Ray Train.
       * Stream training data from Hugging Face datasets with Ray Data.
       * Save and load distributed checkpoints.
       * Scale from a single node to a multi-node cluster with minimal code changes.
       * Monitor training with the Ray dashboard.

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.9+.
       * Ray Train (``ray[train]``) v2.52.1+.
       * ``tiktoken`` and ``datasets`` (Hugging Face).
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

To install the dependencies, run ``pip install "ray[train]" torch tiktoken datasets``.

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
print(train_ds)

###############################################################################
# Ray divides the data into **blocks** and dispatches them to workers.
# This block-based architecture enables **streaming execution**: as soon as
# a stage outputs a block, the next stage can begin processing it
# immediately without waiting for previous stages to finish the entire
# dataset.
# TODO: move the above text elsewhere; we should discuss .schema() lets you inspect the data

print(train_ds.schema())

# TODO discuss the output of schema.

###############################################################################
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
        return {"input_ids": np.array([], dtype=np.int64).reshape(0, BLOCK_SIZE),
                "labels": np.array([], dtype=np.int64).reshape(0, BLOCK_SIZE)}

    tokens_array = np.array(all_tokens, dtype=np.int64).reshape(num_chunks, chunk_len)
    return {
        "input_ids": tokens_array[:, :-1],
        "labels": tokens_array[:, 1:],
    }



###############################################################################
# Apply the tokenization with ``map_batches()``. This operation is **lazy**,
# meaning that Ray Data defers execution until a downstream consumer requests the
# results. Lazy execution lets Ray optimize the entire pipeline before any
# work begins.

train_ds = train_ds.map_batches(tokenize_and_chunk, batch_format="numpy")
val_ds = val_ds.map_batches(tokenize_and_chunk, batch_format="numpy")
print(train_ds.schema())


###############################################################################
# Define the transformer model
# ----------------------------
#
# The model is a decoder-only transformer language model, similar to GPT-2,
# built entirely from standard PyTorch modules. It has approximately 117
# million parameters.
#
# The architecture:
#
# * **Token embedding** maps token IDs to dense vectors.
# * **Positional embedding** encodes position information.
# * **Transformer encoder** with a causal (triangular) attention mask ensures
#   that each token can only attend to preceding tokens. Note: PyTorch's
#   ``TransformerEncoder`` with a causal mask is functionally equivalent to a
#   decoder-only transformer.
# * **Output head** projects the hidden states back to the vocabulary.

class TransformerLM(nn.Module):
    """Decoder-only transformer language model (~117M parameters)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        max_seq_len: int = BLOCK_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm architecture for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output head
        self.output_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Create causal mask so each token only attends to previous tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.output_head(x)
        return logits



###############################################################################
# Verify the model size:

model = TransformerLM()
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
del model  # Free memory before distributed training


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

    # --- Data -----------------------------------------------------------
    # Each worker gets an automatic shard of the dataset.
    train_data_shard = ray.train.get_dataset_shard("train")
    val_data_shard = ray.train.get_dataset_shard("validation")

    # --- Model ----------------------------------------------------------
    model = TransformerLM()
    # prepare_model wraps the model in DistributedDataParallel and places
    # it on the correct device.
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()

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

            logits = model(input_ids)
            # Flatten for cross-entropy: (batch * seq_len, vocab_size) vs (batch * seq_len,)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

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

                logits = model(input_ids)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_perplexity = math.exp(min(avg_val_loss, 20))  # cap to avoid overflow

        # --- Checkpointing ----------------------------------------------
        # Save a checkpoint at the end of each epoch.
        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.save(
                model.module.state_dict(),  # .module unwraps DDP
                os.path.join(tmp_dir, "model.pt"),
            )
            checkpoint = ray.train.Checkpoint.from_directory(tmp_dir)

            ray.train.report(
                metrics={
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_perplexity": val_perplexity,
                    "epoch": epoch,
                },
                checkpoint=checkpoint,
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
# and ``DistributedDataParallel`` wrapping behind the scenes.

NUM_WORKERS = 8  # One worker per GPU on this machine
NUM_EPOCHS = 2
BATCH_SIZE_PER_WORKER = 16

trainer = TorchTrainer(
    train_loop_per_worker=train_func_per_worker,
    train_loop_config={
        "lr": 3e-4,
        "epochs": NUM_EPOCHS,
        "batch_size_per_worker": BATCH_SIZE_PER_WORKER,
    },
    datasets={"train": train_ds, "validation": val_ds},
    scaling_config=ScalingConfig(
        num_workers=NUM_WORKERS,
        use_gpu=True,
    ),
    run_config=RunConfig(
        # Keep the best 2 checkpoints by validation loss
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
        ),
    ),
)

result = trainer.fit()

###############################################################################
# Inspect results
# ---------------
#
# After training, the ``Result`` object contains the final metrics and
# the path to the best checkpoint.

print(f"\nTraining finished!")
print(f"Final metrics: {result.metrics}")
print(f"Best checkpoint path: {result.checkpoint.path}")


###############################################################################
# Load a checkpoint and generate text
# ------------------------------------
#
# As a sanity check, load the trained model from the checkpoint and
# generate a few tokens using greedy decoding:

def generate(model: TransformerLM, prompt_tokens: list[int], max_new_tokens: int = 50) -> list[int]:
    """Generate tokens autoregressively using greedy decoding."""
    model.eval()
    device = next(model.parameters()).device
    tokens = prompt_tokens[:]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Only use the last BLOCK_SIZE tokens if the sequence is too long
            input_ids = torch.tensor([tokens[-BLOCK_SIZE:]], device=device)
            logits = model(input_ids)
            next_token = logits[0, -1, :].argmax().item()
            tokens.append(next_token)

    return tokens


# Load the model from the checkpoint
checkpoint_path = os.path.join(result.checkpoint.path, "model.pt")
trained_model = TransformerLM()
trained_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = trained_model.to(device)

# Generate text from a prompt
prompt = "The history of science"
prompt_tokens = encoding.encode_ordinary(prompt)
generated_tokens = generate(trained_model, prompt_tokens, max_new_tokens=50)
generated_text = encoding.decode(generated_tokens)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")

###############################################################################
# .. note::
#
#    With only 2 epochs of training on Wikitext-103, the generated text
#    will be mostly incoherent. This is expected for a tutorial that
#    prioritizes demonstrating the distributed training workflow over
#    producing a fully-trained model. In a real pre-training run, you would
#    train for many more epochs with a learning rate schedule and a larger
#    dataset.
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
# * Built a ~117M-parameter decoder-only transformer language model
#   using pure PyTorch.
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
