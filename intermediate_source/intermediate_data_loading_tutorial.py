# -*- coding: utf-8 -*-
"""
Data Loading Optimization in PyTorch
==============================================

**Authors**: `Divyansh Khanna <https://github.com/divyanshk>`_, `Ramanish Singh <https://github.com/ramanishsingh>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to optimize DataLoader configuration for maximum throughput
       * Best practices for ``batch_size``, ``num_workers``, and ``pin_memory``
       * Advanced techniques for overlapping data transfers with GPU compute
       * Configuring shared memory strategies and handling ``/dev/shm`` issues

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.0+
       * Basic understanding of PyTorch DataLoader
       * (Optional) A CUDA-capable GPU for GPU-specific optimizations

Introduction
------------

Data loading is often a critical bottleneck in deep learning pipelines. While
GPUs can process batches extremely quickly, inefficient data loading can leave
expensive hardware idle, waiting for the next batch of data. This tutorial
covers best practices and some techniques for optimizing your data loading configuration to
maximize training throughput.

We'll explore the key parameters of PyTorch's DataLoader and provide practical
guidance on tuning them for your specific workload. Rather than showing each
optimization in isolation, we'll build up from a baseline training loop and
progressively apply optimizations, measuring the cumulative speedup at each
step.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a fixed seed for reproducibility
torch.manual_seed(42)

######################################################################
# Creating a Sample Dataset
# -------------------------
#
# First, let's create a simple dataset that simulates expensive
# transformations. This will help us demonstrate the impact of
# various DataLoader configurations.


class SyntheticDataset(Dataset):
    """A synthetic dataset that simulates expensive data transformations."""

    def __init__(self, size=10000, feature_dim=224, transform_delay=0.001):
        self.size = size
        self.feature_dim = feature_dim
        self.transform_delay = transform_delay

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate data lazily to avoid pre-allocating large tensors
        data = torch.randn(3, self.feature_dim, self.feature_dim)
        label = torch.randint(0, 10, (1,)).item()
        if self.transform_delay > 0:
            time.sleep(self.transform_delay)
        return data, label


class SyntheticDatasetBatched(Dataset):
    """Same as SyntheticDataset but with __getitems__ for batched fetching."""

    def __init__(self, size=10000, feature_dim=224, transform_delay=0.001):
        self.size = size
        self.feature_dim = feature_dim
        self.transform_delay = transform_delay

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = torch.randn(3, self.feature_dim, self.feature_dim)
        label = torch.randint(0, 10, (1,)).item()
        if self.transform_delay > 0:
            time.sleep(self.transform_delay)
        return data, label

    def __getitems__(self, indices):
        """Fetch multiple items at once — enables vectorized generation.

        Instead of N individual __getitem__ calls (each with its own
        overhead), this generates the entire batch in one shot using
        vectorized tensor operations.
        """
        n = len(indices)
        # Vectorized generation: one call instead of N individual ones
        data = torch.randn(n, 3, self.feature_dim, self.feature_dim)
        labels = torch.randint(0, 10, (n,))
        # Simulate batch-level I/O: one sleep for the whole batch,
        # not one per sample (e.g., one DB query for N rows)
        if self.transform_delay > 0:
            time.sleep(self.transform_delay)
        return [(data[i], labels[i].item()) for i in range(n)]


######################################################################
# Shared Training Infrastructure
# ------------------------------
#
# To measure the real-world impact of each optimization, we define a
# reusable training loop that accepts a DataLoader and returns timing
# and loss. This avoids duplicating the training loop for every
# benchmark.
#
# We use a **small dataset (500 samples)** with a **high transform
# delay (5ms)** to ensure the pipeline remains data-bound throughout.
# The small dataset means short epochs (16 batches each), so we run
# many epochs — making persistent_workers' benefit visible across
# epoch boundaries.

benchmark_dataset = SyntheticDataset(size=512, feature_dim=224, transform_delay=0.005)


class SmallTransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)  # (B, 64, 7, 7)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, 64)
        x = self.transformer(x)  # (B, 49, 64)
        x = x.mean(dim=1)  # (B, 64)
        return self.classifier(x)


def create_model():
    """Create a conv+transformer model for benchmarking."""
    return SmallTransformerModel().to(device)


def train_and_benchmark(loader, max_batches=160, epochs=10, prefetch_device=None):
    """Train a model over multiple epochs and return elapsed time and average loss.

    Running multiple epochs (10) with a small dataset ensures many epoch
    boundaries, making persistent_workers' startup savings visible.

    Args:
        loader: A DataLoader to iterate over.
        max_batches: Maximum total number of batches to process across all epochs.
        epochs: Number of epochs to iterate (re-iterates the loader each epoch).
        prefetch_device: If set, wraps the loader in a DataPrefetcher each epoch
            for overlapping H2D transfers. Data arrives already on device.

    Returns:
        Tuple of (elapsed_seconds, average_loss).
    """
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        if prefetch_device is not None:
            data_iter = DataPrefetcher(loader, prefetch_device)
        else:
            data_iter = loader

        for data, labels in data_iter:
            if prefetch_device is None:
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            output = model(data)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches >= max_batches:
                break
        if num_batches >= max_batches:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    return elapsed, total_loss / num_batches


######################################################################
# Baseline Training Loop
# ----------------------
#
# Our starting point: a simple DataLoader with no multiprocessing,
# no pinned memory, and default settings. This establishes the
# performance floor we'll improve upon.

baseline_loader = DataLoader(
    benchmark_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

print("\n=== Progressive Optimization Results ===")
print("\nBaseline (num_workers=0, pin_memory=False):")
baseline_time, baseline_loss = train_and_benchmark(baseline_loader)
print(f"  Time: {baseline_time:.4f}s | Loss: {baseline_loss:.4f}")
prev_time = baseline_time

######################################################################
# Batch Size Optimization
# -----------------------
#
# The ``batch_size`` parameter controls how many samples are processed
# together. Choosing the right batch size involves balancing several factors:
#
# **Memory Considerations:**
#
# - Larger batch sizes require more GPU memory for storing inputs,
#   activations, and gradients
# - Out-of-memory (OOM) errors are common with large batch sizes
# - Moderate batch sizes (32-128) often provide the best balance
#
# **Training Dynamics:**
#
# - Batch size changes affect the effective learning rate, typically requiring tuning
# - Larger batches provide more stable gradient estimates but may
#   generalize differently
#
# .. note::
#    When changing batch size, remember to tune your optimizer parameters,
#    especially the learning rate schedule, unless you're doing inference
#
# Since batch size is model-dependent (not a "just add it" optimization),
# we benchmark it in isolation rather than folding it into the progressive
# optimization chain.

# Example: Testing different batch sizes
batch_dataset = SyntheticDataset(size=1000, transform_delay=0)


def benchmark_batch_size(batch_size, num_batches=10):
    """Benchmark data loading with a specific batch size."""
    loader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=True)
    start = time.perf_counter()
    for i, (data, labels) in enumerate(loader):
        if i >= num_batches:
            break
        data = data.to(device, non_blocking=True)
        _ = data.sum()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


# Benchmark different batch sizes
print("\nBatch size comparison (isolated benchmark):")
for bs in [16, 32, 64, 128]:
    elapsed = benchmark_batch_size(bs)
    print(f"  Batch size {bs:3d}: {elapsed:.4f}s for 10 batches")

######################################################################
# Number of Workers (``num_workers``)
# -----------------------------------
#
# The ``num_workers`` parameter controls how many subprocesses are used
# for data loading. This is crucial for parallelizing expensive data
# transformations.
#
# **How it works:**
#
# - Each worker maintains a queue of batches (controlled by ``prefetch_factor``)
# - Workers prepare batches in parallel and transfer them to the main process
# - If ``in_order=True`` (default), batches are returned in order
#
# **When to increase ``num_workers``:**
#
# - When transforms are computationally expensive (augmentations, decoding)
# - When data is loaded from slow storage (network drives, HDD)
# - When you observe GPU idle time due to data loading
#
# **When ``num_workers=0`` might be faster:**
#
# - When transforms are cheap (simple tensor operations)
# - When data is already in memory
# - The overhead of inter-process communication (IPC) exceeds the
#   parallelization benefits
#
# .. note::
#    Finding the optimal ``num_workers`` requires tuning: increase workers
#    until throughput plateaus. Too many workers waste CPU
#    memory (each worker holds its own copy of the dataset object and
#    prefetched batches) and can cause ``/dev/shm`` exhaustion. A good
#    starting point is 2-4 workers per GPU; profile with different values
#    to find the sweet spot for your workload.
#
# Let's add ``num_workers=4`` and ``prefetch_factor=2`` to our training
# loop and measure the improvement:

workers_loader = DataLoader(
    benchmark_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=False,
)

print("\n+ num_workers=4, prefetch_factor=2:")
workers_time, workers_loss = train_and_benchmark(workers_loader)
print(f"  Time: {workers_time:.4f}s | Loss: {workers_loss:.4f}")
print(
    f"  Speedup vs baseline: {baseline_time / workers_time:.2f}x | vs previous: {prev_time / workers_time:.2f}x"
)
prev_time = workers_time

######################################################################
# Understanding ``pin_memory``
# ----------------------------
#
# The ``pin_memory`` parameter enables faster CPU-to-GPU data transfers
# by using page-locked (pinned) memory.
#
# **How pinned memory works:**
#
# - Pinned memory cannot be swapped to disk by the OS
# - This enables faster DMA (Direct Memory Access) transfers to GPU
# - The CPU-to-GPU transfer can happen asynchronously
#
# **Best practices:**
#
# 1. Use ``pin_memory=True`` in the DataLoader (recommended approach)
# 2. Combine with ``non_blocking=True`` when moving data to GPU
# 3. Avoid manually calling ``tensor.pin_memory()`` followed by
#    ``.to(device, non_blocking=True)`` - this is slower because
#    ``pin_memory()`` is blocking
#
# **The safe pattern:**
#
# .. code-block:: python
#
#     # Recommended: Let DataLoader handle pinning
#     loader = DataLoader(dataset, pin_memory=True)
#     for data, labels in loader:
#         data = data.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)
#
# .. seealso::
#    For more details, see the
#    `pin_memory tutorial <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`_
#
# Let's add ``pin_memory=True`` to our configuration:

pinmem_loader = DataLoader(
    benchmark_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=torch.cuda.is_available(),
)

if torch.cuda.is_available():
    print("\n+ pin_memory=True:")
    pinmem_time, pinmem_loss = train_and_benchmark(pinmem_loader)
    print(f"  Time: {pinmem_time:.4f}s | Loss: {pinmem_loss:.4f}")
    print(
        f"  Speedup vs baseline: {baseline_time / pinmem_time:.2f}x | vs previous: {prev_time / pinmem_time:.2f}x"
    )
    print(
        "  (pin_memory benefit is modest here because CPU transform time dominates H2D transfer)"
    )
    prev_time = pinmem_time
else:
    print("\n+ pin_memory: skipped (CUDA not available)")
    pinmem_time = workers_time

######################################################################
# Persistent Workers
# ------------------
#
# By default, worker processes are shut down and restarted between
# epochs. This incurs startup overhead (importing modules, forking
# processes, re-initializing datasets) on every epoch boundary.
#
# Setting ``persistent_workers=True`` keeps the workers alive across
# epochs, eliminating this repeated startup cost.
#
# **When it helps most:**
#
# - Training for many epochs on smaller datasets
# - When dataset ``__init__`` is expensive (e.g., loading metadata)
# - When combined with high ``num_workers``
#
# Let's compare with and without persistent workers over multiple epochs:

non_persistent_loader = DataLoader(
    benchmark_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

persistent_loader = DataLoader(
    benchmark_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)

print("\n+ persistent_workers=True (10 epochs):")
non_persistent_time, _ = train_and_benchmark(non_persistent_loader)
persistent_time, persistent_loss = train_and_benchmark(persistent_loader)
print(f"  Without persistent_workers: {non_persistent_time:.4f}s")
print(f"  With persistent_workers:    {persistent_time:.4f}s")
print(
    f"  Speedup vs baseline: {baseline_time / persistent_time:.2f}x | vs previous: {prev_time / persistent_time:.2f}x"
)
prev_time = persistent_time

######################################################################
# Overlapping H2D Transfer with GPU Compute
# ---------------------------------------------------
#
# For maximum throughput, you can overlap Host-to-Device (H2D) data
# transfers with GPU computation. This ensures the GPU is never idle
# waiting for data.
#
# The idea is to prefetch the next batch to GPU while the current batch
# is being processed.
#
# .. note::
#    The DataPrefetcher shows its greatest benefit when H2D transfer
#    time overlaps meaningfully with GPU compute. If data loading is
#    already fast, the stream synchronization overhead may exceed the benefit.


class DataPrefetcher:
    """Prefetches data to GPU while previous batch is being processed."""

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.next_data = None
        self.next_labels = None
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_labels = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_labels = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_data = self.next_data.to(self.device, non_blocking=True)
                self.next_labels = self.next_labels.to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        data = self.next_data
        labels = self.next_labels

        if data is None:
            raise StopIteration

        # Ensure tensors are ready
        if self.stream is not None:
            data.record_stream(torch.cuda.current_stream())
            labels.record_stream(torch.cuda.current_stream())

        self.preload()
        return data, labels


# Integrate prefetcher into the training loop.
if torch.cuda.is_available():
    print("\n+ DataPrefetcher (overlapping H2D transfer):")
    prefetch_time, prefetch_loss = train_and_benchmark(
        persistent_loader, prefetch_device=device
    )
    print(f"  Time: {prefetch_time:.4f}s | Loss: {prefetch_loss:.4f}")
    print(
        f"  Speedup vs baseline: {baseline_time / prefetch_time:.2f}x | vs previous: {prev_time / prefetch_time:.2f}x"
    )
    prev_time = prefetch_time
else:
    print("\n+ DataPrefetcher: skipped (CUDA not available)")
    prefetch_time = persistent_time

######################################################################
# Dataset-Level Optimization: ``__getitems__``
# --------------------------------------------
#
# Beyond tuning DataLoader parameters, you can optimize the dataset
# itself. PyTorch's DataLoader supports a batched fetching protocol via
# ``__getitems__``: if your dataset defines this method, the fetcher
# calls it once with a list of indices instead of calling ``__getitem__``
# repeatedly for each sample.
#
# **How it works:**
#
# - The default fetcher does: ``[dataset[idx] for idx in batch_indices]``
# - With ``__getitems__``: ``dataset.__getitems__(batch_indices)``
#
# **When this helps:**
#
# - When per-sample overhead is significant (e.g., opening connections,
#   parsing headers, acquiring locks)
# - When data can be fetched in bulk more efficiently (e.g., one SQL query
#   for N rows instead of N queries, or vectorized tensor generation)
# - When the transform has a fixed setup cost that can be amortized
#   across the batch
#
# **Expected signature:**
#
# .. code-block:: python
#
#     def __getitems__(self, indices: list[int]) -> list:
#         # Fetch all items at once and return as a list
#         ...
#
# Our ``SyntheticDatasetBatched`` implements ``__getitems__`` to generate
# the entire batch in one vectorized call (with a single amortized delay)
# rather than N individual calls each with their own delay.
# Let's add this to our cumulative configuration:

benchmark_dataset_batched = SyntheticDatasetBatched(
    size=512, feature_dim=224, transform_delay=0.005
)

batched_loader = DataLoader(
    benchmark_dataset_batched,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)

print("\n+ __getitems__ (batched dataset fetching):")
batched_time, batched_loss = train_and_benchmark(batched_loader)
print(f"  Time: {batched_time:.4f}s | Loss: {batched_loss:.4f}")
print(
    f"  Speedup vs baseline: {baseline_time / batched_time:.2f}x | vs previous: {prev_time / batched_time:.2f}x"
)
prev_time = batched_time

######################################################################
# ``in_order`` parameter
# --------------------------
#
# By default (``in_order=True``), the DataLoader returns batches in
# the same order as the dataset indices. This requires caching batches
# that arrive out of order from workers.
#
# **When to consider ``in_order=False``:**
#
# - When you don't need deterministic ordering (e.g., not checkpointing)
# - When you observe training spikes due to batch caching
# - When maximizing throughput is more important than reproducibility
#
# .. note::
#    ``in_order=False`` might not increase average throughput, but it
#    can reduce variance and eliminate occasional slow batches caused
#    by head-of-line blocking when one worker is slower than others.

######################################################################
# Snapshot Frequency (``snapshot_every_n_steps``)
# -----------------------------------------------
#
# When using torchdata's StatefulDataLoader (for checkpointing), the
# ``snapshot_every_n_steps`` parameter controls how often the
# DataLoader state is saved.
#
# **Trade-offs:**
#
# - **Higher frequency (smaller n):** More overhead, but less data loss
#   on job failure
# - **Lower frequency (larger n):** Less overhead, but more replayed
#   samples on recovery
#
# Choose based on your fault tolerance requirements and the cost of
# reprocessing data.

######################################################################
# Shared Memory and ``set_sharing_strategy``
# ------------------------------------------
#
# When using multiprocessing with ``num_workers > 0``, PyTorch needs to
# transfer tensors between worker processes and the main process. The
# sharing strategy determines how this is done.
#
# **Available Strategies:**
#
# PyTorch provides two sharing strategies via
# ``torch.multiprocessing.set_sharing_strategy()``:
#
# 1. **file_descriptor** (default on most systems)
#
#    - Uses file descriptors to share memory
#    - Limited by system's open file descriptor limit (``ulimit -n``)
#    - More efficient for small tensors
#
# 2. **file_system**
#
#    - Uses shared memory files in ``/dev/shm``
#    - Not limited by file descriptor count
#    - Better for large numbers of tensors
#    - Low transform costs

######################################################################
# **How to Change the Strategy:**
#
# .. code-block:: python
#
#     import torch.multiprocessing as mp
#
#     # Switch to file_system strategy
#     # Must be called before creating any DataLoader workers
#     mp.set_sharing_strategy('file_system')
#
# **Choosing the Right Strategy:**
#
# +-------------------+---------------------------+---------------------------+
# | Scenario          | Recommended Strategy      | Reason                    |
# +===================+===========================+===========================+
# | Many small tensors| file_descriptor (default) | Lower overhead per tensor |
# +-------------------+---------------------------+---------------------------+
# | Few large tensors | file_system               | Avoids fd limits          |
# +-------------------+---------------------------+---------------------------+
# | High num_workers  | file_system               | Avoids fd exhaustion      |
# +-------------------+---------------------------+---------------------------+
#
# .. warning::
#    ``set_sharing_strategy()`` must be called **before** creating any
#    DataLoader with ``num_workers > 0``. Changing it afterward has no
#    effect on existing workers.

######################################################################
# Handling Insufficient Shared Memory (``/dev/shm``)
# --------------------------------------------------
#
# When using ``num_workers > 0``, PyTorch uses shared memory (``/dev/shm``)
# to efficiently pass data between worker processes and the main process.
# If you encounter errors like:
#
# .. code-block:: text
#
#     RuntimeError: unable to open shared memory object </torch_xxx>
#     ERROR: Unexpected bus error encountered in worker
#
# This typically means you've exhausted the shared memory allocation.
#
# **Solutions:**
#
# **1. Increase /dev/shm size (if you can)**
#
# **2. Reduce memory pressure from DataLoader:**
#
# .. code-block:: python
#
#     # Reduce number of workers
#     DataLoader(dataset, num_workers=2)  # Instead of 8+
#
#     # Reduce prefetch factor
#     DataLoader(dataset, num_workers=4, prefetch_factor=1)  # Instead of 2
#
#     # Use smaller batch sizes
#     DataLoader(dataset, batch_size=16)  # Smaller batches = less shm
#
# **3. Switch sharing strategy:**
#
# .. code-block:: python
#
#     import torch.multiprocessing as mp
#     mp.set_sharing_strategy('file_system')
#
# **4. Clean up leaked shared memory:**
#
# .. code-block:: bash
#
#     # List shared memory segments
#     ls -la /dev/shm/
#
#     # Remove orphaned PyTorch segments (be careful!)
#     rm /dev/shm/torch_*
#
# .. note::
#    Shared memory leaks can occur if worker processes crash without
#    proper cleanup.
#

######################################################################
# Final Summary
# -------------
#
# Here's the cumulative effect of each optimization we applied to
# our training loop. Each row includes all optimizations from previous
# rows:
#
# .. rst-class:: summary-table
#
# .. list-table::
#    :header-rows: 1
#    :widths: 55 20 20
#
#    * - Configuration
#      - vs Baseline
#      - vs Previous
#    * - Baseline (num_workers=0, no pinning)
#      - 1.00x
#      - —
#    * - \+ num_workers=4, prefetch_factor=2
#      - ~2.7x
#      - ~2.7x
#    * - \+ pin_memory=True
#      - ~2.8x
#      - ~1.0x
#    * - \+ persistent_workers=True
#      - ~3.7x
#      - ~1.3x
#    * - \+ DataPrefetcher (H2D overlap)
#      - ~3.6x
#      - ~1.0x
#    * - \+ __getitems__ (batched fetching)
#      - ~10x
#      - ~2.9x
#
# .. note::
#    These results are based on our benchmark dataset.
#    Actual speedups will vary depending on your specific
#    workload, hardware, dataset size, and transform complexity.

######################################################################
# Summary and Best Practices
# --------------------------
#
# 1. **Start with moderate batch sizes** (32-128) and scale up if memory
#    allows.
#
# 2. **Use ``num_workers > 0``** when transforms are expensive. Start with
#    2-4 workers and increase based on memory capacity. Higher is not always better.
#
# 3. **Enable ``pin_memory=True``** when using an accelerator.
#
# 4. **Use ``persistent_workers=True``** to avoid worker restart overhead
#    between epochs.
#
# 5. **Profile your pipeline** with to identify CPU bottlenecks during
#    dataset access, transformations, etc.
#
# 6. **Implement data prefetching** for GPU workloads to overlap data
#    transfer with computation.
#
# 7. **Use ``file_system`` sharing strategy** when hitting file descriptor limits.
#

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we learned how to progressively optimize a PyTorch
# data loading pipeline — from a naive single-process baseline to a
# fully optimized configuration using multiprocessing workers, pinned
# memory, persistent workers, CUDA stream-based prefetching, and batched
# dataset fetching with ``__getitems__``. Each optimization targets a
# different bottleneck, and together they can yield an order-of-magnitude
# improvement in throughput. These should be considered best practices
# and performance is dependent on the specific workload.

######################################################################
# Additional Resources
# --------------------
#
# - `PyTorch DataLoader documentation <https://pytorch.org/docs/stable/data.html>`_
# - `Pin Memory and Non-blocking Transfer Tutorial <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`_
# - `PyTorch Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_
