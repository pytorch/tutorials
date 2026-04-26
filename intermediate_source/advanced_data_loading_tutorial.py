# -*- coding: utf-8 -*-
"""
Advanced Data Loading Optimization in PyTorch
==============================================

**Author**: `PyTorch Team <https://pytorch.org>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to optimize DataLoader configuration for maximum throughput
       * Best practices for ``batch_size``, ``num_workers``, and ``pin_memory``
       * Advanced techniques for overlapping data transfers with GPU compute
       * Understanding Python multiprocessing startup methods
       * Using py-spy to profile and isolate CPU-side bottlenecks
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
covers advanced techniques for optimizing your data loading configuration to
maximize training throughput.

We'll explore the key parameters of PyTorch's DataLoader and provide practical
guidance on tuning them for your specific workload.
"""

import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
# - Batch size changes affect the effective learning rate
# - When increasing batch size, you typically need to adjust the learning
#   rate accordingly (linear scaling rule)
# - Larger batches provide more stable gradient estimates but may
#   generalize differently
#
# .. note::
#    When changing batch size, remember to tune your optimizer parameters,
#    especially the learning rate schedule, unless you're doing offline
#    inference.

# Example: Testing different batch sizes
dataset = SyntheticDataset(size=1000, transform_delay=0)


def benchmark_batch_size(batch_size, num_batches=10):
    """Benchmark data loading with a specific batch size."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    start = time.perf_counter()
    for i, (data, labels) in enumerate(loader):
        if i >= num_batches:
            break
        data = data.to(device)
        _ = data.sum()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


# Benchmark different batch sizes
for bs in [16, 32, 64, 128]:
    elapsed = benchmark_batch_size(bs)
    print(f"Batch size {bs:3d}: {elapsed:.4f}s for 10 batches")

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


def benchmark_num_workers(num_workers, batch_size=32, num_batches=20):
    """Benchmark data loading with different number of workers."""
    # Use a dataset with simulated expensive transforms
    expensive_dataset = SyntheticDataset(size=1000, transform_delay=0.005)
    loader = DataLoader(
        expensive_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    start = time.perf_counter()
    for i, (data, labels) in enumerate(loader):
        if i >= num_batches:
            break
    elapsed = time.perf_counter() - start
    return elapsed


# Benchmark different worker counts (keeping it small for tutorial)
print("\nBenchmarking num_workers with expensive transforms:")
for nw in [0, 2, 4]:
    elapsed = benchmark_num_workers(nw)
    print(f"num_workers={nw}: {elapsed:.4f}s for 20 batches")

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


def benchmark_pin_memory(pin_memory, batch_size=64, num_batches=50):
    """Benchmark the effect of pin_memory on data transfer speed."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping pin_memory benchmark")
        return None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=2,
    )

    # Warm up
    for i, (data, labels) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        if i >= 5:
            break

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i, (data, labels) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if i >= num_batches:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


if torch.cuda.is_available():
    print("\nBenchmarking pin_memory:")
    for pm in [False, True]:
        elapsed = benchmark_pin_memory(pm)
        print(f"pin_memory={pm}: {elapsed:.4f}s for 50 batches")
else:
    print("\nSkipping pin_memory benchmark (CUDA not available)")

######################################################################
# The ``in_order`` Parameter
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
#    can reduce variance and eliminate occasional slow batches.

# Example: Comparing in_order=True vs in_order=False
in_order_dataset = SyntheticDataset(size=500, transform_delay=0.002)


def benchmark_in_order(in_order, num_batches=15):
    """Benchmark the effect of in_order on batch delivery timing."""
    loader = DataLoader(
        in_order_dataset,
        batch_size=16,
        num_workers=4,
        prefetch_factor=2,
        in_order=in_order,
    )
    batch_times = []
    prev = time.perf_counter()
    for i, (data, labels) in enumerate(loader):
        if i >= num_batches:
            break
        now = time.perf_counter()
        batch_times.append(now - prev)
        prev = now
    return batch_times


print("\nBenchmarking in_order effect on batch timing variance:")
for order in [True, False]:
    times = benchmark_in_order(order)
    avg = sum(times) / len(times)
    variance = sum((t - avg) ** 2 for t in times) / len(times)
    print(f"  in_order={order}: avg={avg:.4f}s, variance={variance:.6f}")

######################################################################
# Snapshot Frequency (``snapshot_every_n_steps``)
# -----------------------------------------------
#
# When using stateful DataLoaders (for checkpointing), the
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

# Example: Configuring snapshot frequency
# (Note: snapshot_every_n_steps requires a stateful DataLoader)
#
# .. code-block:: python
#
#     from torch.utils.data.dataloader_experimental import DataLoader2
#
#     loader = DataLoader2(
#         dataset,
#         snapshot_every_n_steps=100,  # Snapshot every 100 steps
#     )
#     # Save state for checkpointing
#     state = loader.state_dict()
#     # Restore on resumption
#     loader.load_state_dict(state)

######################################################################
# Advanced: Overlapping H2D Transfer with GPU Compute
# ---------------------------------------------------
#
# For maximum throughput, you can overlap Host-to-Device (H2D) data
# transfers with GPU computation. This ensures the GPU is never idle
# waiting for data.
#
# **Technique: Double Buffering**
#
# The idea is to prefetch the next batch to GPU while the current batch
# is being processed.


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


# Example usage
if torch.cuda.is_available():
    print("\nDemonstrating data prefetching:")
    loader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=2)
    prefetcher = DataPrefetcher(loader, device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i, (data, labels) in enumerate(prefetcher):
        # Your model training here
        _ = data.sum()
        if i >= 5:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Prefetching demonstration complete! ({elapsed:.4f}s for 6 batches)")

######################################################################
# Python Multiprocessing Startup Methods
# --------------------------------------
#
# Python's multiprocessing module supports three startup methods:
#
# **1. spawn (Recommended for most cases)**
#
# - Creates a fresh Python interpreter for each worker
# - Safest option, avoids issues with threads and locks
# - Slightly higher startup time
#
# **2. forkserver**
#
# - Uses a server process to fork workers
# - Good balance of safety and performance
# - Usually the recommended option for production
#
# **3. fork**
#
# - Directly forks the parent process
# - Fastest startup, lowest initial memory
# - **Not recommended:** Can cause issues with threads, locks, and
#   CUDA contexts
#
# .. warning::
#    ``fork`` can lead to deadlocks if the parent process has active
#    threads or holds locks. The child process inherits a potentially
#    corrupted state.
#
# **Setting the startup method:**
#
# .. code-block:: python
#
#     import torch.multiprocessing as mp
#     mp.set_start_method('forkserver')  # Call once at the beginning
#
# .. note::
#    While ``fork`` uses copy-on-write for low initial memory, memory
#    usage can spike during training. Use with caution and only if
#    your pipeline is fork-compatible.
######################################################################
# Profiling CPU Bottlenecks
# -------------------------
#
# When optimizing data loading, it's crucial to identify where CPU time
# is being spent. `py-spy <https://github.com/benfred/py-spy>`_ is a
# sampling profiler for Python that can help isolate bottlenecks,
# especially sections holding the GIL. Use the ``--gil`` flag to track
# GIL contention.

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
#    - Can hit shared memory size limits

######################################################################
# **When to Change the Strategy:**
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
# | Docker/containers | file_system               | fd limits often stricter  |
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
#     OSError: [Errno 28] No space left on device
#     ERROR: Unexpected bus error encountered in worker
#
# This typically means you've exhausted the shared memory allocation.
#
# **Diagnosing the Problem:**

import shutil


def check_shm_status():
    """Check the current shared memory usage."""
    shm_path = "/dev/shm"
    try:
        total, used, free = shutil.disk_usage(shm_path)
        print(f"\n/dev/shm status:")
        print(f"  Total: {total / (1024**3):.2f} GB")
        print(f"  Used:  {used / (1024**3):.2f} GB")
        print(f"  Free:  {free / (1024**3):.2f} GB")
        print(f"  Usage: {used / total * 100:.1f}%")
    except Exception as e:
        print(f"Could not check /dev/shm: {e}")


check_shm_status()

######################################################################
# **Solutions for Insufficient Shared Memory:**
#
# **1. Increase /dev/shm size (if you have root access):**
#
# .. code-block:: bash
#
#     # Temporarily increase to 16GB
#     sudo mount -o remount,size=16G /dev/shm
#
#     # Or permanently in /etc/fstab:
#     # tmpfs /dev/shm tmpfs defaults,size=16G 0 0
#
# **2. For Docker containers, use --shm-size:**
#
# .. code-block:: bash
#
#     # Run with 16GB shared memory
#     docker run --shm-size=16g your_image
#
#     # Or in docker-compose.yml:
#     # services:
#     #   training:
#     #     shm_size: '16gb'
#
# **3. Reduce memory pressure from DataLoader:**
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
# **4. Switch sharing strategy:**
#
# .. code-block:: python
#
#     import torch.multiprocessing as mp
#     mp.set_sharing_strategy('file_system')
#
# **5. Use memory-efficient data formats:**
#
# - Store data in formats that don't require large intermediate tensors
# - Use ``torch.utils.data.IterableDataset`` for streaming data
# - Consider memory-mapped files with ``numpy.memmap`` or ``torch.Storage``
#
# **6. Clean up leaked shared memory:**
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
#    proper cleanup. Using ``persistent_workers=True`` can help reduce
#    this by keeping workers alive longer.


def estimate_shm_usage(batch_size, num_workers, prefetch_factor, tensor_size_mb):
    """Estimate shared memory usage for a DataLoader configuration.

    Args:
        batch_size: Number of samples per batch
        num_workers: Number of worker processes
        prefetch_factor: Batches prefetched per worker
        tensor_size_mb: Size of one sample in MB

    Returns:
        Estimated shared memory usage in GB
    """
    # Each worker prefetches prefetch_factor batches
    batches_in_flight = num_workers * prefetch_factor
    samples_in_flight = batches_in_flight * batch_size
    total_mb = samples_in_flight * tensor_size_mb
    total_gb = total_mb / 1024
    return total_gb


# Example calculation
sample_size_mb = 3 * 224 * 224 * 4 / (1024 * 1024)  # 3x224x224 float32
estimated_usage = estimate_shm_usage(
    batch_size=32, num_workers=8, prefetch_factor=2, tensor_size_mb=sample_size_mb
)
print(f"\nEstimated shm usage for typical ImageNet config:")
print(f"  Sample size: {sample_size_mb:.2f} MB")
print(f"  Config: batch_size=32, num_workers=8, prefetch_factor=2")
print(f"  Estimated usage: {estimated_usage:.2f} GB")

######################################################################
# Summary and Best Practices
# --------------------------
#
# 1. **Start with moderate batch sizes** (32-128) and scale up if memory
#    allows. Remember to adjust learning rate when changing batch size.
#
# 2. **Use ``num_workers > 0``** when transforms are expensive. Start with
#    2-4 workers and increase if data loading is still a bottleneck.
#
# 3. **Enable ``pin_memory=True``** when using CUDA, and always use
#    ``non_blocking=True`` for GPU transfers.
#
# 4. **Use ``persistent_workers=True``** to avoid worker restart overhead
#    between epochs.
#
# 5. **Consider ``forkserver``** as your multiprocessing start method
#    for production workloads.
#
# 6. **Profile your pipeline** with py-spy using ``--subprocesses`` to
#    capture worker activity and identify CPU bottlenecks.
#
# 7. **Implement data prefetching** for GPU workloads to overlap data
#    transfer with computation.
#
# 8. **Monitor ``/dev/shm`` usage** and adjust ``num_workers``,
#    ``prefetch_factor``, or sharing strategy if you hit limits.
#
# 9. **Use ``file_system`` sharing strategy** when hitting file descriptor limits.
#


######################################################################
# Optimizing a Training Loop: A Step-by-Step Story
# -------------------------------------------------
#
# To demonstrate the real-world impact of data loading optimizations,
# let's start with a decent baseline training loop and progressively
# apply optimizations, measuring the speedup at each step.
#
# We'll use our synthetic dataset with simulated expensive transforms
# to make the bottlenecks more apparent.

import torch.nn as nn

# Create a larger dataset for meaningful benchmarks
benchmark_dataset = SyntheticDataset(size=2000, transform_delay=0.002)

######################################################################
# Step 1: Baseline - Simple DataLoader Configuration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our starting point: a reasonable but unoptimized configuration.


def create_baseline_dataloader(dataset):
    """Create a baseline DataLoader with reasonable defaults."""
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # No multiprocessing
        pin_memory=False,  # No pinned memory
    )


def baseline_training_loop(dataset, epochs=5, max_batches=250):
    """Baseline training loop with simple DataLoader."""
    loader = create_baseline_dataloader(dataset)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)

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


print("\n=== Data Loading Optimization Story ===")
print("\nStep 1: Baseline configuration")
baseline_time, baseline_loss = baseline_training_loop(benchmark_dataset)
print(f"  Time: {baseline_time:.4f}s for {250} batches (5 epochs)")
print(f"  Avg loss: {baseline_loss:.4f}")

######################################################################
# Step 2: Add Multiprocessing (num_workers > 0)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The biggest single improvement often comes from enabling multiprocessing.


def create_multiprocess_dataloader(dataset):
    """DataLoader with multiprocessing enabled."""
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # Enable multiprocessing
        pin_memory=False,
        prefetch_factor=2,
    )


def multiprocess_training_loop(dataset, epochs=5, max_batches=250):
    """Training loop with multiprocessing DataLoader."""
    loader = create_multiprocess_dataloader(dataset)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)

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


print("\nStep 2: Add multiprocessing (num_workers=4)")
mp_time, mp_loss = multiprocess_training_loop(benchmark_dataset)
speedup_mp = baseline_time / mp_time
print(f"  Time: {mp_time:.4f}s for {250} batches (5 epochs)")
print(f"  Avg loss: {mp_loss:.4f}")
print(f"  Speedup: {speedup_mp:.2f}x")

######################################################################
# Step 3: Enable Pinned Memory
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For GPU training, pinned memory enables faster data transfers.


def create_pinned_dataloader(dataset):
    """DataLoader with multiprocessing and pinned memory."""
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),  # Enable pinned memory
        prefetch_factor=2,
    )


def pinned_training_loop(dataset, epochs=5, max_batches=250):
    """Training loop with pinned memory."""
    loader = create_pinned_dataloader(dataset)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for data, labels in loader:
            data = data.to(device, non_blocking=True)  # Use non_blocking
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


if torch.cuda.is_available():
    print("\nStep 3: Enable pinned memory")
    pinned_time, pinned_loss = pinned_training_loop(benchmark_dataset)
    speedup_pinned = baseline_time / pinned_time
    print(f"  Time: {pinned_time:.4f}s for {250} batches (5 epochs)")
    print(f"  Avg loss: {pinned_loss:.4f}")
    print(f"  Speedup vs baseline: {speedup_pinned:.2f}x")
else:
    print("\nStep 3: Skipping pinned memory (CUDA not available)")
    pinned_time = mp_time  # Use previous time for comparison

######################################################################
# Step 4: Add Persistent Workers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Persistent workers avoid the overhead of restarting processes between epochs.


def create_persistent_dataloader(dataset):
    """DataLoader with persistent workers."""
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True,  # Keep workers alive
    )


def persistent_training_loop(dataset, epochs=5, max_batches=250):
    """Training loop with persistent workers."""
    loader = create_persistent_dataloader(dataset)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for data, labels in loader:
            data = data.to(device, non_blocking=torch.cuda.is_available())
            labels = labels.to(device, non_blocking=torch.cuda.is_available())

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


print("\nStep 4: Add persistent workers")
persistent_time, persistent_loss = persistent_training_loop(benchmark_dataset)
speedup_persistent = baseline_time / persistent_time
print(f"  Time: {persistent_time:.4f}s for {250} batches (5 epochs)")
print(f"  Avg loss: {persistent_loss:.4f}")
print(f"  Speedup vs baseline: {speedup_persistent:.2f}x")

######################################################################
# Step 5: Fully Optimized with Data Prefetching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Finally, add data prefetching to overlap data transfer with computation.


def optimized_training_loop(dataset, epochs=5, max_batches=250):
    """Fully optimized training loop with prefetching."""
    loader = create_persistent_dataloader(dataset)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        # Use prefetcher for overlapping
        if torch.cuda.is_available():
            data_iter = DataPrefetcher(loader, device)
        else:
            data_iter = loader

        for data, labels in data_iter:
            if not torch.cuda.is_available():
                data = data.to(device)
                labels = labels.to(device)

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


print("\nStep 5: Fully optimized with data prefetching")
optimized_time, optimized_loss = optimized_training_loop(benchmark_dataset)
speedup_optimized = baseline_time / optimized_time
print(f"  Time: {optimized_time:.4f}s for {250} batches (5 epochs)")
print(f"  Avg loss: {optimized_loss:.4f}")
print(f"  Speedup vs baseline: {speedup_optimized:.2f}x")

######################################################################
# Summary of Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's summarize the cumulative speedups achieved:

print("\n=== Optimization Summary ===")
print(f"Baseline:                    {baseline_time:.4f}s")
print(f"+ Multiprocessing:           {mp_time:.4f}s ({speedup_mp:.2f}x)")
if torch.cuda.is_available():
    print(f"+ Pinned memory:             {pinned_time:.4f}s ({speedup_pinned:.2f}x)")
print(f"+ Persistent workers:        {persistent_time:.4f}s ({speedup_persistent:.2f}x)")
print(f"+ Data prefetching:          {optimized_time:.4f}s ({speedup_optimized:.2f}x)")

print("Key takeaways:")
print("- Multiprocessing often provides the biggest single speedup")
print("- Persistent workers reduce overhead when training multiple epochs")
print("- Each optimization builds on the previous ones")
print("- The final configuration can be 2-5x faster than the baseline")
print("- Always benchmark your specific workload and hardware")


######################################################################
# Additional Resources
# --------------------
#
# - `PyTorch DataLoader documentation <https://pytorch.org/docs/stable/data.html>`_
# - `Pin Memory and Non-blocking Transfer Tutorial <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`_
# - `NVIDIA Data Loading Best Practices <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_
# - `PyTorch Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_
