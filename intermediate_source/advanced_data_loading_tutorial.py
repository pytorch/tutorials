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

import torch
from torch.utils.data import DataLoader, Dataset
import time
import os

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
        self.data = torch.randn(size, 3, feature_dim, feature_dim)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate expensive transformation
        if self.transform_delay > 0:
            time.sleep(self.transform_delay)
        return self.data[idx], self.labels[idx]


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
        # Simulate some processing
        _ = data.sum()
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

    for i, (data, labels) in enumerate(prefetcher):
        # Your model training here
        _ = data.sum()
        if i >= 5:
            break
    print("Prefetching demonstration complete!")

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
# Putting It All Together: Optimized DataLoader Configuration
# ------------------------------------------------------------
#
# Here's a template for an optimized DataLoader configuration:


def create_optimized_dataloader(
    dataset,
    batch_size=64,
    num_workers=None,
    pin_memory=None,
    prefetch_factor=2,
):
    """Create an optimized DataLoader with sensible defaults.

    Args:
        dataset: The dataset to load from
        batch_size: Batch size (default: 64)
        num_workers: Number of worker processes. If None, uses a heuristic
                     based on CPU count
        pin_memory: Whether to use pinned memory. If None, auto-detects
                    based on CUDA availability
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        A configured DataLoader
    """
    # Auto-detect optimal settings
    if num_workers is None:
        # Use number of CPU cores, but cap at 8 to avoid overhead
        num_workers = min(os.cpu_count() or 4, 8)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        drop_last=True,  # Avoid smaller last batch
    )

    return loader


# Create and test the optimized loader
optimized_loader = create_optimized_dataloader(dataset, batch_size=32)
print(f"\nOptimized DataLoader settings:")
print(f"  batch_size: {optimized_loader.batch_size}")
print(f"  num_workers: {optimized_loader.num_workers}")
print(f"  pin_memory: {optimized_loader.pin_memory}")

######################################################################
# Profiling CPU Bottlenecks with py-spy - identify slow transforms, downloads
# -------------------------------------
#
# When optimizing data loading, it's crucial to identify where CPU time
# is being spent. `py-spy <https://github.com/benfred/py-spy>`_ is a
# sampling profiler for Python that can attach to running processes
# without requiring code changes.
#
# **Installing py-spy:**
#
# .. code-block:: bash
#
#     pip install py-spy
#
# **Basic Usage - Generating a Flame Graph:**
#
# .. code-block:: bash
#
#     # Record a flame graph while running your training script
#     py-spy record -o profile.svg -- python train.py
#
#     # Or attach to a running process
#     py-spy record -o profile.svg --pid <PID>
#
# **Real-time Top-like View:**
#
# .. code-block:: bash
#
#     # See what functions are consuming CPU in real-time
#     py-spy top --pid <PID>
#
# **Profiling with Subprocesses (for DataLoader workers):**
#
# .. code-block:: bash
#
#     # Include subprocesses (essential for num_workers > 0)
#     py-spy record -o profile.svg --subprocesses -- python train.py
#
# **Interpreting the Results:**
#
# In the flame graph, look for:
#
# - Wide bars in data loading code (Dataset.__getitem__, transforms)
# - Time spent in collate functions
# - IPC overhead between workers and main process
# - Image decoding or other I/O operations
#
# **Example: Identifying a Slow Transform**

import subprocess
import sys


def demonstrate_pyspy_usage():
    """Show how to programmatically check if py-spy is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "py-spy"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("py-spy is installed and ready to use!")
            print("Example commands:")
            print("  py-spy record -o profile.svg --subprocesses -- python train.py")
            print("  py-spy top --pid <YOUR_PID>")
        else:
            print("py-spy not found. Install with: pip install py-spy")
    except Exception as e:
        print(f"Could not check py-spy installation: {e}")


demonstrate_pyspy_usage()

######################################################################
# .. tip::
#    When profiling DataLoader performance, always use ``--subprocesses``
#    flag to capture activity in worker processes. Without it, you'll
#    only see the main process waiting on the queue.
#
# **Common CPU Bottlenecks Revealed by py-spy:**
#
# 1. **Image decoding** - Consider using faster decoders like
#    `torchvision.io` or `pillow-simd`
# 2. **Data augmentation** - Move heavy augmentations to GPU with
#    `kornia` or `torchvision.transforms.v2`
# 3. **Collation** - Custom collate functions might be inefficient
# 4. **File I/O** - Consider using memory-mapped files or faster storage

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

import torch.multiprocessing as mp

# Check current sharing strategy
print(f"\nCurrent sharing strategy: {mp.get_sharing_strategy()}")

# List available strategies
print(f"Available strategies: {mp.get_all_sharing_strategies()}")

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
# 9. **Use ``file_system`` sharing strategy** in Docker containers or
#    when hitting file descriptor limits.
#
# Additional Resources
# --------------------
#
# - `PyTorch DataLoader documentation <https://pytorch.org/docs/stable/data.html>`_
# - `Pin Memory and Non-blocking Transfer Tutorial <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`_
# - `NVIDIA Data Loading Best Practices <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_
# - `PyTorch Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_
