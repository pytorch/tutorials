"""
Performance Tuning Guide
*************************
**Author**: `Szymon Migacz <https://github.com/szmigacz>`_

Performance Tuning Guide is a set of optimizations and best practices which can
accelerate training and inference of deep learning models in PyTorch. Presented
techniques often can be implemented by changing only a few lines of code and can
be applied to a wide range of deep learning models across all domains.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * General optimization techniques for PyTorch models
       * CPU-specific performance optimizations
       * GPU acceleration strategies
       * Distributed training optimizations

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch 2.0 or later
       * Python 3.8 or later
       * CUDA-capable GPU (recommended for GPU optimizations)
       * Linux, macOS, or Windows operating system

Overview
--------

Performance optimization is crucial for efficient deep learning model training and inference.
This tutorial covers a comprehensive set of techniques to accelerate PyTorch workloads across
different hardware configurations and use cases.

General optimizations
---------------------
"""

import torch
import torchvision

###############################################################################
# Enable asynchronous data loading and augmentation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
# supports asynchronous data loading and data augmentation in separate worker
# subprocesses. The default setting for ``DataLoader`` is ``num_workers=0``,
# which means that the data loading is synchronous and done in the main process.
# As a result the main training process has to wait for the data to be available
# to continue the execution.
#
# Setting ``num_workers > 0`` enables asynchronous data loading and overlap
# between the training and data loading. ``num_workers`` should be tuned
# depending on the workload, CPU, GPU, and location of training data.
#
# ``DataLoader`` accepts ``pin_memory`` argument, which defaults to ``False``.
# When using a GPU it's better to set ``pin_memory=True``, this instructs
# ``DataLoader`` to use pinned memory and enables faster and asynchronous memory
# copy from the host to the GPU.

###############################################################################
# Disable gradient calculation for validation or inference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch saves intermediate buffers from all operations which involve tensors
# that require gradients. Typically gradients aren't needed for validation or
# inference.
# `torch.no_grad() <https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad>`_
# context manager can be applied to disable gradient calculation within a
# specified block of code, this accelerates execution and reduces the amount of
# required memory.
# `torch.no_grad() <https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad>`_
# can also be used as a function decorator.

###############################################################################
# Disable bias for convolutions directly followed by a batch norm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.nn.Conv2d() <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_
# has ``bias`` parameter which defaults to ``True`` (the same is true for
# `Conv1d <https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d>`_
# and
# `Conv3d <https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d>`_
# ).
#
# If a ``nn.Conv2d`` layer is directly followed by a ``nn.BatchNorm2d`` layer,
# then the bias in the convolution is not needed, instead use
# ``nn.Conv2d(..., bias=False, ....)``. Bias is not needed because in the first
# step ``BatchNorm`` subtracts the mean, which effectively cancels out the
# effect of bias.
#
# This is also applicable to 1d and 3d convolutions as long as ``BatchNorm`` (or
# other normalization layer) normalizes on the same dimension as convolution's
# bias.
#
# Models available from `torchvision <https://github.com/pytorch/vision>`_
# already implement this optimization.

###############################################################################
# Use parameter.grad = None instead of model.zero_grad() or optimizer.zero_grad()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instead of calling:
model.zero_grad()
# or
optimizer.zero_grad()

###############################################################################
# to zero out gradients, use the following method instead:

for param in model.parameters():
    param.grad = None

###############################################################################
# The second code snippet does not zero the memory of each individual parameter,
# also the subsequent backward pass uses assignment instead of addition to store
# gradients, this reduces the number of memory operations.
#
# Setting gradient to ``None`` has a slightly different numerical behavior than
# setting it to zero, for more details refer to the
# `documentation <https://pytorch.org/docs/master/optim.html#torch.optim.Optimizer.zero_grad>`_.
#
# Alternatively, call ``model`` or
# ``optimizer.zero_grad(set_to_none=True)``.

###############################################################################
# Fuse operations
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Pointwise operations such as elementwise addition, multiplication, and math
# functions like `sin()`, `cos()`, `sigmoid()`, etc., can be combined into a
# single kernel. This fusion helps reduce memory access and kernel launch times.
# Typically, pointwise operations are memory-bound; PyTorch eager-mode initiates
# a separate kernel for each operation, which involves loading data from memory,
# executing the operation (often not the most time-consuming step), and writing
# the results back to memory.
# 
# By using a fused operator, only one kernel is launched for multiple pointwise
# operations, and data is loaded and stored just once. This efficiency is
# particularly beneficial for activation functions, optimizers, and custom RNN cells etc.
#
# PyTorch 2 introduces a compile-mode facilitated by TorchInductor, an underlying compiler
# that automatically fuses kernels. TorchInductor extends its capabilities beyond simple
# element-wise operations, enabling advanced fusion of eligible pointwise and reduction
# operations for optimized performance.
#
# In the simplest case fusion can be enabled by applying
# `torch.compile <https://pytorch.org/docs/stable/generated/torch.compile.html>`_
# decorator to the function definition, for example:

@torch.compile
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

###############################################################################
# Refer to
# `Introduction to torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_
# for more advanced use cases.

###############################################################################
# Enable channels_last memory format for computer vision models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch supports ``channels_last`` memory format for
# convolutional networks. This format is meant to be used in conjunction with
# `AMP <https://pytorch.org/docs/stable/amp.html>`_ to further accelerate
# convolutional neural networks with
# `Tensor Cores <https://www.nvidia.com/en-us/data-center/tensor-cores/>`_.
#
# Support for ``channels_last`` is experimental, but it's expected to work for
# standard computer vision models (e.g. ResNet-50, SSD). To convert models to
# ``channels_last`` format follow
# `Channels Last Memory Format Tutorial <https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_.
# The tutorial includes a section on
# `converting existing models <https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models>`_.

###############################################################################
# Checkpoint intermediate buffers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Buffer checkpointing is a technique to mitigate the memory capacity burden of
# model training. Instead of storing inputs of all layers to compute upstream
# gradients in backward propagation, it stores the inputs of a few layers and
# the others are recomputed during backward pass. The reduced memory
# requirements enables increasing the batch size that can improve utilization.
#
# Checkpointing targets should be selected carefully. The best is not to store
# large layer outputs that have small re-computation cost. The example target
# layers are activation functions (e.g. ``ReLU``, ``Sigmoid``, ``Tanh``),
# up/down sampling and matrix-vector operations with small accumulation depth.
#
# PyTorch supports a native
# `torch.utils.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_
# API to automatically perform checkpointing and recomputation.

###############################################################################
# Disable debugging APIs
# ~~~~~~~~~~~~~~~~~~~~~~
# Many PyTorch APIs are intended for debugging and should be disabled for
# regular training runs:
#
# * anomaly detection:
#   `torch.autograd.detect_anomaly <https://pytorch.org/docs/stable/autograd.html#torch.autograd.detect_anomaly>`_
#   or
#   `torch.autograd.set_detect_anomaly(True) <https://pytorch.org/docs/stable/autograd.html#torch.autograd.set_detect_anomaly>`_
# * profiler related:
#   `torch.autograd.profiler.emit_nvtx <https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx>`_,
#   `torch.autograd.profiler.profile <https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile>`_
# * autograd ``gradcheck``:
#   `torch.autograd.gradcheck <https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradcheck>`_
#   or
#   `torch.autograd.gradgradcheck <https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradgradcheck>`_
#

###############################################################################
# CPU specific optimizations
# --------------------------

###############################################################################
# Utilize Non-Uniform Memory Access (NUMA) Controls
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NUMA or non-uniform memory access is a memory layout design used in data center machines meant to take advantage of locality of memory in multi-socket machines with multiple memory controllers and blocks. Generally speaking, all deep learning workloads, training or inference, get better performance without accessing hardware resources across NUMA nodes. Thus, inference can be run with multiple instances, each instance runs on one socket, to raise throughput. For training tasks on single node, distributed training is recommended to make each training process run on one socket.
#
# In general cases the following command executes a PyTorch script on cores on the Nth node only, and avoids cross-socket memory access to reduce memory access overhead.
#
# .. code-block:: sh
#
#    numactl --cpunodebind=N --membind=N python <pytorch_script>

###############################################################################
# More detailed descriptions can be found `here <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html>`_.

###############################################################################
# Utilize OpenMP
# ~~~~~~~~~~~~~~
# OpenMP is utilized to bring better performance for parallel computation tasks.
# ``OMP_NUM_THREADS`` is the easiest switch that can be used to accelerate computations. It determines number of threads used for OpenMP computations.
# CPU affinity setting controls how workloads are distributed over multiple cores. It affects communication overhead, cache line invalidation overhead, or page thrashing, thus proper setting of CPU affinity brings performance benefits. ``GOMP_CPU_AFFINITY`` or ``KMP_AFFINITY`` determines how to bind OpenMP* threads to physical processing units. Detailed information can be found `here <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html>`_.

###############################################################################
# With the following command, PyTorch run the task on N OpenMP threads.
#
# .. code-block:: sh
#
#    export OMP_NUM_THREADS=N

###############################################################################
# Typically, the following environment variables are used to set for CPU affinity with GNU OpenMP implementation. ``OMP_PROC_BIND`` specifies whether threads may be moved between processors. Setting it to CLOSE keeps OpenMP threads close to the primary thread in contiguous place partitions. ``OMP_SCHEDULE`` determines how OpenMP threads are scheduled. ``GOMP_CPU_AFFINITY`` binds threads to specific CPUs.
# An important tuning parameter is core pinning which prevent the threads of migrating between multiple CPUs, enhancing data location and minimizing inter core communication.
#
# .. code-block:: sh
#
#    export OMP_SCHEDULE=STATIC
#    export OMP_PROC_BIND=CLOSE
#    export GOMP_CPU_AFFINITY="N-M"

###############################################################################
# Intel OpenMP Runtime Library (``libiomp``)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# By default, PyTorch uses GNU OpenMP (GNU ``libgomp``) for parallel computation. On Intel platforms, Intel OpenMP Runtime Library (``libiomp``) provides OpenMP API specification support. It sometimes brings more performance benefits compared to ``libgomp``. Utilizing environment variable ``LD_PRELOAD`` can switch OpenMP library to ``libiomp``:
#
# .. code-block:: sh
#
#    export LD_PRELOAD=<path>/libiomp5.so:$LD_PRELOAD

###############################################################################
# Similar to CPU affinity settings in GNU OpenMP, environment variables are provided in ``libiomp`` to control CPU affinity settings.
# ``KMP_AFFINITY`` binds OpenMP threads to physical processing units. ``KMP_BLOCKTIME`` sets the time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleeping. In most cases, setting ``KMP_BLOCKTIME`` to 1 or 0 yields good performances.
# The following commands show a common settings with Intel OpenMP Runtime Library.
#
# .. code-block:: sh
#
#    export KMP_AFFINITY=granularity=fine,compact,1,0
#    export KMP_BLOCKTIME=1

###############################################################################
# Switch Memory allocator
# ~~~~~~~~~~~~~~~~~~~~~~~
# For deep learning workloads, ``Jemalloc`` or ``TCMalloc`` can get better performance by reusing memory as much as possible than default ``malloc`` function. `Jemalloc <https://github.com/jemalloc/jemalloc>`_ is a general purpose ``malloc`` implementation that emphasizes fragmentation avoidance and scalable concurrency support. `TCMalloc <https://google.github.io/tcmalloc/overview.html>`_ also features a couple of optimizations to speed up program executions. One of them is holding memory in caches to speed up access of commonly-used objects. Holding such caches even after deallocation also helps avoid costly system calls if such memory is later re-allocated.
# Use environment variable ``LD_PRELOAD`` to take advantage of one of them.
#
# .. code-block:: sh
#
#    export LD_PRELOAD=<jemalloc.so/tcmalloc.so>:$LD_PRELOAD


###############################################################################
# Train a model on CPU with PyTorch ``DistributedDataParallel``(DDP) functionality
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For small scale models or memory-bound models, such as DLRM, training on CPU is also a good choice. On a machine with multiple sockets, distributed training brings a high-efficient hardware resource usage to accelerate the training process. `Torch-ccl <https://github.com/intel/torch-ccl>`_, optimized with Intel(R) ``oneCCL`` (collective communications library) for efficient distributed deep learning training implementing such collectives like ``allreduce``, ``allgather``, ``alltoall``, implements PyTorch C10D ``ProcessGroup`` API and can be dynamically loaded as external ``ProcessGroup``. Upon optimizations implemented in PyTorch DDP module, ``torch-ccl`` accelerates communication operations. Beside the optimizations made to communication kernels, ``torch-ccl`` also features simultaneous computation-communication functionality.

###############################################################################
# GPU specific optimizations
# --------------------------

###############################################################################
# Enable Tensor cores
# ~~~~~~~~~~~~~~~~~~~~~~~
# Tensor cores are specialized hardware designed to compute matrix-matrix multiplication
# operations, primarily utilized in deep learning and AI workloads. Tensor cores have
# specific precision requirements which can be adjusted manually or via the Automatic
# Mixed Precision API.
#
# In particular, tensor operations take advantage of lower precision workloads.
# Which can be controlled via ``torch.set_float32_matmul_precision``.
# The default format is set to 'highest,' which utilizes the tensor data type. 
# However, PyTorch offers alternative precision settings: 'high' and 'medium.'
# These options prioritize computational speed over numerical precision."

###############################################################################
# Use CUDA Graphs
# ~~~~~~~~~~~~~~~~~~~~~~~
# At the time of using a GPU, work first must be launched from the CPU and
# in some cases the context switch between CPU and GPU can lead to bad resource
# utilization. CUDA graphs are a way to keep computation within the GPU without
# paying the extra cost of kernel launches and host synchronization.

# It can be enabled using 
torch.compile(m, "reduce-overhead")
# or
torch.compile(m, "max-autotune")

###############################################################################
# Support for CUDA graph is in development, and its usage can incur in increased
# device memory consumption and some models might not compile.

###############################################################################
# Enable cuDNN auto-tuner
# ~~~~~~~~~~~~~~~~~~~~~~~
# `NVIDIA cuDNN <https://developer.nvidia.com/cudnn>`_ supports many algorithms
# to compute a convolution. Autotuner runs a short benchmark and selects the
# kernel with the best performance on a given hardware for a given input size.
#
# For convolutional networks (other types currently not supported), enable cuDNN
# autotuner before launching the training loop by setting:

torch.backends.cudnn.benchmark = True
###############################################################################
#
# * the auto-tuner decisions may be non-deterministic; different algorithm may
#   be selected for different runs.  For more details see
#   `PyTorch: Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html?highlight=determinism>`_
# * in some rare cases, such as with highly variable input sizes,  it's better
#   to run convolutional networks with autotuner disabled to avoid the overhead
#   associated with algorithm selection for each input size.
#

###############################################################################
# Avoid unnecessary CPU-GPU synchronization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Avoid unnecessary synchronizations, to let the CPU run ahead of the
# accelerator as much as possible to make sure that the accelerator work queue
# contains many operations.
#
# When possible, avoid operations which require synchronizations, for example:
#
# * ``print(cuda_tensor)``
# * ``cuda_tensor.item()``
# * memory copies: ``tensor.cuda()``,  ``cuda_tensor.cpu()`` and equivalent
#   ``tensor.to(device)`` calls
# * ``cuda_tensor.nonzero()``
# * python control flow which depends on results of operations performed on CUDA
#   tensors e.g. ``if (cuda_tensor != 0).all()``
#

###############################################################################
# Create tensors directly on the target device
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instead of calling ``torch.rand(size).cuda()`` to generate a random tensor,
# produce the output directly on the target device:
# ``torch.rand(size, device='cuda')``.
#
# This is applicable to all functions which create new tensors and accept
# ``device`` argument:
# `torch.rand() <https://pytorch.org/docs/stable/generated/torch.rand.html#torch.rand>`_,
# `torch.zeros() <https://pytorch.org/docs/stable/generated/torch.zeros.html#torch.zeros>`_,
# `torch.full() <https://pytorch.org/docs/stable/generated/torch.full.html#torch.full>`_
# and similar.

###############################################################################
# Use mixed precision and AMP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mixed precision leverages
# `Tensor Cores <https://www.nvidia.com/en-us/data-center/tensor-cores/>`_
# and offers up to 3x overall speedup on Volta and newer GPU architectures. To
# use Tensor Cores AMP should be enabled and matrix/tensor dimensions should
# satisfy requirements for calling kernels that use Tensor Cores.
#
# To use Tensor Cores:
#
# * set sizes to multiples of 8 (to map onto dimensions of Tensor Cores)
#
#   * see
#     `Deep Learning Performance Documentation
#     <https://docs.nvidia.com/deeplearning/performance/index.html#optimizing-performance>`_
#     for more details and guidelines specific to layer type
#   * if layer size is derived from other parameters rather than fixed, it can
#     still be explicitly padded e.g. vocabulary size in NLP models
#
# * enable AMP
#
#   * Introduction to Mixed Precision Training and AMP:
#     `slides <https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf>`_
#   * native PyTorch AMP is available:
#     `documentation <https://pytorch.org/docs/stable/amp.html>`_,
#     `examples <https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples>`_,
#     `tutorial <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>`_
#
#

###############################################################################
# Preallocate memory in case of variable input length
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Models for speech recognition or for NLP are often trained on input tensors
# with variable sequence length. Variable length can be problematic for PyTorch
# caching allocator and can lead to reduced performance or to unexpected
# out-of-memory errors. If a batch with a short sequence length is followed by
# an another batch with longer sequence length, then PyTorch is forced to
# release intermediate buffers from previous iteration and to re-allocate new
# buffers. This process is time consuming and causes fragmentation in the
# caching allocator which may result in out-of-memory errors.
#
# A typical solution is to implement preallocation. It consists of the
# following steps:
#
# #. generate a (usually random) batch of inputs with maximum sequence length
#    (either corresponding to max length in the training dataset or to some
#    predefined threshold)
# #. execute a forward and a backward pass with the generated batch, do not
#    execute an optimizer or a learning rate scheduler, this step preallocates
#    buffers of maximum size, which can be reused in subsequent
#    training iterations
# #. zero out gradients
# #. proceed to regular training
#

###############################################################################
# Distributed optimizations
# -------------------------

###############################################################################
# Use efficient data-parallel backend
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch has two ways to implement data-parallel training:
#
# * `torch.nn.DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel>`_
# * `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
#
# ``DistributedDataParallel`` offers much better performance and scaling to
# multiple-GPUs. For more information refer to the
# `relevant section of CUDA Best Practices <https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel>`_
# from PyTorch documentation.

###############################################################################
# Skip unnecessary all-reduce if training with ``DistributedDataParallel`` and gradient accumulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# By default
# `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
# executes gradient all-reduce after every backward pass to compute the average
# gradient over all workers participating in the training. If training uses
# gradient accumulation over N steps, then all-reduce is not necessary after
# every training step, it's only required to perform all-reduce after the last
# call to backward, just before the execution of the optimizer.
#
# ``DistributedDataParallel`` provides
# `no_sync() <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync>`_
# context manager which disables gradient all-reduce for particular iteration.
# ``no_sync()`` should be applied to first ``N-1`` iterations of gradient
# accumulation, the last iteration should follow the default execution and
# perform the required gradient all-reduce.

###############################################################################
# Match the order of layers in constructors and during the execution if using ``DistributedDataParallel(find_unused_parameters=True)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
# with ``find_unused_parameters=True`` uses the order of layers and parameters
# from model constructors to build buckets for ``DistributedDataParallel``
# gradient all-reduce. ``DistributedDataParallel`` overlaps all-reduce with the
# backward pass. All-reduce for a particular bucket is asynchronously triggered
# only when all gradients for parameters in a given bucket are available.
#
# To maximize the amount of overlap, the order in model constructors should
# roughly match the order during the execution. If the order doesn't match, then
# all-reduce for the entire bucket waits for the gradient which is the last to
# arrive, this may reduce the overlap between backward pass and all-reduce,
# all-reduce may end up being exposed, which slows down the training.
#
# ``DistributedDataParallel`` with ``find_unused_parameters=False`` (which is
# the default setting) relies on automatic bucket formation based on order of
# operations encountered during the backward pass. With
# ``find_unused_parameters=False`` it's not necessary to reorder layers or
# parameters to achieve optimal performance.

###############################################################################
# Load-balance workload in a distributed setting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load imbalance typically may happen for models processing sequential data
# (speech recognition, translation, language models etc.). If one device
# receives a batch of data with sequence length longer than sequence lengths for
# the remaining devices, then all devices wait for the worker which finishes
# last. Backward pass functions as an implicit synchronization point in a
# distributed setting with
# `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
# backend.
#
# There are multiple ways to solve the load balancing problem. The core idea is
# to distribute workload over all workers as uniformly as possible within each
# global batch. For example Transformer solves imbalance by forming batches with
# approximately constant number of tokens (and variable number of sequences in a
# batch), other models solve imbalance by bucketing samples with similar
# sequence length or even by sorting dataset by sequence length.

###############################################################################
# Conclusion
# ----------
# 
# This tutorial covered a comprehensive set of performance optimization techniques
# for PyTorch models. The key takeaways include:
#
# * **General optimizations**: Enable async data loading, disable gradients for
#   inference, fuse operations with ``torch.compile``, and use efficient memory formats
# * **CPU optimizations**: Leverage NUMA controls, optimize OpenMP settings, and
#   use efficient memory allocators
# * **GPU optimizations**: Enable Tensor cores, use CUDA graphs, enable cuDNN
#   autotuner, and implement mixed precision training
# * **Distributed optimizations**: Use DistributedDataParallel, optimize gradient
#   synchronization, and balance workloads across devices
#
# Many of these optimizations can be applied with minimal code changes and provide
# significant performance improvements across a wide range of deep learning models.
#
# Further Reading
# ---------------
#
# * `PyTorch Performance Tuning Documentation <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_
# * `CUDA Best Practices <https://pytorch.org/docs/stable/notes/cuda.html>`_
# * `Distributed Training Documentation <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_
# * `Mixed Precision Training <https://pytorch.org/docs/stable/amp.html>`_
# * `torch.compile Tutorial <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_
