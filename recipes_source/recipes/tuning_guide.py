"""
Performance Tuning Guide
*************************
**Author**: `Szymon Migacz <https://github.com/szmigacz>`_

Performance Tuning Guide is a set of optimizations and best practices which can
accelerate training and inference of deep learning models in PyTorch. Presented
techniques often can be implemented by changing only a few lines of code and can
be applied to a wide range of deep learning models across all domains.

General optimizations
---------------------
"""

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
# Alternatively, starting from PyTorch 1.7, call ``model`` or
# ``optimizer.zero_grad(set_to_none=True)``.

###############################################################################
# Fuse pointwise operations
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Pointwise operations (elementwise addition, multiplication, math functions -
# ``sin()``, ``cos()``, ``sigmoid()`` etc.) can be fused into a single kernel
# to amortize memory access time and kernel launch time.
#
# `PyTorch JIT <https://pytorch.org/docs/stable/jit.html>`_ can fuse kernels
# automatically, although there could be additional fusion opportunities not yet
# implemented in the compiler, and not all device types are supported equally.
#
# Pointwise operations are memory-bound, for each operation PyTorch launches a
# separate kernel. Each kernel loads data from the memory, performs computation
# (this step is usually inexpensive) and stores results back into the memory.
#
# Fused operator launches only one kernel for multiple fused pointwise ops and
# loads/stores data only once to the memory. This makes JIT very useful for
# activation functions, optimizers, custom RNN cells etc.
#
# In the simplest case fusion can be enabled by applying
# `torch.jit.script <https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script>`_
# decorator to the function definition, for example:

@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

###############################################################################
# Refer to
# `TorchScript documentation <https://pytorch.org/docs/stable/jit.html>`_
# for more advanced use cases.

###############################################################################
# Enable channels_last memory format for computer vision models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch 1.5 introduced support for ``channels_last`` memory format for
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
# Use oneDNN Graph with TorchScript for inference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# oneDNN Graph can significantly boost inference performance. It fuses some compute-intensive operations such as convolution, matmul with their neighbor operations.
# In PyTorch 2.0, it is supported as a beta feature for ``Float32`` & ``BFloat16`` data-types.
# oneDNN Graph receives the model’s graph and identifies candidates for operator-fusion with respect to the shape of the example input.
# A model should be JIT-traced using an example input.
# Speed-up would then be observed after a couple of warm-up iterations for inputs with the same shape as the example input.
# The example code-snippets below are for resnet50, but they can very well be extended to use oneDNN Graph with custom models as well.

# Only this extra line of code is required to use oneDNN Graph
torch.jit.enable_onednn_fusion(True)

###############################################################################
# Using the oneDNN Graph API requires just one extra line of code for inference with Float32.
# If you are using oneDNN Graph, please avoid calling ``torch.jit.optimize_for_inference``.

# sample input should be of the same shape as expected inputs
sample_input = [torch.rand(32, 3, 224, 224)]
# Using resnet50 from torchvision in this example for illustrative purposes,
# but the line below can indeed be modified to use custom models as well.
model = getattr(torchvision.models, "resnet50")().eval()
# Tracing the model with example input
traced_model = torch.jit.trace(model, sample_input)
# Invoking torch.jit.freeze
traced_model = torch.jit.freeze(traced_model)

###############################################################################
# Once a model is JIT-traced with a sample input, it can then be used for inference after a couple of warm-up runs.

with torch.no_grad():
    # a couple of warm-up runs
    traced_model(*sample_input)
    traced_model(*sample_input)
    # speedup would be observed after warm-up runs
    traced_model(*sample_input)

###############################################################################
# While the JIT fuser for oneDNN Graph also supports inference with ``BFloat16`` datatype,
# performance benefit with oneDNN Graph is only exhibited by machines with AVX512_BF16
# instruction set architecture (ISA).
# The following code snippets serves as an example of using ``BFloat16`` datatype for inference with oneDNN Graph:

# AMP for JIT mode is enabled by default, and is divergent with its eager mode counterpart
torch._C._jit_set_autocast_mode(False)

with torch.no_grad(), torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16):
    # Conv-BatchNorm folding for CNN-based Vision Models should be done with ``torch.fx.experimental.optimization.fuse`` when AMP is used
    import torch.fx.experimental.optimization as optimization
    # Please note that optimization.fuse need not be called when AMP is not used
    model = optimization.fuse(model)
    model = torch.jit.trace(model, (example_input))
    model = torch.jit.freeze(model)
    # a couple of warm-up runs
    model(example_input)
    model(example_input)
    # speedup would be observed in subsequent runs.
    model(example_input)


###############################################################################
# Train a model on CPU with PyTorch ``DistributedDataParallel``(DDP) functionality
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For small scale models or memory-bound models, such as DLRM, training on CPU is also a good choice. On a machine with multiple sockets, distributed training brings a high-efficient hardware resource usage to accelerate the training process. `Torch-ccl <https://github.com/intel/torch-ccl>`_, optimized with Intel(R) ``oneCCL`` (collective communications library) for efficient distributed deep learning training implementing such collectives like ``allreduce``, ``allgather``, ``alltoall``, implements PyTorch C10D ``ProcessGroup`` API and can be dynamically loaded as external ``ProcessGroup``. Upon optimizations implemented in PyTorch DDP module, ``torch-ccl`` accelerates communication operations. Beside the optimizations made to communication kernels, ``torch-ccl`` also features simultaneous computation-communication functionality.

###############################################################################
# GPU specific optimizations
# --------------------------

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
#     `video <https://www.youtube.com/watch?v=jF4-_ZK_tyc&feature=youtu.be>`_,
#     `slides <https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf>`_
#   * native PyTorch AMP is available starting from PyTorch 1.6:
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
