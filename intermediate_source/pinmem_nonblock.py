# -*- coding: utf-8 -*-
"""
A guide on good usage of `non_blocking` and `pin_memory()` in PyTorch
=====================================================================

TL;DR
-----

Sending tensors from CPU to GPU can be made faster by using asynchronous transfer and memory pinning, but:

- Calling `tensor.pin_memory().to(device, non_blocking=True)` can be as twice as slow as a plain `tensor.to(device)`;
- `tensor.to(device, non_blocking=True)` is usually a good choice;
- `cpu_tensor.to("cuda", non_blocking=True).mean()` will work, but `cuda_tensor.to("cpu", non_blocking=True).mean()`
  will produce garbage.

"""

import torch

assert torch.cuda.is_available(), "A cuda device is required to run this tutorial"


######################################################################
# Introduction
# ------------
#
# Sending data from CPU to GPU is a cornerstone of many applications that use PyTorch.
# Given this, users should have a good understanding of what tools and options they should be using
# when moving data from one device to another.
#
# This tutorial focuses on two aspects of device-to-device transfer: `Tensor.pin_memory()` and `Tensor.to(device,
# non_blocking=True)`.
# We start by outlining the theory surrounding these concepts, and then move to concrete test examples of the features.
#
# - [Background](#background)
#   - [Memory management basics](#memory-management-basics)
#   - [CUDA and (non-)pageable memory](#cuda-and-non-pageable-memory)
#   - [Asynchronous vs synchronous operations](#asynchronous-vs-synchronous-operations)
# - [Deep dive](#deep-dive)
#   - [`pin_memory()`](#pin_memory)
#   - [`non_blocking=True`](#non_blockingtrue)
#   - [Synergies](#synergies)
#   - [Other directions (GPU -> CPU etc.)](#other-directions)
# - [Practical recommendations](#practical-recommendations)
# - [Case studies](#case-studies)
# - [Conclusion](#conclusion)
# - [Additional resources](#additional-resources)
#
#
# Background
# ----------
#
# Memory management basics
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# When one creates a CPU tensor in PyTorch, the content of this tensor needs to be placed
# in memory. The memory we talk about here is a rather complex concept worth looking at carefully.
# We distinguish two types of memories that are handled by the Memory Management Unit: the main memory (for simplicity)
# and the disk (which may or may not be the hard drive). Together, the available space in disk and RAM (physical memory)
# make up the virtual memory, which is an abstraction of the total resources available.
# In short, the virtual memory makes it so that the available space is larger than what can be found on RAM in isolation
# and creates the illusion that the main memory is larger than it actually is.
#
# In normal circumstances, a regular CPU tensor is _paged_, which means that it is divided in blocks called _pages_ that
# can live anywhere in the virtual memory (both in RAM or on disk). As mentioned earlier, this has the advantage that
# the memory seems larger than what the main memory actually is.
#
# Typically, when a program accesses a page that is not in RAM, a "page fault" occurs and the operating system (OS) then brings
# back this page into RAM (_swap in_ or _page in_).
# In turn, the OS may have to _swap out_ (or _page out_) another page to make room for the new page.
#
# In contrast to pageable memory, a _pinned_ (or _page-locked_ or _non-pageable_) memory is a type of memory that cannot
# be swapped out to disk.
# It allows for faster and more predictable access times, but has the downside that it is more limited than the
# pageable memory (aka the main memory).
#
# .. figure:: /_static/img/pinmem.png
#    :alt:
#
# CUDA and (non-)pageable memory
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To understand how CUDA copies a tensor from CPU to CUDA, let's consider the two scenarios above:
# - If the memory is page-locked, the device can access the memory directly in the main memory. The memory addresses are well
#   defined and functions that need to read these data can be significantly accelerated.
# - If the memory is pageable, all the pages will have to be brought to the main memory before being sent to the GPU.
#   This operation may take time and is less predictable than when executed on page-locked tensors.
#
# More precisely, when CUDA sends pageable data from CPU to GPU, it must first create a page-locked copy of that data
# before making the transfer.
#
# Asynchronous vs. Synchronous Operations with `non_blocking=True` (CUDA `cudaMemcpyAsync`)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When executing a copy from a host (e.g., CPU) to a device (e.g., GPU), the CUDA toolkit offers modalities to do these
# operations synchronously or asynchronously with respect to the host. In the synchronous case, the call to `cudaMemcpy`
# that is queries by `tensor.to(device)` is blocking in the python main thread, which means that the code will stop until
# the data has been transferred to the device.
#
# When calling `tensor.to(device)`, PyTorch always makes a call to
# [`cudaMemcpyAsync`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79).
# If `non_blocking=False` (default), a `cudaStreamSynchronize` will be called after each and every `cudaMemcpyAsync`.
# If `non_blocking=True`, no synchronization is triggered, and the main thread on the host is not blocked.
# Therefore, from the host perspective, multiple tensors can be sent to the device simultaneously in the latter case,
# as the thread does not need for one transfer to be completed to initiate the other.
#
# .. note:: In general, the transfer is blocking on the device size even if it's not on the host side: the copy on the device cannot
#   occur while another operation is being executed. However, in some advanced scenarios, multiple copies or copy and kernel
#   executions can be done simultaneously on the GPU side. To enable this, three requirements must be met:
#
# 1. The device must have at least one free DMA (Direct Memory Access) engine. Modern GPU architectures such as Volterra,
#    Tesla or H100 devices have more than one DMA engine.
#
# 2. The transfer must be done on a separate, non-default cuda stream. In PyTorch, cuda streams can be handles using
#    `torch.cuda.Stream`.
#
# 3. The source data must be in pinned memory.
#
#
# A PyTorch perspective
# ---------------------
#
# `pin_memory()`
# ~~~~~~~~~~~~~~
#
# PyTorch offers the possibility to create and send tensors to page-locked memory through the `pin_memory` functions and
# arguments.
# Any cpu tensor on a machine where a cuda is initialized can be sent to pinned memory through the `pin_memory`
# method. Importantly, `pin_memory` is blocking on the host: the main thread will wait for the tensor to be copied to
# page-locked memory before executing the next operation.
# New tensors can be directly created in pinned memory with functions like `torch.zeros`, `torch.ones` and other
# constructors.
#
# Let us check the speed of pinning memory and sending tensors to cuda:


import torch
import gc
from torch.utils.benchmark import Timer

tensor_pageable = torch.randn(100_000)

tensor_pinned = torch.randn(100_000, pin_memory=True)

print(
    "Regular to(device)",
    Timer("tensor_pageable.to('cuda:0')", globals=globals()).adaptive_autorange(),
)
print(
    "Pinned to(device)",
    Timer("tensor_pinned.to('cuda:0')", globals=globals()).adaptive_autorange(),
)
print(
    "pin_memory() along",
    Timer("tensor_pageable.pin_memory()", globals=globals()).adaptive_autorange(),
)
print(
    "pin_memory() + to(device)",
    Timer(
        "tensor_pageable.pin_memory().to('cuda:0')", globals=globals()
    ).adaptive_autorange(),
)
del tensor_pageable, tensor_pinned
gc.collect()


######################################################################
# We can observe that casting a pinned-memory tensor to GPU is indeed much faster than a pageable tensor, because under
# the hood, a pageable tensor must be copied to pinned memory before being sent to GPU.
#
# However, calling `pin_memory()` on a pageable tensor before casting it to GPU does not bring any speed-up, on the
# contrary this call is actually slower than just executing the transfer. Again, this makes sense, since we're actually
# asking python to execute an operation that CUDA will perform anyway before copying the data from host to device.
#
# `non_blocking=True`
# ~~~~~~~~~~~~~~~~~~~
#
# As mentioned earlier, many PyTorch operations have the option of being executed asynchronously with respect to the host
# through the `non_blocking` argument.
# Here, to account accurately of the benefits of using `non_blocking`, we will design a slightly more involved experiment
# since we want to assess how fast it is to send multiple tensors to GPU with and without calling `non_blocking`.
#


def copy_to_device(*tensors, display_peak_mem=False):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0"))
    return result


def copy_to_device_nonblocking(*tensors, display_peak_mem=False):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0", non_blocking=True))
    # We need to synchronize
    torch.cuda.synchronize()
    return result


tensors = [torch.randn(1000) for _ in range(1000)]
print(
    "Call to `to(device)`",
    Timer("copy_to_device(*tensors)", globals=globals()).adaptive_autorange(),
)
print(
    "Call to `to(device, non_blocking=True)`",
    Timer(
        "copy_to_device_nonblocking(*tensors)", globals=globals()
    ).adaptive_autorange(),
)


######################################################################
# To get a better sense of what is happening here, let us run a profiling of these two code executions:


from torch.profiler import profile, record_function, ProfilerActivity


def profile_mem(cmd):
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        exec(cmd)
    print(cmd)
    print(prof.key_averages().table(row_limit=10))


print("Call to `to(device)`", profile_mem("copy_to_device(*tensors)"))
print(
    "Call to `to(device, non_blocking=True)`",
    profile_mem("copy_to_device_nonblocking(*tensors)"),
)


######################################################################
# The results are without any doubt better when using `non_blocking=True`, as all transfers are initiated simultaneously
# on the host side.
# Note that, interestingly, `to("cuda")` actually performs the same asynchronous device casting operation as the one with
# `non_blocking=True` with a synchronization point after each copy.
#
# The benefit will vary depending on the number and the size of the tensors as well as depending on the hardware being used.
#
# Synergies
# ~~~~~~~~~
#
# Now that we have made the point that data transfer of tensors already in pinned memory to GPU is faster than from
# pageable memory, and that we know that doing these transfers asynchronously is also faster than synchronously, we can
# benchmark the various combinations at hand:


def pin_copy_to_device(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.pin_memory().to("cuda:0"))
    return result


def pin_copy_to_device_nonblocking(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.pin_memory().to("cuda:0", non_blocking=True))
    # We need to synchronize
    torch.cuda.synchronize()
    return result


print("\nCall to `pin_memory()` + `to(device)`")
print(
    "pin_memory().to(device)",
    Timer("pin_copy_to_device(*tensors)", globals=globals()).adaptive_autorange(),
)
print(
    "pin_memory().to(device, non_blocking=True)",
    Timer(
        "pin_copy_to_device_nonblocking(*tensors)", globals=globals()
    ).adaptive_autorange(),
)

print("\nCall to `to(device)`")
print(
    "to(device)",
    Timer("copy_to_device(*tensors)", globals=globals()).adaptive_autorange(),
)
print(
    "to(device, non_blocking=True)",
    Timer(
        "copy_to_device_nonblocking(*tensors)", globals=globals()
    ).adaptive_autorange(),
)

print("\nCall to `to(device)` from pinned tensors")
tensors_pinned = [torch.zeros(1000, pin_memory=True) for _ in range(1000)]
print(
    "tensor_pinned.to(device)",
    Timer("copy_to_device(*tensors_pinned)", globals=globals()).adaptive_autorange(),
)
print(
    "tensor_pinned.to(device, non_blocking=True)",
    Timer(
        "copy_to_device_nonblocking(*tensors_pinned)", globals=globals()
    ).adaptive_autorange(),
)

del tensors, tensors_pinned
gc.collect()


######################################################################
# Other directions (GPU -> CPU, CPU -> MPS etc.)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# So far, we have assumed that doing asynchronous copies from CPU to GPU was safe.
# Indeed, it is a safe thing to do because CUDA will synchronize whenever it is needed to make sure that the data being
# read is not garbage.
# However, any other copy (e.g., from GPU to CPU) has no guarantee whatsoever that the copy will be completed when the
# data is read. In fact, if no explicit synchronization is done, the data on the host can be garbage:
#


tensor = (
    torch.arange(1, 1_000_000, dtype=torch.double, device="cuda")
    .expand(100, 999999)
    .clone()
)
torch.testing.assert_close(
    tensor.mean(), torch.tensor(500_000, dtype=torch.double, device="cuda")
), tensor.mean()
try:
    i = -1
    for i in range(100):
        cpu_tensor = tensor.to("cpu", non_blocking=True)
        torch.testing.assert_close(
            cpu_tensor.mean(), torch.tensor(500_000, dtype=torch.double)
        )
    print("No test failed with non_blocking")
except AssertionError:
    print(f"One test failed with non_blocking: {i}th assertion!")
try:
    i = -1
    for i in range(100):
        cpu_tensor = tensor.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            cpu_tensor.mean(), torch.tensor(500_000, dtype=torch.double)
        )
    print("No test failed with synchronize")
except AssertionError:
    print(f"One test failed with synchronize: {i}th assertion!")


######################################################################
# The same observation could be made with copies from CPU to a non-CUDA device such as MPS.
#
# In summary, copying data from CPU to GPU is safe when using `non_blocking=True`, but for any other direction,
# `non_blocking=True` can still be used but the user must make sure that a device synchronization is executed after
# the data is accessed.
#
# Practical recommendations
# -------------------------
#
# We can now wrap up some early recommendations based on our observations:
# In general, `non_blocking=True` will provide a good speed of transfer, regardless of whether the original tensor is or
# isn't in pinned memory. If the tensor is already in pinned memory, the transfer can be accelerated, but sending it to
# pin memory manually is a blocking operation on the host and hence will annihilate much of the benefit of using
# `non_blocking=True` (and CUDA does the `pin_memory` transfer anyway).
#
# One might now legitimately ask what use there is for the `pin_memory()` method within the `torch.Tensor` class. In the
# following section, we will explore further how this can be used to accelerate the data transfer even more.
#
# Additional considerations
# -------------------------
#
# PyTorch notoriously provides a `DataLoader` class that accepts a `pin_memory` argument.
# Given everything we have said so far about calls to `pin_memory`, how does the dataloader manage to accelerate data
# transfers?
#
# The answer is resides in the fact that the dataloader reserves a separate thread to copy the data from pageable to
# pinned memory, thereby avoiding to block the main thread with this. Consider the following example, where we send a list of
# tensors to cuda after calling pin_memory on a separate thread:
#
# A more isolated example of this is the TensorDict primitive from the homonymous library: when calling `TensorDict.to(device)`,
# the default behavior is to send these tensors to the device asynchronously and make a `device.synchronize()` call after.
# `TensorDict.to()` also offers a `non_blocking_pin` argument which will spawn multiple threads to do the calls to `pin_memory()`
# before launching the calls to `to(device)`.
# This can further speed up the copies as the following example shows:
#
# .. code-block:: bash
#
#    !pip3 install https://github.com/pytorch/tensordict
#

from tensordict import TensorDict
import torch
from torch.utils.benchmark import Timer

td = TensorDict({str(i): torch.randn(1_000_000) for i in range(100)})

print(
    Timer("td.to('cuda:0', non_blocking=False)", globals=globals()).adaptive_autorange()
)
print(Timer("td.to('cuda:0')", globals=globals()).adaptive_autorange())
print(
    Timer(
        "td.to('cuda:0', non_blocking=True, non_blocking_pin=True)", globals=globals()
    ).adaptive_autorange()
)


######################################################################
# As a side note, it may be tempting to create everlasting buffers in pinned memory and copy tensors from pageable memory
# to pinned memory, and use these as shuttle before sending the data to GPU.
# Unfortunately, this does not speed up computation because the bottleneck of copying data to pinned memory is still present.
#
# Another consideration is that transferring data that is stored on disk (shared memory or files) to GPU will usually
# require the data to be copied to pinned memory (which is on RAM) as an intermediate step.
#
# Using `non_blocking` in these context for large amount of data may have devastating effects on RAM consumption.
# In practice, there is no silver bullet, and the performance of any combination of multithreaded pin_memory and
# non_blocking will depend on multiple factors such as the system being used, the OS, the hardware and the tasks being performed.
#
# Finally, creating a large number of tensors or a few large tensors in pinned memory will effectively reserve more RAM
# than pageable tensors would, thereby lowering the amount of available RAM for other operations (such as swapping pages
# in and out), which can have a negative impact over the overall runtime of an algorithm.

######################################################################
# ## Conclusion
#
# ## Additional resources
#
