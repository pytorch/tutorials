# -*- coding: utf-8 -*-
"""
A guide on good usage of `non_blocking` and `pin_memory()` in PyTorch
=====================================================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

Introduction
------------

Transferring data from the CPU to the GPU is fundamental in many PyTorch applications.
It's crucial for users to understand the most effective tools and options available for moving data between devices.
This tutorial examines two key methods for device-to-device data transfer in PyTorch:
:meth:`~torch.Tensor.pin_memory` and :meth:`~torch.Tensor.to` with the `non_blocking=True` option.

Key Learnings
~~~~~~~~~~~~~
Optimizing the transfer of tensors from the CPU to the GPU can be achieved through asynchronous transfers and memory
pinning. However, there are important considerations:
- Using `tensor.pin_memory().to(device, non_blocking=True)` can be up to twice as slow as a straightforward `tensor.to(device)`.
- Generally, `tensor.to(device, non_blocking=True)` is an effective choice for enhancing transfer speed.
- While `cpu_tensor.to("cuda", non_blocking=True).mean()` executes correctly, attempting
  `cuda_tensor.to("cpu", non_blocking=True).mean()` will result in erroneous outputs.

"""

import torch

assert torch.cuda.is_available(), "A cuda device is required to run this tutorial"


######################################################################
#
# We start by outlining the theory surrounding these concepts, and then move to concrete test examples of the features.
#
# - :ref:`Background <pinmem_background>`
#   - :ref:`Memory management basics <pinmem_mem>`
#   - :ref:`CUDA and (non-)pageable memory <pinmem_cuda_pageable_mem>`
#   - :ref:`Asynchronous vs. Synchronous Operations with `non_blocking=True` <pinmem_async_sync>`
# - :ref:`A PyTorch perspective <pinmem_pt_perspective>`
#   - :ref:`pin_memory <pinmem_pinmem>`
#   - :ref:`non_blocking=True <pinmem_nb>`
#   - :ref:`Synergies <synergies>`
#   - :ref:`Other directions (GPU -> CPU) <pinmem_otherdir>`
# - :ref:`Practical recommendations <pinmem_recom>`
# - :ref:`Additional considerations <pinmem_considerations>`
# - :ref:`Conclusion <pinmem_conclusion>`
# - :ref:`Additional resources <pinmem_resources>`
#
#
# Background
# ----------
#
#   .. _pinmem_background:
#
# Memory management basics
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
#   .. _pinmem_mem:
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
#   .. _pinmem_cuda_pageable_mem:
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
#   .. _pinmem_async_sync:
#
# When executing a copy from a host (e.g., CPU) to a device (e.g., GPU), the CUDA toolkit offers modalities to do these
# operations synchronously or asynchronously with respect to the host.
#
# In practice, when calling :meth:`~torch.Tensor.to`, PyTorch always makes a call to
# [`cudaMemcpyAsync`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79).
# If `non_blocking=False` (default), a `cudaStreamSynchronize` will be called after each and every `cudaMemcpyAsync`, making
# the call to :meth:`~torch.Tensor.to` blocking in the main thread.
# If `non_blocking=True`, no synchronization is triggered, and the main thread on the host is not blocked.
# Therefore, from the host perspective, multiple tensors can be sent to the device simultaneously,
# as the thread does not need to wait for one transfer to be completed to initiate the other.
#
# .. note:: In general, the transfer is blocking on the device side (even if it isn't on the host side):
#   the copy on the device cannot occur while another operation is being executed.
#   However, in some advanced scenarios, multiple copies or copy and kernel
#   executions can be done simultaneously on the GPU side. To enable this, three requirements must be met:
#
#   1. The device must have at least one free DMA (Direct Memory Access) engine. Modern GPU architectures such as Volterra,
#      Tesla or H100 devices have more than one DMA engine.
#
#   2. The transfer must be done on a separate, non-default cuda stream. In PyTorch, cuda streams can be handles using
#      :class:`~torch.cuda.Stream`.
#
#   3. The source data must be in pinned memory.
#
#
# A PyTorch perspective
# ---------------------
#
#   .. _pinmem_pt_perspective:
#
# `pin_memory()`
# ~~~~~~~~~~~~~~
#
#   .. _pinmem_pinmem:
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
import matplotlib.pyplot as plt


def timer(cmd):
    return Timer(cmd, globals=globals()).adaptive_autorange().median * 1000


pageable_tensor = torch.randn(1_000_000)

pinned_tensor = torch.randn(1_000_000, pin_memory=True)

pageable_to_device = timer("pageable_tensor.to('cuda:0')")
pinned_to_device = timer("pinned_tensor.to('cuda:0')")
pin_mem = timer("pageable_tensor.pin_memory()")
pin_mem_to_device = timer("pageable_tensor.pin_memory().to('cuda:0')")
r1 = pinned_to_device / pageable_to_device
r2 = pin_mem_to_device / pageable_to_device

fig, ax = plt.subplots()

xlabels = [0, 1, 2]
bar_labels = [
    "pageable_tensor.to(device) (1x)",
    f"pinned_tensor.to(device) ({r1:4.2f}x)",
    f"pageable_tensor.pin_memory().to(device) ({r2:4.2f}x)",
]
values = [pageable_to_device, pinned_to_device, pin_mem_to_device]
colors = ["tab:blue", "tab:red", "tab:orange"]
ax.bar(xlabels, values, label=bar_labels, color=colors)

ax.set_ylabel("Runtime (ms)")
ax.set_title("Device casting runtime (pin-memory)")
ax.set_xticks([])
ax.legend()

plt.show()

del pageable_tensor, pinned_tensor
gc.collect()

######################################################################
#
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
#   .. _pinmem_nb:
#
# As mentioned earlier, many PyTorch operations have the option of being executed asynchronously with respect to the host
# through the `non_blocking` argument.
# Here, to account accurately of the benefits of using `non_blocking`, we will design a slightly more involved experiment
# since we want to assess how fast it is to send multiple tensors to GPU with and without calling `non_blocking`.
#


def copy_to_device(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0"))
    return result


def copy_to_device_nonblocking(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0", non_blocking=True))
    # We need to synchronize
    torch.cuda.synchronize()
    return result


tensors = [torch.randn(1000) for _ in range(1000)]
to_device = timer("copy_to_device(*tensors)")
to_device_nonblocking = timer("copy_to_device_nonblocking(*tensors)")

r1 = to_device_nonblocking / to_device

fig, ax = plt.subplots()

xlabels = [0, 1]
bar_labels = [f"to(device) (1x)", f"to(device, non_blocking=True) ({r1:4.2f}x)"]
colors = ["tab:blue", "tab:red"]
values = [to_device, to_device_nonblocking]

ax.bar(xlabels, values, label=bar_labels, color=colors)

ax.set_ylabel("Runtime (ms)")
ax.set_title("Device casting runtime (non-blocking)")
ax.set_xticks([])
ax.legend()

plt.show()


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
#   .. _pinmem_synergies:
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


tensors_pinned = [torch.randn(1000, pin_memory=True) for _ in range(1000)]

pin_and_copy = timer("pin_copy_to_device(*tensors)")
pin_and_copy_nb = timer("pin_copy_to_device_nonblocking(*tensors)")

page_copy = timer("copy_to_device(*tensors)")
page_copy_nb = timer("copy_to_device_nonblocking(*tensors)")

pinned_copy = timer("copy_to_device(*tensors_pinned)")
pinned_copy_nb = timer("copy_to_device_nonblocking(*tensors_pinned)")

strategies = ("pageable copy", "pinned copy", "pin and copy")
blocking = {
    "blocking": [page_copy, pinned_copy, pin_and_copy],
    "non-blocking": [page_copy_nb, pinned_copy_nb, pin_and_copy_nb],
}

x = torch.arange(3)
width = 0.25
multiplier = 0


fig, ax = plt.subplots(layout="constrained")

for attribute, runtimes in blocking.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, runtimes, width, label=attribute)
    ax.bar_label(rects, padding=3, fmt="%.2f")
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Runtime (ms)")
ax.set_title("Runtime (pin-mem and non-blocking)")
ax.set_xticks([])
ax.legend(loc="upper left", ncols=3)

plt.show()

del tensors, tensors_pinned
gc.collect()


######################################################################
# Other directions (GPU -> CPU, CPU -> MPS etc.)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   .. _pinmem_otherdir:
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
#   .. _pinmem_recom:
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
#   .. _pinmem_considerations:
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

copy_blocking = timer("td.to('cuda:0', non_blocking=False)")
copy_non_blocking = timer("td.to('cuda:0')")
copy_pin_nb = timer("td.to('cuda:0', non_blocking_pin=True, num_threads=0)")
copy_pin_multithread_nb = timer("td.to('cuda:0', non_blocking_pin=True, num_threads=4)")


r1 = copy_non_blocking / copy_blocking
r2 = copy_pin_nb / copy_blocking
r3 = copy_pin_multithread_nb / copy_blocking

fig, ax = plt.subplots()

xlabels = [0, 1, 2, 3]
bar_labels = [
    "Blocking copy (1x)",
    f"Non-blocking copy ({r1:4.2f}x)",
    f"Blocking pin, non-blocking copy ({r2:4.2f}x)",
    f"Non-blocking pin, non-blocking copy ({r3:4.2f}x)",
]
values = [copy_blocking, copy_non_blocking, copy_pin_nb, copy_pin_multithread_nb]
colors = ["tab:blue", "tab:red", "tab:orange", "tab:green"]

ax.bar(xlabels, values, label=bar_labels, color=colors)

ax.set_ylabel("Runtime (ms)")
ax.set_title("Device casting runtime")
ax.set_xticks([])
ax.legend()

plt.show()

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
#   .. _pinmem_conclusion:
#
# ## Additional resources
#
#   .. _pinmem_resources:
#
