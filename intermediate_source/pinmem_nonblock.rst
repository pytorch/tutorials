A guide on good usage of ``non_blocking`` and ``pin_memory()`` in PyTorch
=========================================================================

TL;DR
-----

Sending tensors from CPU to GPU can be made faster by using asynchronous
transfer and memory pinning, but:

-  Calling ``tensor.pin_memory().to(device, non_blocking=True)`` can be
   as twice as slow as a plain ``tensor.to(device)``;
-  ``tensor.to(device, non_blocking=True)`` is usually a good choice;
-  ``cpu_tensor.to("cuda", non_blocking=True).mean()`` is ok, but
   ``cuda_tensor.to("cpu", non_blocking=True).mean()`` will produce
   garbage.

.. code:: ipython3

    import torch
    assert torch.cuda.is_available(), "A cuda device is required to run this tutorial"

Introduction
------------

Sending data from CPU to GPU is a cornerstone of many applications that
use PyTorch. Given this, users should have a good understanding of what
tools and options they should be using when moving data from one device
to another. This tutorial focuses on two aspects of device-to-device
transfer: ``Tensor.pin_memory()`` and
``Tensor.to(device, non_blocking=True)``. We start by outlining the
theory surrounding these concepts, and then move to concrete test
examples of the features.

-  `Background <#background>`__

   -  `Memory management basics <#memory-management-basics>`__
   -  `CUDA and (non-)pageable memory <#cuda-and-non-pageable-memory>`__
   -  `Asynchronous vs synchronous
      operations <#asynchronous-vs-synchronous-operations>`__

-  `Deep dive <#deep-dive>`__

   -  ```pin_memory()`` <#pin_memory>`__
   -  ```non_blocking=True`` <#non_blockingtrue>`__
   -  `Synergies <#synergies>`__
   -  `Other directions (GPU -> CPU etc.) <#other-directions>`__

-  `Practical recommendations <#practical-recommendations>`__
-  `Case studies <#case-studies>`__
-  `Conclusion <#conclusion>`__
-  `Additional resources <#additional-resources>`__

Background
----------

Memory management basics
~~~~~~~~~~~~~~~~~~~~~~~~

When one creates a CPU tensor in PyTorch, the content of this tensor
needs to be placed in memory. The memory we talk about here is a rather
complex concept worth looking at carefully. We distinguish two types of
memories that are handled by the Memory Management Unit: the main memory
(for simplicity) and the disk (which may or may not be the hard drive).
Together, the available space in disk and RAM (physical memory) make up
the virtual memory, which is an abstraction of the total resources
available. In short, the virtual memory makes it so that the available
space is larger than what can be found on RAM in isolation and creates
the illusion that the main memory is larger than it actually is.

In normal circumstances, a regular CPU tensor is *paged*, which means
that it is divided in blocks called *pages* that can live anywhere in
the virtual memory (both in RAM or on disk). As mentioned earlier, this
has the advantage that the memory seems larger than what the main memory
actually is.

Typically, when a program accesses a page that is not in RAM, a “page
fault” occurs and the operating system (OS) then brings back this page
into RAM (*swap in* or *page in*). In turn, the OS may have to *swap
out* (or *page out*) another page to make room for the new page.

In contrast to pageable memory, a *pinned* (or *page-locked* or
*non-pegeable*) memory is a type of memory that cannot be swapped out to
disk. It allows for faster and more predictable access times, but has
the downside that it is more limited than the pageable memory (aka the
main memory).

.. figure:: /_static/img/pinmem.png
   :alt:

CUDA and (non-)pageable memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To understand how CUDA copies a tensor from CPU to CUDA, let’s consider
the two scenarios above: - If the memory is page-locked, the device can
access the memory directly in the main memory. The memory addresses are
well defined and functions that need to read these data can be
significantly accelerated. - If the memory is pageable, all the pages
will have to be brought to the main memory before being sent to the GPU.
This operation may take time and is less predictable than when executed
on page-locked tensors.

More precisely, when CUDA sends pageable data from CPU to GPU, it must
first create a page-locked copy of that data before making the transfer.

Asynchronous vs. Synchronous Operations with ``non_blocking=True`` (CUDA ``cudaMemcpyAsync``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When executing a copy from a host (e.g., CPU) to a device (e.g., GPU),
the CUDA toolkit offers modalities to do these operations synchronously
or asynchronously with respect to the host. In the synchronous case, the
call to ``cudaMemcpy`` that is queries by ``tensor.to(device)`` is
blocking in the python main thread, which means that the code will stop
until the data has been transferred to the device.

When calling ``tensor.to(device)``, PyTorch always makes a call to
```cudaMemcpyAsync`` <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79>`__.
If ``non_blocking=False`` (default), a ``cudaStreamSynchronize`` will be
called after each and every ``cudaMemcpyAsync``. If
``non_blocking=True``, no synchronization is triggered, and the main
thread on the host is not blocked. Therefore, from the host perspective,
multiple tensors can be sent to the device simultaneously in the latter
case, as the thread does not need for one transfer to be completed to
initiate the other.

Note
^^^^

In general, the transfer is blocking on the device size even if it’s not
on the host side: the copy on the device cannot occur while another
operation is being executed. However, in some advanced scenarios,
multiple copies or copy and kernel executions can be done simultaneously
on the GPU side. To enable this, three requirements must be met:

1. The device must have at least one free DMA (Direct Memory Access)
   engine. Modern GPU architectures such as Volterra, Tesla or H100
   devices have more than one DMA engine.
2. The transfer must be done on a separate, non-default cuda stream. In
   PyTorch, cuda streams can be handles using ``torch.cuda.Stream``.
3. The source data must be in pinned memory.

A PyTorch perspective
---------------------

``pin_memory()``
~~~~~~~~~~~~~~~~

PyTorch offers the possibility to create and send tensors to page-locked
memory through the ``pin_memory`` functions and arguments. Any cpu
tensor on a machine where a cuda is initialized can be sent to pinned
memory through the ``pin_memory`` method. Importantly, ``pin_memory`` is
blocking on the host: the main thread will wait for the tensor to be
copied to page-locked memory before executing the next operation. New
tensors can be directly created in pinned memory with functions like
``torch.zeros``, ``torch.ones`` and other constructors.

Let us check the speed of pinning memory and sending tensors to cuda:

.. code:: ipython3

    import torch
    import gc
    from torch.utils.benchmark import Timer
    
    tensor_pageable = torch.randn(100_000)
    
    tensor_pinned = torch.randn(100_000, pin_memory=True)
    
    print("Regular to(device)", 
          Timer("tensor_pageable.to('cuda:0')", globals=globals()).adaptive_autorange())
    print("Pinned to(device)", 
          Timer("tensor_pinned.to('cuda:0')", globals=globals()).adaptive_autorange())
    print("pin_memory() along", 
          Timer("tensor_pageable.pin_memory()", globals=globals()).adaptive_autorange())
    print("pin_memory() + to(device)", 
          Timer("tensor_pageable.pin_memory().to('cuda:0')", globals=globals()).adaptive_autorange())
    del tensor_pageable, tensor_pinned
    gc.collect()



.. parsed-literal::

    Regular to(device) <torch.utils.benchmark.utils.common.Measurement object at 0x7f354e7e32b0>
    tensor_pageable.to('cuda:0')
      Median: 35.26 us
      IQR:    0.04 us (35.23 to 35.28)
      4 measurements, 10000 runs per measurement, 1 thread
    Pinned to(device) <torch.utils.benchmark.utils.common.Measurement object at 0x7f3855d540a0>
    tensor_pinned.to('cuda:0')
      Median: 19.70 us
      IQR:    0.03 us (19.69 to 19.72)
      4 measurements, 10000 runs per measurement, 1 thread
    pin_memory() along <torch.utils.benchmark.utils.common.Measurement object at 0x7f383c0fffa0>
    tensor_pageable.pin_memory()
      Median: 11.82 us
      IQR:    0.03 us (11.80 to 11.83)
      4 measurements, 10000 runs per measurement, 1 thread
    pin_memory() + to(device) <torch.utils.benchmark.utils.common.Measurement object at 0x7f383c0ffb20>
    tensor_pageable.pin_memory().to('cuda:0')
      Median: 40.84 us
      IQR:    0.14 us (40.78 to 40.93)
      4 measurements, 10000 runs per measurement, 1 thread




.. parsed-literal::

    12



We can observe that casting a pinned-memory tensor to GPU is indeed much
faster than a pageable tensor, because under the hood, a pageable tensor
must be copied to pinned memory before being sent to GPU.

However, calling ``pin_memory()`` on a pageable tensor before casting it
to GPU does not bring any speed-up, on the contrary this call is
actually slower than just executing the transfer. Again, this makes
sense, since we’re actually asking python to execute an operation that
CUDA will perform anyway before copying the data from host to device.

``non_blocking=True``
~~~~~~~~~~~~~~~~~~~~~

As mentioned earlier, many PyTorch operations have the option of being
executed asynchronously with respect to the host through the
``non_blocking`` argument. Here, to account accurately of the benefits
of using ``non_blocking``, we will design a slightly more involved
experiment since we want to assess how fast it is to send multiple
tensors to GPU with and without calling ``non_blocking``.

.. code:: ipython3

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
    print("Call to `to(device)`", Timer("copy_to_device(*tensors)", globals=globals()).adaptive_autorange())
    print("Call to `to(device, non_blocking=True)`", Timer("copy_to_device_nonblocking(*tensors)",
                                                 globals=globals()).adaptive_autorange())


.. parsed-literal::

    Call to `to(device)` <torch.utils.benchmark.utils.common.Measurement object at 0x7f354e7e22f0>
    copy_to_device(*tensors)
      Median: 11.03 ms
      IQR:    0.09 ms (10.98 to 11.07)
      4 measurements, 10 runs per measurement, 1 thread
    Call to `to(device, non_blocking=True)` <torch.utils.benchmark.utils.common.Measurement object at 0x7f3855d1b7f0>
    copy_to_device_nonblocking(*tensors)
      Median: 5.88 ms
      IQR:    0.38 ms (5.82 to 6.20)
      4 measurements, 10 runs per measurement, 1 thread


To get a better sense of what is happening here, let us run a profiling
of these two code executions:

.. code:: ipython3

    from torch.profiler import profile, record_function, ProfilerActivity
    
    def profile_mem(cmd):
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            exec(cmd)
        print(cmd)
        print(prof.key_averages().table(row_limit=10))
    
    print("Call to `to(device)`", profile_mem("copy_to_device(*tensors)"))
    print("Call to `to(device, non_blocking=True)`", profile_mem("copy_to_device_nonblocking(*tensors)"))


.. parsed-literal::

    copy_to_device(*tensors)


.. parsed-literal::

    STAGE:2024-07-24 08:29:14 2357923:2357923 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
    STAGE:2024-07-24 08:29:14 2357923:2357923 ActivityProfilerController.cpp:320] Completed Stage: Collection
    STAGE:2024-07-24 08:29:14 2357923:2357923 ActivityProfilerController.cpp:324] Completed Stage: Post Processing


.. parsed-literal::

    -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     aten::to        11.39%       2.118ms        88.36%      16.432ms      16.432us          1000  
               aten::_to_copy        11.87%       2.208ms        86.48%      16.083ms      16.083us          1000  
          aten::empty_strided        26.90%       5.002ms        26.90%       5.002ms       5.002us          1000  
                  aten::copy_        16.27%       3.026ms        49.84%       9.269ms       9.269us          1000  
              cudaMemcpyAsync        11.25%       2.092ms        11.25%       2.092ms       2.092us          1000  
        cudaStreamSynchronize        22.32%       4.151ms        22.32%       4.151ms       4.151us          1000  
    -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 18.597ms
    
    Call to `to(device)` None


.. parsed-literal::

    STAGE:2024-07-24 08:29:14 2357923:2357923 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
    STAGE:2024-07-24 08:29:14 2357923:2357923 ActivityProfilerController.cpp:320] Completed Stage: Collection
    STAGE:2024-07-24 08:29:14 2357923:2357923 ActivityProfilerController.cpp:324] Completed Stage: Post Processing


.. parsed-literal::

    copy_to_device_nonblocking(*tensors)
    -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     aten::to        12.48%       1.621ms        88.23%      11.457ms      11.457us          1000  
               aten::_to_copy        15.72%       2.042ms        85.95%      11.162ms      11.162us          1000  
          aten::empty_strided        35.68%       4.633ms        35.68%       4.633ms       4.633us          1000  
                  aten::copy_        16.69%       2.167ms        35.85%       4.655ms       4.655us          1000  
              cudaMemcpyAsync        19.30%       2.506ms        19.30%       2.506ms       2.506us          1000  
        cudaDeviceSynchronize         0.13%      17.000us         0.13%      17.000us      17.000us             1  
    -------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 12.986ms
    
    Call to `to(device, non_blocking=True)` None


The results are without any doubt better when using
``non_blocking=True``, as all transfers are initiated simultaneously on
the host side. Note that, interestingly, ``to("cuda")`` actually
performs the same asynchrous device casting operation as the one with
``non_blocking=True`` with a synchronization point after each copy.

The benefit will vary depending on the number and the size of the
tensors as well as depending on the hardware being used.

Synergies
~~~~~~~~~

Now that we have made the point that data transfer of tensors already in
pinned memory to GPU is faster than from pageable memory, and that we
know that doing these transfers asynchronously is also faster than
synchronously, we can benchmark the various combinations at hand:

.. code:: ipython3

    
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
    print("pin_memory().to(device)", 
          Timer("pin_copy_to_device(*tensors)", globals=globals()).adaptive_autorange())
    print("pin_memory().to(device, non_blocking=True)", 
          Timer("pin_copy_to_device_nonblocking(*tensors)",
                                                 globals=globals()).adaptive_autorange())
    
    print("\nCall to `to(device)`")
    print("to(device)", 
          Timer("copy_to_device(*tensors)", globals=globals()).adaptive_autorange())
    print("to(device, non_blocking=True)", 
          Timer("copy_to_device_nonblocking(*tensors)",
                                                 globals=globals()).adaptive_autorange())
    
    print("\nCall to `to(device)` from pinned tensors")
    tensors_pinned = [torch.zeros(1000, pin_memory=True) for _ in range(1000)]
    print("tensor_pinned.to(device)", 
          Timer("copy_to_device(*tensors_pinned)", globals=globals()).adaptive_autorange())
    print("tensor_pinned.to(device, non_blocking=True)", 
          Timer("copy_to_device_nonblocking(*tensors_pinned)",
                                                 globals=globals()).adaptive_autorange())
    
    del tensors, tensors_pinned
    gc.collect()



.. parsed-literal::

    
    Call to `pin_memory()` + `to(device)`
    pin_memory().to(device) <torch.utils.benchmark.utils.common.Measurement object at 0x7f3855d56200>
    pin_copy_to_device(*tensors)
      Median: 17.18 ms
      IQR:    0.04 ms (17.16 to 17.20)
      4 measurements, 10 runs per measurement, 1 thread
    pin_memory().to(device, non_blocking=True) <torch.utils.benchmark.utils.common.Measurement object at 0x7f3855d558d0>
    pin_copy_to_device_nonblocking(*tensors)
      Median: 15.42 ms
      IQR:    0.08 ms (15.38 to 15.47)
      4 measurements, 10 runs per measurement, 1 thread
    
    Call to `to(device)`
    to(device) <torch.utils.benchmark.utils.common.Measurement object at 0x7f3855d57f70>
    copy_to_device(*tensors)
      Median: 13.15 ms
      IQR:    0.06 ms (13.13 to 13.20)
      4 measurements, 10 runs per measurement, 1 thread
    to(device, non_blocking=True) <torch.utils.benchmark.utils.common.Measurement object at 0x7f3543d2b7f0>
    copy_to_device_nonblocking(*tensors)
      Median: 8.26 ms
      IQR:    0.05 ms (8.23 to 8.27)
      4 measurements, 10 runs per measurement, 1 thread
    
    Call to `to(device)` from pinned tensors
    tensor_pinned.to(device) <torch.utils.benchmark.utils.common.Measurement object at 0x7f3855d558d0>
    copy_to_device(*tensors_pinned)
      Median: 13.28 ms
      IQR:    0.35 ms (13.13 to 13.48)
      4 measurements, 10 runs per measurement, 1 thread
    tensor_pinned.to(device, non_blocking=True) <torch.utils.benchmark.utils.common.Measurement object at 0x7f3855d56200>
    copy_to_device_nonblocking(*tensors_pinned)
      Median: 8.16 ms
      IQR:    0.08 ms (8.15 to 8.23)
      4 measurements, 10 runs per measurement, 1 thread




.. parsed-literal::

    40087



Other directions (GPU -> CPU, CPU -> MPS etc.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far, we have assumed that doing asynchronous copies from CPU to GPU
was safe. Indeed, it is a safe thing to do because CUDA will synchronize
whenever it is needed to make sure that the data being read is not
garbage. However, any other copy (e.g., from GPU to CPU) has no
guarantee whatsoever that the copy will be completed when the data is
read. In fact, if no explicit synchronization is done, the data on the
host can be garbage:

.. code:: ipython3

    tensor = torch.arange(1, 1_000_000, dtype=torch.double, device="cuda").expand(100, 999999).clone()
    torch.testing.assert_close(tensor.mean(), torch.tensor(500_000, dtype=torch.double, device="cuda")), tensor.mean()
    try:
        i = -1
        for i in range(100):
            cpu_tensor = tensor.to("cpu", non_blocking=True)
            torch.testing.assert_close(cpu_tensor.mean(), torch.tensor(500_000, dtype=torch.double))
        print("No test failed with non_blocking")
    except AssertionError:
        print(f"One test failed with non_blocking: {i}th assertion!")
    try:
        i = -1
        for i in range(100):
            cpu_tensor = tensor.to("cpu", non_blocking=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(cpu_tensor.mean(), torch.tensor(500_000, dtype=torch.double))
        print("No test failed with synchronize")
    except AssertionError:
        print(f"One test failed with synchronize: {i}th assertion!")



.. parsed-literal::

    One test failed with non_blocking: 0th assertion!
    No test failed with synchronize


The same observation could be made with copies from CPU to a non-CUDA
device such as MPS.

In summary, copying data from CPU to GPU is safe when using
``non_blocking=True``, but for any other direction,
``non_blocking=True`` can still be used but the user must make sure that
a device synchronization is executed after the data is accessed.

Practical recommendations
-------------------------

We can now wrap up some early recommendations based on our observations:
In general, ``non_blocking=True`` will provide a good speed of transfer,
regardless of whether the original tensor is or isn’t in pinned memory.
If the tensor is already in pinned memory, the transfer can be
accelerated, but sending it to pin memory manually is a blocking
operation on the host and hence will anihilate much of the benefit of
using ``non_blocking=True`` (and CUDA does the ``pin_memory`` transfer
anyway).

One might now legitimetely ask what use there is for the
``pin_memory()`` method within the ``torch.Tensor`` class. In the
following section, we will explore further how this can be used to
accelerate the data transfer even more.

Additional considerations
-------------------------

PyTorch notoriously provides a ``DataLoader`` class that accepts a
``pin_memory`` argument. Given everything we have said so far about
calls to ``pin_memory``, how does the dataloader manage to accelerate
data transfers?

The answer is resides in the fact that the dataloader reserves a
separate thread to copy the data from pageable to pinned memory, thereby
avoiding to block the main thread with this. Consider the following
example, where we send a list of tensors to cuda after calling
pin_memory on a separate thread:

A more isolated example of this is the TensorDict primitive from the
homonymous library: when calling ``TensorDict.to(device)``, the default
behaviour is to send these tensors to the device asynchronously and make
a ``device.synchronize()`` call after. ``TensorDict.to()`` also offers a
``non_blocking_pin`` argument which will spawn multiple threads to do
the calls to ``pin_memory()`` before launching the calls to
``to(device)``. This can further speed up the copies as the following
example shows:

.. code:: ipython3

    from tensordict import TensorDict
    import torch
    from torch.utils.benchmark import Timer
    
    td = TensorDict({str(i): torch.randn(1_000_000) for i in range(100)})
    
    print(Timer("td.to('cuda:0', non_blocking=False)", globals=globals()).adaptive_autorange())
    print(Timer("td.to('cuda:0')", globals=globals()).adaptive_autorange())
    print(Timer("td.to('cuda:0', non_blocking=True, non_blocking_pin=True)", globals=globals()).adaptive_autorange())


.. parsed-literal::

    /home/vmoens/.conda/envs/torchrl/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


.. parsed-literal::

    <torch.utils.benchmark.utils.common.Measurement object at 0x7f353770c4c0>
    td.to('cuda:0', non_blocking=False)
      Median: 35.55 ms
      IQR:    0.38 ms (35.43 to 35.81)
      4 measurements, 10 runs per measurement, 1 thread
    <torch.utils.benchmark.utils.common.Measurement object at 0x7f3539d7a920>
    td.to('cuda:0')
      Median: 32.59 ms
      IQR:    0.10 ms (32.55 to 32.65)
      4 measurements, 10 runs per measurement, 1 thread
    <torch.utils.benchmark.utils.common.Measurement object at 0x7f354d9cc430>
    td.to('cuda:0', non_blocking=True, non_blocking_pin=True)
      Median: 23.63 ms
      IQR:    0.39 ms (23.45 to 23.84)
      4 measurements, 1 runs per measurement, 1 thread


As a side note, it may be tempting to create everlasting buffers in
pinned memory and copy tensors from pageable memory to pinned memory,
and use these as shuttle before sending the data to GPU. Unfortunately,
this does not speed up computation because the bottleneck of copying
data to pinned memory is still present.

Another consideration is that transferring data that is stored on disk
(shared memory or files) to GPU will usually require the data to be
copied to pinned memory (which is on RAM) as an intermediate step.

Using ``non_blocking`` in these context for large amount of data may
have devastating effects on RAM consumption. In practice, there is no
silver bullet, and the performance of any combination of multithreaded
pin_memory and non_blocking will depend on multiple factors such as the
system being used, the OS, the hardware and the tasks being performed.

Finally, creating a large number of tensors or a few large tensors in
pinned memory will effectively reserve more RAM than pageable tensors
would, thereby lowering the amount of available RAM for other operations
(such as swapping pages in and out), which can have a negative impact
over the overall runtime of an algorithm.

Conclusion
----------

Additional resources
--------------------

