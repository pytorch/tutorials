"""
Profiling your PyTorch Module
------------
**Author:** `Suraj Subramanian <https://github.com/suraj813>`_

PyTorch includes a profiler API that is useful to identify the time and
memory costs of various PyTorch operations in your code. Profiler can be
easily integrated in your code, and the results can be printed as a table
or retured in a JSON trace file.

.. note::
    Profiler supports multithreaded models. Profiler runs in the
    same thread as the operation but it will also profile child operators
    that might run in another thread. Concurrently-running profilers will be
    scoped to their own thread to prevent mixing of results.

Head on over to `this
recipe <https://pytorch.org/tutorials/recipes/recipes/profiler.html>`__
for a quick walkthrough of the Profiler API.

"""

import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler


######################################################################
# Performance debugging using Profiler
# ~~~~~~~~~~~~~~~~~
#
# Profiler can be useful to identify performance bottlenecks in your
# models. In this example, we build a custom module that performs two
# sub-tasks:
#
# - a linear transformation on the input, and
# - use the transformation result to get indices on a mask tensor.
#
# We wrap the code for each sub-task in separate labelled context managers using
# ``profiler.record_function("label")``. In the profiler output, the
# aggregate performance metrics of all operations in the sub-task will
# show up under its corresponding label.
#
#
# Note that using Profiler incurs some overhead, and is best used only for investigating
# code. Remember to remove it if you are benchmarking runtimes.
#

class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)

        return out, hi_idx


######################################################################
# Profile the forward pass
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We initialize random input and mask tensors, and the ``CustomLinear`` module.
#
# Before we run the profiler, we warm-up CUDA to ensure accurate
# performance benchmarking. We wrap the forward pass of our module in the
# ``profiler.profile`` context manager. The ``with_stack=True`` parameter appends the
# file and line number of the operation in the trace.
#

model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)


######################################################################
# Pretty print profiler results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Finally, we print the profiler results. ``profiler.key_averages``
# aggregates the results by operator name, and optionally by input
# shapes and/or stack trace events.
# Grouping by input shapes is useful to identify which tensor shapes
# are utilized by the model.
#
# Here, we use ``group_by_stack_n=5`` which aggregates runtimes by the
# operation and its traceback (truncated to the most recent 5 events), and
# display the events in the order they are registered. The table can also
# be sorted by passing a ``sort_by`` argument (refer to the
# `docs <https://pytorch.org/docs/stable/autograd.html#profiler>`__ for
# valid sorting keys).
#

print(prof.key_averages(group_by_stack_n=5).table())


######################################################################
# Improve memory performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Note that the most expensive operations - in terms of memory and time -
# are at ``forward (11)`` or where we get the mask indices. Let’s try to
# tackle the memory consumption first. We can see that the ``.to()``
# operation consumes 953.67 Mb. ``mask`` is initialized with a
# ``torch.double`` datatype. Can we reduce the memory footprint by casting
# it to ``torch.float``?
#

model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table())


######################################################################
# Improve time performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# That worked, the memory footprint has halved - but while the time
# consumed has reduced a bit (previously 283ms), it’s still too high at
# 194ms. Turns out copying a matrix from CUDA to CPU is pretty expensive!
# The ``aten::copy_`` operator in ``forward (11)`` copies ``mask`` to CPU
# so that it can use the NumPy ``argwhere`` function. We could eliminate
# the need to copy into a NumPy array if we use a ``torch`` function
# ``nonzero()`` here instead.
#

class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("INDEX SCORE"):
            threshold = out.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)

        return out, hi_idx


model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table())


######################################################################
# Replacing ``np.argwhere`` with ``torch.nonzero`` reduced the total
# runtime to about 36ms (from 195ms) by eliminating the need to copy to
# CPU and instead running on the GPU.
#


######################################################################
# Caveat
# ^^^^^^^^^^^^^^
#
# You might notice introducing ``torch.nonzero`` in our module has shown a
# new memory footprint of 2.59 Gb that wasn’t in the stacktrace when we
# used ``np.argwhere``. This is because Profiler currently tracks only PyTorch tensors
# memory consumption and not NumPy arrays; it is likely that ``np.argwhere``
# occupied a similar amount of memory on the CPU but that will not be reported
# in the trace here.
#


######################################################################
# Further Reading
# ~~~~~~~~~~~~~~~~~
# We have seen how Profiler can be used to identify time and memory bottlenecks in PyTorch models.
# Read more about Profiler here:
#
# - `Profiler Usage Recipe <https://pytorch.org/tutorials/recipes/recipes/profiler.html>`__
# - `Profiling RPC-Based Workloads <https://pytorch.org/tutorials/recipes/distributed_rpc_profiling.html>`__
# - `Profiler API Docs <https://pytorch.org/docs/stable/autograd.html?highlight=profiler#profiler>`__
