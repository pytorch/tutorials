# -*- coding: utf-8 -*-

"""
Introduction to ``torch.compile``
=================================
**Author:** William Wen
"""

######################################################################
# ``torch.compile`` is the new way to speed up your PyTorch code!
# ``torch.compile`` makes PyTorch code run faster by
# JIT-compiling PyTorch code into optimized kernels,
# while requiring minimal code changes.
#
# ``torch.compile`` accomplishes this by tracing through
# your Python code, looking for PyTorch operations.
# Code that is difficult to trace will result a
# **graph break**, which are lost optimization opportunities, rather
# than errors or silent incorrectness.
#
# ``torch.compile`` is available in PyTorch 2.0 and later.
#
# This introduction covers basic ``torch.compile`` usage
# and demonstrates the advantages of ``torch.compile`` over
# our previous PyTorch compiler solution,
# `TorchScript <https://pytorch.org/docs/stable/jit.html>`__.
#
# For an end-to-end example on a real model, check out our `end-to-end torch.compile tutorial <https://pytorch.org/tutorials/intermediate/torch_compile_full_example.html>`__.
#
# To troubleshoot issues and to gain a deeper understanding of how to apply ``torch.compile`` to your code, check out `the torch.compile programming model <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html>`__.
#
# **Contents**
#
# .. contents::
#     :local:
#
# **Required pip dependencies for this tutorial**
#
# - ``torch >= 2.0``
# - ``numpy``
# - ``scipy``
#
# **System requirements**
# - A C++ compiler, such as ``g++``
# - Python development package (``python-devel``/``python-dev``)

######################################################################
# Basic Usage
# ------------
#
# We turn on some logging to help us to see what ``torch.compile`` is doing
# under the hood in this tutorial.
# The following code will print out the PyTorch ops that ``torch.compile`` traced.

import torch

# sphinx_gallery_start_ignore
# to clear torch logs format
import os
os.environ["TORCH_LOGS_FORMAT"] = ""
torch._logging._internal.DEFAULT_FORMATTER = (
    torch._logging._internal._default_formatter()
)
torch._logging._internal._init_logs()
# sphinx_gallery_end_ignore

torch._logging.set_logs(graph_code=True)

######################################################################
# ``torch.compile`` is a decorator that takes an arbitrary Python function.


def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(3, 3), torch.randn(3, 3)))


@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


print(opt_foo2(torch.randn(3, 3), torch.randn(3, 3)))

######################################################################
# ``torch.compile`` is applied recursively, so nested function calls
# within the top-level compiled function will also be compiled.


def inner(x):
    return torch.sin(x)


@torch.compile
def outer(x, y):
    a = inner(x)
    b = torch.cos(y)
    return a + b


print(outer(torch.randn(3, 3), torch.randn(3, 3)))


######################################################################
# We can also optimize ``torch.nn.Module`` instances by either calling
# its ``.compile()`` method or by directly ``torch.compile``-ing the module.
# This is equivalent to ``torch.compile``-ing the module's ``__call__`` method
# (which indirectly calls ``forward``).

t = torch.randn(10, 100)


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))


mod1 = MyModule()
mod1.compile()
print(mod1(torch.randn(3, 3)))

mod2 = MyModule()
mod2 = torch.compile(mod2)
print(mod2(torch.randn(3, 3)))


######################################################################
# Demonstrating Speedups
# -----------------------
#
# Now let's demonstrate how ``torch.compile`` speeds up a simple PyTorch example.
# For a demonstration on a more complex model, see our `end-to-end torch.compile tutorial <https://pytorch.org/tutorials/intermediate/torch_compile_full_example.html>`__.


def foo3(x):
    y = x + 1
    z = torch.nn.functional.relu(y)
    u = z * 2
    return u


opt_foo3 = torch.compile(foo3)


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


inp = torch.randn(4096, 4096).cuda()
print("compile:", timed(lambda: opt_foo3(inp))[1])
print("eager:", timed(lambda: foo3(inp))[1])

######################################################################
# Notice that ``torch.compile`` appears to take a lot longer to complete
# compared to eager. This is because ``torch.compile`` takes extra time to compile
# the model on the first few executions.
# ``torch.compile`` re-uses compiled code whever possible,
# so if we run our optimized model several more times, we should
# see a significant improvement compared to eager.

# turn off logging for now to prevent spam
torch._logging.set_logs(graph_code=False)

eager_times = []
for i in range(10):
    _, eager_time = timed(lambda: foo3(inp))
    eager_times.append(eager_time)
    print(f"eager time {i}: {eager_time}")
print("~" * 10)

compile_times = []
for i in range(10):
    _, compile_time = timed(lambda: opt_foo3(inp))
    compile_times.append(compile_time)
    print(f"compile time {i}: {compile_time}")
print("~" * 10)

import numpy as np

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert speedup > 1
print(
    f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
)
print("~" * 10)

######################################################################
# And indeed, we can see that running our model with ``torch.compile``
# results in a significant speedup. Speedup mainly comes from reducing Python overhead and
# GPU read/writes, and so the observed speedup may vary on factors such as model
# architecture and batch size. For example, if a model's architecture is simple
# and the amount of data is large, then the bottleneck would be
# GPU compute and the observed speedup may be less significant.
#
# To see speedups on a real model, check out our `end-to-end torch.compile tutorial <https://pytorch.org/tutorials/intermediate/torch_compile_full_example.html>`__.

######################################################################
# Benefits over TorchScript
# -------------------------
#
# Why should we use ``torch.compile`` over TorchScript? Primarily, the
# advantage of ``torch.compile`` lies in its ability to handle
# arbitrary Python code with minimal changes to existing code.
#
# Compare to TorchScript, which has a tracing mode (``torch.jit.trace``) and
# a scripting mode (``torch.jit.script``). Tracing mode is susceptible to
# silent incorrectness, while scripting mode requires significant code changes
# and will raise errors on unsupported Python code.
#
# For example, TorchScript tracing silently fails on data-dependent control flow
# (the ``if x.sum() < 0:`` line below)
# because only the actual control flow path is traced.
# In comparison, ``torch.compile`` is able to correctly handle it.


def f1(x, y):
    if x.sum() < 0:
        return -y
    return y


# Test that `fn1` and `fn2` return the same result, given the same arguments `args`.
def test_fns(fn1, fn2, args):
    out1 = fn1(*args)
    out2 = fn2(*args)
    return torch.allclose(out1, out2)


inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)

traced_f1 = torch.jit.trace(f1, (inp1, inp2))
print("traced 1, 1:", test_fns(f1, traced_f1, (inp1, inp2)))
print("traced 1, 2:", test_fns(f1, traced_f1, (-inp1, inp2)))

compile_f1 = torch.compile(f1)
print("compile 1, 1:", test_fns(f1, compile_f1, (inp1, inp2)))
print("compile 1, 2:", test_fns(f1, compile_f1, (-inp1, inp2)))
print("~" * 10)

######################################################################
# TorchScript scripting can handle data-dependent control flow,
# but it can require major code changes and will raise errors when unsupported Python
# is used.
#
# In the example below, we forget TorchScript type annotations and we receive
# a TorchScript error because the input type for argument ``y``, an ``int``,
# does not match with the default argument type, ``torch.Tensor``.
# In comparison, ``torch.compile`` works without requiring any type annotations.


import traceback as tb

torch._logging.set_logs(graph_code=True)


def f2(x, y):
    return x + y


inp1 = torch.randn(5, 5)
inp2 = 3

script_f2 = torch.jit.script(f2)
try:
    script_f2(inp1, inp2)
except:
    tb.print_exc()

compile_f2 = torch.compile(f2)
print("compile 2:", test_fns(f2, compile_f2, (inp1, inp2)))
print("~" * 10)

######################################################################
# Graph Breaks
# ------------------------------------
# The graph break is one of the most fundamental concepts within ``torch.compile``.
# It allows ``torch.compile`` to handle arbitrary Python code by interrupting
# compilation, running the unsupported code, then resuming compilation.
# The term "graph break" comes from the fact that ``torch.compile`` attempts
# to capture and optimize the PyTorch operation graph. When unsupported Python code is encountered,
# then this graph must be "broken".
# Graph breaks result in lost optimization opportunities, which may still be undesirable,
# but this is better than silent incorrectness or a hard crash.
#
# Let's look at a data-dependent control flow example to better see how graph breaks work.


def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


opt_bar = torch.compile(bar)
inp1 = torch.ones(10)
inp2 = torch.ones(10)
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)

######################################################################
# The first time we run ``bar``, we see that ``torch.compile`` traced 2 graphs
# corresponding to the following code (noting that ``b.sum() < 0`` is False):
#
# 1. ``x = a / (torch.abs(a) + 1); b.sum()``
# 2. ``return x * b``
#
# The second time we run ``bar``, we take the other branch of the if statement
# and we get 1 traced graph corresponding to the code ``b = b * -1; return x * b``.
# We do not see a graph of ``x = a / (torch.abs(a) + 1)`` outputted the second time
# since ``torch.compile`` cached this graph from the first run and re-used it.
#
# Let's investigate by example how TorchDynamo would step through ``bar``.
# If ``b.sum() < 0``, then TorchDynamo would run graph 1, let
# Python determine the result of the conditional, then run
# graph 2. On the other hand, if ``not b.sum() < 0``, then TorchDynamo
# would run graph 1, let Python determine the result of the conditional, then
# run graph 3.
#
# We can see all graph breaks by using ``torch._logging.set_logs(graph_breaks=True)``.

# Reset to clear the torch.compile cache
torch._dynamo.reset()
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)

######################################################################
# In order to maximize speedup, graph breaks should be limited.
# We can force TorchDynamo to raise an error upon the first graph
# break encountered by using ``fullgraph=True``:

# Reset to clear the torch.compile cache
torch._dynamo.reset()

opt_bar_fullgraph = torch.compile(bar, fullgraph=True)
try:
    opt_bar_fullgraph(torch.randn(10), torch.randn(10))
except:
    tb.print_exc()

######################################################################
# In our example above, we can work around this graph break by replacing
# the if statement with a ``torch.cond``:

from functorch.experimental.control_flow import cond


@torch.compile(fullgraph=True)
def bar_fixed(a, b):
    x = a / (torch.abs(a) + 1)

    def true_branch(y):
        return y * -1

    def false_branch(y):
        # NOTE: torch.cond doesn't allow aliased outputs
        return y.clone()

    b = cond(b.sum() < 0, true_branch, false_branch, (b,))
    return x * b


bar_fixed(inp1, inp2)
bar_fixed(inp1, -inp2)


######################################################################
# In order to serialize graphs or to run graphs on different (i.e. Python-less)
# environments, consider using ``torch.export`` instead (from PyTorch 2.1+).
# One important restriction is that ``torch.export`` does not support graph breaks. Please check
# `the torch.export tutorial <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`__
# for more details on ``torch.export``.
#
# Check out our `section on graph breaks in the torch.compile programming model <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html>`__
# for tips on how to work around graph breaks.

######################################################################
# Troubleshooting
# ---------------
# Is ``torch.compile`` failing to speed up your model? Is compile time unreasonably long?
# Is your code recompiling excessively? Are you having difficulties dealing with graph breaks?
# Are you looking for tips on how to best use ``torch.compile``?
# Or maybe you simply want to learn more about the inner workings of ``torch.compile``?
#
# Check out `the torch.compile programming model <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html>`__.

######################################################################
# Conclusion
# ------------
#
# In this tutorial, we introduced ``torch.compile`` by covering
# basic usage, demonstrating speedups over eager mode, comparing to TorchScript,
# and briefly describing graph breaks.
#
# For an end-to-end example on a real model, check out our `end-to-end torch.compile tutorial <https://pytorch.org/tutorials/intermediate/torch_compile_full_example.html>`__.
#
# To troubleshoot issues and to gain a deeper understanding of how to apply ``torch.compile`` to your code, check out `the torch.compile programming model <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html>`__.
#
# We hope that you will give ``torch.compile`` a try!
