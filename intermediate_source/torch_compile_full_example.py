# -*- coding: utf-8 -*-

"""
``torch.compile`` End-to-End Tutorial
=================================
**Author:** William Wen
"""

######################################################################
# ``torch.compile`` is the new way to speed up your PyTorch code!
# ``torch.compile`` makes PyTorch code run faster by
# JIT-compiling PyTorch code into optimized kernels,
# while requiring minimal code changes.
#
# This tutorial covers an end-to-end example of training and evaluating a
# real model with ``torch.compile``. For a gentle introduction to ``torch.compile``,
# please check out `the introduction to torch.compile tutorial <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__.
#
# **Required pip Dependencies**
#
# - ``torch >= 2.0``
# - ``torchvision``
#
# .. grid:: 2
#
#     .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
#        :class-card: card-prerequisites
#
#        * How to apply ``torch.compile`` to a real model
#        * ``torch.compile`` speedups on a real model
#        * ``torch.compile``'s first few iterations are expected to be slower due to compilation overhead
#
#     .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
#        :class-card: card-prerequisites
#
#        * `Introduction to torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__

# NOTE: a modern NVIDIA GPU (H100, A100, or V100) is recommended for this tutorial in
# order to reproduce the speedup numbers shown below and documented elsewhere.

import torch
import warnings

gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )


######################################################################
# Let's demonstrate how using ``torch.compile`` can speed up a real model.
# We will compare standard eager mode and
# ``torch.compile`` by evaluating and training a ``torchvision`` model on random data.
#
# Before we start, we need to define some utility functions.


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


# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to().cuda(),
        torch.randint(1000, (b,)).cuda(),
    )


N_ITERS = 10

from torchvision.models import densenet121


def init_model():
    return densenet121().cuda()


######################################################################
# First, let's compare inference.
#
# Note that in the call to ``torch.compile``, we have the additional
# ``mode`` argument, which we will discuss below.

model = init_model()

# Note that we generally recommend directly compiling a torch.nn.Module by calling
# its .compile() method.
model_opt = init_model()
model_opt.compile(mode="reduce-overhead")

inp = generate_data(16)[0]
with torch.no_grad():
    print("eager:", timed(lambda: model(inp))[1])
    print("compile:", timed(lambda: model_opt(inp))[1])

######################################################################
# Notice that ``torch.compile`` takes a lot longer to complete
# compared to eager. This is because ``torch.compile`` compiles
# the model into optimized kernels as it executes. In our example, the
# structure of the model doesn't change, and so recompilation is not
# needed. So if we run our optimized model several more times, we should
# see a significant improvement compared to eager.

eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    with torch.no_grad():
        _, eager_time = timed(lambda: model(inp))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    with torch.no_grad():
        _, compile_time = timed(lambda: model_opt(inp))
    compile_times.append(compile_time)
    print(f"compile eval time {i}: {compile_time}")
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
# You may also see different speedup results depending on the chosen ``mode``
# argument. The ``"reduce-overhead"`` mode uses CUDA graphs to further reduce
# the overhead of Python. For your own models,
# you may need to experiment with different modes to maximize speedup. You can
# read more about modes `here <https://pytorch.org/get-started/pytorch-2.0/#user-experience>`__.
#
# You may might also notice that the second time we run our model with ``torch.compile`` is significantly
# slower than the other runs, although it is much faster than the first run. This is because the ``"reduce-overhead"``
# mode runs a few warm-up iterations for CUDA graphs.
#
# Now, let's consider comparing training.

model = init_model()
opt = torch.optim.Adam(model.parameters())


def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()


eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, eager_time = timed(lambda: train(model, inp))
    eager_times.append(eager_time)
    print(f"eager train time {i}: {eager_time}")
print("~" * 10)

model = init_model()
opt = torch.optim.Adam(model.parameters())

# Note that because we are compiling a regular Python function, we do not
# call any .compile() method.
train_opt = torch.compile(train, mode="reduce-overhead")

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, compile_time = timed(lambda: train_opt(model, inp))
    compile_times.append(compile_time)
    print(f"compile train time {i}: {compile_time}")
print("~" * 10)

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert speedup > 1
print(
    f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
)
print("~" * 10)

######################################################################
# Again, we can see that ``torch.compile`` takes longer in the first
# iteration, as it must compile the model, but in subsequent iterations, we see
# significant speedups compared to eager.
#
# We remark that the speedup numbers presented in this tutorial are for
# demonstration purposes only. Official speedup values can be seen at the
# `TorchInductor performance dashboard <https://hud.pytorch.org/benchmark/compilers>`__.

######################################################################
# Conclusion
# ------------
#
# In this tutorial, we applied ``torch.compile`` to training and inference on a real model,
# demonstrating speedups.
#
# Importantly, we note that the first few iterations of a compiled model
# are slower than eager mode due to compilation overhead, but subsequent iterations are expected to
# have speedups.
#
# For a gentle introduction to ``torch.compile``, please check out `the introduction to torch.compile tutorial <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__.
#
# To troubleshoot issues and to gain a deeper understanding of how to apply ``torch.compile`` to your code, check out `the torch.compile programming model <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.html>`__.
#
# We hope that you will give ``torch.compile`` a try!
