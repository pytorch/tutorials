# -*- coding: utf-8 -*-
"""
Automatic Mixed Precision in PyTorch
************************************
**Author**: `Michael Carilli <https://github.com/mcarilli>`_

`torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_ provides convenience methods for mixed precision,
where some operations use the ``torch.float32`` (``float``) datatype and other operations
use ``torch.float16`` (``half``). Some ops, like linear layers and convolutions,
are much faster in ``float16``. Other ops, like reductions, often require the dynamic
range of ``float32``.  Mixed precision tries to match each op to its appropriate datatype.
which can reduce your network's runtime and memory footprint.

Ordinarily, "automatic mixed precision training" uses `torch.cuda.amp.autocast <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast>`_ and
`torch.cuda.amp.GradScaler <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`_ together.
This tutorial measures the performance of a simple network in default precision,
then walks through adding ``autocast`` and ``GradScaler`` to run the same network in
mixed precision with improved performance.

You may download and run this tutorial as a standalone Python script.
The only requirements are Pytorch 1.6+ and a CUDA-capable GPU.

.. contents:: :local:
"""

import torch, time, gc

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

##########################################################
# A simple network
# ----------------
#
# The following sequence of linear layers and ReLUs should show a nice speedup with mixed precision.

def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()

##########################################################
# ``batch_size``, ``in_size``, ``out_size``, and ``num_layers`` are chosen to be large enough to saturate the GPU with work.
# Typically, mixed precision provides the greatest speedup when GPU is saturated.
# Small networks may be CPU bound, in which case mixed precision won't improve performance.
# Sizes are also chosen such that linear layers' participating dimensions are multiples of 8,
# to permit Tensor Core usage on Tensor Core-capable GPUs (see :ref:`Troubleshooting<troubleshooting>` below).
#
# Exercise: Vary participating sizes and see how the mixed precision speedup changes.

batch_size = 512 # Try, for example, 128, 256, 513.
in_size = 4096
out_size = 4096
num_layers = 3
num_batches = 50
epochs = 3

# Creates data in default precision.
# The same data is used for both default and mixed precision trials below.
# You don't need to manually change the type of input data when enabling mixed precision.
data = [torch.randn(batch_size, in_size, device="cuda") for _ in range(num_batches)]
targets = [torch.randn(batch_size, out_size, device="cuda") for _ in range(num_batches)]

loss_fn = torch.nn.MSELoss().cuda()

##########################################################
# Default Precision
# -----------------
# Without torch.cuda.amp, the following simple network executes all ops in default precision (``torch.float32``):

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("With default precision:")

##########################################################
# Adding autocast
# ---------------
# Instances of `torch.cuda.amp.autocast <https://pytorch.org/docs/stable/amp.html#autocasting>`_ serve as context managers that allow regions of your script to run
# in mixed precision.
#
# In these regions, CUDA ops run in a dtype chosen by autocast
# to improve performance while maintaining accuracy.
# See the :ref:`Autocast Op Reference<autocast-op-reference>` for details on what precision
# autocast chooses for each op, and under what circumstances.

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        # Runs the forward pass under autocast.
        with torch.cuda.amp.autocast(enabled=try_amp):
            output = net(input)
            # output is float16 because linear layers autocast to float16.
            assert output.dtype is torch.float16

            loss = loss_fn(output, target)
            # loss is float32 because mse_loss layers autocast to float32.
            assert loss.dtype is torch.float32

        # Exits autocast before backward().
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance

##########################################################
# Adding GradScaler
# -----------------
# `Gradient scaling <https://pytorch.org/docs/stable/amp.html#gradient-scaling>`_
# helps prevent gradients with small magnitudes from flushing to zero
# ("underflowing") when training with mixed precision.
#
# ``torch.cuda.amp.GradScaler`` performs the steps of gradient scaling conveniently.

# Constructs scaler once, at the beginning of the convergence run, using default args.
# If your network fails to converge with default GradScaler args, please file an issue.
# The same GradScaler instance should be used for the entire convergence run.
# If you perform multiple convergence runs in the same script, each run should use
# a dedicated fresh GradScaler instance.  GradScaler instances are lightweight.

scaler = torch.cuda.amp.GradScaler()

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.cuda.amp.autocast():
            output = net(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(opt)

        # Updates the scale for next iteration.
        scaler.update()

        opt.zero_grad()

##########################################################
# All together
# ------------
#
# The following also demonstrates ``enabled``, an optional convenience argument to ``autocast`` and ``GradScaler``.
# If False, ``autocast`` and ``GradScaler``\ 's calls become no-ops.
# This allows switching between default precision and mixed precision without if/else statements.

use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
end_timer_and_print("With mixed precision:")

##########################################################
# Inspecting/modifying gradients (e.g., gradient clipping)
# --------------------------------------------------------
#
# All gradients produced by ``scaler.scale(loss).backward()`` are scaled.  If you wish to modify or inspect
# the parameters' ``.grad`` attributes between ``backward()`` and ``scaler.step(optimizer)``,  you should
# unscale them first using ``scaler.unscale_(optimizer)``.

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.cuda.amp.autocast():
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(opt)

        # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
        # You may use the same value for max_norm here as you would without gradient scaling.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

##########################################################
# Advanced topics
# ---------------
#
# See the `Automatic Mixed Precision Examples <https://pytorch.org/docs/stable/notes/amp_examples.html>`_ for advanced use cases including:
#
# * Gradient accumulation
# * Gradient penalty/double backward
# * Networks with multiple models, optimizers, or losses
# * Multiple GPUs (``torch.nn.DataParallel`` or ``torch.nn.parallel.DistributedDataParallel``)
# * Custom autograd functions (subclasses of ``torch.autograd.Function``)
#
# .. _troubleshooting:
#
# Troubleshooting
# ---------------
#
# Speedup with Amp is minor
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Your network may fail to saturate the GPU(s) with work, and is therefore CPU bound. Amp's effect on GPU performance
#    won't matter.
#
#    * A rough rule of thumb to saturate the GPU is to increase batch and/or network size(s)
#      as much as you can without running OOM.
#    * Try to avoid excessive CPU-GPU synchronization (``.item()`` calls, or printing values from CUDA tensors).
#    * Try to avoid sequences of many small CUDA ops (coalesce these into a few large CUDA ops if you can).
# 2. Your network may be compute bound (lots of matmuls/convolutions) but your GPU does not have Tensor Cores.
#    In this case a more modest speedup is expected.
# 3. Matmul dimensions are not Tensor Core-friendly.  Make sure matmuls' participating sizes are multiples of 8.
#    (For NLP models with encoders/decoders, this can be subtle.  Also. convolutions used to have similar size constraints
#    for Tensor Core use, but for CuDNN versions 7.3 and later, no such constraints exist.  See
#    `here <https://github.com/NVIDIA/apex/issues/221#issuecomment-478084841>`_ for guidance.)
#
# Loss is inf/NaN
# ~~~~~~~~~~~~~~~
# First, check if your network fits an advanced use case in the `Automatic Mixed Precision Examples <https://pytorch.org/docs/stable/notes/amp_examples.html>`_.
# See also `Prefer binary_cross_entropy_with_logits over binary_cross_entropy <https://pytorch.org/docs/stable/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy>`_.
#
# If you're confident your Amp usage is correct, you may need to file an issue, but before doing so, it's helpful to gather the following information:
#
# 1. Try disabling ``autocast`` or ``GradScaler`` individually (by passing ``enabled=False`` to their constructor) and see if infs/NaNs persist.
# 2. If you suspect some region of your network overflows (e.g., a complex loss function), run that forward region in ``float32``.
#    `The autocast docstring <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast>`_'s last code snippet
#    shows running a subregion in ``float32`` (by locally disabling autocast and casting the subregion's inputs).
#
# Type mismatch error (may manifest as CUDNN_STATUS_BAD_PARAM)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Autocast tries to cover all ops that benefit from or require casting.  The
# `ops that receive explicit coverage <https://pytorch.org/docs/stable/amp.html#autocast-op-reference>`_
# are based on reasoning about numerical properties, but also on experience.
# If you see a type mismatch error in an autocast-enabled forward region or a backward pass following that region,
# it's possible autocast missed an op.
#
# Please file an issue with the error backtrace.  ``export TORCH_SHOW_CPP_STACKTRACES=1`` before running your script to provide
# more fine-grained information on which backend op is failing.
