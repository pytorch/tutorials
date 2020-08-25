# -*- coding: utf-8 -*-
"""
Automatic Mixed Precision in PyTorch
*******************************************************
**Author**: `Michael Carilli <https://github.com/mcarilli>`_

`torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_ provides convenience methods for mixed precision,
where some operations use the ``torch.float32`` (``float``) datatype and other operations
use ``torch.float16`` (``half``). Some ops, like linear layers and convolutions,
are much faster in ``float16``. Other ops, like reductions, often require the dynamic
range of ``float32``.  Mixed precision tries to match each op to its appropriate datatype.
which can reduce your network's runtime and memory footprint.

Ordinarily, "automatic mixed precision training" uses `torch.cuda.amp.autocast <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast>`_ and
`torch.cuda.amp.GradScaler <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`_ together.
Here we'll walk through adding ``autocast`` and ``GradScaler`` to a toy network.
First we'll cover typical use, then describe more advanced cases.

.. contents:: :local:
"""

import torch, time, gc

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

def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()

# batch_size, in_size, out_size, and num_layers are chosen to be large enough to saturate the GPU.
# Typically, mixed precision provides the greatest speedup when GPU is working hard.
# Small networks may be CPU bound, in which case mixed precision won't improve performance.
# Sizes are also chosen such that linear layers' participating dimensions are multiples of 8,
# to permit Tensor Core usage on Tensor Core-capable GPUs (see :ref:`Troubleshooting <Troubleshooting>`).
#
# Exercise: Vary participating sizes and see how the mixed precision speedup changes.
batch_size = 512 # Try, for example, 128, 256, 513.
in_size = 4096
out_size = 4096
num_layers = 3
num_batches = 50
epochs = 3

# Creates data in default precision.  The same data is used for both default and mixed precision trials below.
# You don't need to manually change the type of input data when enabling mixed precision.
data = [torch.randn(batch_size, in_size, device="cuda") for _ in range(num_batches)]
targets = [torch.randn(batch_size, out_size, device="cuda") for _ in range(num_batches)]
loss_fn = torch.nn.MSELoss().cuda()

######################################################################
# Default Precision (Baseline)
# ----------------------------
#
# Without torch.cuda.amp, the following simple network executes all ops in default precision (torch.float32):

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

######################################################################
# Adding autocast
# ---------------
#
for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        # Runs the forward pass under autocast
        with torch.cuda.amp.autocast():
            output = net(input)
            # Linear layers with ``float32`` inputs `autocast to float16 <https://pytorch.org/docs/stable/amp.html#ops-that-can-autocast-to-float16>`_
            assert output.dtype is torch.float16

            loss = loss_fn(output, target)
            # ``mse_loss`` layers with ``float16`` inputs `autocast to float32 <https://pytorch.org/docs/stable/amp.html#ops-that-can-autocast-to-float16>`_
            assert loss.dtype is torch.float32

        # Exits autocast before backward().
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance

######################################################################
# Adding GradScaler
# -----------------
#
# See `Gradient Scaling <https://pytorch.org/docs/stable/amp.html#gradient-scaling>`_
# for a full explanation of each step.

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

######################################################################
# All together
# ------------

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.cuda.amp.autocast():
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
end_timer_and_print("With mixed precision:")


######################################################################
# Inspecting/modifying gradients (e.g., gradient clipping)
# --------------------------------------------------------
#
# All gradients produced by ``scaler.scale(loss).backward()`` are scaled.  If you wish to modify or inspect
# the parameters' ``.grad`` attributes between ``backward()`` and ``scaler.step(optimizer)``,  you should
# unscale them first using `scaler.unscale_(optimizer)`.

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

######################################################################
# Advanced topics
# ---------------
#
# See the `Automatic Mixed Precision Examples <https://pytorch.org/docs/stable/notes/amp_examples.html>`_ for advanced use cases including:
# * Gradient penalty/double backward
# * Networks with multiple models, optimizers, or losses
# * Multiple GPUs (``torch.nn.DataParallel`` or ``torch.nn.parallel.DistributedDataParallel``)
# * Custom autograd functions (subclasses of ``torch.autograd.Function``)

######################################################################
# Troubleshooting
# ---------------
#
# Speedup with Amp is minor
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Your network may not be saturating the GPU(s) with work, and is therefore CPU bound. Amp's effect on GPU performance
#    won't matter.  A rough rule of thumb to saturate the GPU is to increase batch and/or network size(s)
#    as much as you can without running OOM.  Also, try to avoid excessive CPU-GPU synchronization (``.item()`` calls, or
#    printing values from CUDA tensors), and try to avoid sequences of many small CUDA ops (coalesce these into a few
#    large CUDA ops if you can).
# 2. Your network may be compute bound (lots of matmuls/convolutions) but your GPU does not have Tensor Cores.
#    In this case a more modest speedup is expected.
# 3. Matmul dimensions are not Tensor Core-friendly.  Make sure matmuls' participating sizes are multiples of 8.
#    (For NLP models with encoders/decoders, this can be subtle.  Also. convolutions used to have similar size constraints
#    for Tensor Core use, but for CuDNN versions 7.3 and later, no such constraints exist.  See `here <https://github.com/NVIDIA/apex/issues/221#issuecomment-478084841>` for details).
#
#
# Loss is inf/NaN
# ~~~~~~~~~~~~~~~
# First, check if your network fits an advanced use case in the `Automatic Mixed Precision Examples <https://pytorch.org/docs/stable/notes/amp_examples.html>`_.
# If you're confident your Amp usage is correct, you may need to file an issue, but before doing so, it's helpful to gather the following information:
# 1. Try disabling ``autocast`` or ``GradScaler`` individually (by passing ``enabled=False`` to their constructor) and see if inf/NaN persist.
# 2. ???
# 3. profit
