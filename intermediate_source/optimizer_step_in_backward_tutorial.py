"""

How to save memory by fusing the optimizer step into the backward pass
======================================================================

Hello there! This tutorial aims to showcase one way of reducing the
memory footprint of a training loop by reducing the memory taken by
the gradients. Say you have a model and you're interested in ways to
optimize memory to avoid OOMing or simply to ooze more out of your GPU.
Well, you _might_ be in luck! We will explore

1. What takes up memory during your training or finetuning loop,
2. Capturing and visualizing memory snapshots to determine the memory bottleneck,
3. The new `tensor.post_accumulate_grad_hook(hook)` API, and finally, if relevant,
4. How everything fits together in 10 lines to achieve memory savings

The ingredients and tools required:
1.  PyTorch 2.1.0 or newer with torchvision
2.  A CUDA GPU

Let us start by importing the required modules and models. We will use a
vision transformer model from torchvision, but feel free to substitute with
your own model. We will also use `torch.optim.Adam` as our optimizer, but,
again, feel free to substitute with your own optimizer.

"""

import torch
from torchvision import models
from pickle import dump

model = models.vit_l_16(weights='DEFAULT').cuda()
optimizer = torch.optim.Adam(model.parameters())

###############################################################################
# Now let's define our typical training loop. You should use real images when
# training, but for the purposes of this tutorial, we are passing in fake
# inputs and not worrying about loading actual data.

IMAGE_SIZE = 224

def train(model, optimizer):
  # create our fake image input: tensor shape is batch_size, channels, height, width
  fake_image = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()

  # call our forward and backward
  loss = model.forward(fake_image)
  loss.sum().backward()

  # optimizer update
  optimizer.step()
  optimizer.zero_grad()

###############################################################################
# So what comprises the memory usage during training?
# """""""""""""""""""""""""""""""""""""""""""""""""""
# We are about to look at some memory snapshots, so we should be prepared to
# analyze them properly. People normally consider training memory to consist of
#
# 1. Model parameters (size P)
# 2. Activations (size A)
# 3. Gradients, which are the same size as the model parameters, so size G = P
# 4. Optimizer state, which is usually a relation to the model parameters. In
#    this case, Adam state requires 2x the model parameters, so size O = 2P
# 5. Intermediate tensors, which are allocated throughout the compute. We will
#    not worry about them for now as they are usually small and ephemeral.
#
# Let's get us a memory snapshot! As your code runs, consider what you may expect
# the CUDA memory timeline to look like.

# tell CUDA to start recording memory allocations
torch.cuda.memory._record_memory_history()

# train 3 steps
train(model, optimizer)

# save a snapshot of the memory allocations
s = torch.cuda.memory._snapshot()
with open(f"snapshot.pickle", "wb") as f:
    dump(s, f)

raise RuntimeError("Stop here and open up the snapshot in Zach Devito's CUDA Memory Visualizer")

###############################################################################
# Now open up the snapshot in Zach Devito's [CUDA Memory Visualizer](
# https://zdevito.github.io/assets/viz/) by dragging the snapshot.pickle file.
# Does the memory timeline match your expectations?
# 
# The model parameters have already been loaded in memory before the training
# step, so we anticipate seeing a chunk of memory devoted to the weights right
# off the bat. As we start our forward, memory should be allocated gradually
# for the activations, or the tensors we are saving to be able to compute gradients
# in the backward. Once we start the backward, the activations should be gradually
# freed while memory of the gradients start building up.
# 
# Lastly, as the optimizer kicks in, its state will be lazily initialized, so we 
# should see the optimizer state memory gradually increase during the end of the
# first training loop only. In future loops, the optimizer memory will remain and
# be inplace updated. The memory for the gradients should be freed accordingly
# by the end of every training loop.

m = SomeModule()

###############################################################################
# This allocates memory for all parameters/buffers and initializes them per
# the default initialization schemes defined in `SomeModule.__init__()`, which
# is wasteful when we want to load a checkpoint as
#     1. We are running the initialization kernels where the results are
#        immediately overwritten by `load_state_dict()`
#     2. We are allocating memory for these parameters/buffers in RAM while
#        `torch.load` of the saved state dictionary also allocates memory for
#        the parameters/buffers in the checkpoint.
#
# In order to solve these two problems, we can use the `torch.device()`
# context manager with `device='meta'` when we instantiate the `nn.Module()`.

with torch.device('meta'):
  meta_m = SomeModule()

###############################################################################
# The [`torch.device()`](https://pytorch.org/docs/main/tensor_attributes.html#torch-device)
# context manager makes sure that factory calls will be performed as if they
# were passed device as an argument. However, it does not affect factory
# function calls which are called with an explicit device argument.
#
# Tensors on the `meta` device do not carry data. However, they possess all
# other metadata a tensor carries such as `.size()` and `.stride()`,
# `.requires_grad` etc.
#
# Next, we consider the loading of the state dictionary.

m.load_state_dict(state_dict)

###############################################################################
# `nn.Module.load_state_dict()` is usually implemented via an in-place
# `param_in_model.copy_(param_in_state_dict)` (i.e. a copy from the
# parameter/buffer with the corresponding key in the state dictionary into
# the parameters/buffers in the `nn.Module`).
#
# However, an in-place copy into a tensor on the `meta` device is a no-op.
# In order to avoid this, we can pass the `assign=True` keyword argument to
# `load_state_dict()`.

meta_m.load_state_dict(state_dict, assign=True)

###############################################################################
# Another caveat here is that since optimizers hold a reference to
# `nn.Module.parameters()`, the optimizer must be initialized after the module
# is loaded from state dict if `assign=True` is passed.

###############################################################################
# To recap, in this tutorial, we learned about `torch.load(mmap=True)`, the
# `torch.device()` context manager with `device=meta` and the
# `nn.Module.load_state_dict(assign=True)` and how these tools could be used
# to aid when loading a model from a checkpoint.