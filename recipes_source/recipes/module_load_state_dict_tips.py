"""

Tips for Loading an `nn.Module` from a Checkpoint
=================================================

In this tutorial, we will share some tips for loading a model from a checkpoint.
In particular, we will discuss


1.  The `mmap` keyword argument on `torch.load`
2.  The `torch.device()` context manager
3.  The `assign` keyword argument on `nn.Module.load_state_dict()`

The following snippet of code illustrates the use of the above three utilities.
"""

import torch
from torch import nn

class SomeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(1, 1) for i in range(10)])

    def forward(self, x):
        return self.linears(x)


m = SomeModule()
torch.save(m.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth', mmap=True)
with torch.device('meta'):
  meta_m = SomeModule()
meta_m.load_state_dict(state_dict, assign=True)

###############################################################################
# Taking a step back, let us inspect the following more vanilla code snippet
# that does not use any of the features listed above

state_dict = torch.load('checkpoint.pth')
m = SomeModule()
m.load_state_dict(state_dict)

###############################################################################
# At `torch.save` time, tensor storages are tagged with the device they are
# saved on. At `torch.load` time, tensor storages will be loaded to the device
# they were tagged with (unless this behavior is overridden using the
# `map_location` flag). For ease of explanation, let us assume that the tensors
# were saved on CPU. This means that on the first line

state_dict = torch.load('checkpoint.pth')

###############################################################################
# All tensor storages will be loaded into CPU RAM, which can be problematic when
#     1. CPU RAM is smaller than the size of the checkpoint
#     2. waiting for the entire checkpoint to be loaded into RAM before
#        doing for example some per-tensor processing
#
# The `mmap` keyword argument to `torch.load` attempts to solve the above two
# problems by using an [`mmap` call](https://man7.org/linux/man-pages/man2/mmap.2.html)
# on the checkpoint, so that tensor storages are memory-mapped and when they are
# fetched from disk to memory is managed by the OS.

state_dict = torch.load('checkpoint.pth', mmap=True)


################################################################################
# For example

# def my_special_routine(t):
#     # for example, post training quantization and move t to device
#     pass

# def my_processing_function(key, device):
#     t = state_dict[key]
#     processed_t = my_special_routine(t)
#     del t
#     return processed_t

# for key in state_dict.keys():
#     device = 'cuda' + str(int(key[0]) % 8)
#     state_dict[key] = my_processing_function(key, device)


# Next, we consider the creation of the module.

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
# other metadata a tensor carries such as ```.size()`` and ```.stride()``,
# ``.requires_grad`` etc.
#
# Next, we consider the loading of the state dictionary.

m.load_state_dict(state_dict)

###############################################################################
# ``nn.Module.load_state_dict()`` is usually implemented via an in-place
# ``param_in_model.copy_(param_in_state_dict)`` (i.e. a copy from the
# parameter/buffer with the corresponding key in the state dictionary into
# the parameters/buffers in the `nn.Module`).
#
# However, an in-place copy into a tensor on the ``meta``` device is a no-op.
# In order to avoid this, we can pass the `assign=True` keyword argument to
# ``load_state_dict()``.

meta_m.load_state_dict(state_dict, assign=True)

###############################################################################
# A caveat here is that since optimizers hold a reference to
# ``nn.Module.parameters()``, the optimizer must be initialized after the module
# is loaded from state dict if `assign=True` is passed.

###############################################################################
# To recap, in this tutorial, we learned about ``torch.load(mmap=True)``, the
# ``torch.device()`` context manager with ``device=meta`` and the
# ``nn.Module.load_state_dict(assign=True)`` and how these tools could be used
# to aid when loading a model from a checkpoint.
