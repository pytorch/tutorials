"""

Tips for Loading an ``nn.Module`` from a Checkpoint
===================================================

In this tutorial, we will share some tips for loading a model from a checkpoint.
In particular, we will discuss


1.  The ``mmap`` keyword argument on ``torch.load``
2.  The ``torch.device()`` context manager
3.  The ``assign`` keyword argument on ``nn.Module.load_state_dict()``

.. note::
   This recipe requires PyTorch 2.1.0 or later.
"""


########################################
# Let us consider a simple ``nn.Module``
import torch
from torch import nn
import time

class SomeModule(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(size, size) for i in range(10)])

    def forward(self, x):
        return self.linears(x)


m = SomeModule(1000)
torch.save(m.state_dict(), 'checkpoint.pth')


#################################################################
# The follow snippet demonstrates the use of the three utilities.

state_dict = torch.load('checkpoint.pth', mmap=True)
with torch.device('meta'):
  meta_m = SomeModule(1000)
meta_m.load_state_dict(state_dict, assign=True)

#############################################################################
# Taking a step back, let us inspect the following more vanilla code snippet
# that does not use any of the features listed above:

state_dict = torch.load('checkpoint.pth')
m = SomeModule(1000)
m.load_state_dict(state_dict)

#################################################################################
# At ``torch.save`` time, tensor storages are tagged with the device they are
# saved on. At ``torch.load`` time, tensor storages will be loaded to the device
# they were tagged with (unless this behavior is overridden using the
# ``map_location`` flag). For ease of explanation, let us assume that the tensors
# were saved on CPU. This means that on the first line

start_time = time.time()
state_dict = torch.load('checkpoint.pth')
end_time = time.time()
print(f"loading time without mmap={end_time - start_time}")

################################################################################
# All tensor storages will be loaded into CPU RAM, which can be problematic when
#     1. CPU RAM is smaller than the size of the checkpoint
#     2. waiting for the entire checkpoint to be loaded into RAM before
#        doing for example some per-tensor processing
#
# The ``mmap`` keyword argument to ``torch.load`` attempts to solve the above two
# problems. As its name implies, the ``mmap`` keyword argument to ``torch.load``
# makes use of an `mmap call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_
# which maps a file on disk into virtual memory and lets the OS handle loading and
# unloading into physical memory automatically. When this flag is passed, tensor
# storages will be memory-mapped.

start_time = time.time()
state_dict = torch.load('checkpoint.pth', mmap=True)
end_time = time.time()
print(f"loading time with mmap={end_time - start_time}")

################################################################################
# As mentioned above, one can use this argument to do per-tensor processing on a
# checkpoint without loading all tensor storages into memory upfront. For example,

def my_special_routine(t, device):
    # this could be a much fancier operation
    return t.to(dtype=torch.bfloat16, device=device)

def my_processing_function(key, device):
    t = state_dict[key]
    processed_t = my_special_routine(t, device)
    del t
    return processed_t

for key in state_dict.keys():
    device = torch.device('cuda:' + str(int(key.lstrip("linears.")[0]) % 8))
    state_dict[key] = my_processing_function(key, device)

##############################################
# Next, we consider the creation of the module.
m = SomeModule(1000)

###############################################################################
# This allocates memory for all parameters/buffers and initializes them per
# the default initialization schemes defined in ``SomeModule.__init__()``, which
# is wasteful when we want to load a checkpoint as
#     1. The result of the initialization kernels will be overwritten by
#        `load_state_dict()` without ever being used, so initialization is
#         wasteful.
#     2. We are allocating memory for these parameters/buffers in RAM while
#        ``torch.load`` of the saved state dictionary also allocates memory for
#        the parameters/buffers in the checkpoint.
#
# In order to solve these two problems, we can use the ``torch.device()``
# context manager with ``device='meta'`` when we instantiate the ``nn.Module()``.

with torch.device('meta'):
  new_m = SomeModule(1000)

############################################################################################
# The `torch.device() <https://pytorch.org/docs/main/tensor_attributes.html#torch-device>` _
# context manager makes sure that factory calls will be performed as if they
# were passed device as an argument.
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
# the parameters/buffers in the ``nn.Module```).
#
# However, an in-place copy into a tensor on the ``meta`` device is a no-op.
# In order to avoid this, we can pass the ``assign=True`` keyword argument to
# ``load_state_dict()``.
#
# A caveat here is that since optimizers hold a reference to
# ``nn.Module.parameters()``, the optimizer must be initialized after the module
# is loaded from state dict if ``assign=True`` is passed.

new_m.load_state_dict(state_dict, assign=True)
# This MUST be done AFTER the load_state_dict with assign.
opt = torch.optim.SGD(new_m.parameters(), lr=1e-3)

###############################################################################
# To recap, in this tutorial, we learned about ``torch.load(mmap=True)``, the
# ``torch.device()`` context manager with ``device=meta`` and the
# ``nn.Module.load_state_dict(assign=True)`` and how these tools could be used
# to aid when loading a model from a checkpoint.
