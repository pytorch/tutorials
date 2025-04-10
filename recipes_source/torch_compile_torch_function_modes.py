"""
(beta) Utilizing Torch Function modes with torch.compile
============================================================

**Author:** `Michael Lazos <https://github.com/mlazos>`_
"""

#########################################################
#  This recipe covers how to use a key torch extensibility point, 
#  torch function modes, in tandem with ``torch.compile`` to override 
#  the behavior of torch operators, also know as **ops**, at trace time, with no runtime overhead.
#
# .. note::
#
#    This recipe requires PyTorch 2.7.0 or later.


#####################################################################
# Rewriting a torch op (torch.add -> torch.mul)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For this example, we'll use torch function modes to rewrite occurences
# of addition with multiply instead. This type of override can be common 
# if a certain backend has a custom implementation that should be dispatched
# for a given op. 
import torch

# exit cleanly if we are on a device that doesn't support ``torch.compile``
if torch.cuda.get_device_capability() < (7, 0):
    print("Exiting because torch.compile is not supported on this device.")
    import sys
    sys.exit(0)

from torch.overrides import BaseTorchFunctionMode

# Define our mode, Note: ``BaseTorchFunctionMode``
# implements the actual invocation of func(..)
class AddToMultiplyMode(BaseTorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func == torch.Tensor.add:
            func = torch.mul

        return super().__torch_function__(func, types, args, kwargs)

@torch.compile()
def test_fn(x, y):
    return x + y * x # Note: infix operators map to torch.Tensor.* methods

x = torch.rand(2, 2)
y = torch.rand_like(x)

with AddToMultiplyMode():
    z = test_fn(x, y)

assert torch.allclose(z, x * y * x)

# The mode can also be used within the compiled region as well like this:

@torch.compile()
def test_fn(x, y):
    with AddToMultiplyMode():
        return x + y * x # Note: infix operators map to torch.Tensor.* methods

x = torch.rand(2, 2)
y = torch.rand_like(x)
z = test_fn(x, y)

assert torch.allclose(z, x * y * x)

######################################################################
# Conclusion
# ~~~~~~~~~~
# In this tutorial we demonstrated how to override the behavior of torch.* operators
# using torch function modes from within torch.compile. This enables users to utilize
# the extensibility benefits of torch function modes without the runtime overhead
# of calling torch function on every op invocation. 
# 
# * `Extending Torch API with Modes <https://pytorch.org/docs/stable/notes/extending.html#extending-all-torch-api-with-modes>`__ - Other examples and backgroun on Torch Function modes.
