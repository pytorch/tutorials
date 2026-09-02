# -*- coding: utf-8 -*-

"""
.. _python-custom-ops-registrations:

Adding Training and Other Registrations to Python Custom Operators
==================================================================

Start here after a base operator passes ``torch.library.opcheck``:

* :ref:`python-custom-ops-functional`
* :ref:`python-custom-ops-mutable`

Registrations do not change the base contract. After adding one, rerun
``torch.library.opcheck`` on representative inputs for that subsystem.
"""

######################################################################
# Adding training support for NumPy sin
# -------------------------------------
# Use ``torch.library.register_autograd`` to add training support for an
# operator. Prefer this over directly using ``torch.autograd.Function``; some
# compositions of ``autograd.Function`` with PyTorch operator registration APIs
# can lead to (and has led to) silent incorrectness when composed with
# ``torch.compile``.
#
# If you don't need training support, there is no need to use
# ``torch.library.register_autograd``. If you end up training with a
# ``custom_op`` that doesn't have an autograd registration, we'll raise an error
# message.
#
# This page uses the same ``numpy.sin`` operation as the functional and mutable
# pages so the only new concept is the autograd registration.

import numpy as np
import torch
from torch import Tensor


@torch.library.custom_op(
    "mylib_training::numpy_sin",
    mutates_args=(),
    device_types="cpu",
)
def numpy_sin(x: Tensor) -> Tensor:
    result = torch.empty_like(x)
    np.sin(x.detach().numpy(), out=result.numpy())
    return result


@numpy_sin.register_fake
def _(x):
    return torch.empty_like(x)


######################################################################
# The fake kernel must describe the same output metadata as the real kernel,
# including shape, strides, dtype, device, layout, and storage offset when
# relevant. Here the real kernel returns ``torch.empty_like(x)``, so the fake
# kernel does the same.
#
# The gradient formula for ``sin(x)`` is ``cos(x)``. The backward formula must
# be written in terms of PyTorch-understood operations or other custom
# operators. Do not directly use non-traceable Python or NumPy code from the
# backward formula.


def numpy_sin_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def numpy_sin_backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    return grad_output * x.cos()


######################################################################
# Register the backward formula and the context setup function:


numpy_sin.register_autograd(
    numpy_sin_backward,
    setup_context=numpy_sin_setup_context,
)


x = torch.randn(5, requires_grad=True)
y = numpy_sin(x)
y.sum().backward()
torch.testing.assert_close(x.grad, x.detach().cos())

######################################################################
# Testing autograd registration
# -----------------------------
# ``opcheck`` verifies that autograd was registered in a supported way, but it
# does not prove that the gradient formula is mathematically correct. Use
# separate numerical tests for that, either manual ones or
# ``torch.autograd.gradcheck``.


gradcheck_input = torch.randn(3, dtype=torch.double, requires_grad=True)
torch.autograd.gradcheck(numpy_sin, (gradcheck_input,))

examples = [
    (torch.randn(5),),
    (torch.randn(0, 3),),
    (torch.randn(4, requires_grad=True),),
    (torch.randn(2, dtype=torch.double, requires_grad=True),),
    (torch.randn(2, 3).t(),),
    (torch.randn(8)[1:],),
]

for example in examples:
    torch.library.opcheck(numpy_sin, example)


######################################################################
# Other registrations
# -------------------
# Add these only when users need them.
#
# * **Multiple device kernels:** pass ``device_types="cpu"`` or
#   ``device_types="cuda"`` if the implementation only works on one device.
#   Register device-specific kernels when devices need different code.
# * **``torch.vmap``:** register a vmap rule with ``torch.library.register_vmap``
#   when batching over the operator should do something different from a Python
#   loop over the batch dimension.
# * **Tensor subclasses or modes:** use ``torch.library.register_torch_dispatch``
#   when a Tensor subclass or ``TorchDispatchMode`` needs special behavior.
# * **Autocast:** for C++/CUDA operators that should participate in autocast,
#   add an autocast registration as described in the C++ custom operator guide.
