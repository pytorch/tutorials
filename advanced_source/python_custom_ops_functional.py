# -*- coding: utf-8 -*-

"""
.. _python-custom-ops-functional:

Functional Python Custom Operators
==================================

Use this path when the operator mutates no Tensor inputs and returns fresh
Tensor outputs.

If the operator must work with ``torch.compile`` or ``torch.export``,
:ref:`register a fake kernel <python-custom-ops-functional-register-fake>`.
The fake kernel describes output metadata without running the real kernel.

Before writing the operator, read the required schema and mutation/aliasing
contract rules in :ref:`python-custom-ops-schema-contract`.

Checklist:

* use ``mutates_args=()``;
* return tensors that do not alias any input;
* register a fake kernel for ``torch.compile`` and ``torch.export``;
* validate the operator with ``torch.library.opcheck``.
"""

######################################################################
# Example: wrapping NumPy sin into a custom operator
# --------------------------------------------------
# Let's say that we are using NumPy's ``sin`` operation. This is an ordinary
# Python function from PyTorch's point of view: it converts the Tensor to a
# NumPy array, calls NumPy, and returns a fresh Tensor.

import numpy as np
import torch
from torch import Tensor


def numpy_sin_impl(x: Tensor) -> Tensor:
    result = torch.empty_like(x)
    np.sin(x.detach().numpy(), out=result.numpy())
    return result


x = torch.randn(5)
torch.testing.assert_close(numpy_sin_impl(x), x.sin())

# This small example focuses on the custom-operator mechanics. More complex
# Python or third-party library calls may not be handled effectively
# out-of-the-box by ``torch.compile``: ``torch.compile`` may induce a
# `"graph break" <https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks>`_
# on functions it is unable to handle, and graph breaks are bad for performance.
# A custom operator gives PyTorch an explicit boundary for such code.
#
# To make ``numpy_sin_impl`` available as a custom operator that works with
# ``torch.compile`` and ``torch.export``, we need to do two things:
#
# 1. wrap the function into a PyTorch custom operator.
# 2. add a "``FakeTensor`` kernel" (aka "meta kernel") to the operator.
#    Given some ``FakeTensors`` inputs (dummy Tensors that don't have storage),
#    this function should return dummy Tensors of your choice with the correct
#    Tensor metadata (shape/strides/``dtype``/device).


@torch.library.custom_op(
    "mylib_functional::numpy_sin",
    mutates_args=(),
    device_types="cpu",
)
def numpy_sin(x: Tensor) -> Tensor:
    result = torch.empty_like(x)
    np.sin(x.detach().numpy(), out=result.numpy())
    return result


######################################################################
# .. _python-custom-ops-functional-register-fake:
#
# Use ``register_fake`` to add a ``FakeTensor`` kernel for the operator.
# ``numpy_sin`` returns one Tensor with the same shape, strides, dtype, device,
# and storage offset as ``torch.empty_like(x)``, so the fake kernel can return
# ``empty_like(x)``. In general, the fake kernel must match all output metadata,
# including storage offset when relevant.


@numpy_sin.register_fake
def _(x):
    return torch.empty_like(x)


######################################################################
# After this, ``numpy_sin`` can be used under ``torch.compile``:


@torch.compile(fullgraph=True)
def f(x):
    return numpy_sin(x)


result = f(x)
torch.testing.assert_close(result, x.sin())

######################################################################
# A PIL image transform, Python binding to a C++ extension, or another
# third-party library call follows the same pattern. If it returns tensors,
# write the fake kernel to match the real output metadata exactly: shape,
# strides, dtype, device, layout, and storage offset when relevant.

######################################################################
# Example: fake kernels must match strides
# ----------------------------------------
# The fake kernel must match the real output strides, not only the shape. This
# operator returns a fresh Tensor with the same shape as ``x`` but different
# strides.


def numpy_sin_strided_impl(x: Tensor) -> Tensor:
    result = torch.empty_strided(
        x.shape,
        tuple(reversed(x.stride())),
        dtype=x.dtype,
        device=x.device,
    )
    np.sin(x.detach().numpy(), out=result.numpy())
    return result


@torch.library.custom_op(
    "mylib_functional::numpy_sin_strided_bad",
    mutates_args=(),
    device_types="cpu",
)
def numpy_sin_strided_bad(x: Tensor) -> Tensor:
    return numpy_sin_strided_impl(x)


@numpy_sin_strided_bad.register_fake
def _(x):
    return torch.empty_like(x)


try:
    torch.library.opcheck(numpy_sin_strided_bad, (torch.randn(2, 3),))
except Exception as exc:
    print(f"opcheck caught incorrect fake kernel metadata: {type(exc).__name__}")
else:
    torch_version = tuple(
        int(part) for part in torch.__version__.split("+")[0].split(".")[:2]
    )
    if torch_version >= (2, 13):
        raise AssertionError("Expected opcheck to fail")
    print("PyTorch versions before 2.13 may not catch this metadata mismatch")


@torch.library.custom_op(
    "mylib_functional::numpy_sin_strided",
    mutates_args=(),
    device_types="cpu",
)
def numpy_sin_strided(x: Tensor) -> Tensor:
    return numpy_sin_strided_impl(x)


@numpy_sin_strided.register_fake
def _(x):
    return torch.empty_strided(
        x.shape,
        tuple(reversed(x.stride())),
        dtype=x.dtype,
        device=x.device,
    )


torch.library.opcheck(numpy_sin_strided, (torch.randn(2, 3),))

######################################################################
# Testing Python custom operators
# -------------------------------
# Use ``torch.library.opcheck`` to test that the custom operator was registered
# correctly. This does not test numerical correctness; write separate tests for
# that.
#
# To use ``opcheck``, pass it a set of example inputs to test against. If your
# operator supports training, then the examples should include Tensors that
# require grad. If your operator supports multiple devices, then the examples
# should include Tensors from each device.


examples = [
    (torch.randn(5),),
    (torch.randn(0, 3),),
    (torch.randn(2, 3, dtype=torch.double),),
    (torch.randn(2, 3).t(),),
    (torch.randn(8)[1:],),
]

for example in examples:
    torch.library.opcheck(numpy_sin, example)

######################################################################
# To add autograd, ``torch.vmap``, or other subsystem support, continue to
# :ref:`python-custom-ops-registrations`.
