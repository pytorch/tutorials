# -*- coding: utf-8 -*-

"""
.. _python-custom-ops-mutable:

Mutable Python Custom Operators
===============================

:ref:`Functional custom operators <python-custom-ops-functional>` showed
``numpy.sin`` as an operator that returns a fresh Tensor. This page shows the
mutable version: a kernel that writes ``sin(x)`` into an existing output Tensor.
Mutable operators have a different contract from functional operators.

Before writing the operator, read the required schema and mutation/aliasing
contract rules in :ref:`python-custom-ops-schema-contract`.

Checklist:

* choose one mutation pattern and keep it stable;
* list every mutated Tensor argument in ``mutates_args``;
* do not return mutated inputs unless you are using a tagged in-place or
  ``out=`` operator (available starting in PyTorch 2.13);
* validate the operator with ``torch.library.opcheck``.
"""

######################################################################
# Choose one mutation contract
# ----------------------------
# Choose the mutation behavior before adding optional registrations. PyTorch
# needs this contract for functionalization in ``torch.compile``
# and autograd.
#
# If the operator does not mutate any Tensor input, use the functional operator
# path instead.
#
# If the operator mutates the first positional Tensor and returns it, use a
# tagged in-place operator, starting in PyTorch 2.13.
#
# If the operator accepts write-only keyword-only ``out=`` Tensor arguments and
# returns them, use a tagged ``out=`` operator, starting in PyTorch 2.13.
#
# For other mutable operators, list every mutated argument in ``mutates_args``
# and do not return mutated inputs or their aliases.

import numpy as np
import torch
from torch import Tensor


######################################################################
# Example: write NumPy sin into an output buffer
# ----------------------------------------------
# Functions that mutate inputs are common because that is how many low-level
# kernels are written; for example, a kernel that computes ``sin`` may take in
# the input and an output tensor and write ``input.sin()`` to the output tensor.
#
# This operator writes ``sin(x)`` into ``out`` and returns ``None``.


@torch.library.custom_op(
    "mylib_mutable::numpy_sin_out",
    mutates_args={"out"},
    device_types="cpu",
)
def numpy_sin_out(x: Tensor, out: Tensor) -> None:
    if x.shape != out.shape:
        raise RuntimeError("x and out must have the same shape")
    if x.dtype != out.dtype:
        raise RuntimeError("x and out must have the same dtype")
    if x.device != out.device:
        raise RuntimeError("x and out must be on the same device")
    np.sin(x.detach().numpy(), out=out.numpy())


x = torch.randn(5)
out = torch.empty_like(x)
numpy_sin_out(x, out)
torch.testing.assert_close(out, x.sin())

######################################################################
# Because the operator doesn't return anything, there is no need to register a
# ``FakeTensor`` kernel (meta kernel) to get it to work with ``torch.compile``.
# If a mutable operator also returns a fresh Tensor, register a fake kernel for
# that output.


@torch.compile(fullgraph=True)
def compiled_numpy_sin_out(x):
    out = torch.empty_like(x)
    numpy_sin_out(x, out)
    return out


torch.testing.assert_close(compiled_numpy_sin_out(x), x.sin())

######################################################################
# PyTorch-style in-place and out= operators
# -----------------------------------------
# Starting in PyTorch 2.13, ``torch.library.custom_op`` supports tagged
# in-place and ``out=`` custom operators.
# Tagged in-place operators return the same Tensor they mutate. Tagged ``out=``
# operators return their keyword-only output buffers in declaration order.
# This example uses ``mylib_mutable::sin_`` for a tagged in-place custom
# operator and ``mylib_mutable::sin_out`` for a tagged ``out=`` custom operator.


supports_tagged_mutable_ops = (
    hasattr(torch, "Tag")
    and hasattr(torch.Tag, "inplace")
    and hasattr(torch.Tag, "out")
)

if supports_tagged_mutable_ops:

    @torch.library.custom_op(
        "mylib_mutable::sin_",
        mutates_args={"x"},
        tags=torch.Tag.inplace,
    )
    def sin_(x: Tensor) -> Tensor:
        x.sin_()
        return x


    @torch.library.custom_op(
        "mylib_mutable::sin_out",
        mutates_args={"out"},
        tags=torch.Tag.out,
    )
    def sin_out(x: Tensor, *, out: Tensor) -> Tensor:
        torch.sin(x, out=out)
        return out


    x_for_inplace = torch.randn(3)
    expected = x_for_inplace.sin()
    torch.testing.assert_close(sin_(x_for_inplace), expected)

    out_for_sin = torch.empty_like(x)
    torch.testing.assert_close(
        sin_out(x, out=out_for_sin),
        x.sin(),
    )
    torch.testing.assert_close(out_for_sin, x.sin())

    torch.library.opcheck(sin_, (torch.randn(3),))
    torch.library.opcheck(
        sin_out,
        (torch.randn(3),),
        {"out": torch.empty(3)},
    )
else:
    print("Tagged in-place and out= custom operators require PyTorch 2.13 or later.")


######################################################################
# Validate the operator
# ---------------------
# And here's an ``opcheck`` run telling us that we did indeed register the
# operator correctly. ``opcheck`` would error out if we forgot to add ``out`` to
# ``mutates_args``, for example.


examples = [
    (torch.randn(5), torch.empty(5)),
    (torch.randn(0, 3), torch.empty(0, 3)),
    (
        torch.randn(2, 3, dtype=torch.double),
        torch.empty(2, 3, dtype=torch.double),
    ),
    (
        torch.randn(2, 3).t(),
        torch.empty_strided((3, 2), (1, 3)),
    ),
]

for example in examples:
    torch.library.opcheck(numpy_sin_out, example)

######################################################################
# For autograd, ``torch.vmap``, or other subsystem behavior, continue to
# :ref:`python-custom-ops-registrations`.
