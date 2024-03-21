# -*- coding: utf-8 -*-

"""
Using User Defined Triton Kernels with ``torch.compile``
=================================
**Author:** `Oguz Ulgen <https://github.com/oulgen>`_
"""

######################################################################
# This tutorial explains how to use user defined Triton kernels with ``torch.compile``.
#
# .. note::
#   This tutorial requires PyTorch 2.3 or later and a GPU that supports Triton.
#

import torch
from torch.utils._triton import has_triton

######################################################################
# Basic Usage
# ------------
#
# In this example, we will use a simple vector addition kernel from the Triton documentation
# with ``torch.compile``.
# Reference: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
#

if not has_triton:
    print("Skipping because triton is not supported on this device.")
else:
    import triton
    from triton import language as tl

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @torch.compile(fullgraph=True)
    def add_fn(x, y):
        output = torch.zeros_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
        return output

    x = torch.randn(4, device="cuda")
    y = torch.randn(4, device="cuda")
    out = add_fn(x, y)
    print(f"Vector addition of\nX:\t{x}\nY:\t{y}\nis equal to\n{out}")

######################################################################
# Advanced Usage
# ------------
#
# It is also possible to triton.autotune with ``torch.compile``.
#
# .. note::
#
#   ``torch.compile`` only supports configs and key arguments to ``triton.autotune``.

if not has_triton:
    print("Skipping because triton is not supported on this device.")
else:
    import triton
    from triton import language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 4}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 4}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=4, num_warps=4),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @torch.compile(fullgraph=True)
    def add_fn(x, y):
        output = torch.zeros_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel_autotuned[grid](x, y, output, n_elements)
        return output

    x = torch.randn(4, device="cuda")
    y = torch.randn(4, device="cuda")
    out = add_fn(x, y)
    print(f"Vector addition of\nX:\t{x}\nY:\t{y}\nis equal to\n{out}")

######################################################################
# Composibility and Limitations
# ------------
#
# As for PyTorch 2.3, the user defined triton kernel support in ``torch.compile``
# composes with dynamic shapes, ``torch.autograd.Function``, JIT inductor and
# AOT inductor.
#
# The support for tensor subclasses and other advanced features currently do
# not exist.
# Support for ``triton.heuristics`` exists when it is used by itself or before
# ``triton.autotune``; however, support for using ``triton.heuristic`` after
# ``triton.autotune`` is not yet supported.
