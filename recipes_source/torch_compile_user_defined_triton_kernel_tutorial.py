# -*- coding: utf-8 -*-

"""
Using User-Defined Triton Kernels with ``torch.compile``
=========================================================
**Author:** `Oguz Ulgen <https://github.com/oulgen>`_
"""

######################################################################
# User-defined Triton kernels can be used to optimize specific parts of your
# model's computation. These kernels are written in Triton's language, which is designed
# to make it easier to achieve peak hardware performance. By using user-defined Triton
# kernels with ``torch.compile``, you can integrate these optimized computations into
# your PyTorch model, potentially achieving significant performance improvements.
#
# This recipes demonstrates how you can use user-defined Triton kernels with ``torch.compile``.
#
# Prerequisites
# -------------------
#
# Before starting this recipe, make sure that you have the following:
#
# * Basic understanding of ``torch.compile`` and Triton. See:
#
#   * `torch.compiler API documentation <https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler>`__
#   * `Introduction to torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__
#   * `Triton language documentation <https://triton-lang.org/main/index.html>`__
#
# * PyTorch 2.3 or later
# * A GPU that supports Triton
#

import torch
from torch.utils._triton import has_triton

######################################################################
# Basic Usage
# --------------------
#
# In this example, we will use a simple vector addition kernel from the Triton documentation
# with ``torch.compile``.
# For reference, see `Triton documentation <https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html>`__.
#

if not has_triton():
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
# -------------------------------------------------------------------
#
# Triton's autotune feature is a powerful tool that automatically optimizes the configuration
# parameters of your Triton kernels. It explores a range of possible configurations and
# selects the one that delivers the best performance for your specific use case.
#
# When used with ``torch.compile``, ``triton.autotune`` can help ensure that your PyTorch
# model is running as efficiently as possible. Here is an example of using ``torch.compile``
# and ``triton.autotune``.
#
# .. note::
#
#   ``torch.compile`` only supports configs and key arguments to ``triton.autotune``.

if not has_triton():
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
# --------------------------------------------------------------------
#
# As of PyTorch 2.3, the support for user-defined Triton kernels in ``torch.compile``
# includes dynamic shapes, ``torch.autograd.Function``, JIT inductor, and AOT inductor.
# You can use these features together to build complex, high-performance models.
#
# However, there are certain limitations to be aware of:
#
# * **Tensor Subclasses:** Currently, there is no support for
#   tensor subclasses and other advanced features.
# * **Triton Features:** While ``triton.heuristics`` can be used either standalone or
#   before ``triton.autotune``, it cannot be used after ```triton.autotune``. This
#   implies that if ``triton.heuristics`` and ``triton.autotune`` are to be used
#   together, ``triton.heuristics`` must be used first.
#
# Conclusion
# -----------
# In this recipe, we explored how to utilize user-defined Triton kernels
# with ``torch.compile``. We delved into the basic usage of a simple
# vector addition kernel and advanced usage involving Triton's autotune
# feature. We also discussed the composability of user-defined Triton
# kernels with other PyTorch features and highlighted some current limitations.
#
# See Also
# ---------
#
# * `Compiling the Optimizers <https://pytorch.org/tutorials/recipes/compiling_optimizer.html>`__
# * `Implementing High-Performance Transformers with Scaled Dot Product Attention <https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>`__
