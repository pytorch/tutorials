# -*- coding: utf-8 -*-

"""
(Prototype) Efficiently writing "sparse" semantics for Adagrad with MaskedTensor
================================================================================
"""

######################################################################
# Before working through this tutorial, please review the MaskedTensor
# `Overview <https://pytorch.org/tutorials/prototype/maskedtensor_overview.html>`__ and
# `Sparsity <https://pytorch.org/tutorials/prototype/maskedtensor_sparsity.html>`__ tutorials.
#
# Introduction and Motivation
# ---------------------------
# `Issue 1369 <https://github.com/pytorch/pytorch/issues/1369>`__ discussed the additional lines of code
# that were introduced while writing "sparse" semantics for Adagrad, but really,
# the code uses sparsity as a proxy for masked semantics rather than the intended use case of sparsity:
# a compression and optimization technique.
# Previously, we worked around the lack of formal masked semantics by introducing one-off semantics and operators
# while forcing users to be aware of storage details such as indices and values.
#
# Now that we have masked semantics, we are better equipped to point out when sparsity is used as a semantic extension.
# We'll also compare and contrast this with equivalent code written using MaskedTensor.
# In the end the code snippets are repeated without additional comments to show the difference in brevity.
#
# Preparation
# -----------
#

import torch
import warnings

# Disable prototype warnings and such
warnings.filterwarnings(action='ignore', category=UserWarning)

# Some hyperparameters
eps = 1e-10
clr = 0.1

i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
grad = torch.sparse_coo_tensor(i, v, [2, 4])

######################################################################
# Simpler Code with MaskedTensor
# ------------------------------
#
# Before we get too far in the weeds, let's introduce the problem a bit more concretely. We will be taking a look
# into the `Adagrad (functional) <https://github.com/pytorch/pytorch/blob/6c2f235d368b697072699e5ca9485fd97d0b9bcc/torch/optim/_functional.py#L16-L51>`__
# implementation in PyTorch with the ultimate goal of simplifying and more faithfully representing the masked approach.
#
# For reference, this is the regular, dense code path without masked gradients or sparsity:
#
# .. code-block:: python
#
#     state_sum.addcmul_(grad, grad, value=1)
#     std = state_sum.sqrt().add_(eps)
#     param.addcdiv_(grad, std, value=-clr)
#
# The vanilla tensor implementation for sparse is:
#
# .. code-block:: python
#
#     def _make_sparse(grad, grad_indices, values):
#         size = grad.size()
#         if grad_indices.numel() == 0 or values.numel() == 0:
#             return torch.empty_like(grad)
#         return torch.sparse_coo_tensor(grad_indices, values, size)
#
#     grad = grad.coalesce()  # the update is non-linear so indices must be unique
#     grad_indices = grad._indices()
#     grad_values = grad._values()
#
#     state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))   # a different _make_sparse per layout
#     std = state_sum.sparse_mask(grad)
#     std_values = std._values().sqrt_().add_(eps)
#     param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
#
# while :class:`MaskedTensor` minimizes the code to the snippet:
#
# .. code-block:: python
#
#     state_sum2 = state_sum2 + masked_grad.pow(2).get_data()
#     std2 = masked_tensor(state_sum2.to_sparse(), mask)
#     std2 = std2.sqrt().add(eps)
#     param2 = param2.add((masked_grad / std2).get_data(), alpha=-clr)
#
# In this tutorial, we will go through each implementation line by line, but at first glance, we can notice
# (1) how much shorter the MaskedTensor implementation is, and
# (2) how it avoids conversions between dense and sparse tensors.
#

######################################################################
# Original Sparse Implementation
# ------------------------------
#
# Now, let's break down the code with some inline comments:
#

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)

# We don't support sparse gradients
param = torch.arange(8).reshape(2, 4).float()
state_sum = torch.full_like(param, 0.5)  # initial value for state sum

grad = grad.coalesce()  # the update is non-linear so indices must be unique
grad_indices = grad._indices()
grad_values = grad._values()
# pow(2) has the same semantics for both sparse and dense memory layouts since 0^2 is zero
state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))

# We take care to make std sparse, even though state_sum clearly is not.
# This means that we're only applying the gradient to parts of the state_sum
# for which it is specified. This further drives the point home that the passed gradient is not sparse, but masked.
# We currently dodge all these concerns using the private method `_values`.
std = state_sum.sparse_mask(grad)
std_values = std._values().sqrt_().add_(eps)

# Note here that we currently don't support div for sparse Tensors because zero / zero is not well defined,
# so we're forced to perform `grad_values / std_values` outside the sparse semantic and then convert back to a
# sparse tensor with `make_sparse`.
# We'll later see that MaskedTensor will actually handle these operations for us as well as properly denote
# undefined / undefined = undefined!
param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)

######################################################################
# The third to last line -- `std = state_sum.sparse_mask(grad)` -- is where we have a very important divergence.
#
# The addition of eps should technically be applied to all values but instead is only applied to specified values.
# Here we're using sparsity as a semantic extension and to enforce a certain pattern of defined and undefined values.
# If parts of the values of the gradient are zero, they are still included if materialized even though they
# could be compressed by other sparse storage layouts. This is theoretically quite brittle!
# That said, one could argue that eps is always very small, so it might not matter so much in practice.
#
# Moreover, an implementation `add_` for sparsity as a storage layout and compression scheme
# should cause densification, but we force it not to for performance.
# For this one-off case it is fine.. until we want to introduce new compression scheme, such as
# `CSC <https://pytorch.org/docs/master/sparse.html#sparse-csc-docs>`__,
# `BSR <https://pytorch.org/docs/master/sparse.html#sparse-bsr-docs>`__,
# or `BSC <https://pytorch.org/docs/master/sparse.html#sparse-bsc-docs>`__.
# We will then need to introduce separate Tensor types for each and write variations for gradients compressed
# using different storage formats, which is inconvenient and not quite scalable nor clean.
#
# MaskedTensor Sparse Implementation
# ----------------------------------
#
# We've been conflating sparsity as an optimization with sparsity as a semantic extension to PyTorch.
# MaskedTensor proposes to disentangle the sparsity optimization from the semantic extension; for example,
# currently we can't have dense semantics with sparse storage or masked semantics with dense storage.
# MaskedTensor enables these ideas by purposefully separating the storage from the semantics.
#
# Consider the above example using a masked gradient:
#

# Let's now import MaskedTensor!
from torch.masked import masked_tensor

# Create an entirely new set of parameters to avoid errors
param2 = torch.arange(8).reshape(2, 4).float()
state_sum2 = torch.full_like(param, 0.5)  # initial value for state sum

mask = (grad.to_dense() != 0).to_sparse()
masked_grad = masked_tensor(grad, mask)

state_sum2 = state_sum2 + masked_grad.pow(2).get_data()
std2 = masked_tensor(state_sum2.to_sparse(), mask)

# We can add support for in-place operations later. Notice how this doesn't
# need to access any storage internals and is in general a lot shorter
std2 = std2.sqrt().add(eps)

param2 = param2.add((masked_grad / std2).get_data(), alpha=-clr)

######################################################################
# Note that the implementations look quite similar, but the MaskedTensor implementation is shorter and simpler.
# In particular, much of the boilerplate code around ``_make_sparse``
# (and needing to have a separate implementation per layout) is handled for the user with :class:`MaskedTensor`.
#
# At this point, let's print both this version and original version for easier comparison:
#

print("state_sum:\n", state_sum)
print("state_sum2:\n", state_sum2)

######################################################################
#

print("std:\n", std)
print("std2:\n", std2)

######################################################################
#

print("param:\n", param)
print("param2:\n", param2)

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we've discussed how native masked semantics can enable a cleaner developer experience for
# Adagrad's existing implementation in PyTorch, which used sparsity as a proxy for writing masked semantics.
# But more importantly, allowing masked semantics to be a first class citizen through MaskedTensor
# removes the reliance on sparsity or unreliable hacks to mimic masking, thereby allowing for proper independence
# and development, while enabling sparse semantics, such as this one.
#
# Further Reading
# ---------------
#
# To continue learning more, you can find our final review (for now) on
# `MaskedTensor Advanced Semantics <https://pytorch.org/tutorials/prototype/maskedtensor_advanced_semantics.html>`__
# to see some of the differences in design decisions between :class:`MaskedTensor` and NumPy's MaskedArray, as well
# as reduction semantics.
#
