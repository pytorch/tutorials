# -*- coding: utf-8 -*-

"""
Efficiency of writing "sparse" semantics for Adagrad
====================================================

`Issue 1369 <https://github.com/pytorch/pytorch/issues/1369>`__ discussed the additional lines of code
that were introduce while writing "sparse" semantics for Adagrad.
But really the code doesn't use sparsity as a compression and optimization technique,
it wants to use masked semantics. We worked around this by introducing one-off semantics and operators
that encode this behavior while forcing users to be aware of storage details such as indices and values.

In particular we'll point out when sparsity is used as a semantic extension, i.e. unspecified values are not zero
and when it is just used to compress zeros.
We'll also compare and contrast this with equivalent code written using MaskedTensor.
In the end the code snippets are repeat without additional comments to show the difference in brevity.

""""

import torch
from torch.masked.maskedtensor import masked_tensor

######################################################################
# Original sparse implementation
# ------------------------------
#
# First, let's look at the current implementation of 
# `Adagrad (functional) <https://github.com/pytorch/pytorch/blob/6c2f235d368b697072699e5ca9485fd97d0b9bcc/torch/optim/_functional.py#L16-L51>`__
#

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)

# Some hyperparameters
eps = 1e-10
clr = 0.1

# We don't support sparse gradients
param = torch.arange(8).reshape(2, 4).float()
i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
grad = torch.sparse_coo_tensor(i, v, [2, 4])
state_sum = torch.full_like(param, 0.5) # initial value for state sum

print("param:\n", param)
print("grad:\n", grad.to_dense())
print("state_sum:\n", state_sum)

######################################################################
#

state_sum = torch.full_like(param, 0.5) # initial value for state sum
print(state_sum)

grad = grad.coalesce()  # the update is non-linear so indices must be unique
grad_indices = grad._indices()
grad_values = grad._values()

# pow(2) has the same semantics for both sparse and dense memory layouts since 0^2 is zero
state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
# We take care to make std sparse, even though state_sum clearly is not.
# This means that we're only applying the gradient to parts of the state_sum
# for which it is specified. This even drives the point home a lot more that
# the passed gradient is not sparse, but masked.
std = state_sum.sparse_mask(grad)
print("state_sum:\n", state_sum)
print("std:\n", std.to_dense())

######################################################################
# This is where we have a very important divergence.
# The addition of eps should technically be applied to all values, but instead is only applied to specified values.
# Here we're using sparsity as a semantic extension and to enforce a certain pattern of defined and undefined values.
# If parts of the values of the gradient are zero they are still included if materialized.
# Even though they could be compressed by other sparse storage layouts.
# This is technically quite brittle even though someone could argue that eps is always very small.
#
# Moreover, an implementation add_ for sparsity as a storage layout and compression scheme should cause densification,
# but we force it not to.
# For this one-off case it is fine until we want to introduce new compression schemes
# such as CSR, BSR or 2:4 block sparsity. We'll then need to introduce separate Tensor types for each
# and write variations for gradients compressed using different storage formats.
#

# We currently dodge all these concerns using the private method values.
std_values = std._values().sqrt_().add_(eps)

# We currently don't support div for sparse Tensors because zero / zero is
# not well defined. For a MaskedTensor undefined / undefined is undefined.
param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
print("param:\n", param)

######################################################################
# MaskedTensor sparse implementation
# ----------------------------------
# 
# We've been conflating sparsity as an optimization with sparsity as a semantic extension to PyTorch.
# MaskedTensor proposes to call the semantic extension through sparsity masked. 
# Currently we can't have dense semantics with sparse storage or masked semantics with dense storage, while 
# MaskedTensor fixes that because it separates the storage from the semantics.
# Consider the above example using a masked gradient:
#

# Create an entirely new set of parameters to avoid errors
param2 = torch.arange(8).reshape(2, 4).float()
state_sum2 = torch.full_like(param, 0.5)  # initial value for state sum

mask = (grad.to_dense() != 0).to_sparse()
masked_grad = masked_tensor(grad, mask)
print("masked_grad:\n", masked_grad)

######################################################################
#

state_sum2 = state_sum2 + masked_grad.pow(2).data()
std2 = masked_tensor(state_sum2.to_sparse(), mask)

# Let's print both this version and the regular version for easier comparison
print("state_sum:\n", state_sum)
print("std:\n", std)
print("state_sum2:\n", state_sum2)
print("std2:\n", std2)

######################################################################
#

# We can add support for in-place operations later. Notice how this doesn't
# need to access any storage internals and is in general a lot shorter
std2 = std2.sqrt().add(eps)

print("std:\n", std)
print("std2:\n", std2)

# .data() indeed returns a sparse tensor
param2 = param2.add((masked_grad / std2).data(), alpha=-clr)
print("param2:\n", param2)

######################################################################
# Conclusion: Difference in code
# ------------------------------
#
# For reference, this is the regular, dense code path without masked gradients or sparsity:
# ::
# 
#    state_sum.addcmul_(grad, grad, value=1)
#    std = state_sum.sqrt().add_(eps)
#    param.addcdiv_(grad, std, value=-clr)
# 
# The vanilla tensor implementation for sparse is:
#

grad = grad.coalesce()  # the update is non-linear so indices must be unique
grad_indices = grad._indices()
grad_values = grad._values()
size = grad.size()

state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
std = state_sum.sparse_mask(grad)
std_values = std._values().sqrt_().add_(eps)
param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)

######################################################################
# while MaskedTensor minimizes the code to the snippet:
#

state_sum2 = state_sum2 + masked_grad.pow(2).data()
std2 = masked_tensor(state_sum2.to_sparse(), mask)
std2 = std2.sqrt().add(eps)
param2 = param2.add((masked_grad / std2).data(), alpha=-clr)

######################################################################
# And for good measure, let's make sure the parameters match:
#

print("param:\n", param)
print("param2:\n", param2)
 