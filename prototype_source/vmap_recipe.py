"""
torch.vmap
==========
This tutorial introduces torch.vmap, an autovectorizer for PyTorch operations.
torch.vmap is a prototype feature and cannot handle a number of use cases;
however, we would like to gather use cases for it to inform the design. If you
are considering using torch.vmap or think it would be really cool for something,
please contact us at https://github.com/pytorch/pytorch/issues/42368.

So, what is vmap?
-----------------
vmap is a higher-order function. It accepts a function `func` and returns a new
function that maps `func` over some dimension of the inputs. It is highly
inspired by JAX's vmap.

Semantically, vmap pushes the "map" into PyTorch operations called by `func`,
effectively vectorizing those operations.
"""
import torch
# NB: vmap is only available on nightly builds of PyTorch.
# You can download one at pytorch.org if you're interested in testing it out.
from torch import vmap

####################################################################
# The first use case for vmap is making it easier to handle
# batch dimensions in your code. One can write a function `func`
# that runs on examples and then lift it to a function that can
# take batches of examples with `vmap(func)`. `func` however
# is subject to many restrictions:
#
# - it must be functional (one cannot mutate a Python data structure
#   inside of it), with the exception of in-place PyTorch operations.
# - batches of examples must be provided as Tensors. This means that
#   vmap doesn't handle variable-length sequences out of the box.
#
# One example of using `vmap` is to compute batched dot products. PyTorch
# doesn't provide a batched `torch.dot` API; instead of unsuccessfully
# rummaging through docs, use `vmap` to construct a new function:

torch.dot                            # [D], [D] -> []
batched_dot = torch.vmap(torch.dot)  # [N, D], [N, D] -> [N]
x, y = torch.randn(2, 5), torch.randn(2, 5)
batched_dot(x, y)

####################################################################
# `vmap` can be helpful in hiding batch dimensions, leading to a simpler
# model authoring experience.
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)

# Note that model doesn't work with a batch of feature vectors because
# torch.dot must take 1D tensors. It's pretty easy to rewrite this
# to use `torch.matmul` instead, but if we didn't want to do that or if
# the code is more complicated (e.g., does some advanced indexing
# shenanigins), we can simply call `vmap`. `vmap` batches over ALL
# inputs, unless otherwise specified (with the in_dims argument,
# please see the documentation for more details).
def model(feature_vec):
    # Very simple linear model with activation
    return feature_vec.dot(weights).relu()

examples = torch.randn(batch_size, feature_size)
result = torch.vmap(model)(examples)
expected = torch.stack([model(example) for example in examples.unbind()])
assert torch.allclose(result, expected)

####################################################################
# `vmap` can also help vectorize computations that were previously difficult
# or impossible to batch. This bring us to our second use case: batched
# gradient computation.
#
# - https://github.com/pytorch/pytorch/issues/8304
# - https://github.com/pytorch/pytorch/issues/23475
#
# The PyTorch autograd engine computes vjps (vector-Jacobian products).
# Using vmap, we can compute (batched vector) - jacobian products.
#
# One example of this is computing a full Jacobian matrix (this can also be
# applied to computing a full Hessian matrix).
# Computing a full Jacobian matrix for some function f: R^N -> R^N usually
# requires N calls to `autograd.grad`, one per Jacobian row.

# Setup
N = 5
def f(x):
    return x ** 2

x = torch.randn(N, requires_grad=True)
y = f(x)
basis_vectors = torch.eye(N)

# Sequential approach
jacobian_rows = [torch.autograd.grad(y, x, v, retain_graph=True)[0]
                 for v in basis_vectors.unbind()]
jacobian = torch.stack(jacobian_rows)

# Using `vmap`, we can vectorize the whole computation, computing the
# Jacobian in a single call to `autograd.grad`.
def get_vjp(v):
    return torch.autograd.grad(y, x, v)[0]

jacobian_vmap = vmap(get_vjp)(basis_vectors)
assert torch.allclose(jacobian_vmap, jacobian)

####################################################################
# The third main use case for vmap is computing per-sample-gradients.
# This is something that the vmap prototype cannot handle performantly
# right now. We're not sure what the API for computing per-sample-gradients
# should be, but if you have ideas, please comment in
# https://github.com/pytorch/pytorch/issues/7786.

def model(sample, weight):
    # do something...    
    return torch.dot(sample, weight)

def grad_sample(sample):
    return torch.autograd.functional.vjp(lambda weight: model(sample), weight)[1]

# The following doesn't actually work in the vmap prototype. But it
# could be an API for computing per-sample-gradients.

# batch_of_samples = torch.randn(64, 5)
# vmap(grad_sample)(batch_of_samples)
