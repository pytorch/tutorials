# -*- coding: utf-8 -*-
"""
Jacobians, Hessians, hvp, vhp, and more: composing function transforms
======================================================================

Computing jacobians or hessians are useful in a number of non-traditional
deep learning models. It is difficult (or annoying) to compute these quantities
efficiently using PyTorch's regular autodiff APIs
(``Tensor.backward()``, ``torch.autograd.grad``). PyTorch's 
`JAX-inspired <https://github.com/google/jax>`_
`function transforms API <https://pytorch.org/docs/master/func.html>`_
provides ways of computing various higher-order autodiff quantities
efficiently.

.. note::

   This tutorial requires PyTorch 2.0.0 or later.

Computing the Jacobian
----------------------
"""

import torch
import torch.nn.functional as F
from functools import partial
_ = torch.manual_seed(0)

######################################################################
# Let's start with a function that we'd like to compute the jacobian of.
# This is a simple linear function with non-linear activation.

def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()

######################################################################
# Let's add some dummy data: a weight, a bias, and a feature vector x.

D = 16
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D)  # feature vector

######################################################################
# Let's think of ``predict`` as a function that maps the input ``x`` from :math:`R^D \to R^D`.
# PyTorch Autograd computes vector-Jacobian products. In order to compute the full
# Jacobian of this :math:`R^D \to R^D` function, we would have to compute it row-by-row
# by using a different unit vector each time.

def compute_jac(xp):
    jacobian_rows = [torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0]
                     for vec in unit_vectors]
    return torch.stack(jacobian_rows)

xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)

jacobian = compute_jac(xp)

print(jacobian.shape)
print(jacobian[0])  # show first row

######################################################################
# Instead of computing the jacobian row-by-row, we can use PyTorch's
# ``torch.vmap`` function transform to get rid of the for-loop and vectorize the
# computation. We can’t directly apply ``vmap`` to ``torch.autograd.grad``;
# instead, PyTorch provides a ``torch.func.vjp`` transform that composes with
# ``torch.vmap``:

from torch.func import vmap, vjp

_, vjp_fn = vjp(partial(predict, weight, bias), x)

ft_jacobian, = vmap(vjp_fn)(unit_vectors)

# let's confirm both methods compute the same result
assert torch.allclose(ft_jacobian, jacobian)

######################################################################
# In a later tutorial a composition of reverse-mode AD and ``vmap`` will give us
# per-sample-gradients.
# In this tutorial, composing reverse-mode AD and ``vmap`` gives us Jacobian
# computation!
# Various compositions of ``vmap`` and autodiff transforms can give us different
# interesting quantities.
#
# PyTorch provides ``torch.func.jacrev`` as a convenience function that performs
# the ``vmap-vjp`` composition to compute jacobians. ``jacrev`` accepts an ``argnums``
# argument that says which argument we would like to compute Jacobians with
# respect to.

from torch.func import jacrev

ft_jacobian = jacrev(predict, argnums=2)(weight, bias, x)

# Confirm by running the following:
assert torch.allclose(ft_jacobian, jacobian)

######################################################################
# Let's compare the performance of the two ways to compute the jacobian.
# The function transform version is much faster (and becomes even faster the
# more outputs there are).
#
# In general, we expect that vectorization via ``vmap`` can help eliminate overhead
# and give better utilization of your hardware.
#
# ``vmap`` does this magic by pushing the outer loop down into the function's
# primitive operations in order to obtain better performance.
#
# Let's make a quick function to evaluate performance and deal with
# microseconds and milliseconds measurements:

def get_perf(first, first_descriptor, second, second_descriptor):
    """takes torch.benchmark objects and compares delta of second vs first."""
    faster = second.times[0]
    slower = first.times[0]
    gain = (slower-faster)/slower
    if gain < 0: gain *=-1
    final_gain = gain*100
    print(f" Performance delta: {final_gain:.4f} percent improvement with {second_descriptor} ")

######################################################################
# And then run the performance comparison:

from torch.utils.benchmark import Timer

without_vmap = Timer(stmt="compute_jac(xp)", globals=globals())
with_vmap = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

no_vmap_timer = without_vmap.timeit(500)
with_vmap_timer = with_vmap.timeit(500)

print(no_vmap_timer)
print(with_vmap_timer)

######################################################################
# Let's do a relative performance comparison of the above with our ``get_perf`` function:

get_perf(no_vmap_timer, "without vmap",  with_vmap_timer, "vmap")

######################################################################
# Furthermore, it’s pretty easy to flip the problem around and say we want to
# compute Jacobians of the parameters to our model (weight, bias) instead of the input

# note the change in input via ``argnums`` parameters of 0,1 to map to weight and bias
ft_jac_weight, ft_jac_bias = jacrev(predict, argnums=(0, 1))(weight, bias, x)

######################################################################
# Reverse-mode Jacobian (``jacrev``) vs forward-mode Jacobian (``jacfwd``)
# ------------------------------------------------------------------------
#
# We offer two APIs to compute jacobians: ``jacrev`` and ``jacfwd``:
#
# - ``jacrev`` uses reverse-mode AD. As you saw above it is a composition of our
#   ``vjp`` and ``vmap`` transforms.
# - ``jacfwd`` uses forward-mode AD. It is implemented as a composition of our
#   ``jvp`` and ``vmap`` transforms.
#
# ``jacfwd`` and ``jacrev`` can be substituted for each other but they have different
# performance characteristics.
#
# As a general rule of thumb, if you’re computing the jacobian of an :math:`R^N \to R^M`
# function, and there are many more outputs than inputs (for example, :math:`M > N`) then
# ``jacfwd`` is preferred, otherwise use ``jacrev``. There are exceptions to this rule,
# but a non-rigorous argument for this follows:
#
# In reverse-mode AD, we are computing the jacobian row-by-row, while in
# forward-mode AD (which computes Jacobian-vector products), we are computing
# it column-by-column. The Jacobian matrix has M rows and N columns, so if it
# is taller or wider one way we may prefer the method that deals with fewer
# rows or columns.

from torch.func import jacrev, jacfwd

######################################################################
# First, let's benchmark with more inputs than outputs:

Din = 32
Dout = 2048
weight = torch.randn(Dout, Din)

bias = torch.randn(Dout)
x = torch.randn(Din)

# remember the general rule about taller vs wider... here we have a taller matrix:
print(weight.shape)

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

jacfwd_timing = using_fwd.timeit(500)
jacrev_timing = using_bwd.timeit(500)

print(f'jacfwd time: {jacfwd_timing}')
print(f'jacrev time: {jacrev_timing}')

######################################################################
# and then do a relative benchmark:

get_perf(jacfwd_timing, "jacfwd", jacrev_timing, "jacrev", );

#######################################################################
# and now the reverse - more outputs (M) than inputs (N):

Din = 2048
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

jacfwd_timing = using_fwd.timeit(500)
jacrev_timing = using_bwd.timeit(500)

print(f'jacfwd time: {jacfwd_timing}')
print(f'jacrev time: {jacrev_timing}')

#######################################################################
# and a relative performance comparison:

get_perf(jacrev_timing, "jacrev", jacfwd_timing, "jacfwd")

#######################################################################
# Hessian computation with functorch.hessian
# ------------------------------------------
# We offer a convenience API to compute hessians: ``torch.func.hessiani``.
# Hessians are the jacobian of the jacobian (or the partial derivative of
# the partial derivative, aka second order).
#
# This suggests that one can just compose functorch jacobian transforms to
# compute the Hessian.
# Indeed, under the hood, ``hessian(f)`` is simply ``jacfwd(jacrev(f))``.
#
# Note: to boost performance: depending on your model, you may also want to
# use ``jacfwd(jacfwd(f))`` or ``jacrev(jacrev(f))`` instead to compute hessians
# leveraging the rule of thumb above regarding wider vs taller matrices.

from torch.func import hessian

# lets reduce the size in order not to overwhelm Colab. Hessians require
# significant memory:
Din = 512
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)

hess_api = hessian(predict, argnums=2)(weight, bias, x)
hess_fwdfwd = jacfwd(jacfwd(predict, argnums=2), argnums=2)(weight, bias, x)
hess_revrev = jacrev(jacrev(predict, argnums=2), argnums=2)(weight, bias, x)

#######################################################################
# Let's verify we have the same result regardless of using hessian API or
# using ``jacfwd(jacfwd())``.

torch.allclose(hess_api, hess_fwdfwd)

#######################################################################
# Batch Jacobian and Batch Hessian
# --------------------------------
# In the above examples we’ve been operating with a single feature vector.
# In some cases you might want to take the Jacobian of a batch of outputs
# with respect to a batch of inputs. That is, given a batch of inputs of
# shape ``(B, N)`` and a function that goes from :math:`R^N \to R^M`, we would like
# a Jacobian of shape ``(B, M, N)``.
#
# The easiest way to do this is to use ``vmap``:

batch_size = 64
Din = 31
Dout = 33

weight = torch.randn(Dout, Din)
print(f"weight shape = {weight.shape}")

bias = torch.randn(Dout)

x = torch.randn(batch_size, Din)

compute_batch_jacobian = vmap(jacrev(predict, argnums=2), in_dims=(None, None, 0))
batch_jacobian0 = compute_batch_jacobian(weight, bias, x)

#######################################################################
# If you have a function that goes from (B, N) -> (B, M) instead and are
# certain that each input produces an independent output, then it's also
# sometimes possible to do this without using ``vmap`` by summing the outputs
# and then computing the Jacobian of that function:

def predict_with_output_summed(weight, bias, x):
    return predict(weight, bias, x).sum(0)

batch_jacobian1 = jacrev(predict_with_output_summed, argnums=2)(weight, bias, x).movedim(1, 0)
assert torch.allclose(batch_jacobian0, batch_jacobian1)

#######################################################################
# If you instead have a function that goes from :math:`R^N \to R^M` but inputs that
# are batched, you compose ``vmap`` with ``jacrev`` to compute batched jacobians:
#
# Finally, batch hessians can be computed similarly. It's easiest to think
# about them by using ``vmap`` to batch over hessian computation, but in some
# cases the sum trick also works.

compute_batch_hessian = vmap(hessian(predict, argnums=2), in_dims=(None, None, 0))

batch_hess = compute_batch_hessian(weight, bias, x)
batch_hess.shape

#######################################################################
# Computing Hessian-vector products
# ---------------------------------
# The naive way to compute a Hessian-vector product (hvp) is to materialize
# the full Hessian and perform a dot-product with a vector. We can do better:
# it turns out we don't need to materialize the full Hessian to do this. We'll
# go through two (of many) different strategies to compute Hessian-vector products:
# - composing reverse-mode AD with reverse-mode AD
# - composing reverse-mode AD with forward-mode AD
#
# Composing reverse-mode AD with forward-mode AD (as opposed to reverse-mode
# with reverse-mode) is generally the more memory efficient way to compute a
# hvp because forward-mode AD doesn't need to construct an Autograd graph and
# save intermediates for backward:

from torch.func import jvp, grad, vjp

def hvp(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]

#######################################################################
# Here's some sample usage.

def f(x):
  return x.sin().sum()

x = torch.randn(2048)
tangent = torch.randn(2048)

result = hvp(f, (x,), (tangent,))

#######################################################################
# If PyTorch forward-AD does not have coverage for your operations, then we can
# instead compose reverse-mode AD with reverse-mode AD:

def hvp_revrev(f, primals, tangents):
  _, vjp_fn = vjp(grad(f), *primals)
  return vjp_fn(*tangents)

result_hvp_revrev = hvp_revrev(f, (x,), (tangent,))
assert torch.allclose(result, result_hvp_revrev[0])
