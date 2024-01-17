# -*- coding: utf-8 -*-
"""
Forward-mode Automatic Differentiation (Beta)
=============================================

This tutorial demonstrates how to use forward-mode AD to compute
directional derivatives (or equivalently, Jacobian-vector products).

The tutorial below uses some APIs only available in versions >= 1.11
(or nightly builds).

Also note that forward-mode AD is currently in beta. The API is
subject to change and operator coverage is still incomplete.

Basic Usage
--------------------------------------------------------------------
Unlike reverse-mode AD, forward-mode AD computes gradients eagerly
alongside the forward pass. We can use forward-mode AD to compute a
directional derivative by performing the forward pass as before,
except we first associate our input with another tensor representing
the direction of the directional derivative (or equivalently, the ``v``
in a Jacobian-vector product). When an input, which we call "primal", is
associated with a "direction" tensor, which we call "tangent", the
resultant new tensor object is called a "dual tensor" for its connection
to dual numbers[0].

As the forward pass is performed, if any input tensors are dual tensors,
extra computation is performed to propagate this "sensitivity" of the
function.

"""

import torch
import torch.autograd.forward_ad as fwAD

primal = torch.randn(10, 10)
tangent = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

# All forward AD computation must be performed in the context of
# a ``dual_level`` context. All dual tensors created in such a context
# will have their tangents destroyed upon exit. This is to ensure that
# if the output or intermediate results of this computation are reused
# in a future forward AD computation, their tangents (which are associated
# with this computation) won't be confused with tangents from the later
# computation.
with fwAD.dual_level():
    # To create a dual tensor we associate a tensor, which we call the
    # primal with another tensor of the same size, which we call the tangent.
    # If the layout of the tangent is different from that of the primal,
    # The values of the tangent are copied into a new tensor with the same
    # metadata as the primal. Otherwise, the tangent itself is used as-is.
    #
    # It is also important to note that the dual tensor created by
    # ``make_dual`` is a view of the primal.
    dual_input = fwAD.make_dual(primal, tangent)
    assert fwAD.unpack_dual(dual_input).tangent is tangent

    # To demonstrate the case where the copy of the tangent happens,
    # we pass in a tangent with a layout different from that of the primal
    dual_input_alt = fwAD.make_dual(primal, tangent.T)
    assert fwAD.unpack_dual(dual_input_alt).tangent is not tangent

    # Tensors that do not have an associated tangent are automatically
    # considered to have a zero-filled tangent of the same shape.
    plain_tensor = torch.randn(10, 10)
    dual_output = fn(dual_input, plain_tensor)

    # Unpacking the dual returns a ``namedtuple`` with ``primal`` and ``tangent``
    # as attributes
    jvp = fwAD.unpack_dual(dual_output).tangent

assert fwAD.unpack_dual(dual_output).tangent is None

######################################################################
# Usage with Modules
# --------------------------------------------------------------------
# To use ``nn.Module`` with forward AD, replace the parameters of your
# model with dual tensors before performing the forward pass. At the
# time of writing, it is not possible to create dual tensor
# `nn.Parameter`s. As a workaround, one must register the dual tensor
# as a non-parameter attribute of the module.

import torch.nn as nn

model = nn.Linear(5, 5)
input = torch.randn(16, 5)

params = {name: p for name, p in model.named_parameters()}
tangents = {name: torch.rand_like(p) for name, p in params.items()}

with fwAD.dual_level():
    for name, p in params.items():
        delattr(model, name)
        setattr(model, name, fwAD.make_dual(p, tangents[name]))

    out = model(input)
    jvp = fwAD.unpack_dual(out).tangent

######################################################################
# Using the functional Module API (beta)
# --------------------------------------------------------------------
# Another way to use ``nn.Module`` with forward AD is to utilize
# the functional Module API (also known as the stateless Module API).

from torch.func import functional_call

# We need a fresh module because the functional call requires the
# the model to have parameters registered.
model = nn.Linear(5, 5)

dual_params = {}
with fwAD.dual_level():
    for name, p in params.items():
        # Using the same ``tangents`` from the above section
        dual_params[name] = fwAD.make_dual(p, tangents[name])
    out = functional_call(model, dual_params, input)
    jvp2 = fwAD.unpack_dual(out).tangent

# Check our results
assert torch.allclose(jvp, jvp2)

######################################################################
# Custom autograd Function
# --------------------------------------------------------------------
# Custom Functions also support forward-mode AD. To create custom Function
# supporting forward-mode AD, register the ``jvp()`` static method. It is
# possible, but not mandatory for custom Functions to support both forward
# and backward AD. See the
# `documentation <https://pytorch.org/docs/master/notes/extending.html#forward-mode-ad>`_
# for more information.

class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        result = torch.exp(foo)
        # Tensors stored in ``ctx`` can be used in the subsequent forward grad
        # computation.
        ctx.result = result
        return result

    @staticmethod
    def jvp(ctx, gI):
        gO = gI * ctx.result
        # If the tensor stored in`` ctx`` will not also be used in the backward pass,
        # one can manually free it using ``del``
        del ctx.result
        return gO

fn = Fn.apply

primal = torch.randn(10, 10, dtype=torch.double, requires_grad=True)
tangent = torch.randn(10, 10)

with fwAD.dual_level():
    dual_input = fwAD.make_dual(primal, tangent)
    dual_output = fn(dual_input)
    jvp = fwAD.unpack_dual(dual_output).tangent

# It is important to use ``autograd.gradcheck`` to verify that your
# custom autograd Function computes the gradients correctly. By default,
# ``gradcheck`` only checks the backward-mode (reverse-mode) AD gradients. Specify
# ``check_forward_ad=True`` to also check forward grads. If you did not
# implement the backward formula for your function, you can also tell ``gradcheck``
# to skip the tests that require backward-mode AD by specifying
# ``check_backward_ad=False``, ``check_undefined_grad=False``, and
# ``check_batched_grad=False``.
torch.autograd.gradcheck(Fn.apply, (primal,), check_forward_ad=True,
                         check_backward_ad=False, check_undefined_grad=False,
                         check_batched_grad=False)

######################################################################
# Functional API (beta)
# --------------------------------------------------------------------
# We also offer a higher-level functional API in functorch
# for computing Jacobian-vector products that you may find simpler to use
# depending on your use case.
#
# The benefit of the functional API is that there isn't a need to understand
# or use the lower-level dual tensor API and that you can compose it with
# other `functorch transforms (like vmap) <https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html>`_;
# the downside is that it offers you less control.
#
# Note that the remainder of this tutorial will require functorch
# (https://github.com/pytorch/functorch) to run. Please find installation
# instructions at the specified link.

import functorch as ft

primal0 = torch.randn(10, 10)
tangent0 = torch.randn(10, 10)
primal1 = torch.randn(10, 10)
tangent1 = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

# Here is a basic example to compute the JVP of the above function.
# The ``jvp(func, primals, tangents)`` returns ``func(*primals)`` as well as the
# computed Jacobian-vector product (JVP). Each primal must be associated with a tangent of the same shape.
primal_out, tangent_out = ft.jvp(fn, (primal0, primal1), (tangent0, tangent1))

# ``functorch.jvp`` requires every primal to be associated with a tangent.
# If we only want to associate certain inputs to `fn` with tangents,
# then we'll need to create a new function that captures inputs without tangents:
primal = torch.randn(10, 10)
tangent = torch.randn(10, 10)
y = torch.randn(10, 10)

import functools
new_fn = functools.partial(fn, y=y)
primal_out, tangent_out = ft.jvp(new_fn, (primal,), (tangent,))

######################################################################
# Using the functional API with Modules
# --------------------------------------------------------------------
# To use ``nn.Module`` with ``functorch.jvp`` to compute Jacobian-vector products
# with respect to the model parameters, we need to reformulate the
# ``nn.Module`` as a function that accepts both the model parameters and inputs
# to the module.

model = nn.Linear(5, 5)
input = torch.randn(16, 5)
tangents = tuple([torch.rand_like(p) for p in model.parameters()])

# Given a ``torch.nn.Module``, ``ft.make_functional_with_buffers`` extracts the state
# (``params`` and buffers) and returns a functional version of the model that
# can be invoked like a function.
# That is, the returned ``func`` can be invoked like
# ``func(params, buffers, input)``.
# ``ft.make_functional_with_buffers`` is analogous to the ``nn.Modules`` stateless API
# that you saw previously and we're working on consolidating the two.
func, params, buffers = ft.make_functional_with_buffers(model)

# Because ``jvp`` requires every input to be associated with a tangent, we need to
# create a new function that, when given the parameters, produces the output
def func_params_only(params):
    return func(params, buffers, input)

model_output, jvp_out = ft.jvp(func_params_only, (params,), (tangents,))


######################################################################
# [0] https://en.wikipedia.org/wiki/Dual_number
