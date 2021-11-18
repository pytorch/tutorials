# -*- coding: utf-8 -*-
"""
Forward-mode Auto Differentiation
=================================

This tutorial demonstrates how to compute directional derivatives
(or, equivalently, Jacobian-vector products) with forward-mode AD.

"""

######################################################################
# Basic Usage
# --------------------------------------------------------------------
# Unlike reverse-mode AD, forward-mode AD computes gradients eagerly
# alongside the forward pass. We can compute a directional derivative
# with forward-mode AD by performing the forward pass as we usually do,
# except, prior to calling the function, we first associate with our
# input with another tensor representing the direction of the directional
# derivative, or equivalently, the ``v`` in a Jacobian-vector product.
# We also call this "direction" tensor a tangent tensor.
#
# As the forward pass is performed (if any input tensors have associated
# tangents) extra computation is performed to propogate this "sensitivity"
# of the function.
#
# [0] https://en.wikipedia.org/wiki/Dual_number

import torch
import torch.autograd.forward_ad as fwAD

primal = torch.randn(10, 10)
tangent = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

# All forward AD computation must be performed in the context of
# the a ``dual_level`` context. All dual tensors created in a
# context will have their tangents destoryed upon exit. This is to ensure that
# if the output or intermediate results of this computation are reused
# in a future forward AD computation, their tangents (which are associated
# with this computation) won't be confused with tangents from later computation.
with fwAD.dual_level():
    # To create a dual tensor we associate a tensor, which we call the
    # primal with another tensor of the same size, which call the tangent.
    # If the layout of the tangent is different from that of the primal,
    # The values of the tangent are copied into a new tensor with the same
    # metadata as the primal. Otherwise, the tangent itself is used as-is.
    #
    # It is important to take note that the dual tensor created by
    # ``make_dual``` is a view of the primal.
    dual_input = fwAD.make_dual(primal, tangent)
    assert dual_input._base is primal
    assert fwAD.unpack_dual(dual_input).tangent is tangent

    # Any tensor involved in the computation that do not have an associated tangent,
    # are automatically considered to have a zero-filled tangent.
    plain_tensor = torch.randn(10, 10)
    dual_output = fn(dual_input, plain_tensor)

    # Unpacking the dual returns a namedtuple, with primal and tangent as its
    # attributes
    jvp = fwAD.unpack_dual(dual_output).tangent

assert fwAD.unpack_dual(dual_output).tangent is None
output = fwAD.unpack_dual(dual_output)

######################################################################
# Usage with Modules
# --------------------------------------------------------------------
# To use ``nn.Module``s with forward AD, replace the parameters of your
# model with dual tensors before performing the forward pass.

import torch.nn as nn

model = nn.Linear(10, 10)
input = torch.randn(64, 10)

with fwAD.dual_level():
    for name, p in model.named_parameters():
        # detach to avoid the extra overhead of creating the backward graph
        # print(p)
        # Oh no! This doesn't quite work yet because make_dua
        # I don't think subclassing works with forward AD because...
        #
        # dual_param = fwAD.make_dual(p.detach(), torch.randn_like(p))
        dual_param = fwAD.make_dual(p, torch.rand_like(p))
        setattr(model, name, dual_param)
    print(fwAD.unpack_dual(getattr(model, "weight")))
    out = model(input)

    # print(fwAD.unpack_dual(next(model.parameters())).tangent)

    jvp = fwAD.unpack_dual(out).tangent

    print("2", jvp)

######################################################################
# Using Modules stateless API
# --------------------------------------------------------------------
# Another way to use ``nn.Module``s with forward AD is to utilize
# the stateless API:

from torch.nn.utils._stateless import functional_call

params = {}
with fwAD.dual_level():
    for name, p in model.named_parameters():
        params[name] = fwAD.make_dual(p, torch.randn_like(p))
    out = functional_call(model, params, input)
    jvp = fwAD.unpack_dual(out).tangent


######################################################################
# Custom autograd Function
# --------------------------------------------------------------------
# Hello world

class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return foo * 2

    @staticmethod
    def jvp(ctx, gI):
        torch.randn_like(gI)
        return gI * 2
