# -*- coding: utf-8 -*-
"""
Tutorial of control flow operators
========================================
**Authors:** Yidi Wu, Thomas Ortner, Richard Zou, Edward Yang, Adnan Akhundov, Horace He and Yanan Cao

This tutorial introduces the PyTorch Control Flow Operators: ``cond``, ``while_loop``,
``scan``, ``associative_scan``, and ``map``. These operators enable data-dependent
control flow to be expressed in a functional, differentiable, and exportable
manner. The tutorial is split into two parts:

Part 1: Inference Examples
--------------------------
Demonstrates basic usage of each control flow operator, following the examples
from the paper.

Part 2: Autograd and Differentiation
------------------------------------
Shows how PyTorch's autograd integrates with the control flow operators and how
to compute gradients through them.

References:
- Control flow operator paper (for semantics and detailed implementation notes)
- Template for documentation structure (torch.export tutorial)

Note: The control flow operators are experimental as of PyTorch 2.9 and are
subject to change.
"""

import torch
from torch.export import export

try:
    from functorch.experimental.control_flow import cond
except Exception:
    cond = getattr(torch, "cond", None)

from torch._higher_order_ops.map import map as torch_map
from torch._higher_order_ops.scan import scan
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.while_loop import while_loop

################################################################################
# Part 1: Inference Examples
# ==========================
#
# This section demonstrates the use of control flow operators for inference.
# Each example corresponds to an operator introduced in the paper.
################################################################################

######################################################################
# cond — data-dependent branching
# -------------------------------
#
# The ``cond`` operator performs a data-dependent branch that can be traced and
# exported. Both branches must have the same input and output structure.
######################################################################

class CondExample(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        pred = (x.sum() > 0).unsqueeze(0)

        def true_fn(t: torch.Tensor):
            return (t.cos(),)

        def false_fn(t: torch.Tensor):
            return (t.sin(),)

        out = cond(pred, true_fn, false_fn, (x,))
        return out[0]


x = torch.randn(3, 3)
model = CondExample()
print("cond result:\n", model(x))

exported = export(model, (x,))
print("Exported graph for cond:\n", exported.graph)

######################################################################
# while_loop — iterative computation with a stopping condition
# ------------------------------------------------------------
#
# The ``while_loop`` operator executes a body function repeatedly while a condition
# is met. Both condition and body must preserve the structure of the carry.
######################################################################

class CountdownExample(torch.nn.Module):
    def forward(self, n: torch.Tensor):
        def cond_fn(i):
            return i > 0

        def body_fn(i):
            return i - 1

        (res,) = while_loop(cond_fn, body_fn, (n,))
        return res


n = torch.tensor(5)
countdown = CountdownExample()
print("while_loop result:\n", countdown(n))

######################################################################
# scan — sequential accumulation
# ------------------------------
#
# The ``scan`` operator performs a for-loop style computation and returns both the
# final carry and stacked outputs per iteration.
######################################################################

def combine(carry, x):
    new_carry = carry + x
    out = new_carry
    return new_carry, out

xs = torch.tensor([1.0, 2.0, 3.0, 4.0])
init = torch.tensor(0.0)
carry, outs = scan(combine, init, xs, dim=0)
print("scan cumulative result:\n", outs)

######################################################################
# associative_scan — parallel prefix computation
# ----------------------------------------------
#
# The ``associative_scan`` operator performs an associative accumulation such as a
# prefix product in a parallelizable way.
######################################################################

def mul(a, b):
    return a * b

vals = torch.arange(1.0, 6.0)
res = associative_scan(mul, vals, dim=0, combine_mode="pointwise")
print("associative_scan cumulative products:\n", res)

######################################################################
# map — functional iteration over a leading dimension
# ---------------------------------------------------
#
# The ``map`` operator applies a function to slices of its input along the leading
# dimension, stacking the results.
######################################################################

def body_fn(x, y):
    return x + y

xs = torch.ones(4, 3)
y = torch.tensor(5.0)
result = torch_map(body_fn, xs, y)
print("map result:\n", result)

################################################################################
# Part 2: Autograd and Differentiation
# ====================================
#
# This section shows how control flow operators integrate with PyTorch’s autograd.
# The same operators can be used in differentiable computations.
################################################################################

######################################################################
# Gradients through map
# ---------------------
#
# All control flow operators are differentiable if the operations inside them are.
# Here we compute gradients through a ``map`` call.
######################################################################

def differentiable_body(x, y):
    return x.sin() * y.cos()

xs = torch.randn(3, 4, requires_grad=True)
y = torch.randn(4, requires_grad=True)

out = torch_map(differentiable_body, xs, y)
loss = out.sum()
loss.backward()

print("Gradient wrt xs:\n", xs.grad)
print("Gradient wrt y:\n", y.grad)

######################################################################
# Differentiable scan (RNN-style)
# -------------------------------
#
# Gradients can also propagate through a ``scan`` operation where the carry
# represents a hidden state.
######################################################################

def rnn_combine(carry, x):
    h = torch.tanh(carry + x)
    return h, h

xs = torch.randn(4, 3, requires_grad=True)
init = torch.zeros(3, requires_grad=True)
carry, outs = scan(rnn_combine, init, xs, dim=0)
loss = outs.sum()
loss.backward()
print("Gradient wrt xs:\n", xs.grad)
print("Gradient wrt init:\n", init.grad)

################################################################################
# Conclusion
# ----------
#
# The PyTorch control flow operators enable flexible, differentiable, and
# exportable control flow directly in Python. The main takeaways from the paper
# are:
#
# 1. **Unified semantics**: Each operator has clearly defined input/output rules
#    and pytree invariants that ensure compatibility with ``torch.export``.
# 2. **Differentiability**: Operators like ``map``, ``scan``, and ``cond`` support
#    full autograd propagation, allowing seamless integration with gradient-based
#    methods.
# 3. **Exportability**: Because they are implemented as functional ops, control
#    flow constructs can be traced, serialized, and optimized like standard ops.
# 4. **Efficiency and parallelism**: Operators such as ``associative_scan`` allow
#    parallel prefix computation, unlocking performance gains.
# 5. **Structured control flow**: ``cond`` and ``while_loop`` generalize
#    conditional and iterative logic while preserving graph structure and
#    analyzability.
#
# These operators bridge the gap between dynamic Python control flow and static
# computation graphs, providing a powerful foundation for defining models with
# complex or data-dependent behaviors in PyTorch.
################################################################################
