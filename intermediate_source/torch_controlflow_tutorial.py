# -*- coding: utf-8 -*-
"""
Tutorial of control flow operators
========================================
**Authors:** Yidi Wu, Thomas Ortner, Richard Zou, Edward Yang, Adnan Akhundov, Horace He and Yanan Cao

This tutorial introduces the PyTorch Control Flow Operators: 
``cond``, ``while_loop``, ``scan``, ``associative_scan``, and ``map``. 
These operators enable data-dependent control flow to be expressed in a functional, 
differentiable, and exportable manner. 

The tutorial is split into three parts:

Part 1: Basic Inference Examples
--------------------------
Demonstrates basic usage of each control flow operator.

Part 2: Advanced Examples
-------------------------
Demonstrates selected usecases in more complex scenarios

Part 3: Autograd Examples
-------------------------
Shows how PyTorch's autograd integrates with the control flow operators
and how to compute gradients through them.

Additional Reference:
`Control flow operator paper <https://openreview.net/pdf?id=GMFG27v26J>`__ 
(for semantics and detailed implementation notes)

Note: The control flow operators are experimental as of PyTorch 2.9
and thus may be subject to change.
"""

import torch
from torch._higher_order_ops.map import map
from torch._higher_order_ops.cond import cond
from torch._higher_order_ops.scan import scan
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.while_loop import while_loop

######################################################################
# Part 1: Basic Inference Examples
# ================================
#
# This section demonstrates the use of control flow operators
# for inference. Each example corresponds to an operator
# introduced in the paper.
######################################################################

######################################################################
# cond — data-dependent branching
# -------------------------------
#
# The ``cond`` operator performs a data-dependent branch that
# can be traced and exported. Both branches must have the same input
# and output structure.
######################################################################
class CondExample(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        
        # Define the branch prediction function
        pred = (x.sum() > 0).unsqueeze(0)

        # Define the function for the true branch
        def true_fn(t: torch.Tensor):
            return (t.cos(),)

        # Define the function for the false branch
        def false_fn(t: torch.Tensor):
            return (t.sin(),)

        out = cond(pred, true_fn, false_fn, (x,))
        return out[0]

# Define the input
xs = torch.randn(3, 3)

# Compute the results using cond
model = CondExample()
result = model(xs)

# Compute the results with pure PyTorch
result_pytorch = xs.cos() if xs.sum() > 0 else xs.sin()

print('='*80)
print('Example: cond')
print("Native PyTorch:\n", result_pytorch)
print("Result with cond:\n", result)
torch.testing.assert_close(result_pytorch, result)
print('-'*80)

######################################################################
# while_loop — iterative computation with a stopping condition
# ------------------------------------------------------------
#
# The ``while_loop`` operator executes a body function repeatedly
# while a condition is met. 
# Both, the condition and the body receive the same arguments.
# The body must preserve the structure of the arguments.
######################################################################
class CumulativeSumWithWhileLoopExample(torch.nn.Module):
    def forward(self, cnt: torch.Tensor, s: torch.Tensor):
        def cond_fn(i, s):
            return i < 5

        def body_fn(i, s):
            return (i + 1, s + i)

        cnt_final, cumsum_final = while_loop(cond_fn, body_fn, (cnt, s))
        return cumsum_final

# Define the inputs
cnt = torch.tensor(0)
s = torch.tensor(0)

# Compute ground truth
result_pytorch = torch.cumsum(torch.arange(5), 0)[-1]

model = CumulativeSumWithWhileLoopExample()
result = model(cnt, s)

print('='*80)
print('Example: while_loop')
print("Native PyTorch:\n", result_pytorch)
print("Cumulative sum with while_loop result:\n", result)
torch.testing.assert_close(result_pytorch, result)
print('-'*80)

######################################################################
# scan — sequential accumulation
# ------------------------------
#
# The ``scan`` operator performs a for-loop style computation and returns both the
# final carry and stacked outputs per iteration.
######################################################################
class ScanExample(torch.nn.Module):
    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        def combine(carry, x):
            new_carry = carry + x
            return new_carry, new_carry * 2.

        cumsum_final, _ = scan(combine, init, xs, dim=0)
        return cumsum_final

# Define the inputs
xs = torch.arange(1, 5)
init = torch.tensor(0)

# Compute ground truth
result_pytorch = torch.cumsum(torch.arange(5), 0)[-1]

model = ScanExample()
result = model(init, xs)

print('='*80)
print('Example: scan')
print("Native PyTorch:\n", result_pytorch)
print("Cumulative sum with scan:\n", result)
torch.testing.assert_close(result_pytorch, result)
print('-'*80)

######################################################################
# associative_scan — parallel prefix computation
# ----------------------------------------------
#
# The ``associative_scan`` operator performs an associative accumulation such as a
# prefix product in a parallelizable way.
######################################################################
class AssociativeScanExample(torch.nn.Module):
    def forward(self, xs: torch.Tensor):
        def sum(a, b):
            return a + b
        
        # associative_scan uses two combine modes:
        # 1.) pointwise: In this mode, the combine_fn is required
        #     to be pointwise and all inputs need to be on a CUDA device.
        #     Furthermore, this mode does not support lifted arguments.
        #     However, this mode leverages the triton's associative_scan
        #     and may be more efficient than the generic mode.
        # 2.) generic: In this mode, there are no restrictions on the combine_fn
        #     and lifted arguments are supported as well. However, this mode 
        #     uses pure PyTorch operations and although they get compiled with 
        #     torch.compile, it may not be as efficient as the pointwise mode.
        res_pointwise = associative_scan(combine_fn=sum, xs=xs.cuda(), dim=0, combine_mode="pointwise")[-1]
        res_generic = associative_scan(combine_fn=sum, xs=xs, dim=0, combine_mode="generic")[-1]
        return res_pointwise, res_generic

# Define the inputs
xs = torch.arange(5)

# Compute ground truth
result_pytorch = torch.cumsum(torch.arange(5), 0)[-1]

model = AssociativeScanExample()
res_generic, res_pointwise = model(xs)

print('='*80)
print('Example: associative_scan')
print("Native PyTorch:\n", result_pytorch)
print("Cumulative sum with associative_scan (generic):\n", res_generic)
print("Cumulative sum with associative_scan (pointwise):\n", res_pointwise)
torch.testing.assert_close(result_pytorch, res_generic.to('cpu'))
torch.testing.assert_close(result_pytorch, res_pointwise.to('cpu'))
print('-'*80)

######################################################################
# map — functional iteration over a leading dimension
# ---------------------------------------------------
#
# The ``map`` operator applies a function to slices of its input along 
# the leading dimension, stacking the results.
######################################################################
class MapExample(torch.nn.Module):
    def forward(self, xs: torch.Tensor, y: torch.Tensor):
        def body_fn(x, y):
            return x ** y
        
        result = map(body_fn, xs, y)
        return result

# Define the inputs
xs = torch.arange(5)
y = torch.tensor(2)

# Compute ground truth
result_pytorch = xs ** y

model = MapExample()
result = model(xs, y)

print('='*80)
print('Example: map')
print("Native PyTorch result:\n", result_pytorch)
print("map result:\n", result)
torch.testing.assert_close(result_pytorch, result)
print('-'*80)

###############################################################################
# Part 2: Advanced Examples
# =========================
#
# This section shows more advanced usecases of the control flow operators
###############################################################################

###############################################################################
# Invalid value filtering
# -----------------------
#
# In real applications, tensors may have NaN or infinity due to reasons
# such as invalid mathematical operators (e.g. division by zero), 
# invalid data. Such invalid values can propagate through calculations 
# and can lead to wrong or unstable results. Using ``cond``, those values
# can be removed from tensors. Also, the minimum and maximum values
# of the tensors can be clamped.
###############################################################################
class AdvancedCondExample(torch.nn.Module):
    def forward(self, xs: torch.Tensor):
        def pred(x):
            return torch.isinf(x).any() or torch.isnan(x).any()

        def true_fn(x):
            no_nans_and_infs = torch.nan_to_num(x, nan=0.0, posinf=100, neginf=-100)
            clamped = torch.clamp(no_nans_and_infs, min=min_allowed_value, max=max_allowed_value)
            return clamped

        def false_fn(x):
            return x.clone()

        xs_filtered = cond(pred(xs),
                        true_fn,
                        false_fn,
                        (xs,)
                        )
        
        return xs_filtered

# Define the input with large values and NaNs
xs = torch.tensor([-10., -4, torch.nan, 1, torch.inf])
max_allowed_value = 4
min_allowed_value = -7

model = AdvancedCondExample()
xs_filtered = model(xs)

print('='*80)
print('Example: Value clamping')
print("Unfiltered tensort:\n", xs)
print("Filtered tensor:\n", xs_filtered)
print('-'*80)

######################################################################
# RNN implemented with scan
# -------------------------
#
# RNNs can be implemented efficiently with the ``scan`` operator,
# making them a first-class citizen in PyTorch.
# In this section, we will implement a simple RNN
######################################################################
class AdvancedRNNExample(torch.nn.Module):
    def __init__(self, Wih, bih, Whh, bhh):
        super(AdvancedRNNExample, self).__init__()
        self.Wih = Wih
        self.bih = bih
        self.Whh = Whh
        self.bhh = bhh
        
    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        def rnn_combine(carry, x):
            h = torch.tanh(x @ self.Wih + self.bih + carry @ self.Whh + self.bhh)
            return h + 0., h
        
        carry, outs = scan(rnn_combine, init, xs, dim=0)
        
        return carry, outs

# Define the inputs
xs = torch.randn(4, 3, requires_grad=True)
init = torch.zeros(5, requires_grad=True)

# Define the RNN with pure PyTorch
rnn = torch.nn.RNN(3, 5)
result_pytorch, _ = rnn(xs)

model = AdvancedRNNExample(rnn.weight_ih_l0.T, rnn.bias_ih_l0, 
                           rnn.weight_hh_l0.T, rnn.bias_hh_l0)
_, result = model(init, xs)

print('='*80)
print('Example: RNN with scan')
print("torch.nn.RNN result:\n", result_pytorch)
print("RNN implemented with scan:\n", result)
torch.testing.assert_close(result_pytorch, result)
print('-'*80)

###############################################################################
# Kernel of a state space model implemented with associative_scan
# --------------------------------------------------
#
# The associative_scan operator can be used to implement 
# State Space Models (SSM) such as the S5 model. To do so, one defines the 
# operator used in the SSM the associative_scan and 
###############################################################################
class AdvancedAssociativeScanExample(torch.nn.Module):
    def forward(self, xs: torch.Tensor):
        def s5_operator(x: torch.Tensor, y: torch.Tensor):
            A_i, Bu_i = x
            A_j, Bu_j = y
            return A_j * A_i, A_j * Bu_i + Bu_j

        result = associative_scan(s5_operator, xs, dim=0,)
        return result

# Define the inputs
timesteps = 4
state_dim = 3
A = torch.randn(state_dim, device='cuda')
B = torch.randn(
    timesteps, state_dim, device='cuda'
)
xs = (A.repeat((timesteps, 1)), B)

model = AdvancedAssociativeScanExample()
result = model(xs)

print('='*80)
print('Example: RNN with scan')
print("SSM kernel implemented with associative_scan:\n", result)
print('-'*80)

###############################################################################
# Part 3: Autograd Examples
# =========================
#
# This section shows how control flow operators integrate with PyTorch’s
# autograd feature. Most operators, except the ``while_loop``,
# implement a backward function to compute the gradients. Hence, they can 
# be used in differentiable computations.
###############################################################################

###############################################################################
# Gradients through cond
# ----------------------
#
# This example shows the gradient propagation through a ``cond``.
# To do so, we will reuse the CondExample from above
###############################################################################
# Define the inputs
xs = torch.tensor([-3.], requires_grad=True)

# Compute the ground truth
result_pytorch = xs.cos() if xs.sum() > 0 else xs.sin()

# Compute the cond results
model = CondExample()
result = model(xs)

print('='*80)
print('Example: cond')
print("Native PyTorch:\n", result_pytorch)
print("Result with cond:\n", result)
torch.testing.assert_close(result_pytorch, result)
print("")

# Compute the ground truth gradients.
# The false_fn is used in the example above and the gradient for 
# the sin() function is the cos() function.
# Therefore, we expect the gradients output to be xs.cos()
grad_pytorch = xs.cos()

# Compute the gradients of the cond result
grad = torch.autograd.grad(result, xs)[0]

print("Gradient of PyTorch:\n", grad_pytorch)
print("Gradient of cond:\n", grad)
torch.testing.assert_close(grad_pytorch, grad)
print('-'*80)

###############################################################################
# Gradients through map
# ---------------------
#
# Here we compute gradients through a ``map`` call.
# To do so, we will reuse the MapExample from above
###############################################################################
# Define the inputs
xs = torch.arange(5, dtype=torch.float32, requires_grad=True)
y = torch.tensor(2)

# Compute the ground truth
result_pytorch = xs ** y

# Compute the cond results
model = MapExample()
result = model(xs, y)

print('='*80)
print('Example: map')
print("Native PyTorch result:\n", xs ** y)
print("map result:\n", result)
torch.testing.assert_close(result_pytorch, result)

grad_pytorch = xs * 2
grad_init = torch.ones_like(xs)
grad = torch.autograd.grad(result, xs, grad_init)[0]

# The map function computes x ** y for each element, where y = 2
# Therefore, we expect the correct gradients to be x * 2
print("Gradient of PyTorch:\n", grad_pytorch)
print("Gradient of cond:\n", grad)
torch.testing.assert_close(grad_pytorch, grad)
print('-'*80)

###############################################################################
# Gradient through RNN
# --------------------
#
# In this section, we will demonstrate the gradient computation 
# through an RNN implemented with the scan operator.
# For this example, we will reuse the AdvancedRNNExample 
###############################################################################
# Define the inputs
xs = torch.randn(4, 3, requires_grad=True)
init = torch.zeros(5, requires_grad=True)

# Define the RNN with pure PyTorch
rnn = torch.nn.RNN(3, 5)
result_pytorch, _ = rnn(xs)

model = AdvancedRNNExample(rnn.weight_ih_l0.T, rnn.bias_ih_l0, 
                           rnn.weight_hh_l0.T, rnn.bias_hh_l0)
_, result = model(init, xs)

print('='*80)
print('Example: RNN with scan')
print("torch.nn.RNN result:\n", result_pytorch)
print("RNN implemented with scan:\n", result)
torch.testing.assert_close(result_pytorch, result)

grad_init = torch.ones_like(result_pytorch)
grad_pytorch = torch.autograd.grad(result_pytorch, xs, grad_init)[0]
grad = torch.autograd.grad(result, xs, grad_init)[0]

# The map function computes x ** y for each element, where y = 2
# Therefore, we expect the correct gradients to be x * 2
print("Gradient of PyTorch:\n", grad_pytorch)
print("Gradient of cond:\n", grad)
torch.testing.assert_close(grad_pytorch, grad)

print('-'*80)

################################################################################
# Conclusion
# ----------
#
# In this tutorial we have demonstrated how to use the control flow operators
# in PyTorch. They enable a flexible, differentiable, and exportable 
# way to implement more complex models and functions in PyTorch. 
# In particular, they bridge the gap between dynamic Python control flow 
# and straight-line computational graphs, constructed by torch.compile.
# 
# For further details, please visit the PyTorch documentation or the 
# corresponding `paper <https://openreview.net/pdf?id=GMFG27v26J>`__.
################################################################################
