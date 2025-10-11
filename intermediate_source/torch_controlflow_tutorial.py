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
torch._dynamo.config.capture_scalar_outputs = True

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

###############################################################################
# RNNs implemented with scan
# -------------------------
#
# RNNs can be implemented either with for-loops, the ``scan`` operator,
# or by writing custom CUDA kernels.
# However, depending on the dynamics the RNNs, creating a custom CUDA kernels
# may be very time consuming and error prone. Especially, because not only the
# forward path needs to be implemented, but the backward path as well, which
# can become very complex.
# The ``scan`` operator allows to aleviate this additional effort and makes
# RNNs a first-class citizen in PyTorch.
# In this section, we will show how an LSTM can be implemented in the three
# ways described above. We will also measure the execution time for the 
# forward and backward propagtion.
###############################################################################
class LSTM_forloop(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_forloop, self).__init__()
        
        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        
    # Implementation adopted from 
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        # The input `xs` has the time as the first dimsion
        output = []
        for i in range(xs.size()[0]):
            hx, cx = self.lstm_cell(xs[i], init)
            init = (hx, cx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        return output
    
class LSTM_scan(torch.nn.Module):
    def __init__(self, Wii, bii, Whi, bhi, Wif, bif, Whf, bhf, Wig, big, Whg, bhg, Wio, bio, Who, bho):
        super(LSTM_scan, self).__init__()
        self.Wii = Wii.clone()
        self.bii = bii.clone()
        self.Whi = Whi.clone()
        self.bhi = bhi.clone()
        self.Wif = Wif.clone()
        self.bif = bif.clone()
        self.Whf = Whf.clone()
        self.bhf = bhf.clone()
        self.Wig = Wig.clone()
        self.big = big.clone()
        self.Whg = Whg.clone()
        self.bhg = bhg.clone()
        self.Wio = Wio.clone()
        self.bio = bio.clone()
        self.Who = Who.clone()
        self.bho = bho.clone()
        
    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        def lstm_combine(carry, x):
            h, c = carry
            
            i = torch.sigmoid(x @ self.Wii + self.bii + h @ self.Whi + self.bhi)
            f = torch.sigmoid(x @ self.Wif + self.bif + h @ self.Whf + self.bhf)
            g = torch.tanh(x @ self.Wig + self.big + h @ self.Whg + self.bhg)
            o = torch.sigmoid(x @ self.Wio + self.bio + h @ self.Who + self.bho)
            
            c_new = f * c + i * g
            h_new = o * torch.tanh(c_new)
            
            # return (h_new, c_new.clone()), h_new.clone()
            return (h_new, c_new + 0.), h_new + 0.
        
        carry, outs = scan(lstm_combine, init, xs, dim=0)
        
        return carry, outs
print('='*80)
print('Example: RNN with scan')

from time import perf_counter
def time_fn(fn, args, warm_up=1):
    t_initial = -1.
    for ind in range(warm_up):
        t_start = perf_counter() 
        result = fn(*args)
        t_stop = perf_counter()
        if ind == 0:
            t_initial = t_stop - t_start

    t_start = perf_counter() 
    result = fn(*args)
    t_stop = perf_counter()
    t_run = t_stop - t_start
    return result, t_initial, t_run

# Define the inputs
time_steps = 20
warm_up_cycles = 3
# input_size = 15
input_size = 50
# hidden_size = 20
hidden_size = 200
xs = torch.randn(time_steps, input_size, requires_grad=True)  # (time_steps, batch, input_size)
h = torch.randn(hidden_size)  # (batch, hidden_size)
c = torch.randn(hidden_size)
init = (h, c)

# Define the for-loop LSTM model
lstm_forloop = LSTM_forloop(input_size, hidden_size)
lstm_forloop_comp = torch.compile(lstm_forloop, fullgraph=True)

# Define the LSTM using CUDA kernels
lstm_forloop_state_dict = lstm_forloop.state_dict()
lstm_cuda_state_dict = {}
for key, value in lstm_forloop_state_dict.items():
    new_key = key.replace('lstm_cell.', '') + '_l0'
    lstm_cuda_state_dict[new_key] = value.clone()
lstm_cuda = torch.nn.LSTM(input_size, hidden_size)
lstm_cuda.load_state_dict(lstm_cuda_state_dict)

# Define the LSTM model using scan
Wii, Wif, Wig, Wio = torch.chunk(lstm_cuda.weight_ih_l0, 4)
Whi, Whf, Whg, Who = torch.chunk(lstm_cuda.weight_hh_l0, 4)
bii, bif, big, bio = torch.chunk(lstm_cuda.bias_ih_l0, 4)
bhi, bhf, bhg, bho = torch.chunk(lstm_cuda.bias_hh_l0, 4)
lstm_scan = LSTM_scan(
                Wii.T, bii,
                Whi.T, bhi,
                
                Wif.T, bif,
                Whf.T, bhf,
                
                Wig.T, big,
                Whg.T, bhg,
                
                Wio.T, bio,
                Who.T, bho,
                )
lstm_scan_comp = torch.compile(lstm_scan, fullgraph=True)

# Run the models, time them and check for equivalence
result_forloop, time_initial_forloop, time_run_forloop = time_fn(lstm_forloop, (init, xs), warm_up=warm_up_cycles)
result_forloop_comp, time_initial_forloop_comp, time_run_forloop_comp = time_fn(lstm_forloop_comp, (init, xs), warm_up=warm_up_cycles)
result_cuda, time_initial_cuda, time_run_cuda = time_fn(lstm_cuda, (xs.clone(), (init[0].clone().unsqueeze(0), init[1].clone().unsqueeze(0))), warm_up=warm_up_cycles)
result_scan, time_initial_scan, time_run_scan = time_fn(lstm_scan, ((init[0].clone().unsqueeze(0), init[1].clone().unsqueeze(0)), xs.clone()), warm_up=warm_up_cycles)
result_scan_comp, time_initial_scan_comp, time_run_scan_comp = time_fn(lstm_scan_comp, ((init[0].clone().unsqueeze(0), init[1].clone().unsqueeze(0)), xs.clone()), warm_up=warm_up_cycles)

torch.testing.assert_close(result_forloop, result_forloop_comp)
torch.testing.assert_close(result_forloop, result_cuda[0])
torch.testing.assert_close(result_forloop, result_scan[1][:, 0, :])
torch.testing.assert_close(result_forloop, result_scan_comp[1][:, 0, :])
print('-'*80)
print(f'T={time_steps}:')
print(f'Compile times:\n\
For-Loop        : {time_initial_forloop_comp:.5f}\n\
Scan            : {time_initial_scan_comp:.5f}\n')
print(f'Run times       :\n\
For-Loop        : {time_run_forloop:.5f} \n\
For-Loop compile: {time_run_forloop_comp:.5f} \n\
CUDA            : {time_run_cuda:.5f} \n\
Scan            : {time_run_scan:.5f} \n\
Scan compile    : {time_run_scan_comp:.5f}')

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
print('Example: Advanced associative_scan')
print("SSM kernel implemented with associative_scan:\n", result)
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
