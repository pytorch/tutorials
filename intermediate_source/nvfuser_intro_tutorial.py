# -*- coding: utf-8 -*-
"""
Getting started with nvFuser
****************************

**Authors**: `Christian Sarofeen <https://github.com/csarofeen>`_
`Kevin Stephano <https://github.com/kevinstephano>`_
`Piotr Bialecki <https://github.com/ptrblck>`_
`Neal Vaidya`


Introduction
------------

This tutorial shows how to trace PyTorch programs using TorchScript, FuncTorch,
and TorchDynamo, and then execute them with nvFuser.

Importing Packages and Selecting a Device
-----------------------------------------
In order to run this tutorial and see the benefits of using nvfuser, 
you would need to install the `1.12.0` PyTorch release as well as 
`functorch` `0.2` or newer version of them. 
Additionally, a GPU is required.
"""

import functools
import random
random.seed(42)
import time
from typing import List
from copy import deepcopy

import torch
import torch.nn.functional as F

from torch.utils import collect_env
print(collect_env.get_pretty_env_info())


######################################################################
# TorchScript & nvFuser
# ---------------------
# Let’s start with a series of operations found in the Transformer Block of
# networks like BERT.
#
# .. figure:: /_static/img/nvfuser_intro/transformer_block.png
#
# In particular, let’s focus on the Bias-Dropout-Add-LayerNorm portion.
# First, let’s define the forward pass for this section of our network.
#

def composite_definition(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    normalization_axis: int,
    dropout_prob: float,
) -> torch.Tensor:
    bias1_out = input1 + bias1
    dropout_out = F.dropout(bias1_out, dropout_prob)
    norm_input = dropout_out + input2
    norm_output = F.layer_norm(norm_input, (input1.size(normalization_axis),), weight, bias2)
    return norm_output

######################################################################
# Because of PyTorch’s autograd system, we don’t need to explicitly define the
# backwards pass. So, the only thing left for us to do is initialize our
# parameters, and create some input tensors along with a simulated gradient
# output tensor for the backwards pass.
#
    
# Setup initial tensors and parameters
input_size = [64, 128, 1024]
device = "cuda"
dtype = torch.float32

######################################################################
# To see the speedup factor of nvFuser, we’ll first measure the speed of our
# forward and backward passes in PyTorch’s default eager mode. To get accurate
# and comparable measurements it’s important to run a few warm up iterations,
# as compilation isn’t performed until the third iteration when using nvFuser
# with TorchScript. Since this is not an end to end network, we’re focusing on
# measuring the performance of the generated kernels themselves. For these
# examples the compilation time of TorchScript and nvFuser were approximately 
# 2.4 seconds.
#

# Create sample inputs
input1 = torch.randn(*input_size, device=device, dtype=dtype, requires_grad=True)
input2 = torch.rand_like(input1).requires_grad_()
 
# Precompute a grad output tensor, for this example it's the same size as the inputs
grad_output = torch.rand_like(input1)
 
# Randomly initialize the model parameters
weight = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias1 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias2 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))

parameters = [input1, input2, weight, bias1, bias2]

######################################################################
# To accurately measure the execution time of the iterations we will do
# 100 iterations with proper GPU synchronization and warmup iterations 
# before we take the timing. Each iteration will both run forward and backwards
# so we have the total iteration time of this operation for training.
# We define a helper method called `profile_workload` and wull use `functool.partial`
# to profile the workloads where possible.
#

# util to profile the workload
def profile_workload(forward_func, grad_output, iteration_count=100):
    # Perform warm-up iterations
    for _ in range(3):
        # Run model, forward and backward
        output = forward_func()
        output.backward(grad_output)
        # delete gradiens to avoid profiling the gradient accumulation
        for p in parameters:
            p.grad = None

    # Synchronize the GPU before starting the timer
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iteration_count):
        # Run model, forward and backward
        output = func()
        output.backward(grad_output)
        # delete gradiens to avoid profiling the gradient accumulation
        for p in parameters:
            p.grad = None

    # Synchronize the GPU before stopping the timer
    torch.cuda.synchronize()
    stop = time.perf_counter()
    iters_per_second = iteration_count / (stop - start)
    print("Average iterations per second: {:.2f}".format(iters_per_second))

func = functools.partial(composite_definition, input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1)
profile_workload(func, grad_output, iteration_count=100)

######################################################################
# To turn on nvFuser, allowing it to generate fast kernels and take over
# execution of these operations we’ll start by using TorchScript. We decorate
# the function of interest, which is a common approach given how challenging it
# has been to get entire models run through TorchScript. We do so by adding the
# TorchScript decorator to our function and adding some type information to the
# function. Adding type information is not always required, but it can often
# help to provide this information to TorchScript because it is a strictly
# typed system. To provide TorchScript even more information, we will also let
# it know which axes we’re going to normalize on, the dropout probability,
# and the keep dimension parameter which are constants and will not change from one
# iteration to the next.
#

scripted_composite_definition = torch.jit.script(composite_definition)

func = functools.partial(scripted_composite_definition, input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1)
profile_workload(func, grad_output, iteration_count=100)

######################################################################
# !TODO Add result TODO!
# In [Section] we’ll explore how to improve this even further by optimizing
# which tensors are saved between the forward and backward pass, which
# currently isn’t possible with TorchScript. Even so, nvFuser generates highly
# optimized kernels that provide significant improvements to performance.
# It is important to mention here that nvFuser does not generate the exact same
# sequence of random numbers, therefore validating nvFuser with PyTorch
# execution without nvFuser requires disabling the random number generation
# functions (dropout in this instance).
#

######################################################################
# Dynamic Shapes
# --------------
#
# Dynamic shape support was a feature designed into nvFuser from the beginning.
# The ability to process different sized input tensors are critical to many
# applications, like Natural Language Processing and Graph Neural Networks.
# To test this feature out we’ll change the way we generate inputs and the
# output gradient tensors to make a few different sizes. We only perturb the
# first two dimensions of the input, since the last dimension is shared with
# the parameters and cannot be changed dynamically in `LayerNorm`. Now, let’s
# replace the generation of our inputs and run our network snippet again:
#

SHAPE_COUNT = 20
dyn_size = deepcopy(input_size)
 
inputs1: List[torch.Tensor] = []
inputs2 : List[torch.Tensor]= []
grad_outputs : List[torch.Tensor]= []
 
 
# Create some random shapes1
for _ in range(SHAPE_COUNT):
    dyn_size[0] = input_size[0] + random.randrange(-2, 3)
    dyn_size[1] = input_size[1] + random.randrange(-2, 3)
    input = torch.randn(*dyn_size, device=device, dtype=dtype, requires_grad=True)
    inputs1.append(input)
    inputs2.append(torch.rand_like(input))
    grad_outputs.append(torch.rand_like(input))
 
# Perform warm-up iterations
for _ in range(SHAPE_COUNT):
    input1 = inputs1[0]
    input2 = inputs2[0]
    grad_output = grad_outputs[0]
    # Run model, forward and backward
    output = composite_definition(input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1)
    output.backward(grad_output)

# Profile manually as our helper function expects static inputs
iteration_count = 100
# Synchronize the GPU before starting the timer
torch.cuda.synchronize()
start = time.perf_counter()
for i in range(iteration_count):
    input1 = inputs1[i % SHAPE_COUNT]
    input2 = inputs2[i % SHAPE_COUNT]
    grad_output = grad_outputs[i % SHAPE_COUNT]
    parameters = [input1, input2, weight, bias1, bias2]
 
    # Run model, forward and backward
    output = composite_definition(input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1)
    output.backward(grad_output)
    # delete gradiens to avoid profiling the gradient accumulation
    for p in parameters:
        p.grad = None
 
# Synchronize the GPU before stopping the timer
torch.cuda.synchronize()
stop = time.perf_counter()
iters_per_second = iteration_count / (stop - start)
print("Average iterations per second: {:.2f}".format(iters_per_second))

######################################################################
# !TODO Add result TODO!
# No changes from before are required for running with TorchScript.
# Both timings are a little slower than the previous example with static shapes,
# likely due to misses in the caching allocator, but this overhead is expected
# to resolve quickly after several real training iterations. nvFuser is able to
# reuse the same kernels it compiled from the warmup iterations even with the
# input sizes changing slightly. When sizes change significantly, nvFuser is
# able to generate new kernels to adapt to the shapes being run, generating
# fast kernels across all input sizes.
#
# Note that today,  nvFuser in TorchScript is the only exposure of nvFuser that
# allows for dynamic shape changes, although we are hoping to expand this in
# the future. For more insight into how dynamic shapes are implemented in nvFuser,
# you can view this session from GTC: https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31952/.

######################################################################
# Defining novel operations with nvFuser and FuncTorch
# ----------------------------------------------------
# 
# One of the primary benefits of nvFuser is the ability to define new composite
# functions from PyTorch primitives which are then dynamically compiled into efficient kernels. 
#
# TorchScript typically works well when predefined composite operations are used,
# but if we change the definition of our operations to use only PyTorch primitives
# we can begin to see some negative impacts. Let’s redefine our Transformer operation
# to replace the `torch.nn.functional.layer_norm` function with primitive PyTorch operations to see the effects.
#

def primitive_definition(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    normalization_axis: int,
    dropout_prob: float,
    keepdim: bool,
) -> torch.Tensor:
    bias1_out = input1 + bias1
    dropout_out = F.dropout(bias1_out, dropout_prob)
    norm_input = dropout_out + input2
    mean = norm_input.mean(normalization_axis, keepdim=keepdim)
    diff = (norm_input - mean)
    diff_sq = diff * diff
    var = diff_sq.mean(normalization_axis, keepdim=keepdim)
    pre_shift_scale_norm_output = (norm_input - mean) / torch.sqrt(var + 1e-12)
    norm_output = weight * pre_shift_scale_norm_output + bias2
    return norm_output

# Create sample inputs
input1 = torch.randn(*input_size, device=device, dtype=dtype, requires_grad=True)
input2 = torch.rand_like(input1).requires_grad_()
 
# Precompute a grad output tensor, for this example it's the same size as the inputs
grad_output = torch.rand_like(input1)
 
# Randomly initialize the model parameters
weight = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias1 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias2 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))

parameters = [input1, input2, weight, bias1, bias2]

# Profile primitive definition
func = functools.partial(primitive_definition, input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1, keepdim=True)
profile_workload(func, grad_output, iteration_count=100)

# Profile scripted primitive definition
scripted_primitive_definition = torch.jit.script(primitive_definition)
func = functools.partial(scripted_primitive_definition, input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1, keepdim=True)
profile_workload(func, grad_output, iteration_count=100)

######################################################################
# !TODO Add result TODO!
# While the above is mathematically equivalent to our previous definition,
# benchmarking our new function with the original static shape using TorchScript
# and nvFuser shows the execution time increases – mostly due to the cost of accessing memory for intermediate results.
#
# TorchScript’s application of Autograd saves activations for each operator in
# the fusion for re-use in the backwards pass, but this isn’t always the optimal
# choice. Especially when chaining together multiple simple operations, it is
# often faster to recompute the original tensors than to store and retrieve
# several saved results from memory. It’s possible to optimize execution of the
# network to eliminate some of these unnecessary memory accesses,
# but it requires building a connected forward and backward graph
# which isn’t possible with TorchScript. Fortunately,
# the `memory_efficient_fusion` pass in FuncTorch, is such an optimization pass.
# To use this pass we have to redefine our function for FuncTorch by pulling the constants inside
# (for now this is the easiest approach for direct use of FuncTorch):
#

def primitive_definition_for_memory_efficient_fusion(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    bias1_out = input1 + bias1
    dropout_out = F.dropout(bias1_out, 0.1)
    norm_input = dropout_out + input2
    mean = norm_input.mean(2, keepdim=True)
    diff = (norm_input - mean)
    diff_sq = diff * diff
    var = diff_sq.mean(2, keepdim=True)
    pre_shift_scale_norm_output = (norm_input - mean) / torch.sqrt(var + 1e-12)
    norm_output = weight * pre_shift_scale_norm_output + bias2
    return norm_output

######################################################################
# Now, instead of passing our function to TorchScript we call FuncTorch’s optimization pass:
#

# Optimize the model with FuncTorch tracing and the memory efficiency optimization pass
from functorch.compile import memory_efficient_fusion
memory_efficient_primitive_definition = memory_efficient_fusion(primitive_definition_for_memory_efficient_fusion)

# Profile memory efficient primitive definition
func = functools.partial(memory_efficient_primitive_definition, input1, input2, weight, bias1, bias2)
profile_workload(func, grad_output, iteration_count=100)

######################################################################
# TODO: Add result TODO
# This improves performance, which is still not as fast as TorchScripts original runtime with the composite definition,
# but much faster than running this new definition without nvFuser.

######################################################################
# .. note:: FuncTorch’s memory efficient pass is still actively in development
#           and future versions are expected to achieve performance closer
#           to that of TorchScript with the composite definition.

######################################################################
# The ability to quickly execute chains of simple operations is important as not every operation has 
# a composite operation defined in PyTorch. Previously, this meant researchers either had to define an entirely
# new optimized operation in PyTorch's internals – which can take a lot of time
# and knowledge of lower-level code – or write the operation in simpler PyTorch
# ops and settle for poor performance. For example, let's replace LayerNorm in
# our example with RMS Norm, which doesn’t have an existing operation definition in PyTorch.
#
#Without nvFuser:
#

def with_rms_norm(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    normalization_axis: int,
    dropout_prob: float,
    keepdim: bool,
) -> torch.Tensor:
    bias_out = input1 + bias
    dropout_out = F.dropout(bias_out, dropout_prob)
    norm_input = dropout_out + input2
    var = norm_input.mul(norm_input).mean(normalization_axis, keepdim)
    pre_shift_scale_norm_output = norm_input / torch.sqrt(var + 1e-12)
    norm_output = weight * pre_shift_scale_norm_output
    return norm_output

# Profile rms_norm
func = functools.partial(with_rms_norm, input1, input2, weight, bias1, normalization_axis=2, dropout_prob=0.1, keepdim=True)
profile_workload(func, grad_output=grad_output)

######################################################################
# With nvFuser Through TorchScript:
#

# Profile scripted rms_norm
scripted_with_rms_norm = torch.jit.script(with_rms_norm)
func = functools.partial(scripted_with_rms_norm, input1, input2, weight, bias1, normalization_axis=2, dropout_prob=0.1, keepdim=True)
profile_workload(func, grad_output, iteration_count=100)

######################################################################
# With nvFuser Through Functorch:
#

def with_rms_norm_for_memory_efficient_fusion(input1: torch.Tensor, input2: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    bias_out = input1 + bias
    dropout_out = torch.nn.functional.dropout(bias_out, 0.1)
    norm_input = dropout_out + input2
    var = norm_input.mul(norm_input).mean(2, keepdim=True)
    pre_shift_scale_norm_output = norm_input / torch.sqrt(var + 1e-12)
    norm_output = weight * pre_shift_scale_norm_output
    return norm_output

# Profile memory efficient rms_norm
memory_efficient_rms_norm = memory_efficient_fusion(with_rms_norm_for_memory_efficient_fusion)
func = functools.partial(memory_efficient_rms_norm, input1, input2, weight, bias1)
profile_workload(func, grad_output, iteration_count=100)

######################################################################
# !TODO Add Results TODO!
# The results show a significant increase in the iterations per second by using TorchScript and nvFuser which is further
# improved with FuncTorch's memory efficientr optimization pass –
# nearly matching the performance of the hand-optimized LayerNorm implementation. 
#
# The ability to define novel operations natively in Python and get performance
# that’s close to a highly optimized composite operation will enable research
# into novel network topologies without paying for the sometimes devastating
# effects on speed of training. nvFuser provides this unique ability as it’s able to
# analyze users’ programs to provide performance as fast as a highly hand-tuned implementation,
# regardless of how the operations are defined.

