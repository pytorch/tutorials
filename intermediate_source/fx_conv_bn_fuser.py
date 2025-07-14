# -*- coding: utf-8 -*-
"""
Building a Convolution/Batch Norm fuser with torch.compile
******************************************************************
**Author**: `Horace He <https://github.com/chillee>`__, `Will Feng <https://github.com/yf225>`__

In this tutorial, we are going to use torch.compile and its pattern matching
capabilities to do the following:

1) Find patterns of conv/batch norm in the data dependencies.
2) For the patterns found in 1), fold the batch norm statistics into the convolution weights.

Note that this specific optimization only works for models in inference mode (i.e. `mode.eval()`).
But the pattern matching system in torch.compile works for both training and inference.

We will demonstrate how to register custom fusion patterns with torch.compile's
pattern matcher to optimize model performance.

"""


######################################################################
# First, let's get some imports out of the way (we will be using all
# of these later in the code).

from typing import Type, Dict, Any, Tuple, Iterable
import copy
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# For this tutorial, we are going to create a model consisting of convolutions
# and batch norms. Note that this model has some tricky components - some of
# the conv/batch norm patterns are hidden within Sequentials and one of the
# ``BatchNorms`` is wrapped in another Module.

class WrappedBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = nn.BatchNorm2d(1)
    def forward(self, x):
        return self.mod(x)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.nested = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, 1),
        )
        self.wrapped = WrappedBatchNorm()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.nested(x)
        x = self.wrapped(x)
        return x

model = M().to(device)
model.eval()

######################################################################
# Fusing Convolution with Batch Norm
# -----------------------------------------
# One of the primary challenges with trying to automatically fuse convolution
# and batch norm in PyTorch is that PyTorch does not provide an easy way of
# accessing the computational graph. torch.compile resolves this problem by
# capturing the computational graph during compilation, allowing us to apply
# pattern-based optimizations across the entire model, including operations
# nested within Sequential modules or wrapped in custom modules.
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import register_replacement

######################################################################
# torch.compile will capture a graph representation of our model. During
# compilation, modules hidden within Sequential containers and wrapped
# modules are all inlined into the graph, making them available for
# pattern matching and optimization.


####################################
# Fusing Convolution with Batch Norm
# ----------------------------------
# Unlike some other fusions, fusion of convolution with batch norm does not
# require any new operators. Instead, as batch norm during inference
# consists of a pointwise add and multiply, these operations can be "baked"
# into the preceding convolution's weights. This allows us to remove the batch
# norm entirely from our model! Read
# https://nenadmarkus.com/p/fusing-batchnorm-and-conv/ for further details. The
# code here is copied from
# https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/nn/utils/fusion.py
# clarity purposes.
def fuse_conv_bn_eval(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


####################################
# Pattern Matching with torch.compile
# ------------------------------------
# Now that we have our fusion logic, we need to register a pattern that
# torch.compile's pattern matcher will recognize and replace during
# compilation.

# Define the pattern we want to match: conv2d followed by batch_norm
def conv_bn_pattern(x, conv_weight, conv_bias, bn_mean, bn_var, bn_weight, bn_bias):
    conv_out = torch.nn.functional.conv2d(x, conv_weight, conv_bias)
    bn_out = torch.nn.functional.batch_norm(
        conv_out, bn_mean, bn_var, bn_weight, bn_bias,
        training=False, eps=1e-5
    )
    return bn_out

def conv_bn_replacement(x, conv_weight, conv_bias, bn_mean, bn_var, bn_weight, bn_bias):
    fused_weight, fused_bias = fuse_conv_bn_weights(
        conv_weight, conv_bias, bn_mean, bn_var, 1e-5, bn_weight, bn_bias
    )
    return torch.nn.functional.conv2d(x, fused_weight, fused_bias)

# Example inputs are needed to trace the pattern functions.
# The inputs should match the function signatures of conv_bn_pattern and conv_bn_replacement.
# These are used to trace the pattern functions to create the match template.
# IMPORTANT: The pattern matcher is shape-agnostic! The specific shapes you use here
# don't limit what shapes will be matched - any valid conv2d->batch_norm sequence
# will be matched regardless of channels, kernel size, or spatial dimensions.
# - x: input tensor (batch_size, channels, height, width)
# - conv_weight: (out_channels, in_channels, kernel_h, kernel_w)
# - conv_bias: (out_channels,)
# - bn_mean, bn_var, bn_weight, bn_bias: all have shape (num_features,) matching out_channels
example_inputs = [
    torch.randn(1, 1, 4, 4).to(device),  # x: input tensor
    torch.randn(1, 1, 1, 1).to(device),  # conv_weight: 1 output channel, 1 input channel, 1x1 kernel
    torch.randn(1).to(device),           # conv_bias: 1 output channel
    torch.randn(1).to(device),           # bn_mean: batch norm running mean
    torch.randn(1).to(device),           # bn_var: batch norm running variance
    torch.randn(1).to(device),           # bn_weight: batch norm weight (gamma)
    torch.randn(1).to(device),           # bn_bias: batch norm bias (beta)
]

from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._inductor import config

# Create a pattern matcher pass and register our pattern
patterns = PatternMatcherPass()

register_replacement(
    conv_bn_pattern,
    conv_bn_replacement,
    example_inputs,
    pm.fwd_only,
    patterns,
)

# Create a custom pass function that applies our patterns
def conv_bn_fusion_pass(graph):
    return patterns.apply(graph)

# Set our custom pass in the config
config.post_grad_custom_post_pass = conv_bn_fusion_pass


######################################################################
# .. note::
#       We make some simplifications here for demonstration purposes, such as only
#       matching 2D convolutions. The pattern matcher in torch.compile
#       can handle more complex patterns.

######################################################################
# Testing out our Fusion Pass
# -----------------------------------------
# We can now run this fusion pass on our initial toy model and verify that our
# results are identical. In addition, we can print out the code for our fused
# model and verify that there are no more batch norms.

from torch._dynamo.utils import counters

# Clear the counters before compilation
counters.clear()

# Ensure pattern matcher is enabled
config.pattern_matcher = True

fused_model = torch.compile(model, backend="inductor")
inp = torch.randn(5, 1, 1, 1).to(device)

# Run the model to trigger compilation and pattern matching
with torch.no_grad():
    output = fused_model(inp)
    expected = model(inp)
    torch.testing.assert_close(output, expected)

# Check how many patterns were matched
assert counters['inductor']['pattern_matcher_count'] == 3, "Expected 3 conv-bn patterns to be matched"

# Create a model with different shapes than our example_inputs
test_model_diff_shape = nn.Sequential(
    nn.Conv2d(3, 16, 5),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 32, 7),
    nn.BatchNorm2d(32),
).to(device).eval()

counters.clear()
compiled_diff_shape = torch.compile(test_model_diff_shape, backend="inductor")
test_input_diff_shape = torch.randn(1, 3, 28, 28).to(device)
with torch.no_grad():
    compiled_diff_shape(test_input_diff_shape)

# Check how many patterns were matched
assert counters['inductor']['pattern_matcher_count'] == 2, "Expected 2 conv-bn patterns to be matched"


######################################################################
# Benchmarking our Fusion on ResNet18
# -----------------------------------
# We can test our fusion pass on a larger model like ResNet18 and see how much
# this pass improves inference performance.
import torchvision.models as models
import time

rn18 = models.resnet18().to(device)
rn18.eval()

inp = torch.randn(10, 3, 224, 224).to(device)
output = rn18(inp)

def benchmark(model, iters=20):
    with torch.no_grad():
        for _ in range(10):
            model(inp)
        begin = time.time()
        for _ in range(iters):
            model(inp)
        return str(time.time()-begin)

# Benchmark original model
print("Original model time: ", benchmark(rn18))

# Compile with our custom pattern
compiled_with_pattern_matching = torch.compile(rn18, backend="inductor")

# Benchmark compiled model
print("\ntorch.compile (with conv-bn pattern matching and other fusions): ", benchmark(compiled_with_pattern_matching))


############
# Conclusion
# ----------
# As we can see, torch.compile provides a powerful way to implement
# graph transformations and optimizations through pattern matching.
# By registering custom patterns, we can extend torch.compile's
# optimization capabilities to handle domain-specific transformations.
#
# The conv-bn fusion demonstrated here is just one example of what's
# possible with torch.compile's pattern matching system.