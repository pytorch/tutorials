# -*- coding: utf-8 -*-
"""
Creating Extensions Using NumPy and SciPy
=========================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_

**Updated by**: `Adam Dziedzic <https://github.com/adam-dziedzic>`_

In this tutorial, we shall go through two tasks:

1. Create a neural network layer with no parameters.

    -  This calls into **numpy** as part of its implementation

2. Create a neural network layer that has learnable weights

    -  This calls into **SciPy** as part of its implementation
"""

import torch
from torch.autograd import Function

###############################################################
# Parameter-less example
# ----------------------
#
# This layer doesn’t particularly do anything useful or mathematically
# correct.
#
# It is aptly named ``BadFFTFunction``
#
# **Layer Implementation**

from numpy.fft import rfft2, irfft2


class BadFFTFunction(Function):
    @staticmethod
    def forward(ctx, input):
        numpy_input = input.detach().numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    @staticmethod
    def backward(ctx, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)

# since this layer does not have any parameters, we can
# simply declare this as a function, rather than as an ``nn.Module`` class


def incorrect_fft(input):
    return BadFFTFunction.apply(input)

###############################################################
# **Example usage of the created layer:**

input = torch.randn(8, 8, requires_grad=True)
result = incorrect_fft(input)
print(result)
result.backward(torch.randn(result.size()))
print(input)

###############################################################
# Parametrized example
# --------------------
#
# In deep learning literature, this layer is confusingly referred
# to as convolution while the actual operation is cross-correlation
# (the only difference is that filter is flipped for convolution,
# which is not the case for cross-correlation).
#
# Implementation of a layer with learnable weights, where cross-correlation
# has a filter (kernel) that represents weights.
#
# The backward pass computes the gradient ``wrt`` the input and the gradient ``wrt`` the filter.

from numpy import flip
import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class ScipyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)


###############################################################
# **Example usage:**

module = ScipyConv2d(3, 3)
print("Filter and bias: ", list(module.parameters()))
input = torch.randn(10, 10, requires_grad=True)
output = module(input)
print("Output from the convolution: ", output)
output.backward(torch.randn(8, 8))
print("Gradient for the input map: ", input.grad)

###############################################################
# **Check the gradients:**

from torch.autograd.gradcheck import gradcheck

moduleConv = ScipyConv2d(3, 3)

input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
print("Are the gradients correct: ", test)
