# -*- coding: utf-8 -*-
"""
What is PyTorch?
================

It’s a Python based scientific computing package targeted at two sets of
audiences:

-  A replacement for NumPy to use the power of GPUs
-  a deep learning research platform that provides maximum flexibility
   and speed

Getting Started
---------------

Tensors
^^^^^^^

Tensors are similar to NumPy’s ndarrays, with the addition being that
Tensors can also be used on a GPU to accelerate computing.
"""

from __future__ import print_function
import torch

###############################################################
# Construct a 5x3 matrix, uninitialized:

x = torch.Tensor(5, 3)
print(x)

###############################################################
# Construct a randomly initialized matrix:

x = torch.rand(5, 3)
print(x)

###############################################################
# Get its size:

print(x.size())

###############################################################
# .. note::
#     ``torch.Size`` is in fact a tuple, so it supports all tuple operations.
#
# Operations
# ^^^^^^^^^^
# There are multiple syntaxes for operations. In the following
# example, we will take a look at the addition operation.
#
# Addition: syntax 1
y = torch.rand(5, 3)
print(x + y)

###############################################################
# Addition: syntax 2

print(torch.add(x, y))

###############################################################
# Addition: providing an output tensor as argument
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

###############################################################
# Addition: in-place

# adds x to y
y.add_(x)
print(y)

###############################################################
# .. note::
#     Any operation that mutates a tensor in-place is post-fixed with an ``_``.
#     For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.
#
# You can use standard NumPy-like indexing with all bells and whistles!

print(x[:, 1])

###############################################################
# Resizing: If you want to resize/reshape tensor, you can use ``torch.view``:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

###############################################################
# **Read later:**
#
#
#   100+ Tensor operations, including transposing, indexing, slicing,
#   mathematical operations, linear algebra, random numbers, etc.,
#   are described
#   `here <http://pytorch.org/docs/torch>`_.
#
# NumPy Bridge
# ------------
#
# Converting a Torch Tensor to a NumPy array and vice versa is a breeze.
#
# The Torch Tensor and NumPy array will share their underlying memory
# locations, and changing one will change the other.
#
# Converting a Torch Tensor to a NumPy Array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

a = torch.ones(5)
print(a)

###############################################################
#


b = a.numpy()
print(b)

###############################################################
# See how the numpy array changed in value.

a.add_(1)
print(a)
print(b)

###############################################################
# Converting NumPy Array to Torch Tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# See how changing the np array changed the Torch Tensor automatically

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

###############################################################
# All the Tensors on the CPU except a CharTensor support converting to
# NumPy and back.
#
# CUDA Tensors
# ------------
#
# Tensors can be moved onto GPU using the ``.cuda`` method.

# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y
