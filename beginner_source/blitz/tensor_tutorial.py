# -*- coding: utf-8 -*-
"""
What is PyTorch?
================

It is a open source machine learning framework that accelerates the 
path from research prototyping to production deployment.

PyTorch is built as a Python-based scientific computing package targeted at two sets of
audiences:

-  Those who are looking for a replacement for NumPy to use the power of GPUs.
-  Researchers who want to build with a deep learning platform that provides maximum flexibility
   and speed.

Getting Started
---------------

In this section of the tutorial, we will introduce the concept of a tensor in PyTorch, and it's operations.

Tensors
^^^^^^^

A tensor is a generic n-dimensional array. tensors in PyTorch are similar to NumPy’s ndarrays, 
with the addition being that tensors can also be used on a GPU to accelerate computing.

To see the behavior of tensors, we will first need to import PyTorch into our program.
"""

from __future__ import print_function
import torch

"""
We import ``future`` here to help port our code from Python 2 to Python 3.
For more details, see the `Python-Future technical documentation <https://python-future.org/quickstart.html>`. 

Let's take a look at how we can create tensors.
"""

###############################################################
# First, construct a 5x3 empty matrix:

x = torch.empty(5, 3)
print(x)

"""
``torch.empty`` creates an uninitialized matrix of type tensor.
When an empty tensor is declared, it does not contain definite known values
before you populate it. The values in the empty tensor are those that were in 
the allocated memory at the time of initialization.
"""
 
###############################################################
# Now, construct a randomly initialized matrix:

x = torch.rand(5, 3)
print(x)

"""
``torch.rand`` creates an initialized matrix of type tensor with a random 
sampling of values. 
"""

###############################################################
# Construct a matrix filled zeros and of dtype long:

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

"""
``torch.zeros`` creates an initialized matrix of type tensor with every
index having a value of zero.
"""

###############################################################
# Let's construct a tensor with data that we define ourselves:

x = torch.tensor([5.5, 3])
print(x)

"""
Our tensor can represent all types of data. This data can be an audio waveform, the
pixels of an image, even entities of a language.

PyTorch has packages that support these specific data types. For additional learning, see:
-  `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`
-  `torchtext <https://pytorch.org/text/>`
-  `torchaudio <https://pytorch.org/audio/>`
"""

###############################################################
# You can create a tensor based on an existing tensor. These methods 
# reuse the properties of the input tensor, e.g. ``dtype``, unless
# new values are provided by the user.
#

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

"""
``tensor.new_*`` methods take in the size of the tensor and a ``dtype``, 
returning a tensor filled with ones.

In this example,``torch.randn_like`` creates a new tensor based upon the 
input tensor, and overrides the ``dtype`` to be a float. The output of 
this method is a tensor of the same size and different ``dtype``.
"""

###############################################################
# We can get the size of a tensor as a tuple:

print(x.size())

###############################################################
# .. note::
#     Since ``torch.Size`` is a tuple, it supports all tuple operations.
#
# Operations
# ^^^^^^^^^^
# There are multiple syntaxes for operations that can be performed on tensors.
# In the following example, we will take a look at the addition operation.
#
# First, let's try using the ``+`` operator.

y = torch.rand(5, 3)
print(x + y)

###############################################################
# Using the ``+`` operator should have the same output as using the 
# ``add()`` method. 

print(torch.add(x, y))

###############################################################
# You can also provide a tensor as an argument to the ``add()``
# method that will contain the data of the output operation.

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

###############################################################
# Finally, you can perform this operation in-place.

# adds x to y
y.add_(x)
print(y)

###############################################################
# .. note::
#     Any operation that mutates a tensor in-place is post-fixed with an ``_``.
#     For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.

###############################################################
# Similar to NumPy, tensors can be index using the standard
# Python ``x[i]`` syntax, where ``x`` is the array and ``i`` is the selection.
# 
# That said, you can NumPy-like indexing with all its bells and whistles!

print(x[:, 1])

###############################################################
# Resizing your tensors might be necessary for your data.
# If you want to resize or reshape tensor, you can use ``torch.view``:

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

###############################################################
# You can access the Python number-value of a one-element tensor using ``.item()``.
# If you have a multidimensional tensor, see the 
# `tolist() https://pytorch.org/docs/stable/tensors.html#torch.Tensor.tolist` method.

x = torch.randn(1)
print(x)
print(x.item())

###############################################################
# **Read later:**
#
#
#   This was just a sample of the 100+ Tensor operations you have 
#   access to in PyTorch. There are many others, including transposing,
#   indexing, slicing, mathematical operations, linear algebra,
#   random numbers, and more. Read and explore more about them in our 
#   `technical documentation <https://pytorch.org/docs/torch>`_.
#
# NumPy Bridge
# ------------
#
# As mentioned earlier, one of the benefits of using PyTorch is that it 
# is built to provide a seemless transition from NumPy.
# 
# For example, converting a Torch Tensor to a NumPy array (and vice versa)
# is a breeze.
#
# The Torch Tensor and NumPy array will share their underlying memory
# locations (if the Torch Tensor is on CPU). That means, changing one will change
# the other.
#
# Let's see this in action.
#
# Converting a Torch Tensor to a NumPy Array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First, construct a 1-dimensional tensor populated with ones.

a = torch.ones(5)
print(a)

###############################################################
# Now, let's construct a NumPy array based off of that tensor.

b = a.numpy()
print(b)

###############################################################
# Let's see how they share their memory locations. Add ``1`` to the torch tensor.

a.add_(1)
print(a)
print(b)

###############################################################
# Take note how the numpy array also changed in value.

###############################################################
# Converting NumPy Array to Torch Tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Try the same thing for NumPy to Torch Tensor.
# See how changing the NumPy array changed the Torch Tensor automatically as well.

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

###############################################################
# All the Tensors on the CPU (except a CharTensor) support converting to
# NumPy and back.
#
# CUDA Tensors
# ------------
#
# Tensors can be moved onto any device using the ``.to`` method.
# The following code block can be run by changing the runtime in
# your notebook to "GPU" or greater.

# This cell will run only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

###############################################################
# Now that you have had time to experiment with Tensors in PyTorch, let's take
# a look at Automatic Differentiation.
