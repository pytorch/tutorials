"""
Tensors and Operations
===================

Tensors and Operations
When training neural network models for real world tasks, we need to be able to effectively represent different types of input data: sets of numerical features, images, videos, sounds, etc. All those different input types can be represented as multi-dimensional arrays of numbers that are called tensors.

Tensor is the basic computational unit in PyTorch. It is very similar to NumPy array, and supports similar operations. However, there are two very important features of Torch tensors that make the especially useful for training large-scale neural networks:

 - Tensor operations can be performed on GPU using CUDA
 - Tensor operations support automatic differentiation using `AutoGrad <quickstart/autograd_tutorial.html>`_
 
Conversion between Torch tensors and NumPy arrays can be done easily:
"""

import torch
import numpy as np

np_array = np.arange(10)
tensor = torch.from_numpy(np_array)

print(f"Tensor={tensor}, Array={tensor.numpy()}")

#################################################################
# .. code:: python 
# Output: Tensor=tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32), Array=[0 1 2 3 4 5 6 7 8 9]
#
# .. note:: When using CPU for computations, tensors converted from arrays share the same memory for data. Thus, changing the underlying array will also affect the tensor.
#
#
# Creating Tensors
# -------------
# The fastest way to create a tensor is to define an uninitialized tensor - the values of this tensor are not set, and depend on the whatever data was there in memory:
#

x = torch.empty(3,6)
print(x)

############################################################################
# .. code:: python
# Output: tensor([[-1.3822e-06,  6.5301e-43, -1.3822e-06,  6.5301e-43, -1.4041e-06,
#          6.5301e-43],
#        [-1.3855e-06,  6.5301e-43, -2.9163e-07,  6.5301e-43, -2.9163e-07,
#          6.5301e-43],
#        [-1.4066e-06,  6.5301e-43, -1.3788e-06,  6.5301e-43, -2.9163e-07,
#          6.5301e-43]])
#
#
# In practice, we ofter want to create tensors initialized to some values, such as zeros, ones or random values. Note that you can also specify the type of elements using dtype parameter, and chosing one of torch types:


x = torch.randn(3,5)
print(x)
y = torch.zeros(3,5,dtype=torch.int)
print(y)
z = torch.ones(3,5,dtype=torch.double)
print(z)

######################################################################
# Output:
# tensor([[-1.0166, -0.6828,  1.8886, -1.2115,  0.0202],
#         [-1.1278,  0.7447,  0.4260, -2.1909,  0.5653],
#         [ 0.0562, -0.1393,  0.6145, -0.6181,  0.1879]])
# tensor([[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]], dtype=torch.int32)
# tensor([[1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.]], dtype=torch.float64)
#
#
# You can also create random tensors with values sampled from different distributions, as described `in documentation. <https://pytorch.org/docs/stable/torch.html#random-sampling>`_
# 
#Similarly to NumPy, you can use eye to create a diagonal identity matrix:

print(torch.eye(10))

################################################################
# Output: 
# tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
#
#
# You can also create new tensors with the same properties or size as existing tensors:
#

print(z.new_ones(2,2))
print(torch.zeros_like(x,dtype=torch.long))

############################################################################
# Tensor Operations
# -------------
# Tensors support all basic arithmetic operations, which can be specified in different ways:
# 
#  - Using operators, such as +, -, etc.
#  - Using functions such as add, mult, etc. Functions can either return values, or store them in the specified ouput variable (using out= parameter)
#  - In-place operations, which modify one of the arguments. Those operations have _ appended to their name, eg. add_.
# 
# Complete reference to all tensor operations can be found in documentation.
# 
# Let us see examples of those operations on two tensors, x and y.
# 
#
#
##################################################################
# More help with the FashionMNIST Pytorch Blitz
# ----------------------------------
# `Tensors <tensor_quickstart_tutorial.html>`_
# `DataSets and DataLoaders <data_quickstart_tutorial.html>`_
# `Transformations <transforms_tutorial.html>`_
# `Build Model <build_model_tutorial.html>`_
# `Optimization Loop <optimization_tutorial.html>`_
# `AutoGrad <autograd_quickstart_tutorial.html>`_
# `Back to FashionMNIST main code base <>`_
