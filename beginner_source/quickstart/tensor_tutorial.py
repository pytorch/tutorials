"""
Tensors and Operations
----------------------
**Tensor** is the basic computational unit in PyTorch. It is very
similar to **NumPy array**, and supports similar operations. However,
there are two very important features of Torch tensors that make them
especially useful for training large-scale neural networks:
-  Tensor operations can be performed on GPUs or other specialized hardware to accelerate computing
-  Tensor operations support automatic differentiation using
   `pytorch.autograd engine <autograd_tutorial.html>`__
Conversion between Torch tensors and NumPy arrays can be done easily:
"""

import torch
import numpy as np

np_array = np.arange(10)
tensor = torch.from_numpy(np_array)

print(f"Tensor={tensor}, Array={tensor.numpy()}")


######################################################################
# .. note::
#    When using CPU for computations, tensors converted from arrays
#    share the same memory for data. Thus, changing the underlying array
#    will also affect the tensor.
#


######################################################################
# Creating Tensors
# ~~~~~~~~~~~~~~~~
#
# The fastest way to create a tensor is to define an *uninitialized*
# tensor - the values of this tensor are not set, and depend on the
# whatever data was there in memory:
#

x = torch.empty(3, 6)


######################################################################
# In practice, we often want to create tensors initialized to some values,
# such as zeros, ones or random values. Note that you can also specify the
# type of elements using ``dtype`` parameter, and chosing one of ``torch``
# types:
#

x = torch.randn(3, 5)
y = torch.zeros(3, 5, dtype=torch.int)
z = torch.ones(3, 5, dtype=torch.double)

######################################################################
# You can also create random tensors with values sampled from different
# distributions, as described `in the
# documentation <https://pytorch.org/docs/stable/torch.html#random-sampling>`__.
#
# Similarly to NumPy, you can use ``eye`` to create a diagonal identity
# matrix:
#

I = torch.eye(10)


######################################################################
# You can also create new tensors with the same properties or size as
# existing tensors:
#

print(z.new_ones(2, 2))  # new_ method allows specifying new size
# _like method supports overriding dtype
print(torch.zeros_like(x, dtype=torch.long))


######################################################################
# Size of the tensor can be obtained using ``.size()`` method, which
# returns a tuple-like object:
#

print(z.size())  # Prints [3.0]


######################################################################
# Tensor Operations
# ~~~~~~~~~~~~~~~~~
#
# Tensors support all basic arithmetic operations, which can be specified
# in different ways:
#  - Using operators, such as ``+``, ``-``, etc. \*
#  - Using functions such as ``add``, ``mult``, etc. Functions can either return values, or store them in the specified ouput variable (using ``out=`` parameter)
#  - In-place operations, which modify one of the arguments. Those operations have ``_`` appended to their name, eg. ``add_``.
# Complete reference to all tensor operations can be found `in the
# documentation <https://pytorch.org/docs/stable/torch.html>`__.
#
# Let us see examples of those operations on two tensors, ``x`` and ``y``.
#

x = torch.randn(3, 5)
y = torch.randn(3, 5)


######################################################################
# Using operator notation
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# We can use overloaded arithmetic operators, such as ``+`` and ``*``:
#

z = x*y


######################################################################
# Note, that ``*`` means elementwise product, and not the matrix product.
# To compute matrix product, we need to use `@` operator or ``matmul`` function, as shown
# below.
#
# Using functions
# ^^^^^^^^^^^^^^^
#
# While only some operations are available as Python operators, `many more
# functions <https://pytorch.org/docs/stable/torch.html#math-operations>`__
# can be specified using the full name. In the example below, ``t``
# transposes the matrix, and ``matmul`` means matrix multiplication:
#

z = torch.matmul(x, y.t())


######################################################################
# Simple operations (addition, multiplication, etc.) also have
# corresponsing functions, and can be called either as methods, or as
# functions:
#

z = x.add(y)
z = torch.add(x, y)


######################################################################
# Sometimes it may be more convenient to store the result into specified
# variable, instead of returning it from a function. In this case you can
# use ``out=`` parameter:
#

torch.add(x, y, out=z)


######################################################################
# In-place operations
# ^^^^^^^^^^^^^^^^^^^
#
# When training neural networks, you often need to **modify** the weights,
# i.e. perform some operation and then store the result into the original
# variable. Those operations are called **in-place operations**, and they
# are marked by the ``_`` symbol at the end of their name:
#

x.add_(y)  # x will be modified

######################################################################
# .. note::
#      In-place operations save some memory, but can be problematic when
#      computing derivatives because of an immediate loss
#      of history. Hence, their use is discouraged.


######################################################################
# Resizing and Indexing
# ~~~~~~~~~~~~~~~~~~~~~
#
# Often you need to change the shape of the tensor without modifying
# its values, eg. to add an extra dimension. To do that, you can use
# ``view`` method, which provides a **view** to the same in-memory values
# using different dimensions:
#

print(x.size())  # original size of x is 3x5
print(x.view(5, 3, 1).size())  # will give size 5x3x1
print(x.view(5, -1))  # will result in size 5x3


######################################################################
# The number of elements in a view should be the same as in the
# original tensor. You can use ``-1`` in one of the dimensions to
# figure out this dimension automatically.
#


######################################################################
# .. note:: ``view`` is similar to ``reshape`` operation in NumPy. There
#           is also a ``reshape`` method available in PyTorch, and it is more
#           powerful than ``view``, because it can also reshape non-contiguous
#           arrays by copying them to the new shape. However, in vast majority of
#           cases you can use ``view`` and make sure that no data copying occurs,
#           and the operation is always efficient.
#


######################################################################
# Tensors support all slicing operations that exist in NymPy:
#

print(x.size())  # original size of x is 3x5
print(x[0].size(), x[:, 0].size(), x[..., 1].size())  # will give 5, 3, 3


######################################################################
# If you have a one-element tensor, for example, after aggregating all
# values of the tensor into one value, you can convert it to a Python
# numerical value using ``item()``:
#

val = x.sum().item()  # will compute the sum of all elements


######################################################################
# Hardware-Accelerated Computations
# ~~~~~~~~~~~~~~~~
#
# One of the major benefits of using PyTorch is the ability to perform
# tensor operations on GPUs and some other specialized hardware. To do that,
# we need to explicitly **move** tensors to another computing platform using ``.to`` method.
#
# In most of the cases, we check for the availability of GPU in the beginning
# of the script, and define the ``device`` object accordingly. Then we move all
# tensors to that device before performing the computations:
#

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Doing computations on {}".format(device))

x = torch.randn(3, 5, device=device)  # create tensor on specified device
y = torch.ones_like(x)  # create tensor on CPU
y = y.to(device)  # move tensor to another device
z = x+y  # this is performed on GPU if it is available
print(z)
print(z.to("cpu", torch.double))


######################################################################
# In the last operation, when we move the tensor back to the CPU, we can
# also change the ``dtype``. This does not result in additional
# computational time, because we need to copy and transform the data when
# moving it from GPU anyway.
#
# Next learn how to load built in and custom `datasets with dataloaders <data_quickstart_tutorial.html>`_
#
# .. include:: /beginner_source/quickstart/qs_toc.txt
#
