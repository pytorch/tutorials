"""
Tensors - The building blocks of deep learning
--------------------------------------------

Tensors are a specialized data structure that are very similar to arrays
and matrices. In PyTorch, we use tensors to encode the inputs and
outputs of a model, as well as the model’s parameters.

Tensors are similar to NumPy’s ndarrays, except that tensors can run on
a GPU to accelerate computing. If you’re familiar with ndarrays, you’ll
be right at home with the Tensor API. If not, follow along in this quick
API walkthrough.

"""

import torch
import numpy as np


######################################################################
# Tensor Initialization
# ~~~~~~~~~~~~~~~~~~~~~
#
# Tensors can be initialized in various ways. Take a look at the following examples
#
# **Directly from data / NumPy arrays:**
#

data = [[1, 2],[3, 4]]
np_array = np.array(data)

tnsr_from_data = torch.tensor(data)
tnsr_from_np = torch.from_numpy(np_array)

######################################################################
# **With random or constant values:**
#

shape = (2,3,)
rand_tnsr = torch.rand(shape)
ones_tnsr = torch.ones(shape)
zeros_tnsr = torch.zeros(shape)
print(f"Random Tensor:\n{rand_tnsr}\n\nOnes Tensor:\n{ones_tnsr}\n\nZeros Tensor:\n{zeros_tnsr}")


###############################################################
# **From another tensor:**
# The new tensor retains the properties of the arg tensor, unless explicitly overridden

new_ones_tnsr = torch.ones_like(tnsr_from_data) # 2 x 2 matrix of ones

try:
  new_rand_tnsr = torch.rand_like(tnsr_from_data) # 2 x 2 matrix of random numbers
except RuntimeError as e:
  print(f"RuntimeError thrown: {e}")
  print()
  print(f"Random values in PyTorch are floating points. Datatype passed to torch.rand_like(): {tnsr_from_data.dtype}")


######################################################################
# This works after we override the dtype property
#

new_rand_tnsr = torch.rand_like(tnsr_from_data, dtype=torch.float)
print(new_rand_tnsr)



######################################################################
# --------------
#


######################################################################
# Tensor Attributes
# ~~~~~~~~~~~~~~~~~
#
# Tensors have attributes that describe their contents and functions. They
# also have attributes that autograd uses to keep track of them in the
# computational graph (more on this in the next section).
#
# **Docs Issues** - https://pytorch.org/docs/stable/tensor_attributes.html
# is not comprehensive (missing data, grad, grad_fn, shape). Contains
# ``memory_format`` which is not an attribute
#

tnsr = torch.rand(3,4)

print(f"Data stored in tensor:\n{tnsr.data}\n")
print(f"Shape of tensor: {tnsr.shape}")
print(f"Datatype of tensor: {tnsr.dtype}")
print(f"Device tensor lives on: {tnsr.device}")

print("-------")

# Autograd-related attributes; more on this later
print(f"Is tensor a leaf on computational graph: {tnsr.is_leaf}")
print(f"Does tensor require gradients: {tnsr.requires_grad}")
print(f"Accumulated gradients of tensor: {tnsr.grad}")
print(f"Function that computed this tensor's gradient: {tnsr.grad_fn}")



######################################################################
# --------------
#


######################################################################
# Tensor Operations
# ~~~~~~~~~~~~~~~~~
#
# Over 100 tensor operations, including transposing, indexing, slicing,
# mathematical operations, linear algebra, random sampling, and more are
# comprehensively described
# `here <https://pytorch.org/docs/stable/torch.html>`__.
#
# Each of them can be run on the GPU (at typically higher speeds than on a
# CPU). If you’re using Colab, allocate a GPU by going to Edit > Notebook
# Settings.
#

# We move our tensor to the GPU if available
if torch.cuda.is_available:
  tnsr.to('cuda')


######################################################################
# Try out some of the operations from the list. You may come across multiple
# syntaxes for the same operation. Don’t get confused by the aliases!
#

###############################################################
# **Standard numpy-like indexing and slicing:**

tnsr = torch.ones(4, 4)
tnsr[:,1] = 0
print(tnsr)

######################################################################
# **Both of these are joining ops, but they are subtly different.**

t1 = torch.cat([tnsr, tnsr], dim=1)
t2 = torch.stack([tnsr, tnsr], dim=1)
print(t1)
print()
print(t2)

######################################################################
# **Multiply op - multiple syntaxes**

# both ops compute element-wise product
print(tnsr * tnsr == tnsr.mul(tnsr))

# both ops compute matrix product
print(tnsr @ tnsr.T == tnsr.matmul(tnsr.T))

######################################################################
# .. note::
#     Operations that have a '_' suffix are in-place. For example:
#     ``x.copy_(y)``, ``x.t_()``, will change ``x``. In-place operations
#     don't work well with Autograd, and their use is discouraged.

print(tnsr)
tnsr.t_()
print(tnsr)
