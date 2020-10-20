"""
Tensors
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
# Tensors can be created directly by passing a Python list or sequence using the ``torch.tensor()`` constructor. The data type is automatically inferred from the data.

data = [[1, 2],[3, 4]]
np_array = np.array(data)

tnsr_from_data = torch.tensor(data)
tnsr_from_np = torch.from_numpy(np_array)


###############################################################
# **From another tensor:**
# The new tensor retains the properties of the arg tensor, unless explicitly overridden.

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
# **With random or constant values:**
#

shape = (2,3,)
rand_tnsr = torch.rand(shape)
ones_tnsr = torch.ones(shape)
zeros_tnsr = torch.zeros(shape)
print(f"Random Tensor:\n{rand_tnsr} \n\n \
        Ones Tensor:\n{ones_tnsr} \n\n \
        Zeros Tensor:\n{zeros_tnsr}")




######################################################################
# --------------
#


######################################################################
# Tensor Attributes
# ~~~~~~~~~~~~~~~~~
#
# Tensor attributes describe their shape data type and where they live.

tnsr = torch.rand(3,4)

print(f"Shape of tensor: {tnsr.shape}")
print(f"Datatype of tensor: {tnsr.dtype}")
print(f"Device tensor lives on: {tnsr.device}")


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
if torch.cuda.is_available():
  tnsr = tnsr.to('cuda')


######################################################################
# Try out some of the operations from the list. If you're familiar with the NumPy API, you'll find the Tensor API a breeze to use.
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

######################################################################
# --------------
#


######################################################################
# Bridge with NumPy
# ~~~~~~~~~~~~~~~~~
# Tensors on the CPU and NumPy arrays can share their underlying memory
# locations, and changing one will change	the other.


######################################################################
# Tensor to NumPy array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
a = torch.ones(5)
print(f"a: {a}")
b = a.numpy()
print(f"b: {b}")

######################################################################
# A change in ``a`` reflects in ``b``

a.add_(1)
print(f"a: {a}")
print(f"b: {b}")


######################################################################
# NumPy array to Tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(f"a: {a}")
print(f"b: {b}")
