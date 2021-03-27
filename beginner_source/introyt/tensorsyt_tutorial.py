"""
First, we’ll import pytorch.

"""

import torch


######################################################################
# Let’s see a few basic tensor manipulations. First, just a few of the
# ways to create tensors:
# 

z = torch.zeros(5, 3)
print(z)
print(z.dtype)


######################################################################
# Above, we create a 5x3 matrix filled with zeros, and query its datatype
# to find out that the zeros are 32-bit floating point numbers, which is
# the default PyTorch.
# 
# What if you wanted integers instead? You can always override the
# default:
# 

i = torch.ones((5, 3), dtype=torch.int16)
print(i)


######################################################################
# You can see that when we do change the default, the tensor helpfully
# reports this when printed.
# 
# It’s common to initialize learning weights randomly, often with a
# specific seed for the PRNG for reproducibility of results:
# 

torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)
print('\nA different random tensor:')
print(r2) # new values

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print('\nShould match r1:')
print(r3) # repeats values of r1 because of re-seed


######################################################################
# PyTorch tensors perform arithmetic operations intuitively. Tensors of
# similar shapes may be added, multiplied, etc. Operations with scalars
# are distributed over the tensor:
# 

ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # every element is multiplied by 2
print(twos)

threes = ones + twos       # additon allowed because shapes are similar
print(threes)              # tensors are added element-wise
print(threes.shape)        # this has the same dimensions as input tensors

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# uncomment this line to get a runtime error
# r3 = r1 + r2


######################################################################
# Here’s a small sample of the mathematical operations available:
# 

r = torch.rand(2, 2) - 0.5 * 2 # values between -1 and 1
print('A random matrix, r:')
print(r)

# Common mathematical operations are supported:
print('\nAbsolute value of r:')
print(torch.abs(r))

# ...as are trigonometric functions:
print('\nInverse sine of r:')
print(torch.asin(r))

# ...and linear algebra operations like determinant and singular value decomposition
print('\nDeterminant of r:')
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r))

# ...and statistical and aggregate operations:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))


######################################################################
# There’s a good deal more to know about the power of PyTorch tensors,
# including how to set them up for parallel computations on GPU - we’ll be
# going into more depth in another video.
# 