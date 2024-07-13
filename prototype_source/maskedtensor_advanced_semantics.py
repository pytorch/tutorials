# -*- coding: utf-8 -*-

"""
(Prototype) MaskedTensor Advanced Semantics
===========================================
"""

######################################################################
# 
# Before working on this tutorial, please make sure to review our
# `MaskedTensor Overview tutorial <https://pytorch.org/tutorials/prototype/maskedtensor_overview.html>`.
#
# The purpose of this tutorial is to help users understand how some of the advanced semantics work
# and how they came to be. We will focus on two particular ones:
#
# *. Differences between MaskedTensor and `NumPy's MaskedArray <https://numpy.org/doc/stable/reference/maskedarray.html>`__  
# *. Reduction semantics
#
# Preparation
# -----------
#

import torch
from torch.masked import masked_tensor
import numpy as np
import warnings

# Disable prototype warnings and such
warnings.filterwarnings(action='ignore', category=UserWarning)

######################################################################
# MaskedTensor vs NumPy's MaskedArray
# -----------------------------------
#
# NumPy's ``MaskedArray`` has a few fundamental semantics differences from MaskedTensor.
#
# *. Their factory function and basic definition inverts the mask (similar to ``torch.nn.MHA``); that is, MaskedTensor
#    uses ``True`` to denote "specified" and ``False`` to denote "unspecified", or "valid"/"invalid",
#    whereas NumPy does the opposite. We believe that our mask definition is not only more intuitive,
#    but it also aligns more with the existing semantics in PyTorch as a whole.
# *. Intersection semantics. In NumPy, if one of two elements are masked out, the resulting element will be
#    masked out as well -- in practice, they
#    `apply the logical_or operator <https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1016-L1024>`__.
#

data = torch.arange(5.)
mask = torch.tensor([True, True, False, True, False])
npm0 = np.ma.masked_array(data.numpy(), (~mask).numpy())
npm1 = np.ma.masked_array(data.numpy(), (mask).numpy())

print("npm0:\n", npm0)
print("npm1:\n", npm1)
print("npm0 + npm1:\n", npm0 + npm1)

######################################################################
# Meanwhile, MaskedTensor does not support addition or binary operators with masks that don't match --
# to understand why, please find the :ref:`section on reductions <reduction-semantics>`.
#

mt0 = masked_tensor(data, mask)
mt1 = masked_tensor(data, ~mask)
print("mt0:\n", mt0)
print("mt1:\n", mt1)

try:
    mt0 + mt1
except ValueError as e:
    print ("mt0 + mt1 failed. Error: ", e)

######################################################################
# However, if this behavior is desired, MaskedTensor does support these semantics by giving access to the data and masks
# and conveniently converting a MaskedTensor to a Tensor with masked values filled in using :func:`to_tensor`.
# For example:
#

t0 = mt0.to_tensor(0)
t1 = mt1.to_tensor(0)
mt2 = masked_tensor(t0 + t1, mt0.get_mask() & mt1.get_mask())

print("t0:\n", t0)
print("t1:\n", t1)
print("mt2 (t0 + t1):\n", mt2)

######################################################################
# Note that the mask is `mt0.get_mask() & mt1.get_mask()` since :class:`MaskedTensor`'s mask is the inverse of NumPy's.
#
# .. _reduction-semantics:
#
# Reduction Semantics
# -------------------
#
# Recall in `MaskedTensor's Overview tutorial <https://pytorch.org/tutorials/prototype/maskedtensor_overview.html>`__
# we discussed "Implementing missing torch.nan* ops". Those are examples of reductions -- operators that remove one
# (or more) dimensions from a Tensor and then aggregate the result. In this section, we will use reduction semantics
# to motivate our strict requirements around matching masks from above.
#
# Fundamentally, :class:`MaskedTensor`s perform the same reduction operation while ignoring the masked out
# (unspecified) values. By way of example:
#

data = torch.arange(12, dtype=torch.float).reshape(3, 4)
mask = torch.randint(2, (3, 4), dtype=torch.bool)
mt = masked_tensor(data, mask)

print("data:\n", data)
print("mask:\n", mask)
print("mt:\n", mt)

######################################################################
# Now, the different reductions (all on dim=1):
#

print("torch.sum:\n", torch.sum(mt, 1))
print("torch.mean:\n", torch.mean(mt, 1))
print("torch.prod:\n", torch.prod(mt, 1))
print("torch.amin:\n", torch.amin(mt, 1))
print("torch.amax:\n", torch.amax(mt, 1))

######################################################################
# Of note, the value under a masked out element is not guaranteed to have any specific value, especially if the
# row or column is entirely masked out (the same is true for normalizations).
# For more details on masked semantics, you can find this `RFC <https://github.com/pytorch/rfcs/pull/27>`__.
#
# Now, we can revisit the question: why do we enforce the invariant that masks must match for binary operators?
# In other words, why don't we use the same semantics as ``np.ma.masked_array``? Consider the following example:
#

data0 = torch.arange(10.).reshape(2, 5)
data1 = torch.arange(10.).reshape(2, 5) + 10
mask0 = torch.tensor([[True, True, False, False, False], [False, False, False, True, True]])
mask1 = torch.tensor([[False, False, False, True, True], [True, True, False, False, False]])
npm0 = np.ma.masked_array(data0.numpy(), (mask0).numpy())
npm1 = np.ma.masked_array(data1.numpy(), (mask1).numpy())

print("npm0:", npm0)
print("npm1:", npm1)

######################################################################
# Now, let's try addition:
#

print("(npm0 + npm1).sum(0):\n", (npm0 + npm1).sum(0))
print("npm0.sum(0) + npm1.sum(0):\n", npm0.sum(0) + npm1.sum(0))

######################################################################
# Sum and addition should clearly be associative, but with NumPy's semantics, they are not,
# which can certainly be confusing for the user.
#
# :class:`MaskedTensor`, on the other hand, will simply not allow this operation since `mask0 != mask1`.
# That being said, if the user wishes, there are ways around this
# (for example, filling in the MaskedTensor's undefined elements with 0 values using :func:`to_tensor`
# like shown below), but the user must now be more explicit with their intentions.
#

mt0 = masked_tensor(data0, ~mask0)
mt1 = masked_tensor(data1, ~mask1)

(mt0.to_tensor(0) + mt1.to_tensor(0)).sum(0)

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we have learned about the different design decisions behind MaskedTensor and
# NumPy's MaskedArray, as well as reduction semantics.
# In general, MaskedTensor is designed to avoid ambiguity and confusing semantics (for example, we try to preserve
# the associative property amongst binary operations), which in turn can necessitate the user
# to be more intentional with their code at times, but we believe this to be the better move.
# If you have any thoughts on this, please `let us know <https://github.com/pytorch/pytorch/issues>`__!
# 
