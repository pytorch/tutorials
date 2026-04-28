Note

Go to the end
to download the full example code.

# MaskedTensor Advanced Semantics

Before working on this tutorial, please make sure to review our
MaskedTensor Overview tutorial <https://pytorch.org/tutorials/prototype/maskedtensor_overview.html>.

The purpose of this tutorial is to help users understand how some of the advanced semantics work
and how they came to be. We will focus on two particular ones:

*. Differences between MaskedTensor and [NumPy's MaskedArray](https://numpy.org/doc/stable/reference/maskedarray.html)
*. Reduction semantics

## Preparation

```
# Disable prototype warnings and such
```

## MaskedTensor vs NumPy's MaskedArray

NumPy's `MaskedArray` has a few fundamental semantics differences from MaskedTensor.

*. Their factory function and basic definition inverts the mask (similar to `torch.nn.MHA`); that is, MaskedTensor

uses `True` to denote "specified" and `False` to denote "unspecified", or "valid"/"invalid",
whereas NumPy does the opposite. We believe that our mask definition is not only more intuitive,
but it also aligns more with the existing semantics in PyTorch as a whole.

*. Intersection semantics. In NumPy, if one of two elements are masked out, the resulting element will be

masked out as well - in practice, they
[apply the logical_or operator](https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1016-L1024).

Meanwhile, MaskedTensor does not support addition or binary operators with masks that don't match -
to understand why, please find the section on reductions.

However, if this behavior is desired, MaskedTensor does support these semantics by giving access to the data and masks
and conveniently converting a MaskedTensor to a Tensor with masked values filled in using `to_tensor()`.
For example:

Note that the mask is mt0.get_mask() & mt1.get_mask() since `MaskedTensor`'s mask is the inverse of NumPy's.

## Reduction Semantics

Recall in [MaskedTensor's Overview tutorial](https://pytorch.org/tutorials/prototype/maskedtensor_overview.html)
we discussed "Implementing missing torch.nan* ops". Those are examples of reductions - operators that remove one
(or more) dimensions from a Tensor and then aggregate the result. In this section, we will use reduction semantics
to motivate our strict requirements around matching masks from above.

Fundamentally, :class:`MaskedTensor`s perform the same reduction operation while ignoring the masked out
(unspecified) values. By way of example:

Now, the different reductions (all on dim=1):

Of note, the value under a masked out element is not guaranteed to have any specific value, especially if the
row or column is entirely masked out (the same is true for normalizations).
For more details on masked semantics, you can find this [RFC](https://github.com/pytorch/rfcs/pull/27).

Now, we can revisit the question: why do we enforce the invariant that masks must match for binary operators?
In other words, why don't we use the same semantics as `np.ma.masked_array`? Consider the following example:

Now, let's try addition:

Sum and addition should clearly be associative, but with NumPy's semantics, they are not,
which can certainly be confusing for the user.

`MaskedTensor`, on the other hand, will simply not allow this operation since mask0 != mask1.
That being said, if the user wishes, there are ways around this
(for example, filling in the MaskedTensor's undefined elements with 0 values using `to_tensor()`
like shown below), but the user must now be more explicit with their intentions.

## Conclusion

In this tutorial, we have learned about the different design decisions behind MaskedTensor and
NumPy's MaskedArray, as well as reduction semantics.
In general, MaskedTensor is designed to avoid ambiguity and confusing semantics (for example, we try to preserve
the associative property amongst binary operations), which in turn can necessitate the user
to be more intentional with their code at times, but we believe this to be the better move.
If you have any thoughts on this, please [let us know](https://github.com/pytorch/pytorch/issues)!

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: maskedtensor_advanced_semantics.ipynb`](../_downloads/94b0affe6be3bca0c01d9418b03fbc4e/maskedtensor_advanced_semantics.ipynb)

[`Download Python source code: maskedtensor_advanced_semantics.py`](../_downloads/adab2960fe1f57c773921db23098048e/maskedtensor_advanced_semantics.py)

[`Download zipped: maskedtensor_advanced_semantics.zip`](../_downloads/420817244eaa78865a564d31e9a5fa00/maskedtensor_advanced_semantics.zip)