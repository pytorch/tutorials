Note

Go to the end
to download the full example code.

# MaskedTensor Overview

This tutorial is designed to serve as a starting point for using MaskedTensors
and discuss its masking semantics.

MaskedTensor serves as an extension to [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) that provides the user with the ability to:

- use any masked semantics (for example, variable length tensors, nan* operators, etc.)
- differentiation between 0 and NaN gradients
- various sparse applications (see tutorial below)

For a more detailed introduction on what MaskedTensors are, please find the
[torch.masked documentation](https://pytorch.org/docs/master/masked.html).

## Using MaskedTensor

In this section we discuss how to use MaskedTensor including how to construct, access, the data
and mask, as well as indexing and slicing.

### Preparation

We'll begin by doing the necessary setup for the tutorial:

```
# Disable prototype warnings and such
```

### Construction

There are a few different ways to construct a MaskedTensor:

- The first way is to directly invoke the MaskedTensor class
- The second (and our recommended way) is to use `masked.masked_tensor()` and `masked.as_masked_tensor()`
factory functions, which are analogous to [`torch.tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor) and [`torch.as_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor)

Throughout this tutorial, we will be assuming the import line: from torch.masked import masked_tensor.

### Accessing the data and mask

The underlying fields in a MaskedTensor can be accessed through:

- the `MaskedTensor.get_data()` function
- the `MaskedTensor.get_mask()` function. Recall that `True` indicates "specified" or "valid"
while `False` indicates "unspecified" or "invalid".

In general, the underlying data that is returned may not be valid in the unspecified entries, so we recommend that
when users require a Tensor without any masked entries, that they use `MaskedTensor.to_tensor()` (as shown above) to
return a Tensor with filled values.

### Indexing and slicing

`MaskedTensor` is a Tensor subclass, which means that it inherits the same semantics for indexing and slicing
as [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor). Below are some examples of common indexing and slicing patterns:

```
# float is used for cleaner visualization when being printed
```

## Why is MaskedTensor useful?

Because of `MaskedTensor`'s treatment of specified and unspecified values as a first-class citizen
instead of an afterthought (with filled values, nans, etc.), it is able to solve for several of the shortcomings
that regular Tensors are unable to; indeed, `MaskedTensor` was born in a large part due to these recurring issues.

Below, we will discuss some of the most common issues that are still unresolved in PyTorch today
and illustrate how `MaskedTensor` can solve these problems.

### Distinguishing between 0 and NaN gradient

One issue that [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) runs into is the inability to distinguish between gradients that are
undefined (NaN) vs. gradients that are actually 0. Because PyTorch does not have a way of marking a value
as specified/valid vs. unspecified/invalid, it is forced to rely on NaN or 0 (depending on the use case), leading
to unreliable semantics since many operations aren't meant to handle NaN values properly. What is even more confusing
is that sometimes depending on the order of operations, the gradient could vary (for example, depending on how early
in the chain of operations a NaN value manifests).

`MaskedTensor` is the perfect solution for this!

#### torch.where

In [Issue 10729](https://github.com/pytorch/pytorch/issues/10729), we notice a case where the order of operations
can matter when using [`torch.where()`](https://docs.pytorch.org/docs/stable/generated/torch.where.html#torch.where) because we have trouble differentiating between if the 0 is a real 0
or one from undefined gradients. Therefore, we remain consistent and mask out the results:

Current result:

`MaskedTensor` result:

The gradient here is only provided to the selected subset. Effectively, this changes the gradient of where
to mask out elements instead of setting them to zero.

#### Another torch.where

[Issue 52248](https://github.com/pytorch/pytorch/issues/52248) is another example.

Current result:

`MaskedTensor` result:

This issue is similar (and even links to the next issue below) in that it expresses frustration with
unexpected behavior because of the inability to differentiate "no gradient" vs "zero gradient",
which in turn makes working with other ops difficult to reason about.

#### When using mask, x/0 yields NaN grad

In [Issue 4132](https://github.com/pytorch/pytorch/issues/4132), the user proposes that
x.grad should be [0, 1] instead of the [nan, 1],
whereas `MaskedTensor` makes this very clear by masking out the gradient altogether.

Current result:

`MaskedTensor` result:

### [`torch.nansum()`](https://docs.pytorch.org/docs/stable/generated/torch.nansum.html#torch.nansum) and [`torch.nanmean()`](https://docs.pytorch.org/docs/stable/generated/torch.nanmean.html#torch.nanmean)

In [Issue 67180](https://github.com/pytorch/pytorch/issues/67180),
the gradient isn't calculate properly (a longstanding issue), whereas `MaskedTensor` handles it correctly.

Current result:

`MaskedTensor` result:

### Safe Softmax

Safe softmax is another great example of [an issue](https://github.com/pytorch/pytorch/issues/55056)
that arises frequently. In a nutshell, if there is an entire batch that is "masked out"
or consists entirely of padding (which, in the softmax case, translates to being set -inf),
then this will result in NaNs, which can lead to training divergence.

Luckily, `MaskedTensor` has solved this issue. Consider this setup:

For example, we want to calculate the softmax along dim=0. Note that the second column is "unsafe" (i.e. entirely
masked out), so when the softmax is calculated, the result will yield 0/0 = nan since exp(-inf) = 0.
However, what we would really like is for the gradients to be masked out since they are unspecified and would be
invalid for training.

PyTorch result:

`MaskedTensor` result:

### Implementing missing torch.nan* operators

In [Issue 61474](https://github.com/pytorch/pytorch/issues/61474),
there is a request to add additional operators to cover the various torch.nan* applications,
such as `torch.nanmax`, `torch.nanmin`, etc.

In general, these problems lend themselves more naturally to masked semantics, so instead of introducing additional
operators, we propose using `MaskedTensor` instead.
Since [nanmean has already landed](https://github.com/pytorch/pytorch/issues/21987),
we can use it as a comparison point:

```
# z is just y with the zeros replaced with nan's
```

```
# MaskedTensor successfully ignores the 0's
```

In the above example, we've constructed a y and would like to calculate the mean of the series while ignoring
the zeros. torch.nanmean can be used to do this, but we don't have implementations for the rest of the
torch.nan* operations. `MaskedTensor` solves this issue by being able to use the base operation,
and we already have support for the other operations listed in the issue. For example:

Indeed, the index of the minimum argument when ignoring the 0's is the 1 in index 1.

`MaskedTensor` can also support reductions when the data is fully masked out, which is equivalent
to the case above when the data Tensor is completely `nan`. `nanmean` would return `nan`
(an ambiguous return value), while MaskedTensor would more accurately indicate a masked out result.

This is a similar problem to safe softmax where 0/0 = nan when what we really want is an undefined value.

## Conclusion

In this tutorial, we've introduced what MaskedTensors are, demonstrated how to use them, and motivated their
value through a series of examples and issues that they've helped resolve.

## Further Reading

To continue learning more, you can find our
[MaskedTensor Sparsity tutorial](https://pytorch.org/tutorials/prototype/maskedtensor_sparsity.html)
to see how MaskedTensor enables sparsity and the different storage formats we currently support.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: maskedtensor_overview.ipynb`](../_downloads/098f809744385f9a587451e656698b10/maskedtensor_overview.ipynb)

[`Download Python source code: maskedtensor_overview.py`](../_downloads/4a51bb8a0b86f2086808e1d10395e483/maskedtensor_overview.py)

[`Download zipped: maskedtensor_overview.zip`](../_downloads/a28aa7ceff8a082bfda2e2e235535822/maskedtensor_overview.zip)