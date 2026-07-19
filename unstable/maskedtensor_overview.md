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
import torch
from torch.masked import masked_tensor, as_masked_tensor
import warnings

# Disable prototype warnings and such
warnings.filterwarnings(action='ignore', category=UserWarning)
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
data = torch.arange(24).reshape(2, 3, 4)
mask = data % 2 == 0

print("data:\n", data)
print("mask:\n", mask)
```

```
data:
 tensor([[[ 0, 1, 2, 3],
 [ 4, 5, 6, 7],
 [ 8, 9, 10, 11]],

 [[12, 13, 14, 15],
 [16, 17, 18, 19],
 [20, 21, 22, 23]]])
mask:
 tensor([[[ True, False, True, False],
 [ True, False, True, False],
 [ True, False, True, False]],

 [[ True, False, True, False],
 [ True, False, True, False],
 [ True, False, True, False]]])
```

```
# float is used for cleaner visualization when being printed
mt = masked_tensor(data.float(), mask)

print("mt[0]:\n", mt[0])
print("mt[:, :, 2:4]:\n", mt[:, :, 2:4])
```

```
mt[0]:
 MaskedTensor(
 [
 [ 0.0000, --, 2.0000, --],
 [ 4.0000, --, 6.0000, --],
 [ 8.0000, --, 10.0000, --]
 ]
)
mt[:, :, 2:4]:
 MaskedTensor(
 [
 [
 [ 2.0000, --],
 [ 6.0000, --],
 [ 10.0000, --]
 ],
 [
 [ 14.0000, --],
 [ 18.0000, --],
 [ 22.0000, --]
 ]
 ]
)
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

```
x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True, dtype=torch.float)
y = torch.where(x < 0, torch.exp(x), torch.ones_like(x))
y.sum().backward()
x.grad
```

```
tensor([4.5400e-05, 6.7379e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
 0.0000e+00, 0.0000e+00, 0.0000e+00, nan, nan])
```

`MaskedTensor` result:

```
x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100])
mask = x < 0
mx = masked_tensor(x, mask, requires_grad=True)
my = masked_tensor(torch.ones_like(x), ~mask, requires_grad=True)
y = torch.where(mask, torch.exp(mx), my)
y.sum().backward()
mx.grad
```

```
MaskedTensor(
 [ 0.0000, 0.0067, --, --, --, --, --, --, --, --, --]
)
```

The gradient here is only provided to the selected subset. Effectively, this changes the gradient of where
to mask out elements instead of setting them to zero.

#### Another torch.where

[Issue 52248](https://github.com/pytorch/pytorch/issues/52248) is another example.

Current result:

```
a = torch.randn((), requires_grad=True)
b = torch.tensor(False)
c = torch.ones(())
print("torch.where(b, a/0, c):\n", torch.where(b, a/0, c))
print("torch.autograd.grad(torch.where(b, a/0, c), a):\n", torch.autograd.grad(torch.where(b, a/0, c), a))
```

```
torch.where(b, a/0, c):
 tensor(1., grad_fn=<WhereBackward0>)
torch.autograd.grad(torch.where(b, a/0, c), a):
 (tensor(nan),)
```

`MaskedTensor` result:

```
a = masked_tensor(torch.randn(()), torch.tensor(True), requires_grad=True)
b = torch.tensor(False)
c = torch.ones(())
print("torch.where(b, a/0, c):\n", torch.where(b, a/0, c))
print("torch.autograd.grad(torch.where(b, a/0, c), a):\n", torch.autograd.grad(torch.where(b, a/0, c), a))
```

```
torch.where(b, a/0, c):
 MaskedTensor( 1.0000, True)
torch.autograd.grad(torch.where(b, a/0, c), a):
 (MaskedTensor(--, False),)
```

This issue is similar (and even links to the next issue below) in that it expresses frustration with
unexpected behavior because of the inability to differentiate "no gradient" vs "zero gradient",
which in turn makes working with other ops difficult to reason about.

#### When using mask, x/0 yields NaN grad

In [Issue 4132](https://github.com/pytorch/pytorch/issues/4132), the user proposes that
x.grad should be [0, 1] instead of the [nan, 1],
whereas `MaskedTensor` makes this very clear by masking out the gradient altogether.

Current result:

```
x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])
y = x/div # => y is [inf, 1]
mask = (div != 0) # => mask is [0, 1]
y[mask].backward()
x.grad
```

```
tensor([nan, 1.])
```

`MaskedTensor` result:

```
x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])
y = x/div # => y is [inf, 1]
mask = (div != 0) # => mask is [0, 1]
loss = as_masked_tensor(y, mask)
loss.sum().backward()
x.grad
```

```
MaskedTensor(
 [ --, 1.0000]
)
```

### [`torch.nansum()`](https://docs.pytorch.org/docs/stable/generated/torch.nansum.html#torch.nansum) and [`torch.nanmean()`](https://docs.pytorch.org/docs/stable/generated/torch.nanmean.html#torch.nanmean)

In [Issue 67180](https://github.com/pytorch/pytorch/issues/67180),
the gradient isn't calculate properly (a longstanding issue), whereas `MaskedTensor` handles it correctly.

Current result:

```
a = torch.tensor([1., 2., float('nan')])
b = torch.tensor(1.0, requires_grad=True)
c = a * b
c1 = torch.nansum(c)
bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
bgrad1
```

```
tensor(nan)
```

`MaskedTensor` result:

```
a = torch.tensor([1., 2., float('nan')])
b = torch.tensor(1.0, requires_grad=True)
mt = masked_tensor(a, ~torch.isnan(a))
c = mt * b
c1 = torch.sum(c)
bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
bgrad1
```

```
MaskedTensor( 3.0000, True)
```

### Safe Softmax

Safe softmax is another great example of [an issue](https://github.com/pytorch/pytorch/issues/55056)
that arises frequently. In a nutshell, if there is an entire batch that is "masked out"
or consists entirely of padding (which, in the softmax case, translates to being set -inf),
then this will result in NaNs, which can lead to training divergence.

Luckily, `MaskedTensor` has solved this issue. Consider this setup:

```
data = torch.randn(3, 3)
mask = torch.tensor([[True, False, False], [True, False, True], [False, False, False]])
x = data.masked_fill(~mask, float('-inf'))
mt = masked_tensor(data, mask)
print("x:\n", x)
print("mt:\n", mt)
```

```
x:
 tensor([[ 0.7036, -inf, -inf],
 [-0.1679, -inf, -0.4656],
 [ -inf, -inf, -inf]])
mt:
 MaskedTensor(
 [
 [ 0.7036, --, --],
 [ -0.1679, --, -0.4656],
 [ --, --, --]
 ]
)
```

For example, we want to calculate the softmax along dim=0. Note that the second column is "unsafe" (i.e. entirely
masked out), so when the softmax is calculated, the result will yield 0/0 = nan since exp(-inf) = 0.
However, what we would really like is for the gradients to be masked out since they are unspecified and would be
invalid for training.

PyTorch result:

```
x.softmax(0)
```

```
tensor([[0.7051, nan, 0.0000],
 [0.2949, nan, 1.0000],
 [0.0000, nan, 0.0000]])
```

`MaskedTensor` result:

```
mt.softmax(0)
```

```
MaskedTensor(
 [
 [ 0.7051, --, --],
 [ 0.2949, --, 1.0000],
 [ --, --, --]
 ]
)
```

### Implementing missing torch.nan* operators

In [Issue 61474](https://github.com/pytorch/pytorch/issues/61474),
there is a request to add additional operators to cover the various torch.nan* applications,
such as `torch.nanmax`, `torch.nanmin`, etc.

In general, these problems lend themselves more naturally to masked semantics, so instead of introducing additional
operators, we propose using `MaskedTensor` instead.
Since [nanmean has already landed](https://github.com/pytorch/pytorch/issues/21987),
we can use it as a comparison point:

```
x = torch.arange(16).float()
y = x * x.fmod(4)
z = y.masked_fill(y == 0, float('nan')) # we want to get the mean of y when ignoring the zeros
```

```
print("y:\n", y)
# z is just y with the zeros replaced with nan's
print("z:\n", z)
```

```
y:
 tensor([ 0., 1., 4., 9., 0., 5., 12., 21., 0., 9., 20., 33., 0., 13.,
 28., 45.])
z:
 tensor([nan, 1., 4., 9., nan, 5., 12., 21., nan, 9., 20., 33., nan, 13.,
 28., 45.])
```

```
print("y.mean():\n", y.mean())
print("z.nanmean():\n", z.nanmean())
# MaskedTensor successfully ignores the 0's
print("torch.mean(masked_tensor(y, y != 0)):\n", torch.mean(masked_tensor(y, y != 0)))
```

```
y.mean():
 tensor(12.5000)
z.nanmean():
 tensor(16.6667)
torch.mean(masked_tensor(y, y != 0)):
 MaskedTensor( 16.6667, True)
```

In the above example, we've constructed a y and would like to calculate the mean of the series while ignoring
the zeros. torch.nanmean can be used to do this, but we don't have implementations for the rest of the
torch.nan* operations. `MaskedTensor` solves this issue by being able to use the base operation,
and we already have support for the other operations listed in the issue. For example:

```
torch.argmin(masked_tensor(y, y != 0))
```

```
MaskedTensor( 1.0000, True)
```

Indeed, the index of the minimum argument when ignoring the 0's is the 1 in index 1.

`MaskedTensor` can also support reductions when the data is fully masked out, which is equivalent
to the case above when the data Tensor is completely `nan`. `nanmean` would return `nan`
(an ambiguous return value), while MaskedTensor would more accurately indicate a masked out result.

```
x = torch.empty(16).fill_(float('nan'))
print("x:\n", x)
print("torch.nanmean(x):\n", torch.nanmean(x))
print("torch.nanmean via maskedtensor:\n", torch.mean(masked_tensor(x, ~torch.isnan(x))))
```

```
x:
 tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
torch.nanmean(x):
 tensor(nan)
torch.nanmean via maskedtensor:
 MaskedTensor(--, False)
```

This is a similar problem to safe softmax where 0/0 = nan when what we really want is an undefined value.

## Conclusion

In this tutorial, we've introduced what MaskedTensors are, demonstrated how to use them, and motivated their
value through a series of examples and issues that they've helped resolve.

## Further Reading

To continue learning more, you can find our
[MaskedTensor Sparsity tutorial](https://pytorch.org/tutorials/prototype/maskedtensor_sparsity.html)
to see how MaskedTensor enables sparsity and the different storage formats we currently support.

**Total running time of the script:** (0 minutes 0.164 seconds)

[`Download Jupyter notebook: maskedtensor_overview.ipynb`](../_downloads/098f809744385f9a587451e656698b10/maskedtensor_overview.ipynb)

[`Download Python source code: maskedtensor_overview.py`](../_downloads/4a51bb8a0b86f2086808e1d10395e483/maskedtensor_overview.py)

[`Download zipped: maskedtensor_overview.zip`](../_downloads/a28aa7ceff8a082bfda2e2e235535822/maskedtensor_overview.zip)