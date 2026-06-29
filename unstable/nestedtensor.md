Note

Go to the end
to download the full example code.

# Getting Started with Nested Tensors

Nested tensors generalize the shape of regular dense tensors, allowing for representation
of ragged-sized data.

- for a regular tensor, each dimension is regular and has a size
- for a nested tensor, not all dimensions have regular sizes; some of them are ragged

Nested tensors are a natural solution for representing sequential data within various domains:

- in NLP, sentences can have variable lengths, so a batch of sentences forms a nested tensor
- in CV, images can have variable shapes, so a batch of images forms a nested tensor

In this tutorial, we will demonstrate basic usage of nested tensors and motivate their usefulness
for operating on sequential data of varying lengths with a real-world example. In particular,
they are invaluable for building transformers that can efficiently operate on ragged sequential
inputs. Below, we present an implementation of multi-head attention using nested tensors that,
combined usage of `torch.compile`, out-performs operating naively on tensors with padding.

Nested tensors are currently a prototype feature and are subject to change.

## Nested tensor initialization

From the Python frontend, a nested tensor can be created from a list of tensors.
We denote nt[i] as the ith tensor component of a nestedtensor.

By padding every underlying tensor to the same shape,
a nestedtensor can be converted to a regular tensor.

All tensors posses an attribute for determining if they are nested;

It is common to construct nestedtensors from batches of irregularly shaped tensors.
i.e. dimension 0 is assumed to be the batch dimension.
Indexing dimension 0 gives back the first underlying tensor component.

```
# When indexing a nestedtensor's 0th dimension, the result is a regular tensor.
```

An important note is that slicing in dimension 0 has not been supported yet.
Which means it not currently possible to construct a view that combines the underlying
tensor components.

## Nested Tensor Operations

As each operation must be explicitly implemented for nestedtensors,
operation coverage for nestedtensors is currently narrower than that of regular tensors.
For now, only basic operations such as index, dropout, softmax, transpose, reshape, linear, bmm are covered.
However, coverage is being expanded.
If you need certain operations, please file an [issue](https://github.com/pytorch/pytorch)
to help us prioritize coverage.

**reshape**

The reshape op is for changing the shape of a tensor.
Its full semantics for regular tensors can be found
[here](https://pytorch.org/docs/stable/generated/torch.reshape.html).
For regular tensors, when specifying the new shape,
a single dimension may be -1, in which case it is inferred
from the remaining dimensions and the number of elements.

The semantics for nestedtensors are similar, except that -1 no longer infers.
Instead, it inherits the old size (here 2 for `nt[0]` and 3 for `nt[1]`).
-1 is the only legal size to specify for a jagged dimension.

**transpose**

The transpose op is for swapping two dimensions of a tensor.
Its full semantics can be found
[here](https://pytorch.org/docs/stable/generated/torch.transpose.html).
Note that for nestedtensors dimension 0 is special;
it is assumed to be the batch dimension,
so transposes involving nestedtensor dimension 0 are not supported.

**others**

Other operations have the same semantics as for regular tensors.
Applying the operation on a nestedtensor is equivalent to
applying the operation to the underlying tensor components,
with the result being a nestedtensor as well.

## Why Nested Tensor

When data is sequential, it is often the case that each sample has a different length.
For example, in a batch of sentences, each sentence has a different number of words.
A common technique for handling varying sequences is to manually pad each data tensor
to the same shape in order to form a batch.
For example, we have 2 sentences with different lengths and a vocabulary
In order to represent his as single tensor we pad with 0 to the max length in the batch.

This technique of padding a batch of data to its max length is not optimal.
The padded data is not needed for computation and wastes memory by allocating
larger tensors than necessary.
Further, not all operations have the same semnatics when applied to padded data.
For matrix multiplications in order to ignore the padded entries, one needs to pad
with 0 while for softmax one has to pad with -inf to ignore specific entries.
The primary objective of nested tensor is to facilitate operations on ragged
data using the standard PyTorch tensor UX, thereby eliminating the need
for inefficient and complex padding and masking.

Let us take a look at a practical example: the multi-head attention component
utilized in [Transformers](https://arxiv.org/pdf/1706.03762.pdf).
We can implement this in such a way that it can operate on either padded
or nested tensors.

set hyperparameters following [the Transformer paper](https://arxiv.org/pdf/1706.03762.pdf)

except for dropout probability: set to 0 for correctness check

Let us generate some realistic fake data from Zipf's law.

Create nested tensor batch inputs

Generate padded forms of query, key, value for comparison

Construct the model

Check correctness and performance

```
# padding-specific step: remove output projection bias from padded entries for fair comparison

# warm up compile first...

# ...now benchmark

# warm up compile first...

# ...now benchmark

# padding-specific step: remove output projection bias from padded entries for fair comparison
```

Note that without `torch.compile`, the overhead of the python subclass nested tensor
can make it slower than the equivalent computation on padded tensors. However, once
`torch.compile` is enabled, operating on nested tensors gives a multiple x speedup.
Avoiding wasted computation on padding becomes only more valuable as the percentage
of padding in the batch increases.

## Conclusion

In this tutorial, we have learned how to perform basic operations with nested tensors and
how implement multi-head attention for transformers in a way that avoids computation on padding.
For more information, check out the docs for the
[torch.nested](https://pytorch.org/docs/stable/nested.html) namespace.

## See Also

- [Accelerating PyTorch Transformers by replacing nn.Transformer with Nested Tensors and torch.compile](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: nestedtensor.ipynb`](../_downloads/0e22044ad9c3abd953c575aedd5e4595/nestedtensor.ipynb)

[`Download Python source code: nestedtensor.py`](../_downloads/f7cbd2a4028223851c72d0d81ce67897/nestedtensor.py)

[`Download zipped: nestedtensor.zip`](../_downloads/c7ac4597a28e534417e00d48ed79c97f/nestedtensor.zip)