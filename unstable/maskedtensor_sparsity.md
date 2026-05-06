Note

Go to the end
to download the full example code.

# MaskedTensor Sparsity

Before working on this tutorial, please make sure to review our
MaskedTensor Overview tutorial <https://pytorch.org/tutorials/prototype/maskedtensor_overview.html>.

## Introduction

Sparsity has been an area of rapid growth and importance within PyTorch; if any sparsity terms are confusing below,
please refer to the [sparsity tutorial](https://pytorch.org/docs/stable/sparse.html) for additional details.

Sparse storage formats have been proven to be powerful in a variety of ways. As a primer, the first use case
most practitioners think about is when the majority of elements are equal to zero (a high degree of sparsity),
but even in cases of lower sparsity, certain formats (e.g. BSR) can take advantage of substructures within a matrix.

Note

At the moment, MaskedTensor supports COO and CSR tensors with plans to support additional formats
(such as BSR and CSC) in the future. If you have any requests for additional formats,
please file a feature request [here](https://github.com/pytorch/pytorch/issues)!

## Principles

When creating a `MaskedTensor` with sparse tensors, there are a few principles that must be observed:

1. `data` and `mask` must have the same storage format, whether that's `torch.strided`, `torch.sparse_coo`, or `torch.sparse_csr`
2. `data` and `mask` must have the same size, indicated by `size()`

## Sparse COO tensors

In accordance with Principle #1, a sparse COO MaskedTensor is created by passing in two sparse COO tensors,
which can be initialized by any of its constructors, for example [`torch.sparse_coo_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor).

As a recap of [sparse COO tensors](https://pytorch.org/docs/stable/sparse.html#sparse-coo-tensors), the COO format
stands for "coordinate format", where the specified elements are stored as tuples of their indices and the
corresponding values. That is, the following are provided:

- `indices`: array of size `(ndim, nse)` and dtype `torch.int64`
- `values`: array of size (nse,) with any integer or floating point dtype

where `ndim` is the dimensionality of the tensor and `nse` is the number of specified elements.

For both sparse COO and CSR tensors, you can construct a `MaskedTensor` by doing either:

1. `masked_tensor(sparse_tensor_data, sparse_tensor_mask)`
2. `dense_masked_tensor.to_sparse_coo()` or `dense_masked_tensor.to_sparse_csr()`

The second method is easier to illustrate so we've shown that below, but for more on the first and the nuances behind
the approach, please read the Sparse COO Appendix.

```
# Disable prototype warnings and such
```

## Sparse CSR tensors

Similarly, `MaskedTensor` also supports the
[CSR (Compressed Sparse Row)](https://pytorch.org/docs/stable/sparse.html#sparse-csr-tensor)
sparse tensor format. Instead of storing the tuples of the indices like sparse COO tensors, sparse CSR tensors
aim to decrease the memory requirements by storing compressed row indices.
In particular, a CSR sparse tensor consists of three 1-D tensors:

- `crow_indices`: array of compressed row indices with size `(size[0] + 1,)`. This array indicates which row
a given entry in values lives in. The last element is the number of specified elements,
while crow_indices[i+1] - crow_indices[i] indicates the number of specified elements in row i.
- `col_indices`: array of size `(nnz,)`. Indicates the column indices for each value.
- `values`: array of size `(nnz,)`. Contains the values of the CSR tensor.

Of note, both sparse COO and CSR tensors are in a [beta](https://pytorch.org/docs/stable/index.html) state.

By way of example:

## Supported Operations

### Unary

All [unary operators](https://pytorch.org/docs/master/masked.html#unary-operators) are supported, e.g.:

### Binary

[Binary operators](https://pytorch.org/docs/master/masked.html#unary-operators) are also supported, but the
input masks from the two masked tensors must match. For more information on why this decision was made, please
find our [MaskedTensor: Advanced Semantics tutorial](https://pytorch.org/tutorials/prototype/maskedtensor_advanced_semantics.html).

Please find an example below:

### Reductions

Finally, [reductions](https://pytorch.org/docs/master/masked.html#reductions) are supported:

### MaskedTensor Helper Methods

For convenience, `MaskedTensor` has a number of methods to help convert between the different layouts
and identify the current layout:

Setup:

`MaskedTensor.to_sparse_coo()` / `MaskedTensor.to_sparse_csr()` / `MaskedTensor.to_dense()`
to help convert between the different layouts.

`MaskedTensor.is_sparse()` - this will check if the `MaskedTensor`'s layout
matches any of the supported sparse layouts (currently COO and CSR).

`MaskedTensor.is_sparse_coo()`

`MaskedTensor.is_sparse_csr()`

## Appendix

### Sparse COO Construction

Recall in our original example, we created a `MaskedTensor`
and then converted it to a sparse COO MaskedTensor with `MaskedTensor.to_sparse_coo()`.

Alternatively, we can also construct a sparse COO MaskedTensor directly by passing in two sparse COO tensors:

Instead of using [`torch.Tensor.to_sparse()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to_sparse.html#torch.Tensor.to_sparse), we can also create the sparse COO tensors directly,
which brings us to a warning:

Warning

When using a function like `MaskedTensor.to_sparse_coo()` (analogous to `Tensor.to_sparse()`),
if the user does not specify the indices like in the above example,
then the 0 values will be "unspecified" by default.

Below, we explicitly specify the 0's:

Note that `mt` and `mt2` look identical on the surface, and in the vast majority of operations, will yield the same
result. But this brings us to a detail on the implementation:

`data` and `mask` - only for sparse MaskedTensors - can have a different number of elements (`nnz()`)
**at creation**, but the indices of `mask` must then be a subset of the indices of `data`. In this case,
`data` will assume the shape of `mask` by `data = data.sparse_mask(mask)`; in other words, any of the elements
in `data` that are not `True` in `mask` (that is, not specified) will be thrown away.

Therefore, under the hood, the data looks slightly different; `mt2` has the "4" value masked out and `mt`
is completely without it. Their underlying data has different shapes,
which would make operations like `mt + mt2` invalid.

### Sparse CSR Construction

We can also construct a sparse CSR MaskedTensor using sparse CSR tensors,
and like the example above, this results in a similar treatment under the hood.

## Conclusion

In this tutorial, we have introduced how to use `MaskedTensor` with sparse COO and CSR formats and
discussed some of the subtleties under the hood in case users decide to access the underlying data structures
directly. Sparse storage formats and masked semantics indeed have strong synergies, so much so that they are
sometimes used as proxies for each other (as we will see in the next tutorial). In the future, we certainly plan
to invest and continue developing in this direction.

## Further Reading

To continue learning more, you can find our
[Efficiently writing "sparse" semantics for Adagrad with MaskedTensor tutorial](https://pytorch.org/tutorials/prototype/maskedtensor_adagrad.html)
to see an example of how MaskedTensor can simplify existing workflows with native masking semantics.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: maskedtensor_sparsity.ipynb`](../_downloads/0bfc601b4d08b8eb61463ae0c13808ac/maskedtensor_sparsity.ipynb)

[`Download Python source code: maskedtensor_sparsity.py`](../_downloads/94851de20bbbf9f26e972b282f6e76d9/maskedtensor_sparsity.py)

[`Download zipped: maskedtensor_sparsity.zip`](../_downloads/a246559fd0656137b4655160cc531dcc/maskedtensor_sparsity.zip)