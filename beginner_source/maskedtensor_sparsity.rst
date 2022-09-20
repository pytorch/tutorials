Sparsity
++++++++

Sparsity has been an area of rapid growth and importance within PyTorch; if any sparsity terms are confusing below,
please refer to the `sparsity tutorial <https://pytorch.org/docs/stable/sparse.html>`__ for additional details.

Sparse storage formats have been proven to be powerful in a variety of ways. As a primer, the first use case
most practitioners think about is when the majority of elements are equal to zero (a high degree of sparsity),
but even in cases of lower sparsity, certain formats (e.g. BSR) can take advantage of substructures within a matrix.

.. note::

    At the moment, MaskedTensor supports COO and CSR tensors with plans to support additional formats
    (e.g. BSR and CSC) in the future. If you have any requests for additional formats, please file a feature request!

Principles
----------

When creating a :class:`MaskedTensor` with sparse tensors, there are a few principles that must be observed:

1. ``data`` and ``mask`` must have the same storage format, whether that's :attr:`torch.strided`, :attr:`torch.sparse_coo`, or :attr:`torch.sparse_csr`
2. ``data`` and ``mask`` must have the same size, indicated by :func:`size()`

Sparse COO tensors
------------------

In accordance with Principle #1, a sparse COO MaskedTensor is created by passing in two sparse COO tensors,
which can be initialized by any of its constructors, e.g. :func:`torch.sparse_coo_tensor`.

As a recap of `sparse COO tensors <https://pytorch.org/docs/stable/sparse.html#sparse-coo-tensors>`__, the COO format
stands for "coordinate format", where the specified elements are stored as tuples of their indices and the
corresponding values. That is, the following are provided:

* ``indices``: array of size ``(ndim, nse)`` and dtype ``torch.int64``
* ``values``: array of size `(nse,)` with any integer or floating point dtype

where ``ndim`` is the dimensionality of the tensor and ``nse`` is the number of specified elements

For both sparse COO and CSR tensors, you can construct a :class:`MaskedTensor` by doing either:

1. ``masked_tensor(sparse_tensor_data, sparse_tensor_mask)``
2. ``dense_masked_tensor.to_sparse_coo()`` or ``dense_masked_tensor.to_sparse_csr()``

The second method is easier to illustrate so we've shown that below, but for more on the first and the nuances behind
the approach, please read the :ref:`sparse-coo-appendix`.

    >>> values = torch.tensor([[0, 0, 3], [4, 0, 5]])
    >>> mask = torch.tensor([[False, False, True], [False, False, True]])
    >>> mt = masked_tensor(values, mask)
    >>> sparse_coo_mt = mt.to_sparse_coo()
    >>> mt
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )
    >>> sparse_coo_mt
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )
    >>> sparse_coo_mt.get_data()
    tensor(indices=tensor([[0, 1],
                          [2, 2]]),
          values=tensor([3, 5]),
          size=(2, 3), nnz=2, layout=torch.sparse_coo)

Sparse CSR tensors
------------------

Similarly, :class:`MaskedTensor` also supports the
`CSR (Compressed Sparse Row) <https://pytorch.org/docs/stable/sparse.html#sparse-csr-tensor>`__
sparse tensor format. Instead of storing the tuples of the indices like sparse COO tensors, sparse CSR tensors
aim to decrease the memory requirements by storing compressed row indices.
In particular, a CSR sparse tensor consists of three 1-D tensors:

* ``crow_indices``: array of compressed row indices with size ``(size[0] + 1,)``. This array indicates which row
  a given entry in values lives in. The last element is the number of specified elements,
  while crow_indices[i+1] - crow_indices[i] indicates the number of specified elements in row i.
* ``col_indices``: array of size ``(nnz,)``. Indicates the column indices for each value.
* ``values``: array of size ``(nnz,)``. Contains the values of the CSR tensor.

Of note, both sparse COO and CSR tensors are in a `beta <https://pytorch.org/docs/stable/index.html>`__ state.

By way of example:

    >>> mt_sparse_csr = mt.to_sparse_csr()
    >>> mt_sparse_csr
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )
    >>> mt_sparse_csr.get_data()
    tensor(crow_indices=tensor([0, 1, 2]),
          col_indices=tensor([2, 2]),
          values=tensor([3, 5]), size=(2, 3), nnz=2, layout=torch.sparse_csr)

Appendix
++++++++

.. _sparse-coo-appendix:

Sparse COO construction
-----------------------

Recall in our original example, we created a :class:`MaskedTensor` and then converted it to a sparse COO MaskedTensor
with :meth:`MaskedTensor.to_sparse_coo`.

Alternatively, we can also construct a sparse COO MaskedTensor directly by passing in two sparse COO tensors:

    >>> values = torch.tensor([[0, 0, 3], [4, 0, 5]]).to_sparse()
    >>> mask = torch.tensor([[False, False, True], [False, False, True]]).to_sparse()
    >>> mt = masked_tensor(values, mask)
    >>> values
    tensor(indices=tensor([[0, 1, 1],
                          [2, 0, 2]]),
          values=tensor([3, 4, 5]),
          size=(2, 3), nnz=3, layout=torch.sparse_coo)
    >>> mask
    tensor(indices=tensor([[0, 1],
                          [2, 2]]),
          values=tensor([True, True]),
          size=(2, 3), nnz=2, layout=torch.sparse_coo)
    >>> mt
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )

Instead of using :meth:`torch.Tensor.to_sparse`, we can also create the sparse COO tensors directly, which brings us to a warning:

.. warning::

  When using a function like :meth:`MaskedTensor.to_sparse_coo`, if the user does not specify the indices like in the above
  example, then the 0 values will be "unspecified" by default.

Below, we explicitly specify the 0's:

    >>> values = torch.sparse_coo_tensor(i, v, (2, 3))
    >>> mask = torch.sparse_coo_tensor(i, m, (2, 3))
    >>> mt2 = masked_tensor(values, mask)
    >>> values
    tensor(indices=tensor([[0, 1, 1],
                          [2, 0, 2]]),
          values=tensor([3, 4, 5]),
          size=(2, 3), nnz=3, layout=torch.sparse_coo)
    >>> mask
    tensor(indices=tensor([[0, 1, 1],
                          [2, 0, 2]]),
          values=tensor([ True, False,  True]),
          size=(2, 3), nnz=3, layout=torch.sparse_coo)
    >>> mt2
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )

Note that ``mt`` and ``mt2`` look identical on the surface, and in the vast majority of operations, will yield the same
result. But this brings us to a detail on the implementation:

``data`` and ``mask`` -- only for sparse MaskedTensors -- can have a different number of elements (:func:`nnz`)
**at creation**, but the indices of ``mask`` must then be a subset of the indices of ``data``. In this case,
``data`` will assume the shape of ``mask`` by ``data = data.sparse_mask(mask)``; in other words, any of the elements
in ``data`` that are not ``True`` in ``mask`` (i.e. not specified) will be thrown away.

Therefore, under the hood, the data looks slightly different; ``mt2`` has the "4" value masked out and ``mt`` is completely
without it. Their underlying data has different shapes, which would make operations like ``mt + mt2`` invalid.

    >>> mt.get_data()
    tensor(indices=tensor([[0, 1],
                          [2, 2]]),
          values=tensor([3, 5]),
          size=(2, 3), nnz=2, layout=torch.sparse_coo)
    >>> mt2.get_data()
    tensor(indices=tensor([[0, 1, 1],
                          [2, 0, 2]]),
          values=tensor([3, 4, 5]),
          size=(2, 3), nnz=3, layout=torch.sparse_coo)

.. _sparse-csr-appendix:

Sparse CSR construction
-----------------------

We can also construct a sparse CSR MaskedTensor using sparse CSR tensors,
and like the example above, this results in a similar treatment under the hood.

    >>> crow_indices = torch.tensor([0, 2, 4])
    >>> col_indices = torch.tensor([0, 1, 0, 1])
    >>> values = torch.tensor([1, 2, 3, 4])
    >>> mask_values = torch.tensor([True, False, False, True])
    >>>
    >>> csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
    >>> mask = torch.sparse_csr_tensor(crow_indices, col_indices, mask_values, dtype=torch.bool)
    >>>
    >>> mt = masked_tensor(csr, mask)
    >>> mt
    MaskedTensor(
      [
        [  1.0000,       --],
        [      --,   4.0000]
      ]
    )
    >>> mt.get_data()
    tensor(crow_indices=tensor([0, 2, 4]),
          col_indices=tensor([0, 1, 0, 1]),
          values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,
          dtype=torch.float64, layout=torch.sparse_csr)

