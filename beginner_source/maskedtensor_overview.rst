MaskedTensor Overview
=====================

This tutorial is designed to serve as a starting point for using MaskedTensors
and discuss its masking semantics.

Using MaskedTensor
++++++++++++++++++

Construction
------------

There are a few different ways to construct a MaskedTensor:

* The first way is to directly invoke the MaskedTensor class
* The second (and our recommended way) is to use :func:`masked.masked_tensor` and :func:`masked.as_masked_tensor` factory functions,
  which are analogous to :func:`torch.tensor` and :func:`torch.as_tensor`

  .. autosummary::
    :toctree: generated
    :nosignatures:

    masked.masked_tensor
    masked.as_masked_tensor

Accessing the data and mask
---------------------------

The underlying fields in a MaskedTensor can be accessed through:

* the :meth:`MaskedTensor.get_data` function
* the :meth:`MaskedTensor.get_mask` function. Recall that ``True`` indicates "specified" or "valid" while ``False`` indicates
  "unspecified" or "invalid".

In general, the underlying data that is returned may not be valid in the unspecified entries, so we recommend that
when users require a Tensor without any masked entries, that they use :meth:`MaskedTensor.to_tensor` (as shown above) to
return a Tensor with filled values.

Indexing and slicing
--------------------

:class:`MaskedTensor` is a Tensor subclass, which means that it inherits the same semantics for indexing and slicing
as :class:`torch.Tensor`. Below are some examples of common indexing and slicing patterns:

    >>> data = torch.arange(60).reshape(3, 4, 5)
    >>> mask = data % 2 == 0
    >>> mt = masked_tensor(data.float(), mask)
    >>> mt[0]
    MaskedTensor(
      [
        [  0.0000,       --,   2.0000,       --,   4.0000],
        [      --,   6.0000,       --,   8.0000,       --],
        [ 10.0000,       --,  12.0000,       --,  14.0000],
        [      --,  16.0000,       --,  18.0000,       --]
      ]
    )
    >>> mt[[0,2]]
    MaskedTensor(
      [
        [
          [  0.0000,       --,   2.0000,       --,   4.0000],
          [      --,   6.0000,       --,   8.0000,       --],
          [ 10.0000,       --,  12.0000,       --,  14.0000],
          [      --,  16.0000,       --,  18.0000,       --]
        ],
        [
          [ 40.0000,       --,  42.0000,       --,  44.0000],
          [      --,  46.0000,       --,  48.0000,       --],
          [ 50.0000,       --,  52.0000,       --,  54.0000],
          [      --,  56.0000,       --,  58.0000,       --]
        ]
      ]
    )
    >>> mt[:, :2]
    MaskedTensor(
      [
        [
          [  0.0000,       --,   2.0000,       --,   4.0000],
          [      --,   6.0000,       --,   8.0000,       --]
        ],
        [
          [ 20.0000,       --,  22.0000,       --,  24.0000],
          [      --,  26.0000,       --,  28.0000,       --]
        ],
        [
          [ 40.0000,       --,  42.0000,       --,  44.0000],
          [      --,  46.0000,       --,  48.0000,       --]
        ]
      ]
    )

Semantics
+++++++++

MaskedTensor vs NumPy's MaskedArray
-----------------------------------

NumPy's ``MaskedArray`` has a few fundamental semantics differences from MaskedTensor.

1. Their factory function and basic definition inverts the mask (similar to ``torch.nn.MHA``); that is, MaskedTensor
uses ``True`` to denote "specified" and ``False`` to denote "unspecified", or "valid"/"invalid", whereas NumPy does the
opposite.
2. Intersection semantics. In NumPy, if one of two elements are masked out, the resulting element will be
masked out as well -- in practice, they
`apply the logical_or operator <https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1016-L1024>`__.

    >>> data = torch.arange(5.)
    >>> mask = torch.tensor([True, True, False, True, False])
    >>> npm0 = np.ma.masked_array(data.numpy(), (~mask).numpy())
    >>> npm1 = np.ma.masked_array(data.numpy(), (mask).numpy())
    >>> npm0
    masked_array(data=[0.0, 1.0, --, 3.0, --],
                mask=[False, False,  True, False,  True],
          fill_value=1e+20,
                dtype=float32)
    >>> npm1
    masked_array(data=[--, --, 2.0, --, 4.0],
                mask=[ True,  True, False,  True, False],
          fill_value=1e+20,
                dtype=float32)
    >>> npm0 + npm1
    masked_array(data=[--, --, --, --, --],
                mask=[ True,  True,  True,  True,  True],
          fill_value=1e+20,
                dtype=float32)

Meanwhile, MaskedTensor does not support addition or binary operators with masks that don't match -- to understand why,
please find the section on reductions.

    >>> mt0 = masked_tensor(data, mask)
    >>> mt1 = masked_tensor(data, ~mask)
    >>> m0
    MaskedTensor(
      [  0.0000,   1.0000,       --,   3.0000,       --]
    )
    >>> mt0 = masked_tensor(data, mask)
    >>> mt1 = masked_tensor(data, ~mask)
    >>> mt0
    MaskedTensor(
      [  0.0000,   1.0000,       --,   3.0000,       --]
    )
    >>> mt1
    MaskedTensor(
      [      --,       --,   2.0000,       --,   4.0000]
    )
    >>> mt0 + mt1
    ValueError: Input masks must match. If you need support for this, please open an issue on Github.

However, if this behavior is desired, MaskedTensor does support these semantics by giving access to the data and masks
and conveniently converting a MaskedTensor to a Tensor with masked values filled in using :func:`to_tensor`.

    >>> t0 = mt0.to_tensor(0)
    >>> t1 = mt1.to_tensor(0)
    >>> mt2 = masked_tensor(t0 + t1, mt0.get_mask() & mt1.get_mask())
    >>> t0
    tensor([0., 1., 0., 3., 0.])
    >>> t1
    tensor([0., 0., 2., 0., 4.])
    >>> mt2
    MaskedTensor(
      [      --,       --,       --,       --,       --]

.. _reduction-semantics:

Reduction semantics
-------------------

The basis for reduction semantics `has been documented and discussed at length <https://github.com/pytorch/rfcs/pull/27>`__,
but again, by way of example:

    >>> data = torch.arange(12, dtype=torch.float).reshape(3, 4)
    >>> mask = torch.randint(2, (3, 4), dtype=torch.bool)
    >>> mt = masked_tensor(data, mask)
    >>> mt
    MaskedTensor(
      [
        [      --,   1.0000,       --,       --],
        [      --,   5.0000,   6.0000,   7.0000],
        [  8.0000,   9.0000,       --,  11.0000]
      ]
    )

    >>> torch.sum(mt, 1)
    MaskedTensor(
      [  1.0000,  18.0000,  28.0000]
    )
    >>> torch.mean(mt, 1)
    MaskedTensor(
      [  1.0000,   6.0000,   9.3333]
    )
    >>> torch.prod(mt, 1)
    MaskedTensor(
      [  1.0000, 210.0000, 792.0000]
    )
    >>> torch.amin(mt, 1)
    MaskedTensor(
      [  1.0000,   5.0000,   8.0000]
    )
    >>> torch.amax(mt, 1)
    MaskedTensor(
      [  1.0000,   7.0000,  11.0000]
    )

Now we can revisit the question: why do we enforce the invariant that masks must match for binary operators?
In other words, why don't we use the same semantics as ``np.ma.masked_array``? Consider the following example:

    >>> data0 = torch.arange(10.).reshape(2, 5)
    >>> data1 = torch.arange(10.).reshape(2, 5) + 10
    >>> mask0 = torch.tensor([[True, True, False, False, False], [False, False, False, True, True]])
    >>> mask1 = torch.tensor([[False, False, False, True, True], [True, True, False, False, False]])

    >>> npm0 = np.ma.masked_array(data0.numpy(), (mask0).numpy())
    >>> npm1 = np.ma.masked_array(data1.numpy(), (mask1).numpy())
    >>> npm0
    masked_array(
      data=[[--, --, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, --, --]],
      mask=[[ True,  True, False, False, False],
            [False, False, False,  True,  True]],
      fill_value=1e+20,
      dtype=float32)
    >>> npm1
    masked_array(
      data=[[10.0, 11.0, 12.0, --, --],
            [--, --, 17.0, 18.0, 19.0]],
      mask=[[False, False, False,  True,  True],
            [ True,  True, False, False, False]],
      fill_value=1e+20,
      dtype=float32)
    >>> (npm0 + npm1).sum(0)
    masked_array(data=[--, --, 38.0, --, --],
                mask=[ True,  True, False,  True,  True],
          fill_value=1e+20,
                dtype=float32)
    >>> npm0.sum(0) + npm1.sum(0)
    masked_array(data=[15.0, 17.0, 38.0, 21.0, 23.0],
                mask=[False, False, False, False, False],
          fill_value=1e+20,
                dtype=float32)

Sum and addition should clearly be associative, but with NumPy's semantics, they are allowed to not be,
which can certainly be confusing for the user. That being said, if the user wishes, there are ways around this
(e.g. filling in the MaskedTensor's undefined elements with 0 values using :func:`to_tensor` as shown in a previous
example), but the user must now be more explicit with their intentions.
