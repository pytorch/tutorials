Safe Softmax
------------

One of the issues that frequently comes up is the necessity for a safe softmax -- that is, if there is an entire
batch that is "masked out" or consists entirely of padding (which, in the softmax case, translates to being set `-inf`),
then this will result in NaNs, which can leading to training divergence. For more detail on why this functionality
is necessary, please find refer to 
`Issue 55056 - Feature Request for Safe Softmax <https://github.com/pytorch/pytorch/issues/55056>`__.

Luckily, :class:`MaskedTensor` has solved this issue:

    >>> data = torch.randn(3, 3)
    >>> mask = torch.tensor([[True, False, False], [True, False, True], [False, False, False]])
    >>> x = data.masked_fill(~mask, float('-inf'))
    >>> mt = masked_tensor(data, mask)

PyTorch result:

    >>> x.softmax(0)
    tensor([[0.3548,    nan, 0.0000],
            [0.6452,    nan, 1.0000],
            [0.0000,    nan, 0.0000]])

:class:`MaskedTensor` result:

    >>> mt.softmax(0)
    MaskedTensor(
      [
        [  0.3548,       --,       --],
        [  0.6452,       --,   1.0000],
        [      --,       --,       --]
      ]
    )
