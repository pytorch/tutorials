Implementing missing torch.nan* operators
-----------------------------------------

In the above issue, there is a request to add additional operators to cover the various `torch.nan*` applications,
such as ``torch.nanmax``, ``torch.nanmin``, etc.

In general, these problems lend themselves more naturally to masked semantics, so instead of introducing additional
operators, we propose using MaskedTensors instead. Since
`nanmean has already landed <https://github.com/pytorch/pytorch/issues/21987>`__, we can use it as a comparison point:

    >>> x = torch.arange(16).float()
    >>> y = x * x.fmod(4)
    >>> y = y.masked_fill(y ==0, float('nan'))
    >>> y
    tensor([nan,  1.,  4.,  9., nan,  5., 12., 21., nan,  9., 20., 33., nan, 13.,
            28., 45.])
    >>> y.nanmean()
    tensor(16.6667)
    >>> torch.mean(masked_tensor(y, ~torch.isnan(y)))
    MaskedTensor( 16.6667, True)

:class:`MaskedTensor` can also support reductions when the data is fully masked out, which is equivalent
to the case above when the data Tensor is completely ``nan``. ``nanmean`` would return ``nan``
(an ambiguous return value), while MaskedTensor would more accurately indicate a masked out result.

    >>> x = torch.empty(16).fill_(float('nan'))
    >>> x
    tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
    >>> torch.nanmean(x)
    tensor(nan)
    >>> torch.mean(masked_tensor(x, ~torch.isnan(x)))
    MaskedTensor(--, False)
