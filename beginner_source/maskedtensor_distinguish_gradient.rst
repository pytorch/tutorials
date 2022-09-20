Distinguishing between 0 and NaN gradient
-----------------------------------------

One issue that :class:`torch.Tensor` runs into is the inability to distinguish between gradients that are not
defined (NaN) vs. gradients that are actually 0. By way of example, below are several different issues where
:class:`MaskedTensor` can resolve and/or work around the NaN gradient problem.

`Issue 10729 <https://github.com/pytorch/pytorch/issues/10729>`__ -- torch.where
--------------------------------------------------------------------------------

Current result:

    >>> x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True, dtype=torch.float)
    >>> y = torch.where(x < 0, torch.exp(x), torch.ones_like(x))
    >>> y.sum().backward()
    >>> x.grad
    tensor([4.5400e-05, 6.7379e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 0.0000e+00, 0.0000e+00,        nan,        nan])

:class:`MaskedTensor` result:

    >>> x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100])
    >>> mask = x < 0
    >>> mx = masked_tensor(x, mask, requires_grad=True)
    >>> my = masked_tensor(torch.ones_like(x), ~mask, requires_grad=True)
    >>> y = torch.where(mask, torch.exp(mx), my)
    >>> y.sum().backward()
    >>> mx.grad
    MaskedTensor(
      [  0.0000,   0.0067,       --,       --,       --,       --,       --,       --,       --,       --,       --]
    )

The gradient here is only provided to the selected subset. Effectively, this changes the gradient of `where`
to mask out elements instead of setting them to zero.

`Issue 52248 <https://github.com/pytorch/pytorch/issues/52248>`__ -- another torch.where
----------------------------------------------------------------------------------------

Current result:

    >>> a = torch.randn((), requires_grad=True)
    >>> b = torch.tensor(False)
    >>> c = torch.ones(())
    >>> torch.where(b, a/0, c)
    tensor(1., grad_fn=<WhereBackward0>)
    >>> torch.autograd.grad(torch.where(b, a/0, c), a)
    (tensor(nan),)

:class:`MaskedTensor` result:

    >>> a = masked_tensor(torch.randn(()), torch.tensor(True), requires_grad=True)
    >>> b = torch.tensor(False)
    >>> c = torch.ones(())
    >>> torch.where(b, a/0, c)
    MaskedTensor(  1.0000, True)
    >>> torch.autograd.grad(torch.where(b, a/0, c), a)
    (MaskedTensor(--, False),)

`Issue 67180 <https://github.com/pytorch/pytorch/issues/67180>`__ -- :func:`torch.nansum` and :func:`torch.nanmean`
-------------------------------------------------------------------------------------------------------------------

Current result:

    >>> a = torch.tensor([1., 2., float('nan')])
    >>> b = torch.tensor(1.0, requires_grad=True)
    >>> c = a * b
    >>> c1 = torch.nansum(c)
    >>> bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
    >>> bgrad1
    tensor(nan)

:class:`MaskedTensor` result:

    >>> a = torch.tensor([1., 2., float('nan')])
    >>> b = torch.tensor(1.0, requires_grad=True)
    >>> mt = masked_tensor(a, ~torch.isnan(a))
    >>> c = mt * b
    >>> c1 = torch.sum(c)
    >>> bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
    >>> bgrad1
    MaskedTensor(  3.0000, True)

`Issue 4132 <https://github.com/pytorch/pytorch/issues/4132>`__ -- when using mask, x/0 yields NaN grad
-------------------------------------------------------------------------------------------------------

Current result:

    >>> x = torch.tensor([1., 1.], requires_grad=True)
    >>> div = torch.tensor([0., 1.])
    >>> y = x/div # => y is [inf, 1]
    >>> mask = (div != 0) # => mask is [0, 1]
    >>> y[mask].backward()
    >>> x.grad # grad is [nan, 1], but expected [0, 1]
    tensor([nan, 1.])

:class:`MaskedTensor` result:

    >>> x = torch.tensor([1., 1.], requires_grad=True)
    >>> div = torch.tensor([0., 1.])
    >>> y = x/div # => y is [inf, 1]
    >>>
    >>> mask = (div != 0) # => mask is [0, 1]
    >>> loss = as_masked_tensor(y, mask)
    >>> loss.sum().backward()
    >>> x.grad
    MaskedTensor(
      [      --,   1.0000]
    )