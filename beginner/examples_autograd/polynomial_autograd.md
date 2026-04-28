Note

Go to the end
to download the full example code.

# PyTorch: Tensors and autograd

A third order polynomial, trained to predict \(y=\sin(x)\) from \(-\pi\)
to \(\pi\) by minimizing squared Euclidean distance.

This implementation computes the forward pass using operations on PyTorch
Tensors, and uses PyTorch autograd to compute gradients.

A PyTorch Tensor represents a node in a computational graph. If `x` is a
Tensor that has `x.requires_grad=True` then `x.grad` is another Tensor
holding the gradient of `x` with respect to some scalar value.

```
# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: polynomial_autograd.ipynb`](../../_downloads/0184bded18578d24a48fdfad2c701b09/polynomial_autograd.ipynb)

[`Download Python source code: polynomial_autograd.py`](../../_downloads/a74022a3afc7f1a190f572b6b9e883e4/polynomial_autograd.py)

[`Download zipped: polynomial_autograd.zip`](../../_downloads/b12cd1a62967beccf41c728fef4ac1fa/polynomial_autograd.zip)