Note

Go to the end
to download the full example code.

# PyTorch: Tensors

A third order polynomial, trained to predict \(y=\sin(x)\) from \(-\pi\)
to \(\pi\) by minimizing squared Euclidean distance.

This implementation uses PyTorch tensors to manually compute the forward pass,
loss, and backward pass.

A PyTorch Tensor is basically the same as a numpy array: it does not know
anything about deep learning or computational graphs or gradients, and is just
a generic n-dimensional array to be used for arbitrary numeric computation.

The biggest difference between a numpy array and a PyTorch Tensor is that
a PyTorch Tensor can run on either CPU or GPU. To run operations on the GPU,
just cast the Tensor to a cuda datatype.

```
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data

# Randomly initialize weights

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: polynomial_tensor.ipynb`](../../_downloads/e5fb02f2aefe0d0ecaec99610c21a513/polynomial_tensor.ipynb)

[`Download Python source code: polynomial_tensor.py`](../../_downloads/b1701799d465392dc244e3de4f56744b/polynomial_tensor.py)

[`Download zipped: polynomial_tensor.zip`](../../_downloads/7037985709d0ab0be4b2c8c29ca78504/polynomial_tensor.zip)