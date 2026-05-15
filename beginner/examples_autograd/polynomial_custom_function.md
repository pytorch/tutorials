Note

Go to the end
to download the full example code.

# PyTorch: Defining New autograd Functions

A third order polynomial, trained to predict \(y=\sin(x)\) from \(-\pi\)
to \(\pi\) by minimizing squared Euclidean distance. Instead of writing the
polynomial as \(y=a+bx+cx^2+dx^3\), we write the polynomial as
\(y=a+b P_3(c+dx)\) where \(P_3(x)=\frac{1}{2}\left(5x^3-3x\right)\) is
the [Legendre polynomial](https://en.wikipedia.org/wiki/Legendre_polynomials) of degree three.

This implementation computes the forward pass using operations on PyTorch
Tensors, and uses PyTorch autograd to compute gradients.

In this implementation we implement our own custom autograd function to perform
\(P_3'(x)\). By mathematics, \(P_3'(x)=\frac{3}{2}\left(5x^2-1\right)\)

```
# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.

# Create random Tensors for weights. For this example, we need
# 4 weights: y = a + b * P3(c + d * x), these weights need to be initialized
# not too far from the correct result to ensure convergence.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: polynomial_custom_function.ipynb`](../../_downloads/36007a4d548729b824a1364eb165d070/polynomial_custom_function.ipynb)

[`Download Python source code: polynomial_custom_function.py`](../../_downloads/71d2368d801179561137f31cc6d0a837/polynomial_custom_function.py)

[`Download zipped: polynomial_custom_function.zip`](../../_downloads/22da64fa3f8be20683e32c93ae47130f/polynomial_custom_function.zip)