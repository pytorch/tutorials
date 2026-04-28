Note

Go to the end
to download the full example code.

# PyTorch: optim

A third order polynomial, trained to predict \(y=\sin(x)\) from \(-\pi\)
to \(\pi\) by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.

Rather than manually updating the weights of the model as we have been doing,
we use the optim package to define an Optimizer that will update the weights
for us. The optim package defines many optimization algorithms that are commonly
used for deep learning, including SGD+momentum, RMSProp, Adam, etc.

```
# Create Tensors to hold input and outputs.

# Prepare the input tensor (x, x^2, x^3).

# Use the nn package to define our model and loss function.

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms. The first argument to the RMSprop constructor tells the
# optimizer which Tensors it should update.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: polynomial_optim.ipynb`](../../_downloads/78c1107d6e85ca28c97818261144e084/polynomial_optim.ipynb)

[`Download Python source code: polynomial_optim.py`](../../_downloads/d11842a6d4f2f816ca55dcf6f75ece1f/polynomial_optim.py)

[`Download zipped: polynomial_optim.zip`](../../_downloads/d8a1adbe78aef52de9f1f1d1cc08b992/polynomial_optim.zip)