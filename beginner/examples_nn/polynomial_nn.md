Note

Go to the end
to download the full example code.

# PyTorch: nn

A third order polynomial, trained to predict \(y=\sin(x)\) from \(-\pi\)
to \(\pi\) by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.
PyTorch autograd makes it easy to define computational graphs and take gradients,
but raw autograd can be a bit too low-level for defining complex neural networks;
this is where the nn package can help. The nn package defines a set of Modules,
which you can think of as a neural network layer that produces output from
input and may have some trainable weights.

```
# Create Tensors to hold input and outputs.

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).

# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.

# You can access the first layer of `model` like accessing the first item of a list

# For linear layer, its parameters are stored as `weight` and `bias`.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: polynomial_nn.ipynb`](../../_downloads/702a393951467ef6b030977e60a85afc/polynomial_nn.ipynb)

[`Download Python source code: polynomial_nn.py`](../../_downloads/cae15c367856daa07db2b142cddfe23f/polynomial_nn.py)

[`Download zipped: polynomial_nn.zip`](../../_downloads/4ae97186e2d5fee43da86f46ada6a3db/polynomial_nn.zip)