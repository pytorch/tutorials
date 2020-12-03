# -*- coding: utf-8 -*-
"""
PyTorch: nn
-----------

A third order polynomial, trained to predict :math:`y=\sin(x)` from :math:`-\pi`
to :math:`pi` by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.
PyTorch autograd makes it easy to define computational graphs and take gradients,
but raw autograd can be a bit too low-level for defining complex neural networks;
this is where the nn package can help. The nn package defines a set of Modules,
which you can think of as a neural network layer that has produces output from
input and may have some trainable weights.
"""
import torch
import math


# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Use the nn package to define our model as a single layer or a sequence of layers.
# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a single linear layer neural network.
model = torch.nn.Linear(3, 1)

# If your model has multiple layers, you can use :class:`torch.nn.Sequential` them in
# sequence to produce its output. Something like:
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out),
# )

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
    # In order to use :class:`torch.nn.Linear`, we need to prepare our
    # input and output in a format of (batch, D_in) and (batch, D_out)
    xx = x.unsqueeze(-1).pow(torch.tensor([1, 2, 3]))
    yy = y.unsqueeze(-1)

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(xx)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, yy)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

print(f'Result: y = {model.bias.item()} + {model.weight[:, 0].item()} x + {model.weight[:, 1].item()} x^2 + {model.weight[:, 2].item()} x^3')
