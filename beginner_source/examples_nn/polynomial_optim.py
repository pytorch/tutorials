# -*- coding: utf-8 -*-
"""
PyTorch: optim
--------------

A third order polynomial, trained to predict :math:`y=\sin(x)` from :math:`-\pi`
to :math:`pi` by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.

Rather than manually updating the weights of the model as we have been doing,
we use the optim package to define an Optimizer that will update the weights
for us. The optim package defines many optimization algorithms that are commonly
used for deep learning, including SGD+momentum, RMSProp, Adam, etc.
"""
import torch
import math


# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Use the nn package to define our model and loss function.
model = torch.nn.Linear(3, 1)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for t in range(2000):
    # In order to use :class:`torch.nn.Linear`, we need to prepare our
    # input and output in a format of (batch, D_in) and (batch, D_out)
    xx = x.unsqueeze(-1).pow(torch.tensor([1, 2, 3]))
    yy = y.unsqueeze(-1)

    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(xx)

    # Compute and print loss.
    loss = loss_fn(y_pred, yy)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()


print(f'Result: y = {model.bias.item()} + {model.weight[:, 0].item()} x + {model.weight[:, 1].item()} x^2 + {model.weight[:, 2].item()} x^3')
