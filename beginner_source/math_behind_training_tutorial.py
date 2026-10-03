"""
Visualizing the Mathematics Behind Neural Network Training
==========================================================

**Author:** `Aditya Mehra <https://github.com/AddyM>`_

Every PyTorch training loop runs three lines that carry the weight of
three branches of mathematics:

.. code-block:: python

   prediction = model(x)        # linear algebra
   loss.backward()              # calculus (chain rule)
   optimizer.step()             # optimization (gradient descent)

This tutorial opens each one up. We will print weight matrices, manually
verify gradients, walk the computational graph, and watch parameters
change — turning PyTorch's abstractions from black boxes into glass
boxes.

No GPU required. No advanced math required. Just a willingness to
look inside the machine.

**Prerequisites:**
Basic familiarity with PyTorch tensors and ``nn.Module``.

Based on the talk *"The Math Behind the Magic"* presented at
Grace Hopper Celebration 2025.
"""

import torch
import torch.nn as nn


######################################################################
# Part 1: Linear Algebra — What ``nn.Linear`` Actually Computes
# --------------------------------------------------------------
#
# A neural network layer performs a linear transformation:
#
# .. math::
#    \mathbf{y} = \mathbf{x} \mathbf{W}^T + \mathbf{b}
#
# This is matrix multiplication followed by vector addition.
# ``nn.Linear`` does nothing more. Let's prove it.
#

torch.manual_seed(42)

# A linear layer: 3 input features -> 2 output features
layer = nn.Linear(in_features=3, out_features=2)

# Input: a single sample with 3 features
x = torch.tensor([[1.0, 2.0, 3.0]])

# Forward pass through the layer
y_layer = layer(x)

# Manual computation: y = x @ W^T + b
y_manual = x @ layer.weight.T + layer.bias

print(f"Layer output:  {y_layer.data}")
print(f"Manual output: {y_manual.data}")
print(f"Match: {torch.allclose(y_layer, y_manual)}")

######################################################################
# They match. ``nn.Linear`` is matrix multiplication plus a bias —
# the same operation whether your network has 3 parameters or 3
# billion.
#
# Let's inspect what the layer stores:
#

print(f"Weight matrix W shape: {list(layer.weight.shape)}")
print(layer.weight.data)
print(f"\nBias vector b shape:   {list(layer.bias.shape)}")
print(layer.bias.data)
print(f"\nTotal parameters:      {sum(p.numel() for p in layer.parameters())}")

######################################################################
# These weights are random initial values. Training will adjust them.
# But the operation — matrix multiply plus bias — never changes.
#
# **Why does depth matter then?** Without activation functions,
# stacking linear layers collapses into a single linear
# transformation. We use ``bias=False`` here to isolate the
# matrix multiplication and make the collapse visible:
#

layer1 = nn.Linear(3, 4, bias=False)
layer2 = nn.Linear(4, 2, bias=False)

y_stacked = layer2(layer1(x))
y_collapsed = x @ (layer2.weight @ layer1.weight).T

print(f"\nTwo layers:    {y_stacked.data}")
print(f"Collapsed:     {y_collapsed.data}")
print(f"Match: {torch.allclose(y_stacked, y_collapsed)}")

######################################################################
# Two layers produced the exact same result as one combined matrix.
# Adding depth without nonlinearity adds nothing. This is why
# activation functions exist.


######################################################################
# Part 2: Activation Functions — Breaking Linearity
# ---------------------------------------------------
#
# Activation functions introduce nonlinearity between layers,
# allowing networks to learn patterns that a single matrix
# multiplication never could.
#
# ReLU — the most widely used activation — has a simple rule:
#
# .. math::
#    \text{ReLU}(x) = \max(0, x)
#
# Its derivative is equally simple: 1 for positive inputs, 0 for
# negative. This derivative is what ``autograd`` uses during
# backpropagation, so understanding it here pays off in Part 3.
#

x_range = torch.linspace(-3, 3, steps=7)

relu_output = torch.relu(x_range)
print("Input:       ", x_range.tolist())
print("ReLU output: ", relu_output.tolist())

######################################################################
# Negative values become zero. Positive values pass through
# unchanged. This simple rule is enough to break the linear
# collapse we proved above — now stacking layers creates genuinely
# new representations.
#
# But there is a cost. When a neuron's input is always negative,
# its ReLU output is always zero, its gradient is always zero,
# and it stops learning permanently. These are called **dead
# neurons** — a real problem in practice:
#

# Simulate a layer's output for 100 samples
torch.manual_seed(0)
layer_output = torch.randn(100, 8) - 1.0  # shifted negative
after_relu = torch.relu(layer_output)

dead = (after_relu == 0).all(dim=0)
print(f"\nDead neurons: {dead.sum().item()} out of {after_relu.shape[1]}")
print(f"Dead neuron mask: {dead.tolist()}")


######################################################################
# Part 3: The Chain Rule — What ``backward()`` Computes
# ------------------------------------------------------
#
# Backpropagation is the chain rule applied systematically. For a
# composition of functions :math:`f(g(x))`:
#
# .. math::
#    \frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
#
# PyTorch builds a computational graph during the forward pass and
# traverses it in reverse during ``backward()``. Let's trace this
# with a simple example and then manually verify the result.
#

# A simple computation: z = 3x^2 + 1
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = 3 * y + 1

# autograd computes dz/dx
z.backward()
print(f"x = {x.item()}")
print(f"z = 3x² + 1 = {z.item()}")
print(f"Autograd:  dz/dx = {x.grad.item()}")
print(f"Manual:    dz/dx = 6x = {6 * x.item()}")

######################################################################
# The chain rule step by step:
#
# .. math::
#    \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
#    = 3 \cdot 2x = 6x = 12
#
# Now let's verify this through an actual neural network layer.
# Note: ``backward()`` requires a scalar, so we sum the output to
# produce a single loss value.
#

x = torch.tensor([[2.0, 1.0]])
W = torch.tensor([[0.5, -0.3],
                   [0.8,  0.2]], requires_grad=True)
b = torch.tensor([0.1, -0.1], requires_grad=True)

# Forward: y = x @ W^T + b (same as nn.Linear)
y = x @ W.T + b
loss = y.sum()  # backward() needs a scalar; sum is the simplest way

loss.backward()

print(f"\nForward pass:")
print(f"  x    = {x.data}")
print(f"  y    = {y.data}")
print(f"  loss = {loss.item()}")

print(f"\nAutograd gradients:")
print(f"  dL/dW = {W.grad}")
print(f"  dL/db = {b.grad}")

######################################################################
# Let's verify manually. Since ``loss = sum(x @ W^T + b)``:
#
# - Each row of ``dL/dW`` equals ``x``, because each output element
#   depends on the same input multiplied by its weight row.
# - ``dL/db`` is all ones, because the sum passes a gradient of 1
#   to each bias element.
#

manual_grad_W = torch.ones(2, 1) @ x  # shape: (2, 1) @ (1, 2) -> (2, 2)
manual_grad_b = torch.ones(2)

print(f"\nManual verification:")
print(f"  dL/dW = {manual_grad_W}")
print(f"  dL/db = {manual_grad_b}")
print(f"  W match: {torch.allclose(W.grad, manual_grad_W)}")
print(f"  b match: {torch.allclose(b.grad, manual_grad_b)}")


######################################################################
# Part 4: Inspecting the Computational Graph
# --------------------------------------------
#
# During the forward pass, PyTorch records every operation in a
# directed acyclic graph. Each tensor stores a reference to the
# function that created it in its ``grad_fn`` attribute. Leaf
# tensors (created directly, not by an operation) have
# ``grad_fn = None``.
#

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = 3 * y + 1

print(f"x.grad_fn = {x.grad_fn}")       # None — leaf tensor
print(f"y.grad_fn = {y.grad_fn}")       # PowBackward0
print(f"z.grad_fn = {z.grad_fn}")       # AddBackward0

######################################################################
# Each ``grad_fn`` knows how to compute its local derivative.
# We can walk the graph to see the full chain that ``backward()``
# will traverse:
#


def trace_graph(tensor, indent=0):
    """Walk the computational graph from output to inputs."""
    if tensor.grad_fn is not None:
        print(" " * indent + str(tensor.grad_fn))
        for child, _ in tensor.grad_fn.next_functions:
            if child is not None:
                trace_graph_node(child, indent + 2)


def trace_graph_node(node, indent=0):
    """Recursively print graph nodes."""
    print(" " * indent + str(node))
    for child, _ in node.next_functions:
        if child is not None:
            trace_graph_node(child, indent + 2)


print(f"\nComputational graph for z = 3x² + 1:")
trace_graph(z)

######################################################################
# Reading bottom-up: ``AccumulateGrad`` is the leaf (``x``),
# ``PowBackward0`` computes :math:`x^2`, ``MulBackward0`` computes
# :math:`3 \cdot x^2`, and ``AddBackward0`` computes :math:`+1`.
#
# When ``backward()`` runs, it walks this graph in reverse, calling
# each node's derivative function and multiplying the results — the
# chain rule, automated.
#
# Two operations that affect this graph in practice:
#

# torch.no_grad() stops graph construction (saves memory at inference)
with torch.no_grad():
    y_no_graph = x ** 2
print(f"\nWith torch.no_grad():  grad_fn = {y_no_graph.grad_fn}")

# .detach() removes a tensor from the graph
y_detached = (x ** 2).detach()
print(f"With .detach():        grad_fn = {y_detached.grad_fn}")


######################################################################
# Part 5: Gradient Descent — What ``optimizer.step()`` Does
# ----------------------------------------------------------
#
# The optimizer updates each parameter by moving it in the direction
# that reduces the loss:
#
# .. math::
#    \theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot
#    \nabla_\theta L
#
# where :math:`\eta` is the learning rate. For SGD, this is literal
# subtraction. Let's verify by comparing parameter values before
# and after a single step.
#

torch.manual_seed(0)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Snapshot the weight before stepping
old_weight = model.weight.data.clone()
old_bias = model.bias.data.clone()

# One forward-backward pass
x = torch.tensor([[1.0]])
target = torch.tensor([[3.0]])
prediction = model(x)
loss = nn.functional.mse_loss(prediction, target)

optimizer.zero_grad()
loss.backward()

# The gradient that backward() computed
print(f"Weight gradient: {model.weight.grad.item():.4f}")
print(f"Bias gradient:   {model.bias.grad.item():.4f}")

optimizer.step()

# Verify: new = old - lr * grad
manual_weight = old_weight - 0.1 * model.weight.grad
manual_bias = old_bias - 0.1 * model.bias.grad

print(f"\nWeight after step:  {model.weight.data.item():.4f}")
print(f"Manual computation: {manual_weight.item():.4f}")
print(f"Match: {torch.allclose(model.weight.data, manual_weight)}")

######################################################################
# ``optimizer.step()`` is subtraction. That is all SGD does — read
# the gradient, scale it by the learning rate, subtract it from the
# parameter.
#
# Now let's watch this process converge. We train a single linear
# layer to learn the function :math:`y = 3x + 1`:
#

torch.manual_seed(0)

# Training data: y = 3x + 1
X = torch.linspace(0, 1, 50).unsqueeze(1)
y_true = 3 * X + 1

# Fresh model and optimizer
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print(f"Before training:  W = {model.weight.item():.4f}, "
      f"b = {model.bias.item():.4f}")

for epoch in range(200):
    prediction = model(X)
    loss = nn.functional.mse_loss(prediction, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}  |  Loss: {loss.item():.6f}  |  "
              f"W: {model.weight.item():.4f}  |  "
              f"b: {model.bias.item():.4f}")

print(f"Target:           W = 3.0000, b = 1.0000")


######################################################################
# Part 6: Putting It Together
# ----------------------------
#
# Every training loop maps to three branches of mathematics.
# Here is the complete picture in a single block:
#

torch.manual_seed(42)

# Data
X = torch.randn(100, 3)
y = X @ torch.tensor([2.0, -1.0, 0.5]) + 0.3  # true relationship

# Model and optimizer
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(300):
    # STEP 1: Forward pass — LINEAR ALGEBRA
    # model(X) computes X @ W^T + b
    prediction = model(X).squeeze()

    # STEP 2: Compute loss — DISTANCE METRIC
    # MSE measures how far predictions are from truth
    loss = nn.functional.mse_loss(prediction, y)

    # STEP 3: Backward pass — CALCULUS (chain rule)
    # Traverses the computational graph in reverse,
    # computing dL/dW and dL/db at every node
    optimizer.zero_grad()
    loss.backward()

    # STEP 4: Update parameters — OPTIMIZATION (gradient descent)
    # W_new = W_old - lr * dL/dW
    optimizer.step()

print(f"Learned weights: {model.weight.data.squeeze().tolist()}")
print(f"True weights:    [2.0, -1.0, 0.5]")
print(f"Learned bias:    {model.bias.item():.4f}")
print(f"True bias:       0.3000")

######################################################################
# The model recovered the true weights and bias from data alone,
# using nothing but matrix multiplication, the chain rule, and
# subtraction.
#
# Summary
# -------
#
# .. list-table::
#    :header-rows: 1
#    :widths: 30 35 35
#
#    * - Training Step
#      - Mathematics
#      - PyTorch API
#    * - Forward pass
#      - Matrix multiplication: :math:`y = xW^T + b`
#      - ``model(x)`` via ``nn.Linear``
#    * - Loss computation
#      - Distance metric: :math:`L = \frac{1}{n}\sum(y - t)^2`
#      - ``nn.functional.mse_loss``
#    * - Backward pass
#      - Chain rule: :math:`\frac{\partial L}{\partial W} =
#        \frac{\partial L}{\partial y} \cdot
#        \frac{\partial y}{\partial W}`
#      - ``loss.backward()``
#    * - Parameter update
#      - Gradient descent: :math:`W \leftarrow W - \eta \nabla L`
#      - ``optimizer.step()``
#
# When training fails, this table becomes your debugging map:
#
# - **Loss not decreasing?** Print the gradients after ``backward()``.
#   Near-zero gradients deep in the network mean vanishing gradients.
#   Enormous gradients mean they are exploding.
#
# - **Model not learning?** Compare parameters before and after
#   ``step()``. If they are not changing, check your learning rate
#   and verify ``zero_grad()`` is in the right place.
#
# - **Unexpected outputs?** Print intermediate activations during
#   the forward pass. Check for dead ReLU neurons (all-zero outputs
#   after activation).
#
# The mathematics is not behind the magic. The mathematics *is* the
# machine. And PyTorch lets you watch it run.
#
# Further Reading
# ----------------
#
# - `Autograd Mechanics <https://pytorch.org/docs/stable/notes/autograd.html>`_
# - `torch.nn Tutorial <https://pytorch.org/tutorials/beginner/nn_tutorial.html>`_
# - `Optimizing Model Parameters <https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>`_
#
