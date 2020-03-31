# -*- coding: utf-8 -*-
"""
Autograd: Automatic Differentiation
===================================

Deep learning uses artificial neural networks, which are computing systems
composed of many layers of interconnected units. By passing data through these
interconnected units, a neural network, or model, is able to learn how to 
approximate the computations required to transform that data input into some 
output. We will learn how to fully construct neural networks in the next section 
of this tutorial.

Before we do so, it is important that we introduce some concepts.

In the training phase, models are able to increase their accuracy through gradient decent.
In short, gradient descent is the process of minimizing our loss (or error) by tweaking the 
weights and biases in our model.

This process introduces the concept "automatic differentiation". Automatic differentiation
helps us calculate the derivatives of our loss function to know how much we should
adjust those weights and biases to decrease loss. There are various methods on how to 
perform automatic differentiation, one of the most popular being backpropagation (finding the
loss based on the previous EPOCH, or iteration). 

In the ``autograd`` package in PyTorch, we have access to automatic differentiation for
all operations on tensors. It is a define-by-run framework, which means, for example,
that your backpropagation is defined by how your code is run and that every iteration can be different.


Getting Started
---------------

This entire process is simplified using PyTorch. In this section of the tutorial, we will
see examples of gradient descent, automatic differentiation, and backpropagation in PyTorch.
This will which will introduce to training our first neural network in the following section.

Tensor
^^^^^^^

``torch.Tensor`` is the central class of PyTorch. When you create a tensor, 
if you set its attribute ``.requires_grad`` as ``True``, the package tracks
all operations on it. When your computation is finished, you can then call
``.backward()`` on the tensor, and have all the gradients computed automatically.
The gradient for this tensor will be accumulated into ``.grad`` attribute.

To stop a tensor from tracking history, you can call ``.detach()`` to detach
it from the computation history, and to prevent future computation from being
tracked.

To prevent tracking history (and using memory), you can also wrap the code block
in ``with torch.no_grad():``. This can be particularly helpful when evaluating a
model, where a it may have trainable parameters with
``requires_grad=True``, but whose gradients are not necessary.

Before we dive into code, there’s one more very important class when implementing
``autograd`` - a ``Function``.

``Tensor`` and ``Function`` are interconnected. Together, they build an acyclic
graph that encodes a complete history of computation. Programmatically, this means 
that each tensor has a ``.grad_fn`` attribute, referencing a ``Function`` that created
the tensor (except for when tensors are explicitly created by the developer - their
``grad_fn`` is ``None``).

Let's see this in action. As always, we will first need to import PyTorch into our program.

"""

import torch

###############################################################
# Create a tensor and set ``requires_grad=True``. This will track all operations on the tensor.

x = torch.ones(2, 2, requires_grad=True)
print(x)

###############################################################
# Now, perform a simple tensor operation:

y = x + 2
print(y)

###############################################################
# ``grad_fn`` points to the last operation performed on the tensor.
# In this case it is an addition function, so its value is ``AddBackward0``.

print(y.grad_fn)

###############################################################
# Perform more operations on ``y``:

z = y * y * 3
out = z.mean()

print(z, out)

################################################################
# Notice how the output of ``z`` has a gradient function of ``MulBackward0`` and 
# the output of ``out`` has ``MeanBackward0``. These outputs give us insight to 
# the history of the tensor, where ``MulBackward`` indicates that a multiplication
# operation was performed, and ``MeanBackward`` indicates the mean was calculated
# previously on this tensor.
#
# Now if we perform a multiplication operation on ``z`` again:

print (z.median())

################################################################
# You can see that the ``grad_fn`` was updated to ``MedianBackward``.
#
# You can change the ``requires_grad`` flag in place using the
# ``.requires_grad_( ... )`` function.
# The input flag defaults to ``False`` if not manually specified.

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

###############################################################
# Gradients
# ---------
# Now that we understand how these operations work with tensors, let's practice with
# backpropagation in PyTorch, using ``.backward()``.
#
# To compute the derivative, you can call ``.backward()`` on
# a ``tensor``. If ``tensor`` is a scalar (i.e. it holds a one element
# data), you don’t need to specify any arguments to ``backward()``. If it has more elements,
# you need to specify a ``gradient`` argument, which will be a tensor of matching shape.
# 
# ``loss.backward()`` computes the derivative of the loss (``dloss/dx``) for every 
# parameter ``x`` where ``requires_grad = True``. These are then accumulated into ``x.grad``.
#
# In our examples, ``out`` is the loss for ``x``. Because ``out`` contains a single scalar,
# ``out.backward()`` is equivalent to ``out.backward(torch.tensor(1.))``.

print (x)
print (out)
out.backward()

# Print the gradients d(out)/dx
print(x.grad)

###############################################################
# Let's break this down mathematically for further understanding. 
#
# You should have got a matrix of ``4.5``. Let’s call ``out``
# *Tensor* “:math:`o`”.
# We have that :math:`o = \frac{1}{4}\sum_i z_i`,
# :math:`z_i = 3(x_i+2)^2` and :math:`z_i\bigr\rvert_{x_i=1} = 27`.
# Therefore,
# :math:`\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)`, hence
# :math:`\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5`.

###############################################################
# If you have a vector valued function :math:`\vec{y}=f(\vec{x})`,
# then the gradient of :math:`\vec{y}` with respect to :math:`\vec{x}`
# is a Jacobian matrix:
#
# .. math::
#   J=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)
#
# Generally speaking, ``torch.autograd`` is an engine for computing
# vector-Jacobian product. That is, given any vector
# :math:`v=\left(\begin{array}{cccc} v_{1} & v_{2} & \cdots & v_{m}\end{array}\right)^{T}`,
# compute the product :math:`v^{T}\cdot J`. If :math:`v` happens to be
# the gradient of a scalar function :math:`l=g\left(\vec{y}\right)`,
# that is,
# :math:`v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}`,
# then by the chain rule, the vector-Jacobian product would be the
# gradient of :math:`l` with respect to :math:`\vec{x}`:
#
# .. math::
#   J^{T}\cdot v=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)\left(\begin{array}{c}
#    \frac{\partial l}{\partial y_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial y_{m}}
#    \end{array}\right)=\left(\begin{array}{c}
#    \frac{\partial l}{\partial x_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial x_{n}}
#    \end{array}\right)
#
# (Note that :math:`v^{T}\cdot J` gives a row vector which can be
# treated as a column vector by taking :math:`J^{T}\cdot v`.)
#
# This characteristic of vector-Jacobian product makes it very
# convenient to feed external gradients into a model that has
# non-scalar output.

###############################################################
# Check out this example to see vector-Jacobian product in practice:

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

###############################################################
# Now in this case ``y`` is no longer a scalar. ``torch.autograd``
# could not compute the full Jacobian directly, but if we just
# want the vector-Jacobian product, simply pass the vector to
# ``backward`` as argument:

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

###############################################################
# As mentioned previously, you can stop ``autograd`` from tracking history on tensors
# (via ``.requires_grad=True``) either by wrapping the code block in
# ``with torch.no_grad():``

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

###############################################################
# Or by using ``.detach()``, which yields a new tensor with the same
# content that does not require gradients:

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

###############################################################
# With this understanding of how ``autograd`` works in PyTorch, let's move on to 
# the next section to construct our neural networks.
#
# **Read Later:**
#
# For more information about ``autograd.Function``, check out our documentation:
# https://pytorch.org/docs/stable/autograd.html#function
