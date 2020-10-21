# -*- coding: utf-8 -*-
"""
A Gentle Introduction to Autograd
---------------------------------

Autograd is PyTorch’s automatic differentiation engine that powers
neural network training. In this section, you will get a conceptual
understanding of how autograd works under the hood.

Background
~~~~~~~~~~
Neural networks (NNs) are a collection of nested functions that are
executed on some input data. These functions are defined by *parameters*
(consisting of weights and biases), which in PyTorch are stored in
tensors.

Training a NN happens in two steps:

**Forward Propagation**: In forward prop, the NN makes its best guess
about the correct output. It runs the input data through each of its
functions to make this guess.

**Backward Propagation**: In backprop, the NN adjusts its parameters
proportionate to the error in its guess. It does this by traversing
backwards from the output, collecting the derivatives of the error with
respect to the parameters of the functions (*gradients*), and optimizing
the parameters using **gradient descent**. For a more detailed walkthrough
of backprop, check out this `video from
3Blue1Brown <https://www.youtube.com/watch?v=tIeHLnjs5U8>`__.

Most deep learning frameworks use automatic differentiation for
backprop; in PyTorch, it is handled by Autograd.


Usage in PyTorch
~~~~~~~~~~~
Backward propagation can be kicked off by calling ``.backward()`` on the error tensor.
This collects the gradients for each parameter in the model.
"""

import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
prediction = model(data) # forward pass
loss = (prediction - labels).sum()
loss.backward() # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent

######################################################################
# At this point, you have everything you need to build your neural network.
# The below sections detail the workings of autograd - feel free to skip them.
#


######################################################################
# --------------
#


######################################################################
# Differentiation in Autograd
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The ``requires_grad`` flag lets autograd know
# if we need gradients w.r.t. these tensors. If it is ``True``, autograd
# tracks all operations on them.


import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2
print(Q)


######################################################################
# ``a`` and ``b`` can be viewed as parameters of an NN, with ``Q``
# analogous to the error. In training we want gradients of the error
# w.r.t. parameters, i.e.
#
# .. math::
#
#
#    \frac{\partial Q}{\partial a} = 9a^2
#
# .. math::
#
#
#    \frac{\partial Q}{\partial b} = -2b
#
# Since ``Q`` is a vector, we pass a ``gradient`` argument in
# ``.backward()``.
#
# ``gradient`` is a tensor of the same shape. Here it represents the
# gradient of Q w.r.t. itself, i.e.
#
# .. math::
#
#
#    \frac{dQ}{dQ} = 1
#

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# check if autograd's gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)


######################################################################
# Optional Reading - Vector Calculus in Autograd
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Mathematically, if you have a vector valued function
# :math:`\vec{y}=f(\vec{x})`, then the gradient of :math:`\vec{y}` with
# respect to :math:`\vec{x}` is a Jacobian matrix :math:`J`:
#
# .. math::
#
#
#      J
#      =
#       \left(\begin{array}{cc}
#       \frac{\partial \bf{y}}{\partial x_{1}} &
#       ... &
#       \frac{\partial \bf{y}}{\partial x_{n}}
#       \end{array}\right)
#      =
#      \left(\begin{array}{ccc}
#       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#       \end{array}\right)
#
# Generally speaking, ``torch.autograd`` is an engine for computing
# vector-Jacobian product. That is, given any vector :math:`\vec{v}`, compute the product
# :math:`J^{T}\cdot \vec{v}`
#
# If :math:`v` happens to be the gradient of a scalar function
#
# .. math::
#
#
#    l
#    =
#    g\left(\vec{y}\right)
#    =
#    \left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}
#
# then by the chain rule, the vector-Jacobian product would be the
# gradient of :math:`l` with respect to :math:`\vec{x}`:
#
# .. math::
#
#
#      J^{T}\cdot \vec{v}=\left(\begin{array}{ccc}
#       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#       \end{array}\right)\left(\begin{array}{c}
#       \frac{\partial l}{\partial y_{1}}\\
#       \vdots\\
#       \frac{\partial l}{\partial y_{m}}
#       \end{array}\right)=\left(\begin{array}{c}
#       \frac{\partial l}{\partial x_{1}}\\
#       \vdots\\
#       \frac{\partial l}{\partial x_{n}}
#       \end{array}\right)
#
# This characteristic of vector-Jacobian product is what we use in the above example;
# ``external_grad`` represents :math:`\vec{v}`.
#



######################################################################
# Computational Graph
# ~~~~~~~~~~~~~~~~~~~
#
# Conceptually, autograd keeps a record of data (tensors) & all executed
# operations (along with the resulting new tensors) in a directed acyclic
# graph (DAG) consisting of
# `Function <https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function>`__
# objects. In this DAG, leaves are the input tensors, roots are the output
# tensors. By tracing this graph from roots to leaves, you can
# automatically compute the gradients using the chain rule.
#
# In a forward pass, autograd does two things simultaneously: \* run the
# requested operation to compute a resulting tensor, and \* maintain the
# operation’s *gradient function* in the DAG. This is stored in the
# resulting tensor’s .\ ``grad_fn`` attribute.
#
# The backward pass kicks off when ``.backward()`` is called on the DAG
# root. Autograd then \* computes the gradients from each ``.grad_fn``, \*
# accumulates them in the respective tensor’s ``.grad`` attribute, and \*
# using the chain rule, propagates all the way to the leaf tensors.
#
# .. Note::
#   **Autograd DAGs are dynamic in PyTorch**
#   An important thing to note is that the graph is recreated from scratch; after each
#   ``.backward()`` call, autograd starts populating a new graph. This is
#   exactly what allows you to use control flow statements in your model;
#   you can change the shape, size and operations at every iteration if
#   needed. Autograd does not need you to encode all possible paths
#   beforehand.
#


######################################################################
# Exclusion from the DAG
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Autograd tracks operations on all tensors which have their
# ``requires_grad`` flag set to ``True``. For tensors that don’t require
# gradients, setting this attribute to ``False`` excludes it from the
# gradient computation DAG and increases efficiency.
#
# The output tensor of an operation will require gradients even if only a
# single input tensor has ``requires_grad=True``.
#

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print("Does `a` require gradients?")
print(a.requires_grad==True)
b = x + z
print("Does `b` require gradients?")
print(b.requires_grad==True)


######################################################################
# This is especially useful when you want to freeze part of your model
# (for instance, when you’re fine-tuning a pretrained model), or you know
# in advance that you’re not going to use gradients w.r.t. some
# parameters.
#

from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of nn.Module instances have requires_grad=True by default
model.fc = nn.Linear(512, 100)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)


######################################################################
# The same functionality is available as a context manager in
# `torch.no_grad() <https://pytorch.org/docs/stable/generated/torch.no_grad.html>`__
#

######################################################################
# --------------
#

######################################################################
# Further readings:
# ~~~~~~~~~~~~~~~~~~~
#
# -  `In-place operations & Multithreaded Autograd <https://pytorch.org/docs/stable/notes/autograd.html>`__
# -  `Example implementation of reverse-mode autodiff <https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC>`__
