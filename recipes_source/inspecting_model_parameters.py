# -*- coding: utf-8 -*-

"""
Inspecting Model Parameters Tutorial
====================================

**Author:** `Logan Thomas <https://github.com/loganthomas>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to inspect a model's parameters using `.parameters()` and `.named_parameters()`
      * How to collect the trainable parameters of a model
      * How to use the `torchinfo` package (formerly `torch-summary`) to print a model summary
"""

#########################################################################
# Overview
# --------
#
# When building neural networks, it's helpful to be able to inspect
# parameters (model weights) at intermediate stages of development.
#
# This can help inform model architecture decisions, like how many
# neurons to put in a proceeding layer.
# Or, it can be used for debugging purposes to ensure each model's layer
# has the anticipated number of weights.
#
# Inspecting Parameters of a Simple Neural Network
# ------------------------------------------------
# Let's start with a simple example:
#
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print(model)

#########################################################################
# Layers inside a neural network are parameterized, i.e.
# have associated weights and biases that are optimized during training.
# Subclassing `nn.Module` automatically tracks all fields defined
# inside a model object, and makes all parameters accessible using a
# model’s `parameters()` or `named_parameters()` methods.
#
# To inspect the shape of the parameter's associated with each layer in the model,
# use `model.parameters()`:
print([param.shape for param in model.parameters()])

#########################################################################
# Sometimes, it's more helpful to be able to have a name associated with
# the parameters of each layer. Use `model.named_parameters()` to access
# the parameter name in addition to the shape:
for name, param in model.named_parameters():
    print(name, param.shape)

#########################################################################
# Notice that the parameters are collected from the `nn.Linear` modules
# specified in the network. Because the default behavior for `nn.Linear`
# is to include a bias term, the output shows both a `weight` and `bias`
# parameter for each of the `nn.Linear` modules.
#
# The shape of these parameters relate to the input shape (`in_features`)
# and output shape (`out_features`) specified in each of the model's layers.
#
# Take for example the first `nn.Linear(28*28, 512)` module specified:
layer = nn.Linear(28*28, 512)

for name, param in layer.named_parameters():
    print(name, param.size())

#########################################################################
# The first line from the printed `model.named_parameters()` section
# (`linear_relu_stack.0.weight torch.Size([512, 784])`) specifies
# the `weight` associated with this layer.
# The second line from the printed `model.named_parameters()` section
# (`linear_relu_stack.0.bias torch.Size([512])`) specifies
# the `bias` associated with this layer. The printed statements using `.named_parameters()`
# are *not* meant to report the original shapes of the model's **layers**
# but the shape of the **weights** (and/or **biases**) of the **parameters of the layers**.
# This can cause confusion for new practitioners since the shape of the weights
# seem to invert the input shape and output shape specified for the Linear layers.
# These weights will be **transposed** during the matrix
# multiplication process when the model makes a prediction (as specified in the `nn.Linear <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`__
# docstring).

#########################################################################
# There is also a helpful `.numel()` method that can be used to gather
# the number of elements that are in each model parameter:
for name, param in model.named_parameters():
    print(f'{name=}, {param.size()=}, {param.numel()=}')

#########################################################################
# The number of elements for each parameter is calculated by taking
# the product of the entries of the Size tensor.
# The `.numel()` can be used to find all the parameters in a model by taking
# the sum across all the layer parameters:
print(f'Total model params: {sum(p.numel() for p in model.parameters()):,}')

#########################################################################
# Sometimes, only the *trainable* parameters are of interest.
# Use the `requires_grad` attribute to collect only those parameters
# that require a gradient to be computed (i.e. those parameters that will be optimized during model training):
print(f'Total model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

#########################################################################
# Since all the model weights currently require a gradient, the number
# of trainable parameters are the same as the total number of model
# parameters. Simply for educational purposes, parameters can be frozen
# to show a difference in count. Below, the first linear layer's `weight` parameters are frozen
# by setting `requires_grad=False` which will result in the trainable
# parameters count having 401,408 less parameters.
for name, param in model.named_parameters():
    if name == 'linear_relu_stack.0.weight':
        param.requires_grad = False
    print(f'{name=}, {param.size()=}, {param.numel()=}, {param.requires_grad=}')
print(f'Total model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
#########################################################################
# Inspecting Parameters of a Convolutional Neural Network
# -------------------------------------------------------
# These techniques also work for Convolutional Neural Networks:

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cnn_model = CNN()
print(cnn_model)
print('-'*72)
for name, param in cnn_model.named_parameters():
    print(f'{name=}, {param.size()=}, {param.numel()=}, {param.requires_grad=}')
print('-'*72)
print(f'Total model trainable params: {sum(p.numel() for p in cnn_model.parameters() if p.requires_grad):,}')

######################################################################
# As with the simple network example above, the number of elements per parameter
# is the product of the parameter size:
import numpy as np

for name, param in cnn_model.named_parameters():
    print(f'{name=}, {param.size()=}, {np.prod(param.size())=} == {param.numel()=}')

######################################################################
# For a more robust approach, consider using the `torchinfo package <https://github.com/TylerYep/torchinfo>`__ (formerly `torch-summary`).
# This package provides information complementary to what is provided by `print(model)` in PyTorch,
# similar to Tensorflow's `model.summary()` API to view the visualization of the model.
#
# Notice that the trainable parameters reported by `torchinfo` matches
# the manually gathered trainable parameters.
import torchinfo

# If running from a notebook, use print(torchinfo.summary(model))
torchinfo.summary(model)
print('-'*72)
print(f'Manually gathered model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
######################################################################
# There is one minor, but important, difference in the way `torchinfo` reports the number of parameters per layer.
# Notice that the `weight` and `bias` parameter counts are **combined**
# to report on the *total* number of parameters per layer.
# For example, the first linear layer of the `model` created in the
# "Inspecting Parameters of a Simple Neural Network" section has a
# `weight` parameter with `401,408` elements and a `bias` parameter
# with `512`. Combining these two yields a total
# of `401,920` (`401,408+512`) parameters for the layer -- which is
# equivalent to what the `torchinfo` summary showed.
#
# A similar report can be generated manually by summing parameters per layer:
from collections import defaultdict
layer_params = defaultdict(int)

for name, param in model.named_parameters():
    # combine weight and bias together using layer name
    # linear_relu_stack.0 = linear_relu_stack.0.weight + linear_relu_stack.bias
    layer_params[name.rsplit('.', 1)[0]] += param.numel()

for name, total_params in layer_params.items():
    print(f'{name=} {total_params=:,}')

######################################################################
# These approaches works for the Convolutional Neural Network as well:

# If running from a notebook, use print(torchinfo.summary(model))
torchinfo.summary(cnn_model)
print('-'*72)
print(f'Manually gathered model trainable params: {sum(p.numel() for p in cnn_model.parameters() if p.requires_grad):,}')
print('-'*72)
print('Manually generated total number of parameters per layer:')
cnn_layer_params = defaultdict(int)

for name, param in cnn_model.named_parameters():
    cnn_layer_params[name.rsplit('.', 1)[0]] += param.numel()

for name, total_params in cnn_layer_params.items():
    print(f'{name=} {total_params=:,}')

######################################################################
# Conclusion
# ----------
#
# Layers inside a neural network have associated weights and biases
# that are optimized during training. These parameters (model weights)
# are made accessible using a model’s `parameters()` or `named_parameters()`
# methods. Interacting with these parameters can help inform model
# architecture decisions or support model debugging.
#
# Further Reading
# ---------------
#
# * `torchinfo <https://github.com/TylerYep/torchinfo>`__: provides information complementary to what is provided by `print(model)` in PyTorch, similar to Tensorflow's model.summary() API.

