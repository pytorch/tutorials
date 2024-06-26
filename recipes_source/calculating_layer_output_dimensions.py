# -*- coding: utf-8 -*-

"""
Calculating Output Dimensions for Convolutional and Pooling Layers
==================================================================

**Author:** `Logan Thomas <https://github.com/loganthomas>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to transition from convolution and pooling layers to linear layers in a model
      * How to manually calculate the output dimensions after applying a convolution or pooling layer
      * How to print the shape of internal tensors for inspecting dimensionality changes in a model
      * How to use the ``torchinfo`` package to show output dimensions for all layers in a model
"""

#########################################################################
# Overview
# --------
#
# Suppose you are creating a `Convolutional Neural Network <https://cs231n.github.io/convolutional-networks/>`__ to classify images.
# You know the shape of your images and have an idea of how you want to structure the network, so
# you start to build. After two or three levels of convolutional layers, with a few pooling layers sprinkled in,
# you've realized you've lost sense of the dimensionality of the data.
#
# When it comes time to flatten your tensors, and transition from convolution and pooling layers to linear layers in your model,
# you'll need to know the correct number of input features (``in_features``) to provide the ``torch.nn.Linear()`` layer.
#
# This short recipe tutorial walks you through three different approaches
# for finding out how to make this transition from convolutions to linear layers as smooth as possible.

#########################################################################
# Early Stages of Model Development
# ---------------------------------
# Let's start by dropping into the scenario described in the overview.
#
# Assume we will train on images from the `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`__.
# That is, our data will comprise of images that are 28x28 pixels, have a single channel, and will be
# classified as a handwritten digit between 0-9 (10 possible options).
#
# With this information, we can start to build our neural network

import torch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)


######################################################################
# So far, so good.
#
# Now it's time to transition from the convolutional space to the linear space.
# But, what should the number of input features be for the first linear layer?
# I know that my previous convolutional layer had an output of ``64``, so that has to be included somehow.
# I'm uncertain of the other multiplication terms, so I'll grab a round number like ``10``.
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 128)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return x


######################################################################
# Let's see if I guessed correctly:

model = Net()

# Simulate a single data point from the dataset
x = torch.ones(1, 1, 28, 28)  # batch_size, channels, height, width
try:
    _ = model(x)
except RuntimeError as e:
    print(f"Error occured: {e}")
######################################################################
# Looks like something is off.
# There are shapes that don't align in my model somewhere -- most likely the first linear layer that transitions from convolutional space to linear space.
#
# Let's explore three different approaches for resolving this issue.

######################################################################
# Approach 1: Calculate the output shapes of each layer manually
# ----------
#
# The `torch.nn.Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`__ and
# `torch.nn.MaxPool2d <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>`__
# provide the same mathematical equation for calculating the output shape after employing these layers.
#
# We can use this equation to calculate the output shapes for all our convolution and pooling layers
# and trace the dimensionality shifts as data flows through our model:
import math


def calc_shape(
    c_in,
    h_in,
    w_in,
    c_out=None,
    kernel=(3, 3),
    padding=(0, 0),
    dilation=(1, 1),
    stride=(1, 1),
):
    """
    Helper function to determine output shape after convolution or pool layer.

    Parameter
    ---------
    c_in : int
        Number of channels in the input.
    h_in : int
        Number of rows in the input (height).
    w_in : int
        Number of columns in the input (width).
    c_out : Optional[int]
        Number of channels in the output. If None, uses c_in.
    kernel : Optional[tuple(int, int)]
        Size of the convolving kernel.
    padding : Optional[tuple(int, int)]
        Padding added to all four sides of the input.
    dilation : Optional[tuple(int, int)]
        Spacing between kernel elements.
    stride : Optional[tuple(int, int)]
        Stride of the convolution.
    """
    c_out = c_in if c_out is None else c_out
    h_out = math.floor(
        ((h_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0]) + 1
    )
    w_out = math.floor(
        ((w_in + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1]) + 1
    )
    return c_out, h_out, w_out


######################################################################
# With this helper function in hand, we can trace pseudo-data through
# our model and see what the correct dimensionality should be for our
# first linear layer:

# Start with image that is 28x28 pixels with 1 channel
input_h_w = (1, 28, 28)
print(f"Input shape (c, h, w) : {input_h_w}")

# Simulated Conv2d with 32 channels output
conv1_out = calc_shape(*input_h_w, c_out=32)
print(f"Post Conv2d.0 shape   : {conv1_out}")

# Simulated Conv2d with 64 channels output
conv2_out = calc_shape(*conv1_out, c_out=64)
print(f"Post Conv2d.1 shape   : {conv2_out}")

# Simulated MaxPool2d with 2x2 kernel
# (the default value of stride is the kernel_size)
pool_out = calc_shape(*conv2_out, kernel=(2, 2), stride=(2, 2))
print(f"Post MaxPool2d.0 shape: {pool_out}")


######################################################################
# Looks like our answer is ``64 * 12 * 12``! Let's give it a try:
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * 12 * 12, 128)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return x


model = Net()
x = torch.ones(1, 1, 28, 28)  # batch_size, channels, height, width
try:
    _ = model(x)
except RuntimeError as e:
    print(f"Error occured: {e}")

######################################################################
# Success! No error was reported.
# Now, we can add our final touches to the network and be on our way:


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * 12 * 12, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


######################################################################
# Approach 2: Add temporary internal print statements to report the data.shape
# ----------------------------------------------------------------------------
#
# Sometimes, simple print statements can go a long way.
# With this approach, we inject temporary print statements within the ``forward`` method
# to report the data shape as it passes through the model.
#
# Just remember to clean up afterwards!
class DebugNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        print(f"input shape: {x.shape}")
        x = self.relu(self.conv1(x))
        print(f"after conv1 shape: {x.shape}")
        x = self.relu(self.conv2(x))
        print(f"after conv2 shape: {x.shape}")
        x = self.pool(x)
        print(f"after pool shape: {x.shape}")


model = DebugNet()
x = torch.ones(1, 1, 28, 28)  # batch_size, channels, height, width
try:
    _ = model(x)
except RuntimeError as e:
    print(f"Error occured: {e}")


######################################################################
# As expected, we see that the shape of the data after the pooling layer is ``[1, 64, 12, 12]``.
# With this information, we can update our model, remove the print statements, and get on to training.
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * 12 * 12, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#####################################################################
# Approach 2 with the ``Sequential`` class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The approach injecting print statements is a little different when
# using ``torch.nn.Sequential`` model class. Here, you'd have to break
# up your implementation to separate the "features" (convolution/pooling layers)
# from the "classifier" (linear layers) for the cleanest result.
#
# Notice the shape of the data is no longer printed layer by layer,
# but now printed as the start and at the transition point.


class DebugNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 12 * 12, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        print(f"input shape: {x.shape}")
        x = self.features(x)
        print(f"post features shape: {x.shape}")
        x = self.classifier(x)
        return x


model = DebugNet2()
x = torch.ones(1, 1, 28, 28)  # batch_size, channels, height, width
try:
    _ = model(x)
except RuntimeError as e:
    print(f"Error occured: {e}")

######################################################################
# Approach 3: Use the ``torchinfo`` package
# -----------------------------------------
#
# If you like to have a clean model summary that includes these output shapes,
# checkout the `torchinfo package <https://github.com/TylerYep/torchinfo>`__.
#
# Torchinfo provides information complementary to what is provided by ``print(moel)`` in PyTorch,
# similar to Tensorflow's ``model.summary()`` API to view the visualization of the model,
# which is helpful while debugging your network.
#
# Notice that ``torchinfo.summary()`` provides a nice model summary that includes the output shapes
# calculated for you (given you provide the correct ``input_shape``):
import torchinfo


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)


model = Net()
# If running from a notebook, use print(torchinfo.summary(model, input_size=(1, 1, 28, 28)))
torchinfo.summary(model, input_size=(1, 1, 28, 28))


######################################################################
# Again, from this output, we can see the answer we need is ``64 * 12 * 12``.
# We can update our model and move to training:
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * 12 * 12, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


######################################################################
# Conclusion
# ----------
#
# When building Convolutional Neural Networks,
# it can be hard to keep track of your data dimensionality as it flows through the model.
# New and seasoned practitioners have all encountered an unexpected ``RuntimError`` reporting
# two matrices cannot be multiplied due to a shape mismatch.
#
# This recipe tutorial explored three approaches for tracking your data's shape as it moves through your model:
#   (1) Manually calculating the output shapes of each layer
#   (2) Adding temporary internal print statements to report the data.shape of tensors
#   (3) Using the ``torchinfo`` package to report a model summary that includes output shapes
#
# These approaches are most helpful when confused about how to transition from convolution or pooling layers to linear layers in a model.

######################################################################
# Further Reading
# ---------------
#
# * `torchinfo <https://github.com/TylerYep/torchinfo>`__: provides information complementary to what is provided by `print(model)` in PyTorch, similar to Tensorflow's model.summary() API.
