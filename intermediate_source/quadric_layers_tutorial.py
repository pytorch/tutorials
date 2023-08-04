"""
(beta) Quadric Layers
==================================================================

**Author**: `Dirk Roeckmann <https://github.com/diro5t>`_

Introduction
------------

Quadric layers introduce quadratic functions with second-order decision boundaries (quadric hypersurfaces)
and can be used as 100% drop-in layers for linear layers (torch.nn.Linear) and present a high-level means
to reduce overall model size.

In comparison to linear layers with n weights and 1 bias (if needed) per neuron, a quadric neuron consists of
2n weights (n quadratic weights and n linear weights) and 1 bias (if needed).
Although this means a doubling in weights per neuron, the more powerful decision boundaries per neuron lead 
in many applications to significantly less neurons per layer or even less layers and in total to less model parameters.

In this tutorial, a simple classification application on the MNIST dataset using a non convolutional model is presented.
"""

# imports
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim, Tensor
from torch.nn.parameter import Parameter, UninitializedParameter

import numpy as np
import matplotlib.pyplot as plt
import math

######################################################################
# 1. Load MNIST data
# -------------------
transf = transforms.Compose([transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,)),])

batch_size = 128

train_set = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transf)
test_set = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transf)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

img_iter = iter(train_loader)
images, labels = next(img_iter)

figure = plt.figure()
img_num = 60
for index in range(1, img_num + 1):
    plt.subplot(10, 6, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

######################################################################
# 2. Define the model
# -------------------
#
# Here we define the LSTM model architecture, following the
# `model <https://github.com/pytorch/examples/blob/master/word_language_model/model.py>`_
# from the word language model example.

######################################################################
# Conclusion
# ----------
#
# Quadric layers can easily be used to reduce model size in many applications just by replacing linear layers.
#
# Thanks for reading! Any feedback is highly appreciated. Just create an issue
# `here <https://github.com/pytorch/pytorch/issues>`.


