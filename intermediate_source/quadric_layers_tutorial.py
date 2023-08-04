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

In this tutorial, a simple classification application on the MNIST dataset  using a non convolutional model is presented.
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


