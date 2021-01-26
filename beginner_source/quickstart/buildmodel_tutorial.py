"""

`Learn the Basics <quickstart_tutorial.html>`_ >
`Tensors <tensor_tutorial.html>`_ > 
`Datasets & DataLoaders <dataquickstart_tutorial.html>`_ >
`Transforms <transforms_tutorial.html>`_ >
**Build Model** >
`Autograd <autograd_tutorial.html>`_ >
`Optimization <optimization_tutorial.html>`_ >
`Save & Load Model <saveloadrun_tutorial.html>`_

Build the Neural Network Model
===================

"""

#################################################################
#
# Now that we have loaded and transformed the data, we can build the neural network model. 
# Neural network consists of a number of layers and PyTorch `torch.nn <https://pytorch.org/docs/stable/nn.html>`_ namespace provides predefined layers 
# that helps us build the model.
# 
# The most common way to define a neural network is to use a class inherited 
# from `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_. 
# It provides great parameter management across all nested submodules, which gives us more 
# flexibility, because we can construct layers of any complexity, including ones with shared weights. 
# 
# In the below example, for our FashionMNIST image dataset, we will create a dense multi-layer network.  
# Lets break down the steps to build this model below.
# 

#############################################
# Import the Packages
# --------------------------
#

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx as onnx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#############################################
# Get Device for Training
# -----------------------
# We want to be able to train our model on both CPU and GPU, if it is available. It is common practice to 
# define a variable ``device`` which will designate the device we will be training on.
# We check to see if `torch.cuda <https://pytorch.org/docs/stable/notes/cuda.html>`_ 
# is available to use the GPU, else we will use the CPU. 
#
# Example:
#

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

##############################################
# Define the Class
# -------------------------
#
# First we define the `NeuralNetwork` class which inherits from ``nn.Module``, the base class for 
# building all neural network modules in PyTorch. We use the ``__init__`` function to define and initialize the NN layers that will be then called in the module's ``forward`` function.
# Then we call the ``NeuralNetwork`` class and assign the device. When training 
# the model we will call ``model`` and pass the data (x) into the forward function and 
# through each layer of our network.
#
#

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 512)
        self.layer2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return F.softmax(x, dim=1)

model = NeuralNetwork().to(device)
print(model)

input = torch.rand(5, 28, 28)

# equivalent to model.forward(input)
model(input)

##############################################
# Model Layers
# -------------------------
#
# Lets break down each layer in the FashionMNIST model. To illustrate it, we 
# will take a sample minibatch of 100 images of size 28x28 and see what happens to it as 
# we pass it through the network. The code in the sections below would essentially explain 
# what happens inside the ``forward`` method of our ``NeuralNetwork`` class. 
#

input_image = torch.rand(100,28,28)
print(input_image.size())

##################################################
# nn.Flatten
# -----------------------------------------------
#
# First we call `nn.Flatten  <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>`_  to reduce tensor dimensions to one.
#
# In our case, flatten keeps the minibatch dimension, but two image dimensions are 
# reduced to one:

flatten = nn.Flatten(start_dim=1, end_dim=2)
flat_image = flatten(input_image)
print(flat_image.size())

##############################################
# nn.Linear 
# -------------------------------
#
# Now that we have flattened our tensor dimension we will pass our data through a `linear layer <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_. The linear layer is 
# a module that applies a linear transformation on the input using it's stored weights and biases.
#

layer1 = nn.Linear(in_features=28*28, out_features=512)
hidden1 = layer1(flat_image)
print(hidden1.size())

#################################################
# Activation Functions
# -------------------------
#
# In between layers of a neural network, we need to put non-linear activation functions, 
# such as `nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_ (which is often 
# used in between hidden layers) or `nn.Softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_, 
# which turns the output of the network into probabilities by rescaling values between 0 and 1, and all sum to one.
#

layer2 = nn.Linear(512,512)
output = nn.Linear(512,10)
hidden2 = layer2(F.relu(hidden1))
print('Hidden 2 output size =',hidden2.size())
z = output(F.relu(hidden2))
out = F.softmax(z)
print('Output size =',out.size())

#################################################
# Parameter Tracking
# -------------------------
#
# The main reason to put all code inside a class inherited from ``nn.Module`` is to 
# utilize **parameter tracking**. Most of the layers inside a neural network, 
# in our case all linear layers, have associated weights and biases that need to 
# be adjusted during training. ``nn.Module`` automatically tracks all fields defined 
# inside the class, and makes all parameters accessible using ``parameters()`` 
# or ``named_parameters()`` methods. Let's have a look at the first two parameters of 
# our neural network that were defined in the beginning of this section:
#

print(list(model.named_parameters())[0:2])
