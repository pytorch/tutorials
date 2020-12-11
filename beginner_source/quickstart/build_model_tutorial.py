"""
Build the Neural Network
===================
"""

#################################################################
# Get Started Building the Model
# -----------------
#
# The data has been loaded and transformed we can now build the model. 
# We will leverage `torch.nn <https://pytorch.org/docs/stable/nn.html>`_ predefined layers that PyTorch has that can simplify our code.
# 
# In the below example, for our FashionMNIT image dataset, we are using a `Sequential` 
# container from class `torch.nn. Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ 
# that allows us to define the model layers inline. In the "Sequential" in-line model building format the ``forward()`` 
# method is created for you and the modules you add are passed in as a list or dictionary in the order that are they are defined.
# 
# Another way to bulid this model is with a class 
# using `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html)>`_ 
# A big plus with using a class that inherits ``nn.Module`` is better parameter management across all nested submodules.
# This gives us more flexibility, because we can construct layers of any complexity, including the ones with shared weights. 
# 
# Lets break down the steps to build this model below
# 

##########################################
# Inline nn.Sequential Example:
# ----------------------------
#

import os
import torch
import torch.nn as nn
import torch.onnx as onnx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# model
model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(classes)),
        nn.Softmax(dim=1)
    ).to(device)
    
print(model)

#############################################
# Class nn.Module Example:
# --------------------------
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


#############################################
# Get Device for Training
# -----------------------
# Here we check to see if `torch.cuda <https://pytorch.org/docs/stable/notes/cuda.html>`_ is available to use the GPU, else we will use the CPU. 
#
# Example:
#

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

##############################################
# __init__
# -------------------------
#
# The ``init`` function inherits from ``nn.Module`` which is the base class for 
# building neural network modules. This function defines the layers in your neural network
# then it initializes the modules to be called in the ``forward`` function.
# 

##############################################
# The Model Module Layers
# -------------------------
#
# Lets break down each model layer in the FashionMNIST model.
#

##################################################
# `nn.Flatten <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>`_ 
# -----------------------------------------------
#
# First we call nn.Flatten to reduce tensor dimensions to one.
#
# From the docs:
# ``torch.nn.Flatten(start_dim: int = 1, end_dim: int = -1)``
#
# Here is an example using one of the training_data set items:
tensor = training_data[0][0]
print(tensor.size())

model = nn.Sequential(
    nn.Flatten()
)
flattened_tensor = model(tensor)
flattened_tensor.size()

##############################################
# `nn.Linear <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ to add a linear layer to the model.
# -------------------------------
#
# Now that we have flattened our tensor dimension we will apply a linear layer 
# transform that will calculate/learn the weights and the bias.
#

# From the docs:
# 
# ``torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)``
#

input = training_data[0][0]
print(input.size())
model = nn.Sequential(
    nn.Flatten(),    
    nn.Linear(28*28, 512),
)
output = model(input)
output.size()

#################################################
# Activation Functions
# -------------------------
#
# After the first two linear layer we will call the `nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)>`_ 
# activation function. Then after the third linear layer we call the `nn.Softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ 
# activation to rescale between 0 and 1 and sum to one.
#

model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(classes)),
        nn.Softmax(dim=1)
    ).to(device)
    
print(model)


###################################################
# Forward Function
# --------------------------------
#
# In the class implementation of the neural network we define a ``forward`` function.  
# Then call the ``NeuralNetwork`` class and assign the device. When training the model we will call ``model``
# and pass the data (x) into the forward function and through each layer of our network.
#
#
def forward(self, x):
    x = self.flatten(x)
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = self.output(x)
    return F.softmax(x, dim=1)
model = NeuralNetwork().to(device)


################################################
# In the next section you will learn about how to train the model and the optimization loop for this example.
#
# Next: Learn more about how the `optimzation loop works with this example <optimization_tutorial.html>`_.
#
# .. include:: /beginner_source/quickstart/qs_toc.txt
#
