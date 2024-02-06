"""
What is a state_dict in PyTorch
===============================
In PyTorch, the learnable parameters (i.e. weights and biases) of a
``torch.nn.Module`` model are contained in the model’s parameters
(accessed with ``model.parameters()``). A ``state_dict`` is simply a
Python dictionary object that maps each layer to its parameter tensor.

Introduction
------------
A ``state_dict`` is an integral entity if you are interested in saving
or loading models from PyTorch.
Because ``state_dict`` objects are Python dictionaries, they can be
easily saved, updated, altered, and restored, adding a great deal of
modularity to PyTorch models and optimizers.
Note that only layers with learnable parameters (convolutional layers,
linear layers, etc.) and registered buffers (batchnorm’s running_mean)
have entries in the model’s ``state_dict``. Optimizer objects
(``torch.optim``) also have a ``state_dict``, which contains information
about the optimizer’s state, as well as the hyperparameters used.
In this recipe, we will see how ``state_dict`` is used with a simple
model.

Setup
-----
Before we begin, we need to install ``torch`` if it isn’t already
available.

.. code-block:: sh

   pip install torch

"""



######################################################################
# Steps
# -----
# 
# 1. Import all necessary libraries for loading our data
# 2. Define and initialize the neural network
# 3. Initialize the optimizer
# 4. Access the model and optimizer ``state_dict``
# 
# 1. Import necessary libraries for loading our data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For this recipe, we will use ``torch`` and its subsidiaries ``torch.nn``
# and ``torch.optim``.
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


######################################################################
# 2. Define and initialize the neural network
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For sake of example, we will create a neural network for training
# images. To learn more see the Defining a Neural Network recipe.
# 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)


######################################################################
# 3. Initialize the optimizer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We will use SGD with momentum.
# 

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 4. Access the model and optimizer ``state_dict``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now that we have constructed our model and optimizer, we can understand
# what is preserved in their respective ``state_dict`` properties.
# 

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print()

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


######################################################################
# This information is relevant for saving and loading the model and
# optimizers for future use.
# 
# Congratulations! You have successfully used ``state_dict`` in PyTorch.
# 
# Learn More
# ----------
# 
# Take a look at these other recipes to continue your learning:
# 
# - `Saving and loading models for inference in PyTorch <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html>`__
# - `Saving and loading a general checkpoint in PyTorch <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>`__
