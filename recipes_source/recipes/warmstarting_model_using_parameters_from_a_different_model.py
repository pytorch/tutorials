"""
Warmstarting model using parameters from a different model in PyTorch
=====================================================================
Partially loading a model or loading a partial model are common
scenarios when transfer learning or training a new complex model.
Leveraging trained parameters, even if only a few are usable, will help
to warmstart the training process and hopefully help your model converge
much faster than training from scratch.

Introduction
------------
Whether you are loading from a partial ``state_dict``, which is missing
some keys, or loading a ``state_dict`` with more keys than the model
that you are loading into, you can set the strict argument to ``False``
in the ``load_state_dict()`` function to ignore non-matching keys.
In this recipe, we will experiment with warmstarting a model using
parameters of a different model.

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
# 2. Define and initialize the neural network A and B
# 3. Save model A
# 4. Load into model B
# 
# 1. Import necessary libraries for loading our data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For this recipe, we will use ``torch`` and its subsidiaries ``torch.nn``
# and ``torch.optim``.
# 

import torch
import torch.nn as nn
import torch.optim as optim


######################################################################
# 2. Define and initialize the neural network A and B
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For sake of example, we will create a neural network for training
# images. To learn more see the Defining a Neural Network recipe. We will
# create two neural networks for sake of loading one parameter of type A
# into type B.
# 

class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
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

netA = NetA()

class NetB(nn.Module):
    def __init__(self):
        super(NetB, self).__init__()
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

netB = NetB()


######################################################################
# 3. Save model A
# ~~~~~~~~~~~~~~~~~~~
# 

# Specify a path to save to
PATH = "model.pt"

torch.save(netA.state_dict(), PATH)


######################################################################
# 4. Load into model B
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# If you want to load parameters from one layer to another, but some keys
# do not match, simply change the name of the parameter keys in the
# state_dict that you are loading to match the keys in the model that you
# are loading into.
# 

netB.load_state_dict(torch.load(PATH), strict=False)


######################################################################
# You can see that all keys matched successfully!
# 
# Congratulations! You have successfully warmstarted a model using
# parameters from a different model in PyTorch.
# 
# Learn More
# ----------
# 
# Take a look at these other recipes to continue your learning:
# 
# - `Saving and loading multiple models in one file using PyTorch <https://pytorch.org/tutorials/recipes/recipes/saving_multiple_models_in_one_file.html>`__
# - `Saving and loading models across devices in PyTorch <https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html>`__
