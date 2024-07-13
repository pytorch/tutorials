"""
Saving and loading models across devices in PyTorch
===================================================

There may be instances where you want to save and load your neural
networks across different devices.

Introduction
------------

Saving and loading models across devices is relatively straightforward
using PyTorch. In this recipe, we will experiment with saving and
loading models across CPUs and GPUs.

Setup
-----

In order for every code block to run properly in this recipe, you must
first change the runtime to “GPU” or higher. Once you do, we need to
install ``torch`` if it isn’t already available.

.. code-block:: sh

   pip install torch

"""

######################################################################
# Steps
# -----
# 
# 1. Import all necessary libraries for loading our data
# 2. Define and initialize the neural network
# 3. Save on a GPU, load on a CPU
# 4. Save on a GPU, load on a GPU
# 5. Save on a CPU, load on a GPU
# 6. Saving and loading ``DataParallel`` models
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
# 3. Save on GPU, Load on CPU
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# When loading a model on a CPU that was trained with a GPU, pass
# ``torch.device('cpu')`` to the ``map_location`` argument in the
# ``torch.load()`` function.
# 

# Specify a path to save to
PATH = "model.pt"

# Save
torch.save(net.state_dict(), PATH)

# Load
device = torch.device('cpu')
model = Net()
model.load_state_dict(torch.load(PATH, map_location=device))


######################################################################
# In this case, the storages underlying the tensors are dynamically
# remapped to the CPU device using the ``map_location`` argument.
# 
# 4. Save on GPU, Load on GPU
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# When loading a model on a GPU that was trained and saved on GPU, simply
# convert the initialized model to a CUDA optimized model using
# ``model.to(torch.device('cuda'))``.
# 
# Be sure to use the ``.to(torch.device('cuda'))`` function on all model
# inputs to prepare the data for the model.
# 

# Save
torch.save(net.state_dict(), PATH)

# Load
device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)


######################################################################
# Note that calling ``my_tensor.to(device)`` returns a new copy of
# ``my_tensor`` on GPU. It does NOT overwrite ``my_tensor``. Therefore,
# remember to manually overwrite tensors:
# ``my_tensor = my_tensor.to(torch.device('cuda'))``.
# 
# 5. Save on CPU, Load on GPU
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# When loading a model on a GPU that was trained and saved on CPU, set the
# ``map_location`` argument in the ``torch.load()`` function to
# ``cuda:device_id``. This loads the model to a given GPU device.
# 
# Be sure to call ``model.to(torch.device('cuda'))`` to convert the
# model’s parameter tensors to CUDA tensors.
# 
# Finally, also be sure to use the ``.to(torch.device('cuda'))`` function
# on all model inputs to prepare the data for the CUDA optimized model.
# 

# Save
torch.save(net.state_dict(), PATH)

# Load
device = torch.device("cuda")
model = Net()
# Choose whatever GPU device number you want
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
model.to(device)


######################################################################
# 6. Saving ``torch.nn.DataParallel`` Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# ``torch.nn.DataParallel`` is a model wrapper that enables parallel GPU
# utilization.
# 
# To save a ``DataParallel`` model generically, save the
# ``model.module.state_dict()``. This way, you have the flexibility to
# load the model any way you want to any device you want.
# 

# Save
torch.save(net.module.state_dict(), PATH)

# Load to whatever device you want


######################################################################
# Congratulations! You have successfully saved and loaded models across
# devices in PyTorch.
# 
