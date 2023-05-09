"""
Saving and loading models for inference in PyTorch
==================================================
There are two approaches for saving and loading models for inference in
PyTorch. The first is saving and loading the ``state_dict``, and the
second is saving and loading the entire model.

Introduction
------------
Saving the model’s ``state_dict`` with the ``torch.save()`` function
will give you the most flexibility for restoring the model later. This
is the recommended method for saving models, because it is only really
necessary to save the trained model’s learned parameters.
When saving and loading an entire model, you save the entire module
using Python’s
`pickle <https://docs.python.org/3/library/pickle.html>`__ module. Using
this approach yields the most intuitive syntax and involves the least
amount of code. The disadvantage of this approach is that the serialized
data is bound to the specific classes and the exact directory structure
used when the model is saved. The reason for this is because pickle does
not save the model class itself. Rather, it saves a path to the file
containing the class, which is used during load time. Because of this,
your code can break in various ways when used in other projects or after
refactors.
In this recipe, we will explore both ways on how to save and load models
for inference.

Setup
-----
Before we begin, we need to install ``torch`` if it isn’t already
available.


::

   pip install torch


"""


######################################################################
# Steps
# -----
# 
# 1. Import all necessary libraries for loading our data
# 2. Define and initialize the neural network
# 3. Initialize the optimizer
# 4. Save and load the model via ``state_dict``
# 5. Save and load the entire model
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
# 3. Initialize the optimizer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We will use SGD with momentum.
# 

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 4. Save and load the model via ``state_dict``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Let’s save and load our model using just ``state_dict``.
# 

# Specify a path
PATH = "state_dict_model.pt"

# Save
torch.save(net.state_dict(), PATH)

# Load
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()


######################################################################
# A common PyTorch convention is to save models using either a ``.pt`` or
# ``.pth`` file extension.
# 
# Notice that the ``load_state_dict()`` function takes a dictionary
# object, NOT a path to a saved object. This means that you must
# deserialize the saved state_dict before you pass it to the
# ``load_state_dict()`` function. For example, you CANNOT load using
# ``model.load_state_dict(PATH)``.
# 
# Remember too, that you must call ``model.eval()`` to set dropout and
# batch normalization layers to evaluation mode before running inference.
# Failing to do this will yield inconsistent inference results.
# 
# 5. Save and load entire model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now let’s try the same thing with the entire model.
# 

# Specify a path
PATH = "entire_model.pt"

# Save
torch.save(net, PATH)

# Load
model = torch.load(PATH)
model.eval()


######################################################################
# Again here, remember that you must call ``model.eval()`` to set dropout and
# batch normalization layers to evaluation mode before running inference.
# 
# Congratulations! You have successfully saved and load models for
# inference in PyTorch.
# 
# Learn More
# ----------
# 
# Take a look at these other recipes to continue your learning:
# 
# - `Saving and loading a general checkpoint in PyTorch <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>`__
# - `Saving and loading multiple models in one file using PyTorch <https://pytorch.org/tutorials/recipes/recipes/saving_multiple_models_in_one_file.html>`__
