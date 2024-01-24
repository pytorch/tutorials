"""
Saving and loading multiple models in one file using PyTorch
============================================================
Saving and loading multiple models can be helpful for reusing models
that you have previously trained.

Introduction
------------
When saving a model comprised of multiple ``torch.nn.Modules``, such as
a GAN, a sequence-to-sequence model, or an ensemble of models, you must
save a dictionary of each model’s state_dict and corresponding
optimizer. You can also save any other items that may aid you in
resuming training by simply appending them to the dictionary.
To load the models, first initialize the models and optimizers, then
load the dictionary locally using ``torch.load()``. From here, you can
easily access the saved items by simply querying the dictionary as you
would expect.
In this recipe, we will demonstrate how to save multiple models to one
file using PyTorch.

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
# 4. Save multiple models
# 5. Load multiple models
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
# images. To learn more see the Defining a Neural Network recipe. Build
# two variables for the models to eventually save.
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

netA = Net()
netB = Net()


######################################################################
# 3. Initialize the optimizer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We will use SGD with momentum to build an optimizer for each model we
# created.
# 

optimizerA = optim.SGD(netA.parameters(), lr=0.001, momentum=0.9)
optimizerB = optim.SGD(netB.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 4. Save multiple models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Collect all relevant information and build your dictionary.
# 

# Specify a path to save to
PATH = "model.pt"

torch.save({
            'modelA_state_dict': netA.state_dict(),
            'modelB_state_dict': netB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            }, PATH)


######################################################################
# 4. Load multiple models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Remember to first initialize the models and optimizers, then load the
# dictionary locally.
# 

modelA = Net()
modelB = Net()
optimModelA = optim.SGD(modelA.parameters(), lr=0.001, momentum=0.9)
optimModelB = optim.SGD(modelB.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()


######################################################################
# You must call ``model.eval()`` to set dropout and batch normalization
# layers to evaluation mode before running inference. Failing to do this
# will yield inconsistent inference results.
# 
# If you wish to resuming training, call ``model.train()`` to ensure these
# layers are in training mode.
# 
# Congratulations! You have successfully saved and loaded multiple models
# in PyTorch.
#
