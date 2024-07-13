"""
Saving and loading a general checkpoint in PyTorch
==================================================
Saving and loading a general checkpoint model for inference or 
resuming training can be helpful for picking up where you last left off.
When saving a general checkpoint, you must save more than just the
model’s state_dict. It is important to also save the optimizer’s
state_dict, as this contains buffers and parameters that are updated as
the model trains. Other items that you may want to save are the epoch
you left off on, the latest recorded training loss, external
``torch.nn.Embedding`` layers, and more, based on your own algorithm.

Introduction
------------
To save multiple checkpoints, you must organize them in a dictionary and
use ``torch.save()`` to serialize the dictionary. A common PyTorch
convention is to save these checkpoints using the ``.tar`` file
extension. To load the items, first initialize the model and optimizer,
then load the dictionary locally using torch.load(). From here, you can
easily access the saved items by simply querying the dictionary as you
would expect.

In this recipe, we will explore how to save and load multiple
checkpoints.

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
# 4. Save the general checkpoint
# 5. Load the general checkpoint
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
# 4. Save the general checkpoint
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Collect all relevant information and build your dictionary.
# 

# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)


######################################################################
# 5. Load the general checkpoint
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Remember to first initialize the model and optimizer, then load the
# dictionary locally.
# 

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()


######################################################################
# You must call ``model.eval()`` to set dropout and batch normalization
# layers to evaluation mode before running inference. Failing to do this
# will yield inconsistent inference results.
# 
# If you wish to resuming training, call ``model.train()`` to ensure these
# layers are in training mode.
# 
# Congratulations! You have successfully saved and loaded a general
# checkpoint for inference and/or resuming training in PyTorch.
#
