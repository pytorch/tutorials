"""
Zeroing out gradients in PyTorch
================================
It is beneficial to zero out gradients when building a neural network.
This is because by default, gradients are accumulated in buffers (i.e,
not overwritten) whenever ``.backward()`` is called.

Introduction
------------
When training your neural network, models are able to increase their
accuracy through gradient descent. In short, gradient descent is the
process of minimizing our loss (or error) by tweaking the weights and
biases in our model.

``torch.Tensor`` is the central class of PyTorch. When you create a
tensor, if you set its attribute ``.requires_grad`` as ``True``, the
package tracks all operations on it. This happens on subsequent backward
passes. The gradient for this tensor will be accumulated into ``.grad``
attribute. The accumulation (or sum) of all the gradients is calculated
when .backward() is called on the loss tensor.

There are cases where it may be necessary to zero-out the gradients of a
tensor. For example: when you start your training loop, you should zero
out the gradients so that you can perform this tracking correctly.
In this recipe, we will learn how to zero out gradients using the
PyTorch library. We will demonstrate how to do this by training a neural
network on the ``CIFAR10`` dataset built into PyTorch.

Setup
-----
Since we will be training data in this recipe, if you are in a runnable
notebook, it is best to switch the runtime to GPU or TPU.
Before we begin, we need to install ``torch`` and ``torchvision`` if
they aren’t already available.

.. code-block:: sh

   pip install torchvision


"""


######################################################################
# Steps
# -----
# 
# Steps 1 through 4 set up our data and neural network for training. The
# process of zeroing out the gradients happens in step 5. If you already
# have your data and neural network built, skip to 5.
# 
# 1. Import all necessary libraries for loading our data
# 2. Load and normalize the dataset
# 3. Build the neural network
# 4. Define the loss function
# 5. Zero the gradients while training the network
# 
# 1. Import necessary libraries for loading our data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For this recipe, we will just be using ``torch`` and ``torchvision`` to
# access the dataset.
# 

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


######################################################################
# 2. Load and normalize the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# PyTorch features various built-in datasets (see the Loading Data recipe
# for more information).
# 

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


######################################################################
# 3. Build the neural network
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We will use a convolutional neural network. To learn more see the
# Defining a Neural Network recipe.
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


######################################################################
# 4. Define a Loss function and optimizer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Let’s use a Classification Cross-Entropy loss and SGD with momentum.
# 

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 5. Zero the gradients while training the network
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# This is when things start to get interesting. We simply have to loop
# over our data iterator, and feed the inputs to the network and optimize.
# 
# Notice that for each entity of data, we zero out the gradients. This is
# to ensure that we aren’t tracking any unnecessary information when we
# train our neural network.
# 

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


######################################################################
# You can also use ``model.zero_grad()``. This is the same as using
# ``optimizer.zero_grad()`` as long as all your model parameters are in
# that optimizer. Use your best judgment to decide which one to use.
# 
# Congratulations! You have successfully zeroed out gradients PyTorch.
# 
# Learn More
# ----------
# 
# Take a look at these other recipes to continue your learning:
# 
# - `Loading data in PyTorch <https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html>`__
# - `Saving and loading models across devices in PyTorch <https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html>`__
