# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================
**Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_

.. figure:: /_static/img/stn/FSeq.png

In this tutorial, you will learn how to augument your network using
a visual attention mechanism. You can read more about the spatial transformer 
networks at `DeepMind paper <https://arxiv.org/abs/1506.02025>`__

Spatial transformer networks are a generalisation of differentiable
attention to any spatial transformation. Spatial transformer networks (STN for short)
allows a neural network to learn how to do geometric transformations
to the input image.
For example it can crop a region of interest, scale and correct
the orientation of an image. It can be a useful mechanism because CNN are not 
invariant to rotation and scale and more generally : affine transformation. 
One of the best things about them is that we can plug an STN into any 
existing network with very little effort. We have used them for classification 
tasks as well as in auto-encoders. 
"""
# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

plt.ion()   # interactive mode

######################################################################
# Loading the data
# ----------------
#
# In this post we experiment with the classic MNIST dataset. Using a 
# standard convolutional network augmented with a spatial transformer 
# network.

use_cuda = torch.cuda.is_available()


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)

######################################################################
# Depicting spatial transformer networks
# --------------------------------------
#
# Spatial transformer networks boils down to three main components :
#
# -  The localization network is a regular CNN which regresses the transformation
#    parameters. The transformation is never learned explicitly from this dataset,
#    instead the network learns automatically the spatial transformations that enhances
#    the global accuracy.
# -  The grid generator generates a grid of coordinates in the input image corresponding
#    to each pixel from the output image.
# -  The sampler uses the parameters of the transformation and apply it to the input image.
#
# .. figure:: /_static/img/stn/stn-arch.png
#
# .. Note::
#    We need the latest version of PyTorch that contains 
#    affine_grid and grid_sample modules.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regress the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
if use_cuda:
    model.cuda()

######################################################################
# Training the model
# ------------------
#
# Now, let's use the SGD algorithm to train the model. The network is 
# learning the classifcation task in a supervised way. In the same time 
# the model is learning STN automatically in an end-to-end fashion.

optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
#
# A simple test procedure to measure STN the performances on MNIST.
#

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

######################################################################
# Visualizing the STN results
# ---------------------------
#
# Now, we will inspect the learned visual attention mechanism.
#
# We define a small helper function in order to visualize the
# transfromations while training.

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and the coresponding 
# transformed batch.

def visualize_stn():
    # Get a batch of training data
    data, _ = next(iter(test_loader))
    data = Variable(data, volatile=True)

    if use_cuda:
        data = data.cuda()

    inGrid = convert_image_np(torchvision.utils.make_grid(data.cpu().data))
    outGrid = convert_image_np(torchvision.utils.make_grid(model.stn(data).cpu().data))

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(inGrid)
    axarr[0].set_title('Dataset Images')

    axarr[1].imshow(outGrid)
    axarr[1].set_title('Transformed Images')



for epoch in range(1, 60 + 1):
    train(epoch)
    test()

visualize_stn()

plt.ioff()
plt.show()
