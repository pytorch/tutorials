# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================

**Authors**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`__, `Rabin
Adhikari <https://github.com/rabinadk1>`__

.. figure:: /_static/img/stn/FSeq.png

In this tutorial, you will learn how to augment your network using a
visual attention mechanism called spatial transformer networks. You can
read more about the spatial transformer networks in the `DeepMind
paper <https://arxiv.org/abs/1506.02025>`__.

Spatial transformer networks are a generalization of differentiable
attention to any spatial transformation. Spatial transformer networks
(STN for short) allow a neural network to learn how to perform spatial
transformations on the input image in order to enhance the geometric
invariance of the model. For example, it can crop a region of interest,
scale and correct the orientation of an image. It can be a useful
mechanism because CNNs are not invariant to rotation and scale and more
general affine transformations.

One of the best things about STN is the ability to simply plug it into
any existing CNN with very little modification.

"""

# License: BSD
# Authors: Ghassen Hamrouni, Rabin Adhikari

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

plt.ion()   # interactive mode


######################################################################
# Show GPU specifications, if available
# 

!nvidia-smi


######################################################################
# Loading the data
# ----------------
# 
# In this post we experiment with the classic MNIST dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.
# 

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,), inplace=True)
])

train_ds = datasets.MNIST(root='.', download=True, train=True, transform=image_transform)
test_ds = datasets.MNIST(root='.', train=False, transform=image_transform)

dataloader_common_kwargs = {
    "batch_size": 64,
    "num_workers": min(os.cpu_count(), 4)
}

# Training dataset
train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **dataloader_common_kwargs)

# Test dataset
test_loader = torch.utils.data.DataLoader(test_ds, **dataloader_common_kwargs)


######################################################################
# Depicting spatial transformer networks
# --------------------------------------
# 
# Spatial transformer networks boils down to three main components :
# 
# -  The localization network is a regular CNN which regresses the
#    transformation parameters. The transformation is never learned
#    explicitly from this dataset, instead the network learns
#    automatically the spatial transformations that enhances the global
#    accuracy.
# -  The grid generator generates a grid of coordinates in the input image
#    corresponding to each pixel from the output image.
# -  The sampler uses the parameters of the transformation and applies it
#    to the input image.
# 
# .. figure:: /_static/img/stn/stn-arch.png
# 
# .. Note:: We need the latest version of PyTorch that contains
# affine_grid and grid_sample modules.
# 

class STN(nn.Module):
    def __init__(self):
        super().__init__()

        fc_last_layer = nn.Linear(32, 3 * 2)
        
        # Initialize the weights/bias with identity transformation
        fc_last_layer.weight.data.zero_()
        fc_last_layer.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        
        # Regressor for the 3 * 2 affine matrix
        fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            fc_last_layer
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            fc_loc,
        )

    # Spatial transformer network forward function
    def forward(self, x: torch.Tensor):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x


######################################################################
# Creating a CNN Classifier inlcluding a STN within
# -------------------------------------------------
# 

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Needed later on for visualization
        self.stn = STN()
        
        self.model = nn.Sequential(
            self.stn,
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(inplace=True),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(50, 10)
        )


    def forward(self, x: torch.Tensor):
        return self.model(x)


model = Net().to(device)


######################################################################
# Training the model
# ------------------
# 
# The network is learning the classification task in a supervised way. In
# the same time the model is learning STN automatically in an end-to-end
# fashion.
# 


######################################################################
# Defining Optimizer
# ~~~~~~~~~~~~~~~~~~
# 
# Now, let’s use the Adam optimizer to train the model
# 

optimizer = optim.Adam(model.parameters(), lr=2e-3)


######################################################################
# Defining Train Function
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# Runs for every epoch
# 

def train(epoch: int):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        
        loss = F.cross_entropy(output, target.to(device))
        
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_ds)} ({100 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


######################################################################
# Defining Test Function
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# A simple test procedure to measure the STN performances on MNIST.
# 

@torch.inference_mode()
def test():
    model.eval()
    
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data.to(device))

        target = target.to(device)

        # sum up batch loss
        test_loss += F.cross_entropy(output, target, reduction="sum").item()

        # get the index of the max log-probability
        pred = output.argmax(1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_ds)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_ds)} ({100 * correct / len(test_ds):.0f}%)\n")


######################################################################
# Running Training Loop
# ~~~~~~~~~~~~~~~~~~~~~
# 
# Training the model and testing it for each epoch
# 

for epoch in range(1, 21):
    train(epoch)
    test()


######################################################################
# Visualizing the STN results
# ---------------------------
# 
# Now, we will inspect the results of our learned visual attention
# mechanism.
# 


######################################################################
# Defining function to convert Tensor to Numpy Image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def convert_image_np(inp: torch.Tensor):
    """Convert a Tensor in CPU to numpy image."""
    numpy_inp = inp.permute(1, 2, 0).numpy()
    scaled_inp = std * numpy_inp + mean
    clipped_inp = np.clip(scaled_inp, 0, 1)
    return clipped_inp


######################################################################
# Defining Visualization Function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We want to visualize the output of the spatial transformers layer after
# the training, we visualize a batch of input images and the corresponding
# transformed batch using STN.
# 

@torch.inference_mode()
def visualize_stn():
    # Get a batch of training data
    images, _ = next(iter(test_loader))
    
    input_tensor = images.to(device)

    transformed_input_tensor = model.stn(input_tensor)

    in_grid = convert_image_np(
        torchvision.utils.make_grid(images))

    out_grid = convert_image_np(
        torchvision.utils.make_grid(transformed_input_tensor.cpu()))

    # Plot the results side-by-side
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(in_grid)
    ax1.set_title('Dataset Images')

    ax2.imshow(out_grid)
    ax2.set_title('Transformed Images')


######################################################################
# Runing Visualization Function
# -----------------------------
# 
# Visualize the STN transformation on some input batch
# 

visualize_stn()

plt.ioff()
plt.show()
