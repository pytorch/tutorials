# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================
**Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_

.. figure:: /_static/img/stn/FSeq.png

In this tutorial, you will learn how to augment your network using
a visual attention mechanism called spatial transformer
networks. You can read more about the spatial transformer
networks in the `DeepMind paper <https://arxiv.org/abs/1506.02025>`__

Spatial transformer networks are a generalization of differentiable
attention to any spatial transformation. Spatial transformer networks
(STN for short) allow a neural network to learn how to perform spatial
transformations on the input image in order to enhance the geometric
invariance of the model.
For example, it can crop a region of interest, scale and correct
the orientation of an image. It can be a useful mechanism because CNNs
are not invariant to rotation and scale and more general affine
transformations.

One of the best things about STN is the ability to simply plug it into
any existing CNN with very little modification.

Update for this tutorial:
- Add a distorted MNIST dataset 60*60 to interpret the original approach
using torch.grid_sample with padding_mode = "zeros"
- Add a new Spatial Transformer Network compatible with the distorted MNIST dataset
- Quantify the difference between padding_mode in torch.grid_sample (i.e., "zeros" vs
"boundary")
"""
# License: BSD
# Author: Ghassen Hamrouni

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.ion()  # interactive mode

######################################################################
# Loading the data
# ----------------
#
# In this post we experiment with the classic MNIST dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.

import google_drive_downloader
from google_drive_downloader import GoogleDriveDownloader as GDD
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training dataset
normal_train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=0)
# Test dataset
normal_test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=0)


######################################################################
# Loading the data
# ----------------
#
# In this post we experiment with the classic MNIST dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.
#
# Update: to interpret the Spatial Transformer Network better as the
# updated aims indicate, we also experiment with a distorted MNIST dataset.
# In the distorted MNIST dataset, for an image:
# - The original digits are placed randomly into a black canvas of 60*60.
# - Noises (i.e., patches sampled from other images not identical to the
# specific digit in the image) are placed randomly in the new canvas 60*60 above.
#
# The distorted MNIST dataset with image size 60*60 is loaded from:
# https://github.com/theRealSuperMario/pytorch_stn/blob/master/data/mnist_cluttered_60.npz
#
# A preview of the distorted MNIST dataset with image size 60*60 is loaded from:
# https://drive.google.com/file/d/1txYwNjgY5FxYIUuScE7AKgmeXA4MJB5R/view?usp=drive_linkmo.png
# Credit for this distorted MNISt dataset is given to
# **Author**: `Sandro Braun <https://github.com/theRealSuperMario>`_

# Helper class to load the distorted dataset
class DistortedDataSet(Dataset):
    # TODO: ? transforms may not be required here
    """
    Generate dataset composed of:
    - The original inputs & outputs (using torch DataLoader)
    - Transforms (using torchvision transforms)
    """

    def __init__(self, inputs, outputs, transform):
        super(DistortedDataSet, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.transform = transform

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        input_ = self.inputs[idx]
        input_ = input_[None, :, :]
        output_ = int(self.outputs[idx])
        if self.transform:
            input_ = self.transform(input_)
        return input_, output_


# Load the distorted MNIST dataset first
distorted_file_id = '1txYwNjgY5FxYIUuScE7AKgmeXA4MJB5R'
GDD.download_file_from_google_drive(file_id=distorted_file_id, dest_path='./distorted_mnist_60.npz', unzip=True)
distorted_data = np.load('distorted_mnist_60.npz')

# Training dataset (distorted)
train_images = torch.tensor(distorted_data['X_train'], dtype=torch.float32)
train_digits = torch.tensor(distorted_data['y_train'], dtype=torch.float32)
train_set = DistortedDataSet(inputs=train_images, outputs=train_digits,
                             transform=transforms.Compose([
                                 transforms.Normalize((0.1307,), (0.3081,))]))
distorted_train_loader = DataLoader(
    dataset=train_set, batch_size=64, shuffle=True, num_workers=0)

# Test dataset (distorted)
test_images = torch.tensor(distorted_data['X_test'], dtype=torch.float32)
test_digits = torch.tensor(distorted_data['y_test'], dtype=torch.float32)
test_set = DistortedDataSet(inputs=test_images, outputs=test_digits,
                            transform=transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3081,))]))
distorted_test_loader = DataLoader(
    dataset=test_set, batch_size=64, shuffle=False, num_workers=0)


######################################################################
# Depicting spatial transformer networks
# --------------------------------------
#
# Spatial transformer networks boils down to three main components :
#
# -  The localization network is a regular CNN which regresses the
#    transformation parameters. The transformation is never learned
#    explicitly from this dataset, instead the network learns automatically
#    the spatial transformations that enhances the global accuracy.
# -  The grid generator generates a grid of coordinates in the input
#    image corresponding to each pixel from the output image.
# -  The sampler uses the parameters of the transformation and applies
#    it to the input image.
#
# .. figure:: /_static/img/stn/stn-arch.png
#
# .. Note::
#    We need the latest version of PyTorch that contains
#    affine_grid and grid_sample modules.
#
# Update: to interpret the Spatial Transformer Network better as the
# updated aims indicate:
# - A Spatial Transformer Network that digests the image size 60*60, named Net_60, is added.
# - This Net_60 enables either "zeros" or "boundary" padding_mode in torch.grid_sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net_60(nn.Module):
    def __init__(self, padding_mode):
        super(Net_60, self).__init__()
        self.localization = nn.Sequential(nn.Conv2d(1, 8, kernel_size=7),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(8, 10, kernel_size=5),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2880, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 11 * 11, 32), nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
        self.padding_mode = padding_mode

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 11 * 11)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode=self.padding_mode)
        return x

    def forward(self, x):
        x = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 2880)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model_28 = Net().to(device)
model_60_padding_zeros = Net_60(padding_mode="zeros").to(device)
model_60_padding_boundary = Net_60(padding_mode="boundary").to(device)

######################################################################
# Training the model
# ------------------
#
# Now, let's use the SGD algorithm to train the model. The network is
# learning the classification task in a supervised way. In the same time
# the model is learning STN automatically in an end-to-end fashion.


optimizer_28 = optim.SGD(model_28.parameters(), lr=0.01)
optimizer_60_padding_zeros = optim.SGD(model_60_padding_zeros.parameters(), lr=0.01)
optimizer_60_padding_boundary = optim.SGD(model_60_padding_boundary.parameters(), lr=0.01)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


#
# A simple test procedure to measure the STN performances on MNIST.
#


def test(model, test_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))


######################################################################
# Visualizing the STN results
# ---------------------------
#
# Now, we will inspect the results of our learned visual attention
# mechanism.
#
# We define a small helper function in order to visualize the
# transformations while training.


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.
#
# Update: to interpret the Spatial Transformer Network better as the
# updated aims indicate, this function is modified to take any torch.Dataloader

def visualize_stn(model, test_loader):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


# Update: to interpret the Spatial Transformer Network better as the
# updated aims indicate, now we perform the following:
# 1. Use model to train, test and visualize for th original image (size 28*28)
# 2. Use model_60_padding_zeros to train, test and visualize for the distorted image (size 60*60)
# 3. Use model_60_padding_boundary to train, test and visualize for the distorted image (size 60*60)

# The model for original image size 28*28
for epoch in range(1, 20 + 1):
    train(model_28, normal_train_loader, optimizer_28, epoch)
    test(model_28, normal_test_loader)

# the model for distorted image size 60*60, with padding zeros for torch.grid_sample
for epoch in range(1, 20 + 1):
    train(model_60_padding_zeros, distorted_train_loader, optimizer_60_padding_zeros, epoch)
    test(model_60_padding_zeros, distorted_test_loader)

# the model for distorted image size 60*60, with padding boundary for torch.grid_sample
for epoch in range(1, 20 + 1):
    train(model_60_padding_boundary, distorted_train_loader, optimizer_60_padding_boundary, epoch)
    test(model_60_padding_boundary, distorted_test_loader)

# Visualize the STN transformation on some input batche for model_28
# model_60_padding_zeros, and model_60_padding_boundary, respectively
visualize_stn(model_28, normal_test_loader)
visualize_stn(model_60_padding_zeros, distorted_test_loader)
visualize_stn(model_60_padding_boundary, distorted_test_loader)

plt.ioff()
plt.show()

######################################################################
# Interpreting the STN results
# ---------------------------
#
# With the visualization from the 3 Spatial Transformer Networks above:
#
# -
# -
# -
