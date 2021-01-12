##################################################################
# .. include:: /beginner_source/quickstart/qs_toc.txt
#

"""
.. include:: /beginner_source/quickstart/qs_toc.txt


Transforms
===================

Data does not come ready to be processed in the machine learning algorithm. We need to do different data manipulations or transforms to prepare it for training. There are many types of transformations and it depends on the type of model you are building and the state of your data as to which ones you should use. 

In the below example, for our FashionMNIT image dataset, we are taking our image features (x), turning it into a tensor and normalizing it. Then taking the labels (y) padding with zeros to get a consistent shape. We will break down each of these steps and the why below.

Full Section Example:
"""
import os
import torch
import torch.nn as nn
import torch.onnx as onnx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# image classes
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress",
           "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# data used for training
training_data = datasets.FashionMNIST('data', train=True, download=True,
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]),
                                      target_transform=transforms.Compose([
                                          transforms.Lambda(lambda y: torch.zeros(
                                              10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
                                      ])
                                      )

# data used for testing
test_data = datasets.FashionMNIST('data', train=False, download=True,
                                  transform=transforms.Compose(
                                      [transforms.ToTensor()]),
                                  target_transform=transforms.Compose([
                                      transforms.Lambda(lambda y: torch.zeros(
                                          10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
                                  ])
                                  )

##############################################
# Pytorch Datasets
# --------------------------
#
# We are using the built-in open FashionMNIST datasets from the PyTorch library. For more info on the Datasets and Loaders check out `this <dataquickstart_tutorial.html>`_ resource. The ``Train=True`` indicates we want to download the training dataset from the built-in datasets, ``Train=False`` indicates to download the testing dataset. This way we have data partitioned out for training and testing within the provided PyTorch datasets. We will apply the same transfoms to both the training and testing datasets.
#
# From the docs:
#
# ``torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)``

##############################################
# Transform: Features
# ---------------------------
# Example:
#

transform = transforms.Compose([transforms.ToTensor()])

#####################################################
# Compose
# ------------------------
#
# The `transforms.compose`` allows us to string together different steps of transformations in a sequential order. This allows us to add an array of transforms for both the features and labels when preparing our data for training.
#

#################################################
# ToTensor()
# -------------------------------
#
# For the feature transforms we have an array of transforms to process our image data for training. The first transform in the array is ``transforms.ToTensor()`` this is from class `torchvision.transforms.ToTensor <https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor>`_. We need to take our images and turn them into a tensor. (To learn more about Tensors check out `this <tensor_tutorial.html>`_ resource.) The ``ToTensor()`` transformation is doing more than converting our image into a tensor. Its also normalizing our data for us by scaling the images to be between 0 and 1.
#
#
# .. note:: ToTensor only normalized image data that is in PIL mode of (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8. In the other cases, tensors are returned without scaling.
#
#
# Check out the other `TorchVision Transforms <https://pytorch.org/docs/stable/torchvision/transforms.html>`_
#

##############################################
# Target_Transform: Labels
# -------------------------------
#
# Example:
#

target_transform = transforms.Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

#################################################
# This function is taking the y input and creating a tensor of size 10 with a float datatype. Then its calling scatter `torch.Tensor.scatter_ class <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_>`_ to send each item to torch.zeros, according to the row, index and current item value.
#  - Dim=0  is row wise index
#  - index = torchtensor(y) is the index of the element toscatter
#  - value = 1 is the source elemnt

##############################################
# Using your own data
# --------------------------------------
#
# Below is an example for processing image data using a dataset from a local directory.
#
# Example:
#

data_dir = 'data'
batch_size = 4

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes


##################################################################
# Next learn how to `build the model <buildmodel_tutorial.html>`_
#