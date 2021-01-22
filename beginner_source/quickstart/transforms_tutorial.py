"""
`Quickstart <quickstart_tutorial.html>`_ >
`Tensors <tensor_tutorial.html>`_ > 
`DataSets & DataLoaders <dataquickstart_tutorial.html>`_ >
**Transforms** >
`Build Model <buildmodel_tutorial.html>`_ >
`Autograd <autograd_tutorial.html>`_ >
`Optimization <optimization_tutorial.html>`_ >
`Save & Load Model <saveloadrun_tutorial.html>`_

Transforms
===================

In most cases data does not come in its final processed form that is required for training machine learning algorithms. We need to do different data manipulations or **transformations** to prepare it for training. There are many types of transformations, and it depends on the type of model you are building and the state of your data as to which ones you should use. 

In the example below, let's take FashionMNIST image dataset, which is available from ``torchvision.datasets`` using the following function:
"""
import torchvision

ds = torchvision.datasets.FashionMNIST(
 'data',                # specifies data directory to store data
 train=True,            # specifies training or test dataset to use
 transform=None,        # specifies transforms to apply to features (images) 
 target_transform=None, # specifies transforms to apply to labels
 download=False)        # should the data be downloaded from the Internet

################################
#To prepare data for training we need to take our image (also called features, x), turn it into a tensor and normalize it. Then we need to convert labels (y) into one-hot encoding.
#
#We will break down each of these steps below.

##############################################
# PyTorch Datasets
# --------------------------
#
# We are using the built-in FashionMNIST dataset from the PyTorch library. 
# For more info on the Datasets and Loaders check out `this <dataquickstart_tutorial.html>`_ section of the tutorial. 
# The ``train=True`` argument indicates we want the training split of the dataset (``train=False`` downloads the test split instead). 
# This way we have data partitioned out for training and testing within the provided PyTorch datasets. 
# We will apply the same transforms to both the training and testing datasets.

# import packages
import os
import torch
import torch.nn as nn
import torch.onnx as onnx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Here we define the image classes.
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress",
           "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

##############################################
# Feature Transforms and Label Transforms
# ---------------------------------------
#
# Below is the code to load the FashionMNIST dataset and apply the transforms:

training_data = datasets.FashionMNIST(
    "data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
    target_transform=transforms.Lambda(
        lambda y: torch.zeros(10, dtype=torch.float)
        .scatter_(0, torch.tensor(y), value=1)
    )
)

########################################
# Here we define two transformations:
#
# * ``transform`` is the transformation we apply to features, in our case - to images. The dataset contains images in PIL format so we need to convert them to tensors using the ``ToTensor()`` transform.
# * ``target_transform`` defines a transformation that is applied to labels in the dataset. Here, the  label is a class number from 0 to 9, and we need to convert it to one-hot encoding.

#################################################
# ToTensor()
# -------------------------------
#
# `transforms.ToTensor <https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor>`_ transform is required to prepare an image for training. It takes the PIL image, converts it into a `tensor <tensor_tutorial.html>`_, and normalizes our data by scaling the image pixel intensity values to be between 0 and 1.
#
#

##############################################
# Lambda Transforms
# -------------------------------
#
# We use a **lambda transform** to turn the class number into one-hot encoding. This function takes y as an input and creates a zero tensor of size 10. Then it calls scatter `torch.Tensor.scatter_ class <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_>`_ to take a value 1 and store it into the correct position of the zero vector defined by the class number.

target_transform = transforms.Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

###############################################
# Check out more `torchvision transforms <https://pytorch.org/docs/stable/torchvision/transforms.html>`_
#

#####################################################
# Compose
# ------------------------
#
# In many cases, we need to perform several transformations on the data sequentially. `transforms.Compose <https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Compose>`_ allows us to string together different steps of transformations in a sequential order. We will see an example of using composition transform in the next section.


##############################################
# Using your own data
# --------------------------------------
#
# Below is an example for processing image data using a dataset from a local directory. It assumes that we have ``train`` and ``val`` subdirectories with training and validation dataset. In this example we want to apply different sets of transforms for training and validation dataset:
#
# * For training data, we want to perform some **data augmentation**, and do random croping/resizing of the original image. We also introduce random horizontal flips.
# * For testing, we typically want to be consistent and always use the same images - thus we do not do any augmentation, just resizing. 
#
# We also normalize values by subtracting the mean, which was computed along the whole dataset.
#
# To be able to unify the code for train and validation datasets, we use a special trick and create a dictionary of transforms for the train and validation dataset:
#
# .. code-block:: Python
#
#   data_transforms = {
#        'train': 
#           transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#       ]),
#       'val': 
#           transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#       ]),
#   }
#
# Next, we define a similar dictionary of train and validation datasets by using ``datasets.ImageFolder`` class. This class allows us to create a dataset from all files in a folder, and apply any transformations to them:
#
# .. code-block:: Python
#
#   data_dir = 'data'
#
#   image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                              data_transforms[x])
#                     for x in ['train', 'val']}
# 
# Similarly we define a dictionary of dataloaders to prepare our datasets for training. They allow us to shuffle data and group them into batches of a specified size:
#
# .. code-block:: Python
#
#   batch_size = 4
#
#   dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
#                                                  batch_size=batch_size,
#                                                  shuffle=True, num_workers=4)
#                  for x in ['train', 'val']}
#
#   dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#
#   class_names = image_datasets['train'].classes

#########################################
# `< Previous <dataquickstart_tutorial.html>`_ |
# `Next > <buildmodel_tutorial.html>`_
# 