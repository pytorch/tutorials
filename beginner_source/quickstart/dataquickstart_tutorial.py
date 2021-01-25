"""

`Quickstart <quickstart_tutorial.html>`_ >
`Tensors <tensor_tutorial.html>`_ > 
**Datasets & DataLoaders** >
`Transforms <transforms_tutorial.html>`_ >
`Build Model <buildmodel_tutorial.html>`_ >
`Autograd <autograd_tutorial.html>`_ >
`Optimization <optimization_tutorial.html>`_ >
`Save & Load Model <saveloadrun_tutorial.html>`_

Datasets & Dataloaders
===================

"""

#################################################################
# Getting Started With Data in PyTorch
# -----------------
#
# Before we start building models with PyTorch, let's first learn how to load and process data. Data can be sourced from local files, cloud datastores and database queries. It comes in all sorts of forms and formats from structured tables to image, audio, text, video files and more. 
#

###############################################################
# .. figure:: /_static/img/quickstart/typesdata.png
#    :alt: typesdata
# 

############################################################
# Different data types require different python libraries to load and process such as `openCV <https://opencv.org/>`_ and `PIL <https://pillow.readthedocs.io/en/stable/reference/Image.html>`_ for images, `NLTK <https://www.nltk.org/>`_ and `spaCy <https://spacy.io/>`_ for text and `Librosa <https://librosa.org/doc/latest/index.html>`_ for audio. 
# 
# If not properly organized, code for processing data samples can quickly get messy and become hard to maintain. Since different model architectures can be applied to many data types, we ideally want our dataset code to be decoupled from our model training code. To this end, PyTorch provides a simple Datasets interface for linking and managing collections of data. 
# 
# A whole set of example datasets such as Fashion MNIST that implement this interface are built into PyTorch extension libraries. They are subclasses of `torch.utils.data.Dataset` that have parameters and functions specific to the type of data and the particular dataset. The actual data samples can be downloaded from the internet. These are useful for benchmarking and testing your models before training on your own custom datasets.
# 
# You can find some of these datasets 
# here: `Image Datasets <https://pytorch.org/docs/stable/torchvision/datasets.html>`_,
# `Text Datasets  <https://pytorch.org/text/stable/datasets.html>`_, and
# `Audio Datasets <https://pytorch.org/audio/stable/datasets.html>`_
#

############################################################
# Loading a Dataset
# -------------------
# 
# Here is an example of how to load the `Fashion-MNIST <https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/>`_ dataset from torch vision.
# `Fashion-MNIST <https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/>`_ is a dataset of Zalando’s article images consisting of of 60,000 training examples and 10,000 test examples. 
# Each example is comprised of a 28×28 grayscale image, associated with a label from one of 10 classes. Read more `here <https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist>`_.
#
# To load the FashionMNIST Dataset we need to provide the following three parameters:
#  - ``root`` is the path where the train/test data is stored. 
#  - ``train`` includes the training dataset. 
#  - ``download=True`` downloads the data from the internet if it's not available at root.
#

import torch 
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

clothing = datasets.FashionMNIST(
 'data',                # specifies data directory to store data
 train=True,            # specifies training or test dataset to use
 transform=None,        # specifies transforms to apply to features (images)
 target_transform=None, # specifies transforms to apply to labels
 download=True)        # should the data be downloaded from the Internet


#################################################################
# Iterating and Visualizing the Dataset
# -----------------
# 
# Once we have the ``clothing`` dataset, we can index it manually like a list: ``clothing[index]``. Then use ``matplotlib`` to visualize the dataset.

labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt', 7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols*rows +1):
    sample_idx = np.random.randint(len(clothing))
    img = clothing[sample_idx][0]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[clothing[sample_idx][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

#################################################################
# ..
#  .. figure:: /_static/img/quickstart/fashion_mnist.png
#    :alt: fashion_mnist
#

#################################################################
# Creating a Custom Dataset
# -----------------
#
# To work with your own data, we can implement a custom class that inherits from ``Dataset``. This custom class must implement three functions: `__init__`, `__len__`, and `__getitem__`. Let's look at a custom image dataset implementation. In this example, we have a number of images stored in a directory, and their labels stored separately in a CSV file. Here's what it looks like; in the following sections, we will break down what's happening in each function.
#

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1:]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample 
        
#################################################################
# Import the packages
# -------
# 
# Import ``os`` for file handling, ``torch`` for PyTorch, `pandas <https://pandas.pydata.org/>`_ for loading labels, `torch vision <https://pytorch.org/blog/pytorch-1.7-released/>`_ to read image files, and ``Dataset`` to implement the Dataset interface.
# 
# Example:
#

import os
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#################################################################
# __init__
# -----------------
#
# The __init__ function is run once when instantiating our Dataset object. Here, we use it to load 
# the directory containing the images, and their labels (contained in a csv file). While creating the 
# Dataset object, we can optionally pass it the transform that should be run on the images.
#
# The labels.csv file looks like: ::
#
#     tshirt1.jpg, 0
#     tshirt2.jpg, 0
#     ......
#     ankleboot999.jpg, 9
# 
# Example:
# 

def __init__(self, labels_file, img_dir, transform=None):
    self.img_labels = pd.read_csv(labels_file)
    self.img_dir = img_dir
    self.transform = transform

#################################################################
# __len__
# -----------------
#
# The __len__ function returns the number of samples in our dataset. 
# 
# Example:

def __len__(self):
    return len(self.img_labels)

#################################################################
# __getitem__
# -----------------
#
# The __getitem__ function is the most important function in the Datasets interface. It takes a tensor or an index as input and returns a loaded sample from your dataset at the given indices.
# 
# If provided a tensor as an index, we convert the tensor to a list first. We then load the file at the given index from our image directory, as well as the image label from our pandas annotations DataFrame. This image and label are then wrapped in a single sample dictionary which we can apply a transform on and return. Transforms will be discussed in more detail in the next section: `Transforms <transforms_tutorial.html>`_
# 
# Example:
#

def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    img_path = os.path.join(self.root_dir,
                            self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1:]
    sample = {'image': image, 'label': label}
    if self.transform:
        sample = self.transform(sample)
    return sample 

#################################################################
# Preparing your data for training with DataLoaders
# -------------------------------------------------
#
# Now we have an organized mechanism for managing data which is great, but there is still a lot of manual work we would have to do to train a model with our Dataset. 
# 
# For example we would have to manually maintain the code for: 
#
# * Batching 
# * Shuffling 
# * Parallel batch distribution 
# 
# The PyTorch Dataloader ``torch.utils.data.DataLoader`` is an iterator that handles all of this complexity for us, enabling us to load a dataset and focus on training our model.

dataloader = DataLoader(clothing, batch_size=4, shuffle=True, num_workers=0)

###########################
# Iterate through the Dataset
# --------------------------
#
# We have loaded that dataset into the ``dataloader`` and can iterate through the dataset as needed.
# Below is a simple example of how to iterate and display an image or return a label count:


# Display image and label.
for train_features, train_labels in dataloader.dataset:
    print(train_labels)
    plt.imshow(train_features, cmap='gray')
    plt.show()
    break;

# Count the number of occurances for label number 9 which is for the 'Bag'
count = 0
for train_features, train_labels in dataloader.dataset:
    if(train_labels==9):
        count+=1
print(count)

#################################################################
# With this we have all we need to know to load and process data of any kind in PyTorch to train deep learning models.
# 
