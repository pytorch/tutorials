"""
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
# If not properly organized, code for processing data samples can quickly get messy and become hard to maintain. Since different model architectures can be applied to many data types, we ideally want our dataset code to be decoupled from our model training code. To this end, PyTorch provides a simple Datasets interface for linking managing collections of data. 
# 
# A whole set of example datasets such as Fashion MNIST that implement this interface are built into PyTorch extension libraries. They are subclasses of torch.utils.data.Dataset that have parameters and functions specific to the type of data and the particular dataset. The actual data samples can be downloaded from the internet.These are useful for benchmarking and testing your models before training on your own custom datasets.
# 
# You can find some of them below. 
#
#  - `Image Datasets <https://pytorch.org/docs/stable/torchvision/datasets.html>`_
#  - `Text Datasets  <https://pytorch.org/text/datasets.html>`_
#  - `Audio Datasets <https://pytorch.org/audio/datasets.html>`_
#

#################################################################
# Iterating through a Dataset
# -----------------
# 
# Once we have a Dataset we can index it manually like a list `clothing[index]`. 
# 
# Here is an example of how to load the `Fashion-MNIST<https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/>`_ dataset from torch vision.
# `Fashion-MNIST <https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/>`_ is a dataset of Zalando’s article images consisting of of 60,000 training examples and 10,000 test examples. 
# Each example is comprised of a 28×28 grayscale image, associated with a label from one of 10 classes. Read more `here <https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist>`_.
#
# To load the FashionMNIST Dataset we need to provide the following three parameters:
#  - root is the path where the train/test data is stored. 
#  - train includes the training dataset. 
#  - setting download to true downloads the data from the internet if it's not available at root.


import torch 
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

clothing = datasets.FashionMNIST(root='data', train=True, download=True)
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt', 7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols*rows +1):
    sample_idx = np.random.randint(len(clothing))
    img = clothing[sample_idx][0][0,:,:]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[clothing[sample_idx][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

#################################################################
# .. figure:: /_static/img/quickstart/fashion_mnist.png
#    :alt: fashion_mnist
#

#################################################################
# Creating a Custom Dataset
# -----------------
#
# To work with your own data lets look at the a simple custom image Dataset implementation:
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
# Imports 
# -------
# 
# Import os for file handling, torch for PyTorch, `pandas <https://pandas.pydata.org/>`_ for loading labels, `torch vision <https://pytorch.org/blog/pytorch-1.7-released/>`_ to read image files, and Dataset to implement the Dataset interface.
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
# Init
# -----------------
#
# The init function is used for all the first time operations when our Dataset is loaded. In this case we use it to load our annotation labels to memory and the keep track of directory of our image file. Note that different types of data can take different init inputs you are not limited to just an annotations file, directory_path and transforms but for images this is a standard practice.
# A sample csv annotations file may look as follows:
# tshirt1.jpg, 0
# tshirt2.jpg, 0
#     ......
# ankleboot999.jpg, 9
# 
# Example:
# 

def __init__(self, annotations_file, img_dir, transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform

#################################################################
# __len__
# -----------------
#
# The __len__ function is very simple here we just need to return the number of samples in our dataset. 
# 
# Example:

def __len__(self):
    return len(self.img_labels)

#################################################################
# __getitem__
# -----------------
#
# The __getitem__ function is the most important function in the Datasets interface this. It takes a tensor or an index as input and returns a loaded sample from you dataset at from the given indecies.
# 
# In this sample if provided a tensor we convert the tensor to a list containing our index. We then load the file at the given index from our image directory as well as the image label from our pandas annotations DataFrame. This image and label are then wrapped in a single sample dictionary which we can apply a Transform on and return. To learn more about Transforms see the next section of the Blitz. 
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
# Now we have a organized mechansim for managing data which is great, but there is still a lot of manual work we would have to do train a model with our Dataset. 
# 
# For example we would have to manually maintain the code for: 
# * Batching 
# * Suffling 
# * Parallel batch distribution 
# 
# The PyTorch Dataloader *torch.utils.data.DataLoader* is an iterator that handles all of this complexity for us enabling us to load a dataset and focusing on train our model.

dataloader = DataLoader(clothing, batch_size=4, shuffle=True, num_workers=0)

#################################################################
# With this we have all we need to know to load an process data of any kind in PyTorch to train deep learning models.
# 
# Next: Learn more about how to `transform that data for training <transforms_tutorial.html>`_.
#
# .. include:: /beginner_source/quickstart/qs_toc.txt
#
