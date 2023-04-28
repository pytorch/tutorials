"""
Loading data in PyTorch
=======================
PyTorch features extensive neural network building blocks with a simple,
intuitive, and stable API. PyTorch includes packages to prepare and load
common datasets for your model.

Introduction
------------
At the heart of PyTorch data loading utility is the
`torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__
class. It represents a Python iterable over a dataset. Libraries in
PyTorch offer built-in high-quality datasets for you to use in
`torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__.
These datasets are currently available in:

* `torchvision <https://pytorch.org/vision/stable/datasets.html>`__
* `torchaudio <https://pytorch.org/audio/stable/datasets.html>`__
* `torchtext <https://pytorch.org/text/stable/datasets.html>`__

with more to come.
Using the ``yesno`` dataset from ``torchaudio.datasets.YESNO``, we will
demonstrate how to effectively and efficiently load data from a PyTorch
``Dataset`` into a PyTorch ``DataLoader``.
"""



######################################################################
# Setup
# -----
# Before we begin, we need to install ``torchaudio`` to have access to the
# dataset.

# pip install torchaudio

#######################################################
# To run in Google Colab, uncomment the following line:

# !pip install torchaudio

#############################
# Steps
# -----
#
# 1. Import all necessary libraries for loading our data
# 2. Access the data in the dataset
# 3. Loading the data
# 4. Iterate over the data
# 5. [Optional] Visualize the data
#
#
# 1. Import necessary libraries for loading our data
# ---------------------------------------------------------------
#
# For this recipe, we will use ``torch`` and ``torchaudio``. Depending on
# what built-in datasets you use, you can also install and import
# ``torchvision`` or ``torchtext``.
#

import torch
import torchaudio


######################################################################
# 2. Access the data in the dataset
# ---------------------------------------------------------------
#
# The ``yesno`` dataset in ``torchaudio`` features sixty recordings of one
# individual saying yes or no in Hebrew; with each recording being eight
# words long (`read more here <https://www.openslr.org/1/>`__).
#
# ``torchaudio.datasets.YESNO`` creates a dataset for ``yesno``.
torchaudio.datasets.YESNO(
     root='./',
     url='http://www.openslr.org/resources/1/waves_yesno.tar.gz',
     folder_in_archive='waves_yesno',
     download=True)

###########################################################################
# Each item in the dataset is a tuple of the form: (waveform, sample_rate,
# labels).
#
# You must set a ``root`` for the ``yesno`` dataset, which is where the
# training and testing dataset will exist. The other parameters are
# optional, with their default values shown. Here is some additional
# useful info on the other parameters:

# * ``download``: If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
#
# Let’s access our ``yesno`` data:
#

# A data point in ``yesno`` is a tuple (waveform, sample_rate, labels) where labels
# is a list of integers with 1 for yes and 0 for no.
yesno_data = torchaudio.datasets.YESNO('./', download=True)

# Pick data point number 3 to see an example of the the ``yesno_data``:
n = 3
waveform, sample_rate, labels = yesno_data[n]
print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))


######################################################################
# When using this data in practice, it is best practice to provision the
# data into a “training” dataset and a “testing” dataset. This ensures
# that you have out-of-sample data to test the performance of your model.
#
# 3. Loading the data
# ---------------------------------------------------------------
#
# Now that we have access to the dataset, we must pass it through
# ``torch.utils.data.DataLoader``. The ``DataLoader`` combines the dataset
# and a sampler, returning an iterable over the dataset.
#

data_loader = torch.utils.data.DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=True)


######################################################################
# 4. Iterate over the data
# ---------------------------------------------------------------
#
# Our data is now iterable using the ``data_loader``. This will be
# necessary when we begin training our model! You will notice that now
# each data entry in the ``data_loader`` object is converted to a tensor
# containing tensors representing our waveform, sample rate, and labels.
#

for data in data_loader:
  print("Data: ", data)
  print("Waveform: {}\nSample rate: {}\nLabels: {}".format(data[0], data[1], data[2]))
  break


######################################################################
# 5. [Optional] Visualize the data
# ---------------------------------------------------------------
#
# You can optionally visualize your data to further understand the output
# from your ``DataLoader``.
#

import matplotlib.pyplot as plt

print(data[0][0].numpy())

plt.figure()
plt.plot(waveform.t().numpy())


######################################################################
# Congratulations! You have successfully loaded data in PyTorch.
#
# Learn More
# ----------
#
# Take a look at these other recipes to continue your learning:
#
# - `Defining a Neural Network <https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html>`__
# - `What is a state_dict in PyTorch <https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html>`__
