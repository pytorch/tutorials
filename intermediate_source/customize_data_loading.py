# -*- coding: utf-8 -*-
"""
Customizing Data Loading Tutorial
====================================
**Author**: `Xiang Gao <https://zasdfgbnm.github.io>`_

This tutorial demonstrates how to create custom batch sampler and
collate function for advanced data loading.

Readers of this tutorial are assumed to have read the
`Data Loading and Processing Tutorial <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_
"""

from __future__ import print_function, division
from torch import tensor, full_like, long
from itertools import islice
from os.path import join
from numpy import load
from torch.utils.data import TensorDataset, ConcatDataset
from torch.utils.data.dataloader import default_collate, DataLoader
from math import ceil

######################################################################
# The dataset we are going to deal with is a simplified version of
# `the ANI-1 dataset <https://github.com/isayev/ANI1_dataset>`_. The
# ANI-1 dataset is a dataset of energies of various molecules.
# The dataset itself is complicated and contains many information
# not interesting to this tutorial, so a simplified dataset that only
# contains coordinates and energies of 100 molecules with easier data
# organization is created for the purpose of this tutorial. Note that
# the species (carbon, hydrogen, etc.) information, which is necessary
# for serious machine learning, is discarded in this simplified dataset.
#
# .. note::
#     Download the dataset from `here <https://download.pytorch.org/tutorial/ANI-1-simplified.zip>`_.
#
# In chemistry, people are interested in building deep learning models that
# compute the energy of a molecule from the (x,y,z) coordinates of all its
# atoms.
#
# Here in this dataset, each `.xyz.npy` file stores a numpy ndarray of shape
# (N, atoms, 3). This ndarray stores N different sets of atom coordinates 
# (each set of atom coordinates is called a conformation) of one molecule.
# The values of N can be different from molecules to molecule. The energies
# of these conformations are stored in a ndarray of shape (N,) in the `.energy.npy`
# file with the same basename.
#
# The difficulty is, different molecules have different number of atoms, i.e.
# we can not simply stack N coordinate tensors of shape (atoms, 3) from N
# different molecules to get a tensor of shape (N, atoms, 3) because these 
# tensors have different shape. During training, batching is critical for
# performance, but to construct a size M batch, we can not stack M different
# coordinate tensors from the same molecule either, although there is no shape
# issue, because this gives poor gradients.
#
# The strategy to overcome this problem used here in this this tutorial is to
# create a custom batch sampler and collate function to sample from 4 different
# molecules, each molecules sample a chunk of 64 coordinate tensors, to get a
# size 256 batch.
#
# Dataset
# -------
#
# For the simplified ANI-1 dataset, since each `.npy` file already contains
# data of different conformations of the same molecule, all we need to do is
# to load the ndarrays for each molecule, convert these ndarrays to tensors,
# wrap these tensors with `TensorDataset` in order to be able to iterate on
# different conformations, and finally concat these datasets of different
# molecules using `ConcatDataset`:
#

def load_dataset(root_dir):
    molecules = []
    for i in range(100):
        xyz_file = join(root_dir, '{}.xyz.npy'.format(i))
        energy_file = join(root_dir, '{}.energy.npy'.format(i))
        xyz = tensor(load(xyz_file))
        energy = tensor(load(energy_file))
        molecule_id = full_like(energy, i).type(long)
        molecules.append(TensorDataset(molecule_id, xyz, energy))
    return ConcatDataset(molecules)

######################################################################
# Now let's take a look at the first 5 elements of the dataset we get:
#

dataset = load_dataset('ANI-1-simplified/')
print('Length of data set:', len(dataset))
for molecule_id, xyz, energy in islice(dataset, 5):
    print(molecule_id.item(), xyz.shape, energy.shape)

######################################################################
# Batch Sampler
# -------------
#
# Now it's time to sample batches according to the strategies explained
# before. This job is done by samplers and batch samplers. Samplers and
# batch samplers cooperate with dataset by generating indices of data
# that should be retrieved from dataset. The difference between samplers
# and batch samplers is, a batch sampler generates a list of indices, while
# a sampler should generate a single index.
#
# To implement a sampler, you need to extend `torch.utils.data.sampler.Sampler`
# and implement the `__iter__` and `__len__` methods. To implement a batch
# sampler, you also need to implement these two methods, but you do not
# need to extend any base class.
# 
# Pytorch has many builtin samplers supporting sequential sampling, random
# sampling, subset sampling, etc. Pytorch also has a builtin batch sampler
# that wraps a sampler and sample it for batch_size times to get a minibatch.
# For most use cases, these builtin classes should be enough for use. But
# for the purpose of this tutorial, we have to implement our own batch
# sampler for our advanced requirement.
#
# The the first minibatch of the below batch sampler is constructed by taking
# the first 64 conformations of molecule 0-3, and the second minibatch takes
# the first 64 conformations of molecule 4-8, ... , after the first 64
# conformations of all molecules are iterated, then conformations 64-127 will
# be used for each molecule, and so on.
#

class ANIBatchSampler(object):

    def __init__(self, concat_source):
        self.concat_source = concat_source

    def _concated_index(self, molecule, conformation):
        """
        Get the index in the  dataset of the specified conformation
        of the specified molecule.
        """
        src = self.concat_source
        cumulative_sizes = [0] + src.cumulative_sizes
        return cumulative_sizes[molecule] + conformation

    def __iter__(self):
        src = self.concat_source
        sizes = [len(x) for x in src.datasets]
        """Number of conformations of each molecule"""
        unfinished = list(zip(range(100), [0] * 100))
        """List of pairs (molecule, progress) storing the current progress
        of iterating each molecules."""
        
        batch = []
        batch_molecules = 0
        """The number of molecules already in batch"""
        while len(unfinished) > 0:
            new_unfinished = []
            for molecule, progress in unfinished:
                size = sizes[molecule]
                # the last incomplete chunk is not dropped
                end = min(progress + 64, size)
                if end < size:
                    new_unfinished.append((molecule, end))
                batch += [self._concated_index(molecule, x) for x in range(progress, end)]
                batch_molecules += 1
                if batch_molecules >= 4:
                    yield batch
                    batch = []
                    batch_molecules = 0
            unfinished = new_unfinished

        # the last incomplete batch is not dropped
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(self.concat_source)

######################################################################
# Now let's take a look at the a few elements of the batch sampler:
#

def pretty_print(indices):
    """Pretty print indices"""
    pretty_list = [indices[0], indices[0]]
    for i in indices[1:] + [None]:
        last = pretty_list.pop()
        if i != last + 1:
            first = pretty_list.pop()
            s = '{}...{}'.format(first,last) if first != last else first
            pretty_list += [s, i, i]
        else:
            pretty_list.append(i)
    pretty_list = pretty_list[:-2]
    print(pretty_list)

batch_sampler = ANIBatchSampler(dataset)
for i in islice(batch_sampler, 5):
    pretty_print(i)

######################################################################
# From the result, we can see the indices starts from 0 and increase
# contiguously for 64 times, then there is a jump followed by another
# contiguous increase. This is exactly what we want. The jump means
# we have already done with all the 64 minibatches and its time to
# move on to a new molecule.
#
# Collate function
# ----------------
#
# Each raw minibatch sampled from dataset is a list of something.
# Depending on the dataset, it can be list of tensors, list of
# dictionaries, etc. Each element of the list is obtained by getting
# the item of the dataset using the index from batch sampler. But
# list is not the most convenient form to be used by a model. For
# example, usually if we get a raw minibatch as a list of tensors
# of the same shape, we want to stack these tensors on the first
# dimension. Collate function is a function called by the dataloader
# to obtain a form convenient for model to use. A collate function
# only takes a single parameter which is the list.
# 
# Pytorch comes with a default collate function `torch.utils.data.dataloader.default_collate`.
# This builtin collate function would stack tensors, convert numpy
# ndarray to tensors and then stack them, combine numbers to a
# 1-dimensional tensor, and recursively deal with dictionaries
# and sequences. This behavior can deal with most cases well.
#
# Since we only want to stack tensors from the same molecule, we need
# to create a custom collate function:
#

def collate(batch):
    by_molecules = {}
    for molecule_id, xyz, energy in batch:
        molecule_id = molecule_id.item()
        if molecule_id not in by_molecules:
            by_molecules[molecule_id] = []
        by_molecules[molecule_id].append((xyz, energy))
    for i in by_molecules:
        by_molecules[i] = default_collate(by_molecules[i])
    return by_molecules

######################################################################
# Dataloader
# ----------
#
# We now have all components ready, and it is simple to create a
# dataloader:
#

dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate)

######################################################################
# Note that if you specify a custom batch sampler, you should not specify
# `batch_size`, `shuffle`, `sampler`, and `drop_last`. All these features
# should be implemented in the custom batch sampler.
#
# Now let's iterate the dataloader to see what we get:
#

for i in islice(dataloader, 3):
    print('Batch:')
    for j in i:
        print('molecule', j, i[j][0].shape, i[j][1].shape)
