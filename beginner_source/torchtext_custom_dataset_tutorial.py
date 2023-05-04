# -*- coding: utf-8 -*-
"""
Preaparing custom text dataset using Torchtext
==============================================

**Author**: `Anupam Sharma <https://anp-scp.github.io/>`_

This tutorial is regarding the preparation of a text dataset using Torchtext. In the tutorial, we
will be preparing a  custom dataset that can be further utilized to train a sequence-to-sequence
model for machine translation (something like, in this tutorial: `Sequence to Sequence Learning
with Neural Networks <https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%\
20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb>`_) but using Torchtext 0.15.0 instead
of a legacy version.

In this tutorial, we will learn how to:

* Read a dataset
* Tokenize sentence
* Apply transforms to sentence
* Perform bucket batching

Let us assume that we need to prepare a dataset to train a model that can perform English to
Finnish translation. We will use a tab-delimited Finnish - English sentence pairs provided by
the `Tatoeba Project <https://tatoeba.org/en>`_ which can be downloaded from this link: `Click
Here <https://www.manythings.org/anki/fin-eng.zip>`__
"""

# %%
# Setup
# -----
#
# First, download the dataset, extract the zip, and note the path to the file `fin.txt`.
# The dataset can be downloaded from this link: `Click Here <https://www.manythings.org/anki/fin\
# -eng.zip>`__ .
#
# Ensure that following packages are installed:
#
# * `Torchdata 0.6.0 <https://pytorch.org/data/beta/index.html>`_ (Installation instructions: `C\
#   lick here <https://github.com/pytorch/data>`__)
# * `Torchtext 0.15.0 <https://pytorch.org/text/stable/index.html>`_ (Installation instructions:\
#   `Click here <https://github.com/pytorch/text>`__)
# * Spacy (Docs: `Click here <https://spacy.io/usage>`__)
#
# Here, we are using `Spacy` to tokenize text. In simple words tokenization means to
# convert a sentence to list of words. Spacy is a python package used for various Natural
# Language Processing (NLP) tasks.
#
# Download the English and Finnish models from spacy as shown below: ::
#
#   python -m spacy download en_core_web_sm
#   python -m spacy download fi_core_news_sm


# %%
# Let us start by importing required modules:

import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator
eng = spacy.load("en_core_web_sm") # Load the English model to be used for tokenizing
fin = spacy.load("fi_core_news_sm") # Load the Finnish model to be used for tokenizing

# %%
# Now we will load the dataset

FILE_PATH = 'fin.txt'
dataPipe = dp.iter.IterableWrapper([FILE_PATH])
dataPipe = dp.iter.FileOpener(dataPipe, mode='rb')
dataPipe = dataPipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)

# %%
# In the above code block, we are doing following things:
#
# 1. At line 2, we are creating an iterable of filenames
# 2. At line 3, we pass the iterable to `FileOpener` which then
#    opens the file in read mode
# 3. At line 4, we call a function to parse the file, which
#    again returns an iterable of tuples representing each rows
#    of the tab-delimited file
#
# Data pipes can be thought of something like a dataset object, on which
# we can perform various operations. Check `this tutorial <https://pytorch.org\
# /data/beta/dp_tutorial.html>`_ for more details on data pipes.
#
# We can verify if the iterable has the pair of sentences as shown
# below:

for sample in dataPipe:
    print(sample)
    break

# %%
# Note that we also have attribution details along with pair of sentences. We will
# write a small function to remove the attribution details:

def remove_attribution(row):
    """
    Function to keep the first two elements in a tuple
    """
    return row[:2]
dataPipe = dataPipe.map(remove_attribution)

# %%
# The `map` function at line 2 in above code block can be used to apply some function
# on each elements of data pipe. Now, we can verify that the data pipe only contains
# pair of sentences.


for sample in dataPipe:
    print(sample)
    break

