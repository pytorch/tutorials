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
