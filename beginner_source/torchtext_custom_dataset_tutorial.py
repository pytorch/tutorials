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

