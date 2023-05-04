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

# %%
# Now, let us define few functions to perform tokenization:

def eng_tokenize(text):
    """
    Tokenize an English text and returns list of tokens
    """
    return [token.text for token in eng.tokenizer(text)]

def fin_tokenize(text):
    """
    Tokenize a Finnish text and returns list of tokens
    """
    return [token.text for token in fin.tokenizer(text)]

# %%
# Above function accepts a text and returns a list of words
# as shown below:

print(eng_tokenize("Have a good day!!!"))
print(fin_tokenize("Hyvää päivänjatkoa!!!"))

# %%
# Building the vocabulary
# -----------------------
# Let us consider an English sentence as the source and a Finnish sentence as the target.
#
# Vocabulary can be considered as the set of unique words we have in the dataset.
# We will build vocabulary for both our source and target now.
#
# Let us define a function to get tokens from elements of tuples in the iterator.
# The comments within the function specifies the need and working of it:

def get_tokens(data_iter, place):
    """
    Function to yield tokens from an iterator. Since, our iterator contains
    tuple of sentences (source and target), `place` parameters defines for which
    index to return the tokens for. `place=0` for source and `place=1` for target
    """
    for english, finnish in data_iter:
        if place == 0:
            yield eng_tokenize(english)
        else:
            yield fin_tokenize(finnish)

# %%
# Now, we will build vocabulary for source:

sourceVocab = build_vocab_from_iterator(
    get_tokens(dataPipe,0),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
sourceVocab.set_default_index(sourceVocab['<unk>'])

# %%
# The code above, builds the vocabulary from the iterator. In the above code block:
#
# * At line 2, we call the `get_tokens()` function with `place=0` as we need vocabulary for
#   source sentences.
# * At line 3, we set `min_freq=2`. This means, the function will skip those words that occurs
#   less than 2 times.
# * At line 4, we specify some special tokens:
#
#   * `<sos>` for start of sentence
#   * `<eos>` for end of senetence
#   * `<unk>` for unknown words. An example of unknown word is the one skipped because of
#     `min_freq=2`.
#   * `<pad>` is the padding token. While training, a model we mostly train in batches. In a
#     batch, there can be sentences of different length. So, we pad the shorter sentences with
#     `<pad>` token to make length of all sequences in the batch equal.
#
# * At line 5, we set `special_first=True`. Which means `<pad>` will get index 0, `<sos>` index 1,
#   `<eos>` index 2, and <unk> will get index 3 in the vocabulary.
# * At line 7, we set default index as index of `<unk>`. That means if some word is not in
#   vocbulary, we will use `<unk>` instead of that unknown word.
#
# Similarly, we will build vocabulary for target sentences:

targetVocab = build_vocab_from_iterator(
    get_tokens(dataPipe,1),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
targetVocab.set_default_index(targetVocab['<unk>'])

# %%
# Note that the example above shows how can we add special tokens to our vocabulary. The
# special tokens may change based on the requirements.
#
# Now, we can verify that special tokens are placed at the beginning and then other words.
# In the below code, `sourceVocab.get_itos()` returns a list with tokens at index based on
# vocabulary.

print(sourceVocab.get_itos()[:9])

