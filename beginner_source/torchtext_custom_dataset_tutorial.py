# -*- coding: utf-8 -*-
"""
Pre-process custom text dataset using Torchtext
==============================================

**Author**: `Anupam Sharma <https://anp-scp.github.io/>`_

This tutorial illustrates the usage of torchtext on a dataset that is not built-in. In the tutorial,
we will pre-process a dataset that can be further utilized to train a sequence-to-sequence
model for machine translation (something like, in this tutorial: `Sequence to Sequence Learning
with Neural Networks <https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%\
20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb>`_) but without using legacy version
of torchtext.

In this tutorial, we will learn how to:

* Read a dataset
* Tokenize sentence
* Apply transforms to sentence
* Perform bucket batching

Let us assume that we need to prepare a dataset to train a model that can perform English to
Finnish translation. We will use a tab-delimited Finnish - English sentence pairs provided by
the `Tatoeba Project <https://tatoeba.org/en>`_ which can be downloaded from this link: `Click
Here <https://www.manythings.org/anki/fin-eng.zip>`__.

Sentence pairs for other languages can be found in this link:

Link: `https://www.manythings.org/anki/ <https://www.manythings.org/anki/>`__
"""

# %%
# Setup
# -----
#
# First, download the dataset, extract the zip, and note the path to the file `fin.txt`.
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

FILE_PATH = 'data/fin.txt'
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
# we can perform various operations.
# Check `this tutorial <https://pytorch.org/data/beta/dp_tutorial.html>`_ for more details on
# data pipes.
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

# %%
# Numericalize sentences using vocabulary
# ---------------------------------------
# After building the vocabulary, we need to convert our sentences to corresponding indices.
# Let us define some functions for this:

def get_transform(vocab):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        ## converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        ## Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
        # 1 as seen in previous section
        T.AddToken(1, begin=True),
        ## Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
        # 2 as seen in previous section
        T.AddToken(2, begin=False)
    )
    return text_tranform

# %%
# Now, let us see how to use the above function. The function returns an object of `Transforms`
# which we will use on our sentence. Let us take a random sentence and check the working of
# the transform:

tempList = list(dataPipe)
someSetence = tempList[798][0]
print("Some sentence=", end="")
print(someSetence)
transformedSentence = get_transform(sourceVocab)(eng_tokenize(someSetence))
print("Transformed sentence=", end="")
print(transformedSentence)
indexToString = sourceVocab.get_itos()
for index in transformedSentence:
    print(indexToString[index], end=" ")

# %%
# In the above code,:
#
#   * At line 2, we take a source setence from list that we created from dataPipe at line 1
#   * At line 5, we get a transform based on a source vocabulary and apply it to a tokenized
#     sentence. Note that transforms take list of words and not a sentence.
#   * At line 8, we get the mapping of index to string and then use it get the transformed
#     sentence
#
# Now we will use functions of `dataPipe` to apply transform to all our sentences.
# Let us define some more functions for this.

def apply_transform(sequence_pair):
    """
    Apply transforms to sequence of tokens in a sequence pair
    """

    return (
        get_transform(sourceVocab)(eng_tokenize(sequence_pair[0])),
        get_transform(targetVocab)(fin_tokenize(sequence_pair[1]))
    )
dataPipe = dataPipe.map(apply_transform) ## Apply the function to each element in the iterator
tempList = list(dataPipe)
print(tempList[0])

# %%
# Make batches (with bucket batch)
# --------------------------------
# Generally, we train models in batches. While working for sequence to sequence models, it is
# recommended to keep the length of sequences in a batch similar. For that we will use
# `bucketbatch` function of `dataPipe`.
#
# Let us define some functions that will be used by the `bucketbatch` function.

def sort_bucket(bucket):
    """
    Function to sort a given bucket. Here, we want to sort based on the length of
    source and target sequence.
    """
    return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))

# %%
# Now, we will apply the `bucketbatch` function:

dataPipe = dataPipe.bucketbatch(
    batch_size = 4, batch_num=5,  bucket_num=1,
    use_in_batch_shuffle=False, sort_key=sort_bucket
)

# %%
# In the above code block:
#
#   * We keep batch size = 4.
#   * `batch_num` is the number of batches to keep in a bucket
#   * `bucket_num` is the number of buckets to keep in a pool for shuffling
#   * `sort_key` specifies the function that takes a bucket and sorts it
#
# Now, let us consider a batch of source sentences as `X` and a batch of target sentences as `y`.
# Generally, while training a model, we predict on a batch of `X` and compare the result with `y`.
# But, a batch in our `dataPipe` is of the form `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`:

print(list(dataPipe)[0])
# %%
# So, we will now convert them into the form: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`.
# For this we will write a small function:

def separate_source_target(sequence_pairs):
    """
    input of form: `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`
    output of form: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`
    """
    sources,targets = zip(*sequence_pairs)
    return sources,targets

## Apply the function to each element in the iterator
dataPipe = dataPipe.map(separate_source_target)
print(list(dataPipe)[0])

# %%
# Now, we have the data as desired.
#
# Padding
# -------
# As discussed earlier while building vocabulary, we need to pad shorter sentences in a batch to
# make all the sequences in a batch of equal length. We can perform padding as follows:

def apply_padding(pair_of_sequences):
    """
    Convert sequnces to tensors and apply padding
    """
    return (T.ToTensor(0)(list(pair_of_sequences[0])), T.ToTensor(0)(list(pair_of_sequences[1])))
## `T.ToTensor(0)` returns a transform that converts the sequence to `torch.tensor` and also applies
# padding. Here, `0` is passed to the constructor to specify the index of the `<pad>` token in the
# vocabulary.
dataPipe = dataPipe.map(apply_padding)

# %%
# Now, we can use the index to string mapping to see how the sequence would look with tokens
# instead of indices:

sourceItoS = sourceVocab.get_itos()
targetItoS = targetVocab.get_itos()

def show_some_transformed_sentences(data_pipe):
    """
    Function to show how the senetnces look like after applying all transforms.
    Here we try to print actual words instead of corresponding index
    """
    for sources,targets in data_pipe:
        if sources[0][-1] != 0:
            continue # Just to visualize padding of shorter sentences
        for i in range(4):
            source = ""
            for token in sources[i]:
                source += " " + sourceItoS[token]
            target = ""
            for token in targets[i]:
                target += " " + targetItoS[token]
            print(f"Source: {source}")
            print(f"Traget: {target}")
        break

show_some_transformed_sentences(dataPipe)
# %%
# In the above output we can observe that the shorter sentences are padded with `<pad>`. Now, we
# can use this dataPipe while writing our training function.
#
# Some parts of this tutorial was inspired from this article:
# `Link https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71 \
#  <https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71 >`__.
