# -*- coding: utf-8 -*-
"""
Preprocess custom text dataset using Torchtext
===============================================

**Author**: `Anupam Sharma <https://anp-scp.github.io/>`_

This tutorial illustrates the usage of torchtext on a dataset that is not built-in. In the tutorial,
we will preprocess a dataset that can be further utilized to train a sequence-to-sequence
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
German translation. We will use a tab-delimited German - English sentence pairs provided by
the `Tatoeba Project <https://tatoeba.org/en>`_ which can be downloaded from
`this link <https://www.manythings.org/anki/deu-eng.zip>`__.

Sentence pairs for other languages can be found in `this link <https://www.manythings.org/anki/>`\
__.
"""

# %%
# Setup
# -----
#
# First, download the dataset, extract the zip, and note the path to the file `deu.txt`.
#
# Ensure that following packages are installed:
#
# * `Torchdata 0.6.0 <https://pytorch.org/data/beta/index.html>`_ (`Installation instructions \
#   <https://github.com/pytorch/data>`__)
# * `Torchtext 0.15.0 <https://pytorch.org/text/stable/index.html>`_ (`Installation instructions \
#   <https://github.com/pytorch/text>`__)
# * `Spacy <https://spacy.io/usage>`__
#
# Here, we are using `Spacy` to tokenize text. In simple words tokenization means to
# convert a sentence to list of words. Spacy is a python package used for various Natural
# Language Processing (NLP) tasks.
#
# Download the English and German models from Spacy as shown below:
#
# .. code-block:: shell
#
#    python -m spacy download en_core_web_sm
#    python -m spacy download de_core_news_sm
#


# %%
# Let us start by importing required modules:

import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator
eng = spacy.load("en_core_web_sm") # Load the English model to tokenize English text
de = spacy.load("de_core_news_sm") # Load the German model to tokenize German text

# %%
# Now we will load the dataset

FILE_PATH = 'data/deu.txt'
data_pipe = dp.iter.IterableWrapper([FILE_PATH])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)

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
# DataPipes can be thought of something like a dataset object, on which
# we can perform various operations.
# Check `this tutorial <https://pytorch.org/data/beta/dp_tutorial.html>`_ for more details on
# DataPipes.
#
# We can verify if the iterable has the pair of sentences as shown
# below:

for sample in data_pipe:
    print(sample)
    break

# %%
# Note that we also have attribution details along with pair of sentences. We will
# write a small function to remove the attribution details:

def removeAttribution(row):
    """
    Function to keep the first two elements in a tuple
    """
    return row[:2]
data_pipe = data_pipe.map(removeAttribution)

# %%
# The `map` function at line 6 in above code block can be used to apply some function
# on each elements of `data_pipe`. Now, we can verify that the `data_pipe` only contains
# pair of sentences.


for sample in data_pipe:
    print(sample)
    break

# %%
# Now, let us define few functions to perform tokenization:

def engTokenize(text):
    """
    Tokenize an English text and return a list of tokens
    """
    return [token.text for token in eng.tokenizer(text)]

def deTokenize(text):
    """
    Tokenize a German text and return a list of tokens
    """
    return [token.text for token in de.tokenizer(text)]

# %%
# Above function accepts a text and returns a list of words
# as shown below:

print(engTokenize("Have a good day!!!"))
print(deTokenize("Haben Sie einen guten Tag!!!"))

# %%
# Building the vocabulary
# -----------------------
# Let us consider an English sentence as the source and a German sentence as the target.
#
# Vocabulary can be considered as the set of unique words we have in the dataset.
# We will build vocabulary for both our source and target now.
#
# Let us define a function to get tokens from elements of tuples in the iterator.


def getTokens(data_iter, place):
    """
    Function to yield tokens from an iterator. Since, our iterator contains
    tuple of sentences (source and target), `place` parameters defines for which
    index to return the tokens for. `place=0` for source and `place=1` for target
    """
    for english, german in data_iter:
        if place == 0:
            yield engTokenize(english)
        else:
            yield deTokenize(german)

# %%
# Now, we will build vocabulary for source:

source_vocab = build_vocab_from_iterator(
    getTokens(data_pipe,0),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
source_vocab.set_default_index(source_vocab['<unk>'])

# %%
# The code above, builds the vocabulary from the iterator. In the above code block:
#
# * At line 2, we call the `getTokens()` function with `place=0` as we need vocabulary for
#   source sentences.
# * At line 3, we set `min_freq=2`. This means, the function will skip those words that occurs
#   less than 2 times.
# * At line 4, we specify some special tokens:
#
#   * `<sos>` for start of sentence
#   * `<eos>` for end of sentence
#   * `<unk>` for unknown words. An example of unknown word is the one skipped because of
#     `min_freq=2`.
#   * `<pad>` is the padding token. While training, a model we mostly train in batches. In a
#     batch, there can be sentences of different length. So, we pad the shorter sentences with
#     `<pad>` token to make length of all sequences in the batch equal.
#
# * At line 5, we set `special_first=True`. Which means `<pad>` will get index 0, `<sos>` index 1,
#   `<eos>` index 2, and <unk> will get index 3 in the vocabulary.
# * At line 7, we set default index as index of `<unk>`. That means if some word is not in
#   vocabulary, we will use `<unk>` instead of that unknown word.
#
# Similarly, we will build vocabulary for target sentences:

target_vocab = build_vocab_from_iterator(
    getTokens(data_pipe,1),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
target_vocab.set_default_index(target_vocab['<unk>'])

# %%
# Note that the example above shows how can we add special tokens to our vocabulary. The
# special tokens may change based on the requirements.
#
# Now, we can verify that special tokens are placed at the beginning and then other words.
# In the below code, `source_vocab.get_itos()` returns a list with tokens at index based on
# vocabulary.

print(source_vocab.get_itos()[:9])

# %%
# Numericalize sentences using vocabulary
# ---------------------------------------
# After building the vocabulary, we need to convert our sentences to corresponding indices.
# Let us define some functions for this:

def getTransform(vocab):
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
# which we will use on our sentence. Let us take a random sentence and check how the transform
# works.

temp_list = list(data_pipe)
some_sentence = temp_list[798][0]
print("Some sentence=", end="")
print(some_sentence)
transformed_sentence = getTransform(source_vocab)(engTokenize(some_sentence))
print("Transformed sentence=", end="")
print(transformed_sentence)
index_to_string = source_vocab.get_itos()
for index in transformed_sentence:
    print(index_to_string[index], end=" ")

# %%
# In the above code,:
#
# * At line 2, we take a source sentence from list that we created from `data_pipe` at line 1
# * At line 5, we get a transform based on a source vocabulary and apply it to a tokenized
#   sentence. Note that transforms take list of words and not a sentence.
# * At line 8, we get the mapping of index to string and then use it get the transformed
#   sentence
#
# Now we will use DataPipe functions to apply transform to all our sentences.
# Let us define some more functions for this.

def applyTransform(sequence_pair):
    """
    Apply transforms to sequence of tokens in a sequence pair
    """

    return (
        getTransform(source_vocab)(engTokenize(sequence_pair[0])),
        getTransform(target_vocab)(deTokenize(sequence_pair[1]))
    )
data_pipe = data_pipe.map(applyTransform) ## Apply the function to each element in the iterator
temp_list = list(data_pipe)
print(temp_list[0])

# %%
# Make batches (with bucket batch)
# --------------------------------
# Generally, we train models in batches. While working for sequence to sequence models, it is
# recommended to keep the length of sequences in a batch similar. For that we will use
# `bucketbatch` function of `data_pipe`.
#
# Let us define some functions that will be used by the `bucketbatch` function.

def sortBucket(bucket):
    """
    Function to sort a given bucket. Here, we want to sort based on the length of
    source and target sequence.
    """
    return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))

# %%
# Now, we will apply the `bucketbatch` function:

data_pipe = data_pipe.bucketbatch(
    batch_size = 4, batch_num=5,  bucket_num=1,
    use_in_batch_shuffle=False, sort_key=sortBucket
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
# But, a batch in our `data_pipe` is of the form `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`:

print(list(data_pipe)[0])
# %%
# So, we will now convert them into the form: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`.
# For this we will write a small function:

def separateSourceTarget(sequence_pairs):
    """
    input of form: `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`
    output of form: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`
    """
    sources,targets = zip(*sequence_pairs)
    return sources,targets

## Apply the function to each element in the iterator
data_pipe = data_pipe.map(separateSourceTarget)
print(list(data_pipe)[0])

# %%
# Now, we have the data as desired.
#
# Padding
# -------
# As discussed earlier while building vocabulary, we need to pad shorter sentences in a batch to
# make all the sequences in a batch of equal length. We can perform padding as follows:

def applyPadding(pair_of_sequences):
    """
    Convert sequences to tensors and apply padding
    """
    return (T.ToTensor(0)(list(pair_of_sequences[0])), T.ToTensor(0)(list(pair_of_sequences[1])))
## `T.ToTensor(0)` returns a transform that converts the sequence to `torch.tensor` and also applies
# padding. Here, `0` is passed to the constructor to specify the index of the `<pad>` token in the
# vocabulary.
data_pipe = data_pipe.map(applyPadding)

# %%
# Now, we can use the index to string mapping to see how the sequence would look with tokens
# instead of indices:

source_index_to_string = source_vocab.get_itos()
target_index_to_string = target_vocab.get_itos()

def showSomeTransformedSentences(data_pipe):
    """
    Function to show how the sentences look like after applying all transforms.
    Here we try to print actual words instead of corresponding index
    """
    for sources,targets in data_pipe:
        if sources[0][-1] != 0:
            continue # Just to visualize padding of shorter sentences
        for i in range(4):
            source = ""
            for token in sources[i]:
                source += " " + source_index_to_string[token]
            target = ""
            for token in targets[i]:
                target += " " + target_index_to_string[token]
            print(f"Source: {source}")
            print(f"Traget: {target}")
        break

showSomeTransformedSentences(data_pipe)
# %%
# In the above output we can observe that the shorter sentences are padded with `<pad>`. Now, we
# can use `data_pipe` while writing our training function.
#
# Some parts of this tutorial was inspired from `this article
# <https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71>`__.
