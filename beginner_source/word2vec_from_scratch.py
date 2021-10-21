"""
Word2vec Tutorial
=================

**Author**: `Olha Chernytska <https://github.com/OlgaChernytska>`__

Word Embeddings is the most fundamental concept in Deep Natural Language
Processing. And word2vec is one of the earliest algorithms used to train
word embeddings. In this tutorial, we will implement from scratch the
CBOW version of the word2vec model described in the paper `Efficient
Estimation of Word Representations in Vector
Space <https://arxiv.org/abs/1301.3781>`__.

"""


######################################################################
# Model Architecture
# ------------------
# 


######################################################################
# Word2vec is based on the idea that a word’s meaning is defined by its
# context - N words before and N words after the current word. N is a
# hyperparameter. With larger N we can create better embeddings, but at
# the same time, such a model requires more computational resources. In
# the original paper, N is 4-5, and in my visualizations below, N is 2.
# 


######################################################################
# .. figure:: https://raw.githubusercontent.com/OlgaChernytska/word2vec-pytorch/main/docs/word_context.png
#    :alt: word_context
# 
#    word_context
# 


######################################################################
# CBOW (Continuous Bag-of-Words) model is created to predict a current
# word based on its context words. For instance, it takes “machine”,
# “learning”, “a”, “method” as inputs and returns “is” as an output. CBOW
# is a multi-class classification model by definition. Detailed
# visualization below should make it clear.
# 


######################################################################
# .. figure:: https://raw.githubusercontent.com/OlgaChernytska/word2vec-pytorch/main/docs/cbow_overview.png
#    :alt: cbow_overview
# 
#    cbow_overview
# 


######################################################################
# **What is happening in the black box?**
# 
# The initial step would be to encode all words with their IDs. ID is an
# integer (index) that identifies word position in the vocabulary.
# “Vocabulary” is a term to describe a set of unique words in the text.
# This set may be all words in the text or just the most frequent ones.
# More on that is in Section “Data Preparation”.
# 
# Word2vec model is very simple and has only two layers:
# 
# -  **Embedding layer**, which takes word ID and returns its
#    300-dimensional vector. Word2vec embeddings are 300-dimensional, as
#    authors proved this number to be the best in terms of embedding
#    quality and computational costs. You may think about embedding layer
#    as a simple lookup table with learnable weights, or as a linear layer
#    without bias and activation.
# -  Then comes the **Linear (Dense) layer** with a Softmax activation. We
#    create a model for a multi-class classification task, where the
#    number of classes is equal to the number of words in the vocabulary.
# 
# CBOW model expects output to be a bag-of-words (that’s from where its
# name comes from). Each word goes through the same Embedding layer, and
# then all word embedding vectors are averaged before going into the
# Linear layer. Detailed architecture is in the image below.
# 


######################################################################
# .. figure:: https://raw.githubusercontent.com/OlgaChernytska/word2vec-pytorch/main/docs/cbow_detailed.png
#    :alt: cbow_detailed
# 
#    cbow_detailed
# 


######################################################################
# **Where are the word embeddings?**
# 
# We train the model that is not going to be used directly. We don’t want
# to predict a word from its context. Instead, we want to get word
# vectors. It turns out that these vectors are weights of the Embedding
# layer. More details on that are in Section “Retrieving Embeddings”.
# 


######################################################################
# Defining Model in PyTorch
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Word2vec model in PyTorch is defined in a standard way, by subclassing
# from
# ```nn.Module`` <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module>`__.
# We should initialize layers (two our case, 2 layers - Embedding and
# Linear) and define the forward pass. The model can only be initialized
# after vocabulary is created because it expects parameter ``vocab_size``
# which defines the input size of the Embedding layer and output size of
# the Linear layer.
# 

import torch.nn as nn

EMBED_DIMENSION = 300 
EMBED_MAX_NORM = 1

class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


######################################################################
# Pay attention, there is no Softmax activation in the Linear Layer.
# That’s because PyTorch
# ```CrossEntropyLoss`` <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropy#torch.nn.CrossEntropyLoss>`__
# expects predictions to be raw, unnormalized scores.
# 
# Model input is a list of word IDs because
# ```Embedding`` <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`__
# layer expects words as IDs. Model output is N-dimensional vector, where
# N is vocabulary size.
# 
# ``EMBED_DIMENSION`` is the dimensionality of word embedding. By
# definition, it is 300, but feel free to try smaller and larger values.
# 
# ``EMBED_MAX_NORM`` is the parameter to restrict word embedding norms (to
# be 1, in our case). It works as a regularization parameter and prevents
# weights in the Embedding layer grow uncontrollably. ``EMBED_MAX_NORM``
# is worth experimenting with. What I’ve seen: when restricting embedding
# vector norm, similar words like “mother” and “father” have higher cosine
# similarity, compared to when ``EMBED_MAX_NORM=None``.
# 


######################################################################
# Data Preparation
# ----------------
# 
# Data
# ~~~~
# 
# Word2vec is an unsupervised algorithm, so we need only a text corpus,
# the larger the better. For the study purposes, the
# `WikiText-2 <https://pytorch.org/text/stable/datasets.html#wikitext-2>`__
# dataset available in PyTorch would be a good choice. It contains 36k
# text lines and 2M tokens in the train part (tokens are words +
# punctuation).
# 
# Below is the code for creating the WikiText2 dataset. We are going to
# use only the train part. After initializing the iterator-style dataset
# we convert it to the map-style dataset using function
# ```to_map_style_dataset`` <https://pytorch.org/text/stable/data_functional.html#to-map-style-dataset>`__,
# so we can reuse ``dataset`` several times without creating it again each
# time after it was used.
# 

from torchtext.datasets import WikiText2
from torchtext.data import to_map_style_dataset

dataset = WikiText2(split=("train"))
dataset = to_map_style_dataset(dataset)


######################################################################
# Vocabulary
# ~~~~~~~~~~
# 
# The main step in data preparation is to create a vocabulary. The
# vocabulary contains the words for which embeddings will be trained.
# Vocabulary may be the list of all the unique words within a text corpus,
# but usually, it is not.
# 
# It is better to create vocabulary:
# 
# -  Either by filtering out rare words, that occurred less than N times
#    in the corpus;
# -  Or by choosing the top N most frequent words.
# 
# Such filtering makes much sense because, with a smaller vocabulary, the
# model is faster to train. On the other hand, you probably do not want to
# use embedding for words that appeared only once within the text corpus,
# as these embeddings may not be good enough. To create good word
# embeddings the model should see a word several times and in different
# contexts.
# 
# Each word in the vocabulary has its unique index. Words in vocabulary
# may be sorted alphabetically or based on their frequencies, or may not –
# it should not affect the model training. Vocabulary is usually
# represented as a dictionary data structure:
# 

# example
vocab = {
    "a": 1,
    "analysis": 2,
    "analytical": 3,
    "automates": 4,
    "building": 5,
    "data": 6,
}


######################################################################
# Punctuation marks and other special symbols may be also added to the
# vocabulary, and we train embeddings for them as well. You may lowercase
# all the words, or train separate embeddings for the words “apple” and
# “Apple”; in some cases, it may be useful.
# 
# Depending on what you want your vocabulary (and word embeddings) to be
# like – preprocess the text corpus appropriately. Lowercase or not,
# remove punctuation or not, and tokenize text.
# 


######################################################################
# .. figure:: https://raw.githubusercontent.com/OlgaChernytska/word2vec-pytorch/main/docs/data_preparation.png
#    :alt: data_prepatation
# 
#    data_prepatation
# 


######################################################################
# Creating Vocabulary in PyTorch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# We create vocabulary from the dataset using the function
# ```build_vocab_from_iterator`` <https://pytorch.org/text/stable/vocab.html#build-vocab-from-iterator>`__.
# The WikiText-2 dataset has rare words replaced with token ``<unk>``, we
# add this token as a special symbol with ID=0 and all out-of-vocabulary
# words also encode with ID=0. And let’s use only words that appeared at
# least 50 times within a text.
# 

from torchtext.vocab import build_vocab_from_iterator

MIN_WORD_FREQUENCY = 50 

def build_vocab(dataset, tokenizer):   
    vocab = build_vocab_from_iterator(
        map(tokenizer, dataset),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


######################################################################
# As a tokenizer, we use the “basic_english” tokenizer available in
# PyTorch. It lowercases text, splits it into tokens by whitespace, and
# puts punctuation into separate tokens. You may check its `source
# code <https://pytorch.org/text/stable/_modules/torchtext/data/utils.html>`__
# in case you need more details.
# 

from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english", language="en")


######################################################################
# Let’s finally create a vocabulary and define a text pipeline, that
# incorporated the logic of how the text will be processed: first
# tokenized, and then each word encoded with its ID.
# 

vocab = build_vocab(dataset, tokenizer)

vocab_size = len(vocab.get_stoi())
print(f"Vocabulary size: {vocab_size}")

text_pipeline = lambda x: vocab(tokenizer(x))


######################################################################
# Dataloader
# ~~~~~~~~~~
# 
# Dataloader we create with ``collate_fn`` (here is `its
# documentation <https://pytorch.org/docs/stable/data.html>`__). This
# function implements the logic of how to batch individual samples. When
# looping through WikiText-2 datasets, each sample retrieved is a text
# paragraph.
# 
# In ``collate_fn`` we say:
# 
# -  Take each text paragraph:
# 
#    -  Lowercase it, tokenize it, and encode with IDs (function
#       ``text_pipeline``).
#    -  If the paragraph is too short – skip it. If too long (includes
#       more than 256 tokens) – truncate it to have only the first 256
#       tokens.
#    -  With the moving window of size 9 (4 history words, middle word,
#       and 4 future words) loop through the paragraph.
#    -  All middle words merge into a list – they will be Ys.
#    -  All contexts (history and future words) merge into a list of lists
#       – they will be Xs.
# 
# -  Merge Xs from all paragraphs together – they will be batch Xs.
# -  Merge Ys from all paragraphs together – they will be batch Ys.
# 
# Pay attention, the number of final batches (Xs and Ys) when we call
# ``collate_fn`` will be different from parameter ``batch_size`` specified
# in ``DataLoader``, and will vary for different paragraphs.
# 

import torch

CBOW_N_WORDS = 4
MAX_SEQUENCE_LENGTH = 256

def collate_cbow(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


######################################################################
# And here is how ``collate_fn`` is used with ``DataLoader``. Function
# ``partial`` we use here to be able to pass arguments to
# ``collate_cbow``.
# 
# You choose batch size to fit into the memory. Just remember: that batch
# size is the number of dataset paragraphs, which will be processed into
# input-output pairs, and this number will be much larger.
# 

from torch.utils.data import DataLoader
from functools import partial

BATCH_SIZE = 96

dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_cbow, text_pipeline=text_pipeline),
 )


######################################################################
# Training Details
# ----------------
# 


######################################################################
# Let’s initialize the model, criterion, and optimizer. In the original
# paper of word2vec initial learning rate is 0.025 and it decreases
# linearly every epoch until it reaches 0 at the end of the last epoch. We
# can easily implement this learning rate logic with
# ```LambdaLR`` <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR>`__
# scheduler. For most experiments, word2vec authors choose epochs to be 3,
# let’s stick with this number as well.
# 

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

LR = 0.025
EPOCHS = 3

model = CBOW_Model(vocab_size=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_lambda = lambda epoch: (EPOCHS - epoch) / EPOCHS
lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)


######################################################################
# And let the training begin… If you are training on CPU it should not
# take more than 10 minutes, and on GPU training will be even faster.
# 

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training on:', device)
model = model.to(device)

for epoch in range(EPOCHS):
    running_loss = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())
   
    print('Epoch {}/{}: Train loss = {:.3f}'.format(epoch+1, EPOCHS, np.mean(running_loss)))   
    lr_scheduler.step()


######################################################################
# Retrieving Word Embeddings
# --------------------------
# 
# Word embeddings are stored in the Embedding layer. Embedding layer size
# is (vocab_size, 300), which means there we have embedding for all the
# words in the vocabulary. When trained on the WikiText-2 dataset, the
# Embedding layer size is (4099, 300), and each row is a word vector. Here
# is how to get Embedding layer weights:
# 

embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()
embeddings.shape


######################################################################
# In case you need word order the same as in embedding matrix, it is
# stored in ``vocab``:
# 

words = vocab.get_itos()
words[:5]


######################################################################
# It’s up to you - what to do with trained word embedding next. You may
# create a sentence classification model and before putting words into the
# model, encode them with word2vec embeddings. You may conduct an analysis
# of the text corpus by clustering or visualizing words using embedding
# vectors as sets of features.
# 
# Or, you may use the word2vec model to find synonyms and similar words
# within the text corpus. And I’ll show you how to do that.
# 


######################################################################
# Find Similar Words
# ~~~~~~~~~~~~~~~~~~
# 


######################################################################
# The similarity between two words is calculated as the cosine similarity
# between their vectors. The closer the vectors - the more similar the
# words are. So, to find the most similar words for the word “mother”, we
# calculate cosine similarities between the “mother” vector and all word
# vectors we have. Function
# ```sklearn.metrics.pairwise.cosine_similarity`` <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html>`__
# would be very helpful here.
# 
# After that, we sort all cosine similarities to get words with the
# highest ones - they are the most similar/related words.
# 

from sklearn.metrics.pairwise import cosine_similarity

def get_top_similar(word: str, topN: int = 10):
    word_id = vocab[word]
    if word_id == 0:
        print("Out of vocabulary word")
        return

    word_vec = embeddings[word_id]
    word_vec = np.reshape(word_vec, (1, len(word_vec)))
    
    sims = cosine_similarity(embeddings, word_vec)
    sims = sims.flatten()
    
    topN_ids = np.argsort(-sims)[1 : topN + 1]

    topN_dict = {}
    for sim_word_id in topN_ids:
        sim_word = vocab.lookup_token(sim_word_id)
        topN_dict[sim_word] = sims[sim_word_id]
    return topN_dict


######################################################################
# We can find similar words only for words that are **in** vocabulary
# (because we have vectors for them). For instance, the word “mother” is
# there, but not the word “bastion”. Also, we select similar words
# **only** from vocabulary, so it may happen, that the most similar ones
# look completely irrelevant only because there are no truly relevant
# words within a vocabulary.
# 
# Feel free to run ``get_top_similar`` for various words that come to your
# mind.
# 

get_top_similar('mother')

get_top_similar('bastion')


######################################################################
# To Sum Up
# ---------
# 


######################################################################
# In this tutorial, we trained the CBOW version of word2vec from scratch
# with PyTorch. We went through all the important steps - model
# definition, vocabulary creation, text preprocessing, dataloaders
# initialization, and training. We also discussed how to get trained word
# embeddings from the model and reviewed one of the ways of how to use
# them.
# 
# I hope, it helped you learn more about NLP and PyTorch!
# 