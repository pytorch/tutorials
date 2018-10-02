# -*- coding: utf-8 -*-
"""
Deploying a Seq2Seq Model with the Hybrid Frontend
==================================================
**Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
"""


######################################################################
# This tutorial will walk through the process of transitioning a
# sequence-to-sequence model to Torch Script using PyTorch’s Hybrid
# Frontend. The model that we will convert is the chatbot model from the
# `Chatbot tutorial <https://pytorch.org/tutorials/beginner/chatbot_tutorial.html>`__. 
# You can either treat this tutorial as a “Part 2” to the Chatbot tutorial
# and deploy your own pretrained model, or you can start with this
# document and use a pretrained model that we host. In the latter case,
# you can reference the original Chatbot tutorial for details
# regarding data preprocessing, model theory and definition, and model
# training.
#
# .. attention:: This example requires PyTorch 1.0 (preview) or later.
#    For installation information visit http://pytorch.org/get-started.
#
# What is the Hybrid Frontend?
# ----------------------------
#
# During the research and development phase of a deep learning-based
# project, it is advantageous to interact with an **eager**, imperative
# interface like PyTorch’s. This gives users the ability to write
# familiar, idiomatic Python, allowing for the use of Python data
# structures, control flow operations, print statements, and debugging
# utilities. Although the eager interface is a beneficial tool for
# research and experimentation applications, when it comes time to deploy
# the model in a production environment, having a **graph**-based model
# representation is very beneficial. A deferred graph representation
# allows for optimizations such as out-of-order execution, and the ability
# to target highly optimized hardware architectures. Also, a graph-based
# representation enables framework-agnostic model exportation. PyTorch
# provides mechanisms for incrementally converting eager-mode code into
# Torch Script, a statically analyzable and optimizable subset of Python
# that Torch uses to represent deep learning programs independently from
# the Python runtime.
#
# The API for converting eager-mode PyTorch programs into Torch Script is
# found in the torch.jit module. This module has two core modalities for
# converting an eager-mode model to a Torch Script graph representation:
# **tracing** and **scripting**. The ``torch.jit.trace`` function takes a
# module or function and a set of example inputs. It then runs the example
# input through the function or module while tracing the computational
# steps that are encountered, and outputs a graph-based function that
# performs the traced operations. **Tracing** is great for straightforward
# modules and functions that do not involve data-dependent control flow,
# such as standard convolutional neural networks. However, if a function
# with data-dependent if statements and loops is traced, only the
# operations called along the execution route taken by the example input
# will be recorded. In other words, the control flow itself is not
# captured. To convert modules and functions containing data-dependent
# control flow, a **scripting** mechanism is provided. Scripting
# explicitly converts the module or function code to Torch Script,
# including all possible control flow routes. To use script mode, be sure
# to inherit from the the ``torch.jit.ScriptModule`` base class (instead
# of ``torch.nn.Module``) and add a ``torch.jit.script`` decorator to your
# Python function or a ``torch.jit.script_method`` decorator to your
# module’s methods. The one caveat with using scripting is that it only
# supports a restricted subset of Python. For all details relating to the
# supported features, see the Torch Script `language
# reference <https://pytorch.org/docs/master/jit.html>`__. To provide the
# maximum flexibility, the modes of Torch Script can be composed to
# represent your whole program, and these techniques can be applied
# incrementally.
#
# .. figure:: /_static/img/chatbot/pytorch_workflow.png
#    :align: center
#    :alt: workflow
#



######################################################################
# Acknowledgements
# ----------------
#
# This tutorial was inspired by the following sources:
#
# 1) Yuan-Kuei Wu’s pytorch-chatbot implementation:
#    https://github.com/ywk991112/pytorch-chatbot
#
# 2) Sean Robertson’s practical-pytorch seq2seq-translation example:
#    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
#
# 3) FloydHub’s Cornell Movie Corpus preprocessing code:
#    https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus
#


######################################################################
# Prepare Environment
# -------------------
#
# First, we will import the required modules and set some constants. If
# you are planning on using your own model, be sure that the
# ``MAX_LENGTH`` constant is set correctly. As a reminder, this constant
# defines the maximum allowed sentence length during training and the
# maximum length output that the model is capable of producing.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

device = torch.device("cpu")


MAX_LENGTH = 10  # Maximum sentence length

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


######################################################################
# Model Overview
# --------------
#
# As mentioned, the model that we are using is a
# `sequence-to-sequence <https://arxiv.org/abs/1409.3215>`__ (seq2seq)
# model. This type of model is used in cases when our input is a
# variable-length sequence, and our output is also a variable length
# sequence that is not necessarily a one-to-one mapping of the input. A
# seq2seq model is comprised of two recurrent neural networks (RNNs) that
# work cooperatively: an **encoder** and a **decoder**.
#
# .. figure:: /_static/img/chatbot/seq2seq_ts.png
#    :align: center
#    :alt: model
#
#
# Image source:
# https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
#
# Encoder
# ~~~~~~~
#
# The encoder RNN iterates through the input sentence one token
# (e.g. word) at a time, at each time step outputting an “output” vector
# and a “hidden state” vector. The hidden state vector is then passed to
# the next time step, while the output vector is recorded. The encoder
# transforms the context it saw at each point in the sequence into a set
# of points in a high-dimensional space, which the decoder will use to
# generate a meaningful output for the given task.
#
# Decoder
# ~~~~~~~
#
# The decoder RNN generates the response sentence in a token-by-token
# fashion. It uses the encoder’s context vectors, and internal hidden
# states to generate the next word in the sequence. It continues
# generating words until it outputs an *EOS_token*, representing the end
# of the sentence. We use an `attention
# mechanism <https://arxiv.org/abs/1409.0473>`__ in our decoder to help it
# to “pay attention” to certain parts of the input when generating the
# output. For our model, we implement `Luong et
# al. <https://arxiv.org/abs/1508.04025>`__\ ’s “Global attention” module,
# and use it as a submodule in our decode model.
#


######################################################################
# Data Handling
# -------------
#
# Although our models conceptually deal with sequences of tokens, in
# reality, they deal with numbers like all machine learning models do. In
# this case, every word in the model’s vocabulary, which was established
# before training, is mapped to an integer index. We use a ``Voc`` object
# to contain the mappings from word to index, as well as the total number
# of words in the vocabulary. We will load the object later before we run
# the model.
#
# Also, in order for us to be able to run evaluations, we must provide a
# tool for processing our string inputs. The ``normalizeString`` function
# converts all characters in a string to lowercase and removes all
# non-letter characters. The ``indexesFromSentence`` function takes a
# sentence of words and returns the corresponding sequence of word
# indexes.
#

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens
        for word in keep_words:
            self.addWord(word)


# Lowercase and remove non-letter characters
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Takes string sentence, returns sentence of word indexes
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


######################################################################
# Define Encoder
# --------------
#
# We implement our encoder’s RNN with the ``torch.nn.GRU`` module which we
# feed a batch of sentences (vectors of word embeddings) and it internally
# iterates through the sentences one token at a time calculating the
# hidden states. We initialize this module to be bidirectional, meaning
# that we have two independent GRUs: one that iterates through the
# sequences in chronological order, and another that iterates in reverse
# order. We ultimately return the sum of these two GRUs’ outputs. Since
# our model was trained using batching, our ``EncoderRNN`` model’s
# ``forward`` function expects a padded input batch. To batch
# variable-length sentences, we allow a maximum of *MAX_LENGTH* tokens in
# a sentence, and all sentences in the batch that have less than
# *MAX_LENGTH* tokens are padded at the end with our dedicated *PAD_token*
# tokens. To use padded batches with a PyTorch RNN module, we must wrap
# the forward pass call with ``torch.nn.utils.rnn.pack_padded_sequence``
# and ``torch.nn.utils.rnn.pad_packed_sequence`` data transformations.
# Note that the ``forward`` function also takes an ``input_lengths`` list,
# which contains the length of each sentence in the batch. This input is
# used by the ``torch.nn.utils.rnn.pack_padded_sequence`` function when
# padding.
#
# Hybrid Frontend Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Since the encoder’s ``forward`` function does not contain any
# data-dependent control flow, we will use **tracing** to convert it to
# script mode. When tracing a module, we can leave the module definition
# as-is. We will initialize all models towards the end of this document
# before we run evaluations.
#

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


######################################################################
# Define Decoder’s Attention Module
# ---------------------------------
#
# Next, we’ll define our attention module (``Attn``). Note that this
# module will be used as a submodule in our decoder model. Luong et
# al. consider various “score functions”, which take the current decoder
# RNN output and the entire encoder output, and return attention
# “energies”. This attention energies tensor is the same size as the
# encoder output, and the two are ultimately multiplied, resulting in a
# weighted tensor whose largest values represent the most important parts
# of the query sentence at a particular time-step of decoding.
#

# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


######################################################################
# Define Decoder
# --------------
#
# Similarly to the ``EncoderRNN``, we use the ``torch.nn.GRU`` module for
# our decoder’s RNN. This time, however, we use a unidirectional GRU. It
# is important to note that unlike the encoder, we will feed the decoder
# RNN one word at a time. We start by getting the embedding of the current
# word and applying a
# `dropout <https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.Dropout>`__.
# Next, we forward the embedding and the last hidden state to the GRU and
# obtain a current GRU output and hidden state. We then use our ``Attn``
# module as a layer to obtain the attention weights, which we multiply by
# the encoder’s output to obtain our attended encoder output. We use this
# attended encoder output as our ``context`` tensor, which represents a
# weighted sum indicating what parts of the encoder’s output to pay
# attention to. From here, we use a linear layer and softmax normalization
# to select the next word in the output sequence.
#
# Hybrid Frontend Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Similarly to the ``EncoderRNN``, this module does not contain any
# data-dependent control flow. Therefore, we can once again use
# **tracing** to convert this model to Torch Script after it is
# initialized and its parameters are loaded.
#

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


######################################################################
# Define Evaluation
# -----------------
#
# Greedy Search Decoder
# ~~~~~~~~~~~~~~~~~~~~~
#
# As in the chatbot tutorial, we use a ``GreedySearchDecoder`` module to
# facilitate the actual decoding process. This module has the trained
# encoder and decoder models as attributes, and drives the process of
# encoding an input sentence (a vector of word indexes), and iteratively
# decoding an output response sequence one word (word index) at a time.
#
# Encoding the input sequence is straightforward: simply forward the
# entire sequence tensor and its corresponding lengths vector to the
# ``encoder``. It is important to note that this module only deals with
# one input sequence at a time, **NOT** batches of sequences. Therefore,
# when the constant **1** is used for declaring tensor sizes, this
# corresponds to a batch size of 1. To decode a given decoder output, we
# must iteratively run forward passes through our decoder model, which
# outputs softmax scores corresponding to the probability of each word
# being the correct next word in the decoded sequence. We initialize the
# ``decoder_input`` to a tensor containing an *SOS_token*. After each pass
# through the ``decoder``, we *greedily* append the word with the highest
# softmax probability to the ``decoded_words`` list. We also use this word
# as the ``decoder_input`` for the next iteration. The decoding process
# terminates either if the ``decoded_words`` list has reached a length of
# *MAX_LENGTH* or if the predicted word is the *EOS_token*.
#
# Hybrid Frontend Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The ``forward`` method of this module involves iterating over the range
# of :math:`[0, max\_length)` when decoding an output sequence one word at
# a time. Because of this, we should use **scripting** to convert this
# module to Torch Script. Unlike with our encoder and decoder models,
# which we can trace, we must make some necessary changes to the
# ``GreedySearchDecoder`` module in order to initialize an object without
# error. In other words, we must ensure that our module adheres to the
# rules of the scripting mechanism, and does not utilize any language
# features outside of the subset of Python that Torch Script includes.
#
# To get an idea of some manipulations that may be required, we will go
# over the diffs between the ``GreedySearchDecoder`` implementation from
# the chatbot tutorial and the implementation that we use in the cell
# below. Note that the lines highlighted in red are lines removed from the
# original implementation and the lines highlighted in green are new.
#
# .. figure:: /_static/img/chatbot/diff.png
#    :align: center
#    :alt: diff
#
# Changes:
# ^^^^^^^^
#
# -  ``nn.Module`` -> ``torch.jit.ScriptModule``
#
#    -  In order to use PyTorch’s scripting mechanism on a module, that
#       module must inherit from the ``torch.jit.ScriptModule``.
#
#
# -  Added ``decoder_n_layers`` to the constructor arguments
#
#    -  This change stems from the fact that the encoder and decoder
#       models that we pass to this module will be a child of
#       ``TracedModule`` (not ``Module``). Therefore, we cannot access the
#       decoder’s number of layers with ``decoder.n_layers``. Instead, we
#       plan for this, and pass this value in during module construction.
#
#
# -  Store away new attributes as constants
#
#    -  In the original implementation, we were free to use variables from
#       the surrounding (global) scope in our ``GreedySearchDecoder``\ ’s
#       ``forward`` method. However, now that we are using scripting, we
#       do not have this freedom, as the assumption with scripting is that
#       we cannot necessarily hold on to Python objects, especially when
#       exporting. An easy solution to this is to store these values from
#       the global scope as attributes to the module in the constructor,
#       and add them to a special list called ``__constants__`` so that
#       they can be used as literal values when constructing the graph in
#       the ``forward`` method. An example of this usage is on NEW line
#       19, where instead of using the ``device`` and ``SOS_token`` global
#       values, we use our constant attributes ``self._device`` and
#       ``self._SOS_token``.
#
#
# -  Add the ``torch.jit.script_method`` decorator to the ``forward``
#    method
#
#    -  Adding this decorator lets the JIT compiler know that the function
#       that it is decorating should be scripted.
#
#
# -  Enforce types of ``forward`` method arguments
#
#    -  By default, all parameters to a Torch Script function are assumed
#       to be Tensor. If we need to pass an argument of a different type,
#       we can use function type annotations as introduced in `PEP
#       3107 <https://www.python.org/dev/peps/pep-3107/>`__. In addition,
#       it is possible to declare arguments of different types using
#       MyPy-style type annotations (see
#       `doc <https://pytorch.org/docs/master/jit.html#types>`__).
#
#
# -  Change initialization of ``decoder_input``
#
#    -  In the original implementation, we initialized our
#       ``decoder_input`` tensor with ``torch.LongTensor([[SOS_token]])``.
#       When scripting, we are not allowed to initialize tensors in a
#       literal fashion like this. Instead, we can initialize our tensor
#       with an explicit torch function such as ``torch.ones``. In this
#       case, we can easily replicate the scalar ``decoder_input`` tensor
#       by multiplying 1 by our SOS_token value stored in the constant
#       ``self._SOS_token``.
#

class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    @torch.jit.script_method
    def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores



######################################################################
# Evaluating an Input
# ~~~~~~~~~~~~~~~~~~~
#
# Next, we define some functions for evaluating an input. The ``evaluate``
# function takes a normalized string sentence, processes it to a tensor of
# its corresponding word indexes (with batch size of 1), and passes this
# tensor to a ``GreedySearchDecoder`` instance called ``searcher`` to
# handle the encoding/decoding process. The searcher returns the output
# word index vector and a scores tensor corresponding to the softmax
# scores for each decoded word token. The final step is to convert each
# word index back to its string representation using ``voc.index2word``.
#
# We also define two functions for evaluating an input sentence. The
# ``evaluateInput`` function prompts a user for an input, and evaluates
# it. It will continue to ask for another input until the user enters ‘q’
# or ‘quit’.
#
# The ``evaluateExample`` function simply takes a string input sentence as
# an argument, normalizes it, evaluates it, and prints the response.
#

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


# Evaluate inputs from user input (stdin)
def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

# Normalize input sentence and call evaluate()
def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print("> " + sentence)
    # Normalize sentence
    input_sentence = normalizeString(sentence)
    # Evaluate sentence
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))


######################################################################
# Load Pretrained Parameters
# --------------------------
#
# Ok, its time to load our model!
#
# Use hosted model
# ~~~~~~~~~~~~~~~~
#
# To load the hosted model:
#
# 1) Download the model `here <https://download.pytorch.org/models/tutorials/4000_checkpoint.tar>`__.
#
# 2) Set the ``loadFilename`` variable to the path to the downloaded
#    checkpoint file.
#
# 3) Leave the ``checkpoint = torch.load(loadFilename)`` line uncommented,
#    as the hosted model was trained on CPU.
#
# Use your own model
# ~~~~~~~~~~~~~~~~~~
#
# To load your own pre-trained model:
#
# 1) Set the ``loadFilename`` variable to the path to the checkpoint file
#    that you wish to load. Note that if you followed the convention for
#    saving the model from the chatbot tutorial, this may involve changing
#    the ``model_name``, ``encoder_n_layers``, ``decoder_n_layers``,
#    ``hidden_size``, and ``checkpoint_iter`` (as these values are used in
#    the model path).
#
# 2) If you trained the model on a CPU, make sure that you are opening the
#    checkpoint with the ``checkpoint = torch.load(loadFilename)`` line.
#    If you trained the model on a GPU and are running this tutorial on a
#    CPU, uncomment the
#    ``checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))``
#    line.
#
# Hybrid Frontend Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Notice that we initialize and load parameters into our encoder and
# decoder models as usual. Also, we must call ``.to(device)`` to set the
# device options of the models and ``.eval()`` to set the dropout layers
# to test mode **before** we trace the models. ``TracedModule`` objects do
# not inherit the ``to`` or ``eval`` methods.
#

save_dir = os.path.join("data", "save")
corpus_name = "cornell movie-dialogs corpus"

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# If you're loading your own model
# Set checkpoint to load from
checkpoint_iter = 4000
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                             '{}_checkpoint.tar'.format(checkpoint_iter))

# If you're loading the hosted model
loadFilename = '4000_checkpoint.tar'

# Load model
# Force CPU device options (to match tensors in this tutorial)
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc = Voc(corpus_name)
voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
# Load trained model params
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()
print('Models built and ready to go!')


######################################################################
# Convert Model to Torch Script
# -----------------------------
#
# Encoder
# ~~~~~~~
#
# As previously mentioned, to convert the encoder model to Torch Script,
# we use **tracing**. Tracing any module requires running an example input
# through the model’s ``forward`` method and trace the computational graph
# that the data encounters. The encoder model takes an input sequence and
# a corresponding lengths tensor. Therefore, we create an example input
# sequence tensor ``test_seq``, which is of appropriate size (MAX_LENGTH,
# 1), contains numbers in the appropriate range
# :math:`[0, voc.num\_words)`, and is of the appropriate type (int64). We
# also create a ``test_seq_length`` scalar which realistically contains
# the value corresponding to how many words are in the ``test_seq``. The
# next step is to use the ``torch.jit.trace`` function to trace the model.
# Notice that the first argument we pass is the module that we want to
# trace, and the second is a tuple of arguments to the module’s
# ``forward`` method.
#
# Decoder
# ~~~~~~~
#
# We perform the same process for tracing the decoder as we did for the
# encoder. Notice that we call forward on a set of random inputs to the
# traced_encoder to get the output that we need for the decoder. This is
# not required, as we could also simply manufacture a tensor of the
# correct shape, type, and value range. This method is possible because in
# our case we do not have any constraints on the values of the tensors
# because we do not have any operations that could fault on out-of-range
# inputs.
#
# GreedySearchDecoder
# ~~~~~~~~~~~~~~~~~~~
#
# Recall that we scripted our searcher module due to the presence of
# data-dependent control flow. In the case of scripting, we do the
# conversion work up front by adding the decorator and making sure the
# implementation complies with scripting rules. We initialize the scripted
# searcher the same way that we would initialize an un-scripted variant.
#

### Convert encoder model
# Create artificial inputs
test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words)
test_seq_length = torch.LongTensor([test_seq.size()[0]])
# Trace the model
traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))

### Convert decoder model
# Create and generate artificial inputs
test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
# Trace the model
traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))

### Initialize searcher module
scripted_searcher = GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)


######################################################################
# Print Graphs
# ------------
#
# Now that our models are in Torch Script form, we can print the graphs of
# each to ensure that we captured the computational graph appropriately.
# Since our ``scripted_searcher`` contains our ``traced_encoder`` and
# ``traced_decoder``, these graphs will print inline.
#

print('scripted_searcher graph:\n', scripted_searcher.graph)


######################################################################
# Run Evaluation
# --------------
#
# Finally, we will run evaluation of the chatbot model using the Torch
# Script models. If converted correctly, the models will behave exactly as
# they would in their eager-mode representation.
#
# By default, we evaluate a few common query sentences. If you want to
# chat with the bot yourself, uncomment the ``evaluateInput`` line and
# give it a spin.
#

# Evaluate examples
sentences = ["hello", "what's up?", "who are you?", "where am I?", "where are you from?"]
for s in sentences:
    evaluateExample(s, traced_encoder, traced_decoder, scripted_searcher, voc)

# Evaluate your input
#evaluateInput(traced_encoder, traced_decoder, scripted_searcher, voc)


######################################################################
# Save Model
# ----------
#
# Now that we have successfully converted our model to Torch Script, we
# will serialize it for use in a non-Python deployment environment. To do
# this, we can simply save our ``scripted_searcher`` module, as this is
# the user-facing interface for running inference against the chatbot
# model. When saving a Script module, use script_module.save(PATH) instead
# of torch.save(model, PATH).
#

scripted_searcher.save("scripted_chatbot.pth")
