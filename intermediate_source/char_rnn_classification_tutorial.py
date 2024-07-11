# -*- coding: utf-8 -*-
"""
NLP From Scratch: Classifying Names with a Character-Level RNN
**************************************************************
**Author**: `Sean Robertson <https://github.com/spro>`_
**Updated**: `Matthew Schultz <https://github.com/mgs28>`_ 

We will be building and training a basic character-level Recurrent Neural
Network (RNN) to classify words. This tutorial, along with two other
Natural Language Processing (NLP) "from scratch" tutorials
:doc:`/intermediate/char_rnn_generation_tutorial` and
:doc:`/intermediate/seq2seq_translation_tutorial`, show how to
preprocess data to model NLP. In particular these tutorials do not
use many of the convenience functions of `torchtext`, so you can see how
preprocessing to model NLP works at a low level.

A character-level RNN reads words as a series of characters -
outputting a prediction and "hidden state" at each step, feeding its
previous hidden state into each next step. We take the final prediction
to be the output, i.e. which class the word belongs to.

Specifically, we'll train on a few thousand surnames from 18 languages
of origin, and predict which language a name is from based on the
spelling:

Recommended Preparation
=======================

Before starting this tutorial it is recommended that you have installed PyTorch,
and have a basic understanding of Python programming language and Tensors:

-  https://pytorch.org/ For installation instructions
-  :doc:`/beginner/deep_learning_60min_blitz` to get started with PyTorch in general
   and learn the basics of Tensors
-  :doc:`/beginner/pytorch_with_examples` for a wide and deep overview
-  :doc:`/beginner/former_torchies_tutorial` if you are former Lua Torch user

It would also be useful to know about RNNs and how they work:

-  `The Unreasonable Effectiveness of Recurrent Neural
   Networks <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>`__
   shows a bunch of real life examples
-  `Understanding LSTM
   Networks <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__
   is about LSTMs specifically but also informative about RNNs in
   general
"""

######################################################################
# Preparing Torch 
# ==========================
#
# Set up torch to default to the right device use GPU acceleration depending on your hardware (CPU or CUDA). 
#

import torch 

# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")

######################################################################
# Preparing the Data
# ==================
#
# Download the data from `here <https://download.pytorch.org/tutorial/data.zip>`__ 
# and extract it to the current directory.
#
# Included in the ``data/names`` directory are 18 text files named as
# ``[Language].txt``. Each file contains a bunch of names, one name per
# line, mostly romanized (but we still need to convert from Unicode to
# ASCII).
#
# The first thing we need to define is our data items. In this case, we will create a class called NameData 
# which will have an __init__ function to specify the input fields and some helper functions. Our first 
# helper function will be __str__ to convert objects to strings for easy printing 
#
# There are two key pieces of this that we will flesh out over the course of this tutorial. First is the basic data 
# object which a label and some text. In this instance, label = the country of origin and text = the name. 
#
# However, our data has some issues that we will need to clean up. First off, we need to convert Unicode to plain ASCII to 
# limit the RNN input layers. This is accomplished by converting Unicode strings to ASCII and allowing a small set of allowed characters (allowed_characters)

import string 
import unicodedata

class NameData:
    allowed_characters = string.ascii_letters + " .,;'"
    n_letters = len(allowed_characters) 


    def __init__(self, label, text):
        self.label = label
        self.text = NameData.unicodeToAscii(text) 
    
    def __str__(self):
        return f"label={self.label}, text={self.text}"

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427    
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in NameData.allowed_characters
        )

#########################
#Now we can use that class to create a singe piece of data.
#

print (f"{NameData(label='Polish', text='Ślusàrski')}")

######################################################################
# Turning Names into Tensors
# ==========================
#
# Now that we have all the names organized, we need to turn them into
# Tensors to make any use of them.
#
# To represent a single letter, we use a "one-hot vector" of size
# ``<1 x n_letters>``. A one-hot vector is filled with 0s except for a 1
# at index of the current letter, e.g. ``"b" = <0 1 0 0 0 ...>``.
#
# To make a word we join a bunch of those into a 2D matrix
# ``<line_length x 1 x n_letters>``.
#
# That extra 1 dimension is because PyTorch assumes everything is in
# batches - we're just using a batch size of 1 here.
#
# For this, you'll need to add a couple of capabilities to our NameData object.

import string 
import unicodedata

class NameData:
    allowed_characters = string.ascii_letters + " .,;'"
    n_letters = len(allowed_characters) 


    def __init__(self, label, text):
        self.label = label
        self.text = NameData.unicodeToAscii(text) 
        self.tensor =  NameData.lineToTensor(self.text) 
    
    def __str__(self):
        return f"label={self.label}, text={self.text}\ntensor = {self.tensor}"

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427    
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in NameData.allowed_characters
        )

    # Find letter index from all_letters, e.g. "a" = 0
    def letterToIndex(letter):
        return NameData.allowed_characters.find(letter)

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(line):
        tensor = torch.zeros(len(line), 1, NameData.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][NameData.letterToIndex(letter)] = 1
        return tensor

#########################
# Here are some examples of how to use the NameData object

print (f"{NameData(label='none', text='a')}")
print (f"{NameData(label='Korean', text='Ahn')}")

#########################
# Congratulations, you have built the foundational tensor objects for this learning task! You can use a similar approach 
# for other RNN tasks with text.
#
# Next, we need to combine all our examples into a dataset so we can train, text and validate our models. For this, 
# we will use the `Dataset and DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>` classes 
# to hold our dataset. Each Dataset needs to implement three functions: __init__, __len__, and __getitem__. 

from io import open
import glob
import os
import unicodedata
import string
import time 

import torch
from torch.utils.data import Dataset

class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir #for provenance of the dataset
        self.load_time = time.localtime #for provenance of the dataset 
        labels_set = set() #set of all classes

        self.data = []

        #read all the txt files in the specified directory
        text_files = glob.glob(os.path.join(data_dir, '*.txt'))                           
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines: 
                self.data.append(NameData(label=label, text=name))

        self.labels = list(labels_set)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        label_tensor = torch.tensor([self.labels.index(data_item.label)], dtype=torch.long)
        return label_tensor, data_item.tensor, data_item.label, data_item.text
    

#########################
#Here are some examples of how to use the NamesDataset object


alldata = NamesDataset("data/names")
print(f"loaded {len(alldata)} items of data")
print(f"example = {alldata[0]}")

#########################
#Using the dataset object allows us to easily split the data into train and test sets. Here we create a 80/20 
#split but the torch.utils.data has more useful utilities. Here we specify a generator since we need to use the 
#same device as torch defaults to above. 

train_set, test_set = torch.utils.data.random_split(alldata, [.8, .2], generator=torch.Generator(device=device).manual_seed(1))

print(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")

#########################
#Now we have a basic dataset containing 20074 examples where each example is a pairing of label and name. We have also 
#split the dataset into training and testing so we can validate the model that we build. 


######################################################################
# Creating the Network
# ====================
#
# Before autograd, creating a recurrent neural network in Torch involved
# cloning the parameters of a layer over several timesteps. The layers
# held hidden state and gradients which are now entirely handled by the
# graph itself. This means you can implement a RNN in a very "pure" way,
# as regular feed-forward layers.
#
# This RNN module implements a "vanilla RNN" an is just 3 linear layers 
# which operate on an input and hidden state, with a ``LogSoftmax`` layer 
# after the output.s
#

import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_labels):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_labels = output_labels

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, len(output_labels))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

###########################
#We can then create a RNN with 128 hidden nodes and given our datasets


n_hidden = 128
rnn = RNN(NameData.n_letters, n_hidden, alldata.labels)
print(rnn) 

######################################################################
# To run a step of this network we need to pass a single character input 
# and a hidden state (which we initialize as zeros at first). We'll get to 
# multi-character names next

input = NameData(label='none', text='A').tensor
output, next_hidden = rnn(input[0], torch.zeros(1, n_hidden))
print(output) 

######################################################################
# Scoring Multi-character names 
# --------------------
# Multi-character names require just a little bit more effort which is 
# keeping track of the hidden output and passing it back into the RNN. 
# You can see this updated work defined in the function forward()

import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_labels):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_labels = output_labels

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, len(output_labels))
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, line_tensor):
        hidden = torch.zeros(1, rnn.hidden_size)
        output = torch.zeros(1, len(self.output_labels))

        for i in range(line_tensor.size()[0]):
            input = line_tensor[i]
            hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
            output = self.h2o(hidden)
            output = self.softmax(output)

        return output

    def label_from_output(self, output):
        top_n, top_i = output.topk(1)
        label_i = top_i[0].item()
        return self.output_labels[label_i], label_i


###########################
#Now we can score the output for names!


n_hidden = 128
rnn = RNN(NameData.n_letters, n_hidden, alldata.labels)

input = NameData(label='none', text='Albert').tensor
output = rnn(input) #this is equivalent to output = rnn.forward(input)
print(output) 
print(rnn.label_from_output(output))

######################################################################
#
# Training
# ========


######################################################################
# Training the Network
# --------------------
#
# Now all it takes to train this network is show it a bunch of examples,
# have it make guesses, and tell it if it's wrong.
# 
# We start by defining a function learn_single() which learns from a single
# piece of input data. 
#
# -  Create input and target tensors
# -  Create a zeroed initial hidden state
# -  Read each letter in and
#
#    -  Keep hidden state for next letter
#
# -  Compare final output to target
# -  Back-propagate
# -  Return the output and loss
#
# We also define a learn() function which trains on a given dataset with minibatches

import torch.nn as nn
import torch.nn.functional as F
import random 
import numpy as np 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_labels):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_labels = output_labels

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, len(output_labels))
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, line_tensor):
        hidden = torch.zeros(1, rnn.hidden_size)
        output = torch.zeros(1, len(self.output_labels))

        for i in range(line_tensor.size()[0]):
            input = line_tensor[i]
            hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
            output = self.h2o(hidden)
            output = self.softmax(output)

        return output

    def label_from_output(self, output):
        top_n, top_i = output.topk(1)
        label_i = top_i[0].item()
        return self.output_labels[label_i], label_i
    
    def learn(self, training_data, n_epoch = 250, n_batch_size = 64, report_every = 50, learning_rate = 0.005, criterion = nn.NLLLoss()):
        """
        Learn on a batch of training_data for a specified number of iterations and reporting thresholds
        """
        # Keep track of losses for plotting
        current_loss = 0
        all_losses = []
        self.train() 
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        start = time.time()
        print(f"training on data set with n = {len(training_data)}")

        for iter in range(1, n_epoch + 1): 
            self.zero_grad() # clear the gradients 

            # create some minibatches
            # we cannot use dataloaders because each of our names is a different length
            batches = list(range(len(training_data)))
            random.shuffle(batches)
            batches = np.array_split(batches, len(batches) //n_batch_size )

            for idx, batch in enumerate(batches): 
                batch_loss = 0
                for i in batch: #for each example in this batch
                    (label_tensor, text_tensor, label, text) = training_data[i]
                    output = self.forward(text_tensor)
                    loss = criterion(output, label_tensor)
                    batch_loss += loss

                # optimize parameters
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 3)
                optimizer.step()
                optimizer.zero_grad()

                current_loss += batch_loss.item() / len(batch)
            
            all_losses.append(current_loss / len(batches) )
            if iter % report_every == 0:
                print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
            current_loss = 0
        
        return all_losses

##########################################################################
# We can now train a dataset with mini batches for a specified number of epochs

n_hidden = 128
hidden = torch.zeros(1, n_hidden)
rnn = RNN(NameData.n_letters, n_hidden, alldata.labels)
start = time.time()
all_losses = rnn.learn(train_set, n_epoch=10, learning_rate=0.2, report_every=1)
end = time.time()
print(f"training took {end-start}s")

######################################################################
# Plotting the Results
# --------------------
#
# Plotting the historical loss from ``all_losses`` shows the network
# learning:
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()

######################################################################
# Evaluating the Results
# ======================
#
# To see how well the network performs on different categories, we will
# create a confusion matrix, indicating for every actual language (rows)
# which language the network guesses (columns). To calculate the confusion
# matrix a bunch of samples are run through the network with
# ``evaluate()``, which is the same as ``train()`` minus the backprop.
#

def evaluate(rnn, testing_data):
    confusion = torch.zeros(len(rnn.output_labels), len(rnn.output_labels))
    
    rnn.eval() #set to eval mode
    with torch.no_grad(): # do not record the gradients during eval phase
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn.forward(text_tensor)
            guess, guess_i = rnn.label_from_output(output)
            label_i = rnn.output_labels.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(rnn.output_labels)):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy()) #numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + rnn.output_labels, rotation=90)
    ax.set_yticklabels([''] + rnn.output_labels)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


evaluate(rnn, test_set)


######################################################################
# You can pick out bright spots off the main axis that show which
# languages it guesses incorrectly, e.g. Chinese for Korean, and Spanish
# for Italian. It seems to do very well with Greek, and very poorly with
# English (perhaps because of overlap with other languages).
#


######################################################################
# Exercises
# =========
#
# -  Get better results with a bigger and/or better shaped network
#
#    -  Vary the hyperparameters to improve performance (e.g. 250 epochs, batch size, learning rate ) 
#    -  Add more linear layers
#    -  Try the ``nn.LSTM`` and ``nn.GRU`` layers
#    -  Combine multiple of these RNNs as a higher level network
# 
# -  Try with a different dataset of line -> label, for example:
#
#    -  Any word -> language
#    -  First name -> gender
#    -  Character name -> writer
#    -  Page title -> blog or subreddit