"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
**Building Models** ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

Building Models with PyTorch
============================

Follow along with the video below or on `youtube <https://www.youtube.com/watch?v=OSqIP-mOWOI>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/OSqIP-mOWOI" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

``torch.nn.Module`` and ``torch.nn.Parameter``
----------------------------------------------

In this video, we’ll be discussing some of the tools PyTorch makes
available for building deep learning networks.

Except for ``Parameter``, the classes we discuss in this video are all
subclasses of ``torch.nn.Module``. This is the PyTorch base class meant
to encapsulate behaviors specific to PyTorch Models and their
components.

One important behavior of ``torch.nn.Module`` is registering parameters.
If a particular ``Module`` subclass has learning weights, these weights
are expressed as instances of ``torch.nn.Parameter``. The ``Parameter``
class is a subclass of ``torch.Tensor``, with the special behavior that
when they are assigned as attributes of a ``Module``, they are added to
the list of that modules parameters. These parameters may be accessed
through the ``parameters()`` method on the ``Module`` class.

As a simple example, here’s a very simple model with two linear layers
and an activation function. We’ll create an instance of it and ask it to
report on its parameters:

"""

import torch

class TinyModel(torch.nn.Module):
    
    def __init__(self):
        super(TinyModel, self).__init__()
        
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)


#########################################################################
# This shows the fundamental structure of a PyTorch model: there is an
# ``__init__()`` method that defines the layers and other components of a
# model, and a ``forward()`` method where the computation gets done. Note
# that we can print the model, or any of its submodules, to learn about
# its structure.
# 
# Common Layer Types
# ------------------
# 
# Linear Layers
# ~~~~~~~~~~~~~
# 
# The most basic type of neural network layer is a *linear* or *fully
# connected* layer. This is a layer where every input influences every
# output of the layer to a degree specified by the layer’s weights. If a
# model has *m* inputs and *n* outputs, the weights will be an *m* x *n*
# matrix. For example:
# 

lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)


#########################################################################
# If you do the matrix multiplication of ``x`` by the linear layer’s
# weights, and add the biases, you’ll find that you get the output vector
# ``y``.
# 
# One other important feature to note: When we checked the weights of our
# layer with ``lin.weight``, it reported itself as a ``Parameter`` (which
# is a subclass of ``Tensor``), and let us know that it’s tracking
# gradients with autograd. This is a default behavior for ``Parameter``
# that differs from ``Tensor``.
# 
# Linear layers are used widely in deep learning models. One of the most
# common places you’ll see them is in classifier models, which will
# usually have one or more linear layers at the end, where the last layer
# will have *n* outputs, where *n* is the number of classes the classifier
# addresses.
# 
# Convolutional Layers
# ~~~~~~~~~~~~~~~~~~~~
# 
# *Convolutional* layers are built to handle data with a high degree of
# spatial correlation. They are very commonly used in computer vision,
# where they detect close groupings of features which the compose into
# higher-level features. They pop up in other contexts too - for example,
# in NLP applications, where a word’s immediate context (that is, the
# other words nearby in the sequence) can affect the meaning of a
# sentence.
# 
# We saw convolutional layers in action in LeNet5 in an earlier video:
# 

import torch.functional as F


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


##########################################################################
# Let’s break down what’s happening in the convolutional layers of this
# model. Starting with ``conv1``:
# 
# -  LeNet5 is meant to take in a 1x32x32 black & white image. **The first
#    argument to a convolutional layer’s constructor is the number of
#    input channels.** Here, it is 1. If we were building this model to
#    look at 3-color channels, it would be 3.
# -  A convolutional layer is like a window that scans over the image,
#    looking for a pattern it recognizes. These patterns are called
#    *features,* and one of the parameters of a convolutional layer is the
#    number of features we would like it to learn. **This is the second
#    argument to the constructor is the number of output features.** Here,
#    we’re asking our layer to learn 6 features.
# -  Just above, I likened the convolutional layer to a window - but how
#    big is the window? **The third argument is the window or kernel
#    size.** Here, the “5” means we’ve chosen a 5x5 kernel. (If you want a
#    kernel with height different from width, you can specify a tuple for
#    this argument - e.g., ``(3, 5)`` to get a 3x5 convolution kernel.)
# 
# The output of a convolutional layer is an *activation map* - a spatial
# representation of the presence of features in the input tensor.
# ``conv1`` will give us an output tensor of 6x28x28; 6 is the number of
# features, and 28 is the height and width of our map. (The 28 comes from
# the fact that when scanning a 5-pixel window over a 32-pixel row, there
# are only 28 valid positions.)
# 
# We then pass the output of the convolution through a ReLU activation
# function (more on activation functions later), then through a max
# pooling layer. The max pooling layer takes features near each other in
# the activation map and groups them together. It does this by reducing
# the tensor, merging every 2x2 group of cells in the output into a single
# cell, and assigning that cell the maximum value of the 4 cells that went
# into it. This gives us a lower-resolution version of the activation map,
# with dimensions 6x14x14.
# 
# Our next convolutional layer, ``conv2``, expects 6 input channels
# (corresponding to the 6 features sought by the first layer), has 16
# output channels, and a 3x3 kernel. It puts out a 16x12x12 activation
# map, which is again reduced by a max pooling layer to 16x6x6. Prior to
# passing this output to the linear layers, it is reshaped to a 16 \* 6 \*
# 6 = 576-element vector for consumption by the next layer.
# 
# There are convolutional layers for addressing 1D, 2D, and 3D tensors.
# There are also many more optional arguments for a conv layer
# constructor, including stride length(e.g., only scanning every second or
# every third position) in the input, padding (so you can scan out to the
# edges of the input), and more. See the
# `documentation <https://pytorch.org/docs/stable/nn.html#convolution-layers>`__
# for more information.
# 
# Recurrent Layers
# ~~~~~~~~~~~~~~~~
# 
# *Recurrent neural networks* (or *RNNs)* are used for sequential data -
# anything from time-series measurements from a scientific instrument to
# natural language sentences to DNA nucleotides. An RNN does this by
# maintaining a *hidden state* that acts as a sort of memory for what it
# has seen in the sequence so far.
# 
# The internal structure of an RNN layer - or its variants, the LSTM (long
# short-term memory) and GRU (gated recurrent unit) - is moderately
# complex and beyond the scope of this video, but we’ll show you what one
# looks like in action with an LSTM-based part-of-speech tagger (a type of
# classifier that tells you if a word is a noun, verb, etc.):
# 

class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


########################################################################
# The constructor has four arguments:
# 
# -  ``vocab_size`` is the number of words in the input vocabulary. Each
#    word is a one-hot vector (or unit vector) in a
#    ``vocab_size``-dimensional space.
# -  ``tagset_size`` is the number of tags in the output set.
# -  ``embedding_dim`` is the size of the *embedding* space for the
#    vocabulary. An embedding maps a vocabulary onto a low-dimensional
#    space, where words with similar meanings are close together in the
#    space.
# -  ``hidden_dim`` is the size of the LSTM’s memory.
# 
# The input will be a sentence with the words represented as indices of
# one-hot vectors. The embedding layer will then map these down to an
# ``embedding_dim``-dimensional space. The LSTM takes this sequence of
# embeddings and iterates over it, fielding an output vector of length
# ``hidden_dim``. The final linear layer acts as a classifier; applying
# ``log_softmax()`` to the output of the final layer converts the output
# into a normalized set of estimated probabilities that a given word maps
# to a given tag.
# 
# If you’d like to see this network in action, check out the `Sequence
# Models and LSTM
# Networks <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html>`__
# tutorial on pytorch.org.
# 
# Transformers
# ~~~~~~~~~~~~
# 
# *Transformers* are multi-purpose networks that have taken over the state
# of the art in NLP with models like BERT. A discussion of transformer
# architecture is beyond the scope of this video, but PyTorch has a
# ``Transformer`` class that allows you to define the overall parameters
# of a transformer model - the number of attention heads, the number of
# encoder & decoder layers, dropout and activation functions, etc. (You
# can even build the BERT model from this single class, with the right
# parameters!) The ``torch.nn.Transformer`` class also has classes to
# encapsulate the individual components (``TransformerEncoder``,
# ``TransformerDecoder``) and subcomponents (``TransformerEncoderLayer``,
# ``TransformerDecoderLayer``). For details, check out the
# `documentation <https://pytorch.org/docs/stable/nn.html#transformer-layers>`__
# on transformer classes, and the relevant
# `tutorial <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`__
# on pytorch.org.
# 
# Other Layers and Functions
# --------------------------
# 
# Data Manipulation Layers
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# There are other layer types that perform important functions in models,
# but don’t participate in the learning process themselves.
# 
# **Max pooling** (and its twin, min pooling) reduce a tensor by combining
# cells, and assigning the maximum value of the input cells to the output
# cell (we saw this). For example:
# 

my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))


#########################################################################
# If you look closely at the values above, you’ll see that each of the
# values in the maxpooled output is the maximum value of each quadrant of
# the 6x6 input.
# 
# **Normalization layers** re-center and normalize the output of one layer
# before feeding it to another. Centering and scaling the intermediate
# tensors has a number of beneficial effects, such as letting you use
# higher learning rates without exploding/vanishing gradients.
# 

my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())



##########################################################################
# Running the cell above, we’ve added a large scaling factor and offset to
# an input tensor; you should see the input tensor’s ``mean()`` somewhere
# in the neighborhood of 15. After running it through the normalization
# layer, you can see that the values are smaller, and grouped around zero
# - in fact, the mean should be very small (> 1e-8).
# 
# This is beneficial because many activation functions (discussed below)
# have their strongest gradients near 0, but sometimes suffer from
# vanishing or exploding gradients for inputs that drive them far away
# from zero. Keeping the data centered around the area of steepest
# gradient will tend to mean faster, better learning and higher feasible
# learning rates.
# 
# **Dropout layers** are a tool for encouraging *sparse representations*
# in your model - that is, pushing it to do inference with less data.
# 
# Dropout layers work by randomly setting parts of the input tensor
# *during training* - dropout layers are always turned off for inference.
# This forces the model to learn against this masked or reduced dataset.
# For example:
# 

my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))


##########################################################################
# Above, you can see the effect of dropout on a sample tensor. You can use
# the optional ``p`` argument to set the probability of an individual
# weight dropping out; if you don’t it defaults to 0.5.
# 
# Activation Functions
# ~~~~~~~~~~~~~~~~~~~~
# 
# Activation functions make deep learning possible. A neural network is
# really a program - with many parameters - that *simulates a mathematical
# function*. If all we did was multiple tensors by layer weights
# repeatedly, we could only simulate *linear functions;* further, there
# would be no point to having many layers, as the whole network would
# reduce could be reduced to a single matrix multiplication. Inserting
# *non-linear* activation functions between layers is what allows a deep
# learning model to simulate any function, rather than just linear ones.
# 
# ``torch.nn.Module`` has objects encapsulating all of the major
# activation functions including ReLU and its many variants, Tanh,
# Hardtanh, sigmoid, and more. It also includes other functions, such as
# Softmax, that are most useful at the output stage of a model.
# 
# Loss Functions
# ~~~~~~~~~~~~~~
# 
# Loss functions tell us how far a model’s prediction is from the correct
# answer. PyTorch contains a variety of loss functions, including common
# MSE (mean squared error = L2 norm), Cross Entropy Loss and Negative
# Likelihood Loss (useful for classifiers), and others.
# 
