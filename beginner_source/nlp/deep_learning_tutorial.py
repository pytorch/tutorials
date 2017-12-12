# -*- coding: utf-8 -*-
r"""
Deep Learning with PyTorch
**************************

Deep Learning Building Blocks: Affine maps, non-linearities and objectives
==========================================================================

Deep learning consists of composing linearities with non-linearities in
clever ways. The introduction of non-linearities allows for powerful
models. In this section, we will play with these core components, make
up an objective function, and see how the model is trained.


Affine Maps
~~~~~~~~~~~

One of the core workhorses of deep learning is the affine map, which is
a function :math:`f(x)` where

.. math::  f(x) = Ax + b

for a matrix :math:`A` and vectors :math:`x, b`. The parameters to be
learned here are :math:`A` and :math:`b`. Often, :math:`b` is refered to
as the *bias* term.


Pytorch and most other deep learning frameworks do things a little
differently than traditional linear algebra. It maps the rows of the
input instead of the columns. That is, the :math:`i`'th row of the
output below is the mapping of the :math:`i`'th row of the input under
:math:`A`, plus the bias term. Look at the example below.

"""

# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


######################################################################

lin = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = autograd.Variable(torch.randn(2, 5))
print(lin(data))  # yes


######################################################################
# Non-Linearities
# ~~~~~~~~~~~~~~~
#
# First, note the following fact, which will explain why we need
# non-linearities in the first place. Suppose we have two affine maps
# :math:`f(x) = Ax + b` and :math:`g(x) = Cx + d`. What is
# :math:`f(g(x))`?
#
# .. math::  f(g(x)) = A(Cx + d) + b = ACx + (Ad + b)
#
# :math:`AC` is a matrix and :math:`Ad + b` is a vector, so we see that
# composing affine maps gives you an affine map.
#
# From this, you can see that if you wanted your neural network to be long
# chains of affine compositions, that this adds no new power to your model
# than just doing a single affine map.
#
# If we introduce non-linearities in between the affine layers, this is no
# longer the case, and we can build much more powerful models.
#
# There are a few core non-linearities.
# :math:`\tanh(x), \sigma(x), \text{ReLU}(x)` are the most common. You are
# probably wondering: "why these functions? I can think of plenty of other
# non-linearities." The reason for this is that they have gradients that
# are easy to compute, and computing gradients is essential for learning.
# For example
#
# .. math::  \frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))
#
# A quick note: although you may have learned some neural networks in your
# intro to AI class where :math:`\sigma(x)` was the default non-linearity,
# typically people shy away from it in practice. This is because the
# gradient *vanishes* very quickly as the absolute value of the argument
# grows. Small gradients means it is hard to learn. Most people default to
# tanh or ReLU.
#

# In pytorch, most non-linearities are in torch.functional (we have it imported as F)
# Note that non-linearites typically don't have parameters like affine maps do.
# That is, they don't have weights that are updated during training.
data = autograd.Variable(torch.randn(2, 2))
print(data)
print(F.relu(data))


######################################################################
# Softmax and Probabilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The function :math:`\text{Softmax}(x)` is also just a non-linearity, but
# it is special in that it usually is the last operation done in a
# network. This is because it takes in a vector of real numbers and
# returns a probability distribution. Its definition is as follows. Let
# :math:`x` be a vector of real numbers (positive, negative, whatever,
# there are no constraints). Then the i'th component of
# :math:`\text{Softmax}(x)` is
#
# .. math::  \frac{\exp(x_i)}{\sum_j \exp(x_j)}
#
# It should be clear that the output is a probability distribution: each
# element is non-negative and the sum over all components is 1.
#
# You could also think of it as just applying an element-wise
# exponentiation operator to the input to make everything non-negative and
# then dividing by the normalization constant.
#

# Softmax is also in torch.nn.functional
data = autograd.Variable(torch.randn(5))
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
print(F.log_softmax(data, dim=0))  # theres also log_softmax


######################################################################
# Objective Functions
# ~~~~~~~~~~~~~~~~~~~
#
# The objective function is the function that your network is being
# trained to minimize (in which case it is often called a *loss function*
# or *cost function*). This proceeds by first choosing a training
# instance, running it through your neural network, and then computing the
# loss of the output. The parameters of the model are then updated by
# taking the derivative of the loss function. Intuitively, if your model
# is completely confident in its answer, and its answer is wrong, your
# loss will be high. If it is very confident in its answer, and its answer
# is correct, the loss will be low.
#
# The idea behind minimizing the loss function on your training examples
# is that your network will hopefully generalize well and have small loss
# on unseen examples in your dev set, test set, or in production. An
# example loss function is the *negative log likelihood loss*, which is a
# very common objective for multi-class classification. For supervised
# multi-class classification, this means training the network to minimize
# the negative log probability of the correct output (or equivalently,
# maximize the log probability of the correct output).
#


######################################################################
# Optimization and Training
# =========================
#
# So what we can compute a loss function for an instance? What do we do
# with that? We saw earlier that autograd.Variable's know how to compute
# gradients with respect to the things that were used to compute it. Well,
# since our loss is an autograd.Variable, we can compute gradients with
# respect to all of the parameters used to compute it! Then we can perform
# standard gradient updates. Let :math:`\theta` be our parameters,
# :math:`L(\theta)` the loss function, and :math:`\eta` a positive
# learning rate. Then:
#
# .. math::  \theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta L(\theta)
#
# There are a huge collection of algorithms and active research in
# attempting to do something more than just this vanilla gradient update.
# Many attempt to vary the learning rate based on what is happening at
# train time. You don't need to worry about what specifically these
# algorithms are doing unless you are really interested. Torch provides
# many in the torch.optim package, and they are all completely
# transparent. Using the simplest gradient update is the same as the more
# complicated algorithms. Trying different update algorithms and different
# parameters for the update algorithms (like different initial learning
# rates) is important in optimizing your network's performance. Often,
# just replacing vanilla SGD with an optimizer like Adam or RMSProp will
# boost performance noticably.
#


######################################################################
# Creating Network Components in Pytorch
# ======================================
#
# Before we move on to our focus on NLP, lets do an annotated example of
# building a network in Pytorch using only affine maps and
# non-linearities. We will also see how to compute a loss function, using
# Pytorch's built in negative log likelihood, and update parameters by
# backpropagation.
#
# All network components should inherit from nn.Module and override the
# forward() method. That is about it, as far as the boilerplate is
# concerned. Inheriting from nn.Module provides functionality to your
# component. For example, it makes it keep track of its trainable
# parameters, you can swap it between CPU and GPU with the .cuda() or
# .cpu() functions, etc.
#
# Let's write an annotated example of a network that takes in a sparse
# bag-of-words representation and outputs a probability distribution over
# two labels: "English" and "Spanish". This model is just logistic
# regression.
#


######################################################################
# Example: Logistic Regression Bag-of-Words classifier
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our model will map a sparse BOW representation to log probabilities over
# labels. We assign each word in the vocab an index. For example, say our
# entire vocab is two words "hello" and "world", with indices 0 and 1
# respectively. The BoW vector for the sentence "hello hello hello hello"
# is
#
# .. math::  \left[ 4, 0 \right]
#
# For "hello world world hello", it is
#
# .. math::  \left[ 2, 2 \right]
#
# etc. In general, it is
#
# .. math::  \left[ \text{Count}(\text{hello}), \text{Count}(\text{world}) \right]
#
# Denote this BOW vector as :math:`x`. The output of our network is:
#
# .. math::  \log \text{Softmax}(Ax + b)
#
# That is, we pass the input through an affine map and then do log
# softmax.
#

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# the model knows its parameters.  The first output below is A, the second is b.
# Whenever you assign a component to a class variable in the __init__ function
# of a module, which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the Pytorch devs, your module
# (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
for param in model.parameters():
    print(param)

# To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
sample = data[0]
bow_vector = make_bow_vector(sample[0], word_to_ix)
log_probs = model(autograd.Variable(bow_vector))
print(log_probs)


######################################################################
# Which of the above values corresponds to the log probability of ENGLISH,
# and which to SPANISH? We never defined it, but we need to if we want to
# train the thing.
#

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}


######################################################################
# So lets train! To do this, we pass instances through to get log
# probabilities, compute a loss function, compute the gradient of the loss
# function, and then update the parameters with a gradient step. Loss
# functions are provided by Torch in the nn package. nn.NLLLoss() is the
# negative log likelihood loss we want. It also defines optimization
# functions in torch.optim. Here, we will just use SGD.
#
# Note that the *input* to NLLLoss is a vector of log probabilities, and a
# target label. It doesn't compute the log probabilities for us. This is
# why the last layer of our network is log softmax. The loss function
# nn.CrossEntropyLoss() is the same as NLLLoss(), except it does the log
# softmax for you.
#

# Run on test data before we train, just to see a before-and-after
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in range(100):
    for instance, label in data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Variable as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

# Index corresponding to Spanish goes up, English goes down!
print(next(model.parameters())[:, word_to_ix["creo"]])


######################################################################
# We got the right answer! You can see that the log probability for
# Spanish is much higher in the first example, and the log probability for
# English is much higher in the second for the test data, as it should be.
#
# Now you see how to make a Pytorch component, pass some data through it
# and do gradient updates. We are ready to dig deeper into what deep NLP
# has to offer.
#
