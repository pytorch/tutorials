# -*- coding: utf-8 -*-
"""
Deep Learning for Natural Language Processing with Pytorch
**********************************************************
**Author**: `Robert Guthrie <https://github.com/rguthrie3/DeepLearningForNLPInPytorch>`_

This tutorial will walk you through the key ideas of deep learning
programming using Pytorch. Many of the concepts (such as the computation
graph abstraction and autograd) are not unique to Pytorch and are
relevant to any deep learning tool kit out there.

I am writing this tutorial to focus specifically on NLP for people who
have never written code in any deep learning framework (e.g, TensorFlow,
Theano, Keras, Dynet). It assumes working knowledge of core NLP
problems: part-of-speech tagging, language modeling, etc. It also
assumes familiarity with neural networks at the level of an intro AI
class (such as one from the Russel and Norvig book). Usually, these
courses cover the basic backpropagation algorithm on feed-forward neural
networks, and make the point that they are chains of compositions of
linearities and non-linearities. This tutorial aims to get you started
writing deep learning code, given you have this prerequisite knowledge.

Note this is about *models*, not data. For all of the models, I just
create a few test examples with small dimensionality so you can see how
the weights change as it trains. If you have some real data you want to
try, you should be able to rip out any of the models from this notebook
and use them on it.

"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


######################################################################
# 1. Introduction to Torch's tensor library
# =========================================
#


######################################################################
# All of deep learning is computations on tensors, which are
# generalizations of a matrix that can be indexed in more than 2
# dimensions. We will see exactly what this means in-depth later. First,
# lets look what we can do with tensors.
#


######################################################################
# Creating Tensors
# ~~~~~~~~~~~~~~~~
#
# Tensors can be created from Python lists with the torch.Tensor()
# function.
#

# Create a torch.Tensor object with the given data.  It is a 1D vector
V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
print(V)

# Creates a matrix
M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.Tensor(M_data)
print(M)

# Create a 3D tensor of size 2x2x2.
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.Tensor(T_data)
print(T)


######################################################################
# What is a 3D tensor anyway? Think about it like this. If you have a
# vector, indexing into the vector gives you a scalar. If you have a
# matrix, indexing into the matrix gives you a vector. If you have a 3D
# tensor, then indexing into the tensor gives you a matrix!
#
# A note on terminology:
# when I say "tensor" in this tutorial, it refers
# to any torch.Tensor object. Matrices and vectors are special cases of
# torch.Tensors, where their dimension is 1 and 2 respectively. When I am
# talking about 3D tensors, I will explicitly use the term "3D tensor".
#

# Index into V and get a scalar
print(V[0])

# Index into M and get a vector
print(M[0])

# Index into T and get a matrix
print(T[0])


######################################################################
# You can also create tensors of other datatypes. The default, as you can
# see, is Float. To create a tensor of integer types, try
# torch.LongTensor(). Check the documentation for more data types, but
# Float and Long will be the most common.
#


######################################################################
# You can create a tensor with random data and the supplied dimensionality
# with torch.randn()
#

x = torch.randn((3, 4, 5))
print(x)


######################################################################
# Operations with Tensors
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# You can operate on tensors in the ways you would expect.

x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
z = x + y
print(z)


######################################################################
# See `the documentation <http://pytorch.org/docs/torch.html>`__ for a
# complete list of the massive number of operations available to you. They
# expand beyond just mathematical operations.
#
# One helpful operation that we will make use of later is concatenation.
#

# By default, it concatenates along the first axis (concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# second arg specifies which axis to concat along
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# If your tensors are not compatible, torch will complain.  Uncomment to see the error
# torch.cat([x_1, x_2])


######################################################################
# Reshaping Tensors
# ~~~~~~~~~~~~~~~~~
#
# Use the .view() method to reshape a tensor. This method receives heavy
# use, because many neural network components expect their inputs to have
# a certain shape. Often you will need to reshape before passing your data
# to the component.
#

x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))  # Reshape to 2 rows, 12 columns
# Same as above.  If one of the dimensions is -1, its size can be inferred
print(x.view(2, -1))


######################################################################
# 2. Computation Graphs and Automatic Differentiation
# ===================================================
#


######################################################################
# The concept of a computation graph is essential to efficient deep
# learning programming, because it allows you to not have to write the
# back propagation gradients yourself. A computation graph is simply a
# specification of how your data is combined to give you the output. Since
# the graph totally specifies what parameters were involved with which
# operations, it contains enough information to compute derivatives. This
# probably sounds vague, so lets see what is going on using the
# fundamental class of Pytorch: autograd.Variable.
#
# First, think from a programmers perspective. What is stored in the
# torch.Tensor objects we were creating above? Obviously the data and the
# shape, and maybe a few other things. But when we added two tensors
# together, we got an output tensor. All this output tensor knows is its
# data and shape. It has no idea that it was the sum of two other tensors
# (it could have been read in from a file, it could be the result of some
# other operation, etc.)
#
# The Variable class keeps track of how it was created. Lets see it in
# action.
#

# Variables wrap tensor objects
x = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)
# You can access the data with the .data attribute
print(x.data)

# You can also do all the same operations you did with tensors with Variables.
y = autograd.Variable(torch.Tensor([4., 5., 6]), requires_grad=True)
z = x + y
print(z.data)

# BUT z knows something extra.
print(z.creator)


######################################################################
# So Variables know what created them. z knows that it wasn't read in from
# a file, it wasn't the result of a multiplication or exponential or
# whatever. And if you keep following z.creator, you will find yourself at
# x and y.
#
# But how does that help us compute a gradient?
#

# Lets sum up all the entries in z
s = z.sum()
print(s)
print(s.creator)


######################################################################
# So now, what is the derivative of this sum with respect to the first
# component of x? In math, we want
#
# .. math::
#
#    \frac{\partial s}{\partial x_0}
#
#
#
# Well, s knows that it was created as a sum of the tensor z. z knows
# that it was the sum x + y. So
#
# .. math::  s = \overbrace{x_0 + y_0}^\text{$z_0$} + \overbrace{x_1 + y_1}^\text{$z_1$} + \overbrace{x_2 + y_2}^\text{$z_2$}
#
# And so s contains enough information to determine that the derivative
# we want is 1!
#
# Of course this glosses over the challenge of how to actually compute
# that derivative. The point here is that s is carrying along enough
# information that it is possible to compute it. In reality, the
# developers of Pytorch program the sum() and + operations to know how to
# compute their gradients, and run the back propagation algorithm. An
# in-depth discussion of that algorithm is beyond the scope of this
# tutorial.
#


######################################################################
# Lets have Pytorch compute the gradient, and see that we were right:
# (note if you run this block multiple times, the gradient will increment.
# That is because Pytorch *accumulates* the gradient into the .grad
# property, since for many models this is very convenient.)
#

# calling .backward() on any variable will run backprop, starting from it.
s.backward()
print(x.grad)


######################################################################
# Understanding what is going on in the block below is crucial for being a
# successful programmer in deep learning.
#

x = torch.randn((2, 2))
y = torch.randn((2, 2))
z = x + y  # These are Tensor types, and backprop would not be possible

var_x = autograd.Variable(x)
var_y = autograd.Variable(y)
# var_z contains enough information to compute gradients, as we saw above
var_z = var_x + var_y
print(var_z.creator)

var_z_data = var_z.data  # Get the wrapped Tensor object out of var_z...
# Re-wrap the tensor in a new variable
new_var_z = autograd.Variable(var_z_data)

# ... does new_var_z have information to backprop to x and y?
# NO!
print(new_var_z.creator)
# And how could it?  We yanked the tensor out of var_z (that is 
# what var_z.data is).  This tensor doesn't know anything about
# how it was computed.  We pass it into new_var_z, and this is all the
# information new_var_z gets.  If var_z_data doesn't know how it was 
# computed, theres no way new_var_z will.
# In essence, we have broken the variable away from its past history


######################################################################
# Here is the basic, extremely important rule for computing with
# autograd.Variables (note this is more general than Pytorch. There is an
# equivalent object in every major deep learning toolkit):
#
# **If you want the error from your loss function to backpropogate to a
# component of your network, you MUST NOT break the Variable chain from
# that component to your loss Variable. If you do, the loss will have no
# idea your component exists, and its parameters can't be updated.**
#
# I say this in bold, because this error can creep up on you in very
# subtle ways (I will show some such ways below), and it will not cause
# your code to crash or complain, so you must be careful.
#


######################################################################
# 3. Deep Learning Building Blocks: Affine maps, non-linearities and objectives
# =============================================================================
#


######################################################################
# Deep learning consists of composing linearities with non-linearities in
# clever ways. The introduction of non-linearities allows for powerful
# models. In this section, we will play with these core components, make
# up an objective function, and see how the model is trained.
#


######################################################################
# Affine Maps
# ~~~~~~~~~~~
#
# One of the core workhorses of deep learning is the affine map, which is
# a function :math:`f(x)` where
#
# .. math::  f(x) = Ax + b
#
# for a matrix :math:`A` and vectors :math:`x, b`. The parameters to be
# learned here are :math:`A` and :math:`b`. Often, :math:`b` is refered to
# as the *bias* term.
#


######################################################################
# Pytorch and most other deep learning frameworks do things a little
# differently than traditional linear algebra. It maps the rows of the
# input instead of the columns. That is, the :math:`i`'th row of the
# output below is the mapping of the :math:`i`'th row of the input under
# :math:`A`, plus the bias term. Look at the example below.
#

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

# Softmax is also in torch.functional
data = autograd.Variable(torch.randn(5))
print(data)
print(F.softmax(data))
print(F.softmax(data).sum())  # Sums to 1 because it is a distribution!
print(F.log_softmax(data))  # theres also log_softmax


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
# 4. Optimization and Training
# ============================
#


######################################################################
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
# algorithms are doing unless you are really interested. Torch provies
# many in the torch.optim package, and they are all completely
# transparent. Using the simplest gradient update is the same as the more
# complicated algorithms. Trying different update algorithms and different
# parameters for the update algorithms (like different initial learning
# rates) is important in optimizing your network's performance. Often,
# just replacing vanilla SGD with an optimizer like Adam or RMSProp will
# boost performance noticably.
#


######################################################################
# 5. Creating Network Components in Pytorch
# =========================================
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
        return F.log_softmax(self.linear(bow_vec))


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
#(in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
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


######################################################################
# 6. Word Embeddings: Encoding Lexical Semantics
# ==============================================
#


######################################################################
# Word embeddings are dense vectors of real numbers, one per word in your
# vocabulary. In NLP, it is almost always the case that your features are
# words! But how should you represent a word in a computer? You could
# store its ascii character representation, but that only tells you what
# the word *is*, it doesn't say much about what it *means* (you might be
# able to derive its part of speech from its affixes, or properties from
# its capitalization, but not much). Even more, in what sense could you
# combine these representations? We often want dense outputs from our
# neural networks, where the inputs are :math:`|V|` dimensional, where
# :math:`V` is our vocabulary, but often the outputs are only a few
# dimensional (if we are only predicting a handful of labels, for
# instance). How do we get from a massive dimensional space to a smaller
# dimensional space?
#
# How about instead of ascii representations, we use a one-hot encoding?
# That is, we represent the word :math:`w` by
#
# .. math::  \overbrace{\left[ 0, 0, \dots, 1, \dots, 0, 0 \right]}^\text{|V| elements}
#
# where the 1 is in a location unique to :math:`w`. Any other word will
# have a 1 in some other location, and a 0 everywhere else.
#
# There is an enormous drawback to this representation, besides just how
# huge it is. It basically treats all words as independent entities with
# no relation to each other. What we really want is some notion of
# *similarity* between words. Why? Let's see an example.
#


######################################################################
# Suppose we are building a language model. Suppose we have seen the
# sentences 
# 
# * The mathematician ran to the store. 
# * The physicist ran to the store. 
# * The mathematician solved the open problem.
#
# in our training data. Now suppose we get a new sentence never before
# seen in our training data: 
# 
# * The physicist solved the open problem.
#
# Our language model might do OK on this sentence, but wouldn't it be much
# better if we could use the following two facts: 
# 
# * We have seen  mathematician and physicist in the same role in a sentence. Somehow they
#   have a semantic relation. 
# * We have seen mathematician in the same role  in this new unseen sentence 
#   as we are now seeing physicist.
#
# and then infer that physicist is actually a good fit in the new unseen
# sentence? This is what we mean by a notion of similarity: we mean
# *semantic similarity*, not simply having similar orthographic
# representations. It is a technique to combat the sparsity of linguistic
# data, by connecting the dots between what we have seen and what we
# haven't. This example of course relies on a fundamental linguistic
# assumption: that words appearing in similar contexts are related to each
# other semantically. This is called the `distributional
# hypothesis <https://en.wikipedia.org/wiki/Distributional_semantics>`__.
#


######################################################################
# Getting Dense Word Embeddings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# How can we solve this problem? That is, how could we actually encode
# semantic similarity in words? Maybe we think up some semantic
# attributes. For example, we see that both mathematicians and physicists
# can run, so maybe we give these words a high score for the "is able to
# run" semantic attribute. Think of some other attributes, and imagine
# what you might score some common words on those attributes.
#
# If each attribute is a dimension, then we might give each word a vector,
# like this:
#
# .. math::
#
#     q_\text{mathematician} = \left[ \overbrace{2.3}^\text{can run},
#    \overbrace{9.4}^\text{likes coffee}, \overbrace{-5.5}^\text{majored in Physics}, \dots \right]
#
# .. math::
#
#     q_\text{physicist} = \left[ \overbrace{2.5}^\text{can run},
#    \overbrace{9.1}^\text{likes coffee}, \overbrace{6.4}^\text{majored in Physics}, \dots \right]
#
# Then we can get a measure of similarity between these words by doing:
#
# .. math::  \text{Similarity}(\text{physicist}, \text{mathematician}) = q_\text{physicist} \cdot q_\text{mathematician}
#
# Although it is more common to normalize by the lengths:
#
# .. math::
#
#     \text{Similarity}(\text{physicist}, \text{mathematician}) = \frac{q_\text{physicist} \cdot q_\text{mathematician}}
#    {\| q_\text{\physicist} \| \| q_\text{mathematician} \|} = \cos (\phi)
#
# Where :math:`\phi` is the angle between the two vectors. That way,
# extremely similar words (words whose embeddings point in the same
# direction) will have similarity 1. Extremely dissimilar words should
# have similarity -1.
#


######################################################################
# You can think of the sparse one-hot vectors from the beginning of this
# section as a special case of these new vectors we have defined, where
# each word basically has similarity 0, and we gave each word some unique
# semantic attribute. These new vectors are *dense*, which is to say their
# entries are (typically) non-zero.
#
# But these new vectors are a big pain: you could think of thousands of
# different semantic attributes that might be relevant to determining
# similarity, and how on earth would you set the values of the different
# attributes? Central to the idea of deep learning is that the neural
# network learns representations of the features, rather than requiring
# the programmer to design them herself. So why not just let the word
# embeddings be parameters in our model, and then be updated during
# training? This is exactly what we will do. We will have some *latent
# semantic attributes* that the network can, in principle, learn. Note
# that the word embeddings will probably not be interpretable. That is,
# although with our hand-crafted vectors above we can see that
# mathematicians and physicists are similar in that they both like coffee,
# if we allow a neural network to learn the embeddings and see that both
# mathematicians and physicisits have a large value in the second
# dimension, it is not clear what that means. They are similar in some
# latent semantic dimension, but this probably has no interpretation to
# us.
#


######################################################################
# In summary, **word embeddings are a representation of the *semantics* of
# a word, efficiently encoding semantic information that might be relevant
# to the task at hand**. You can embed other things too: part of speech
# tags, parse trees, anything! The idea of feature embeddings is central
# to the field.
#


######################################################################
# Word Embeddings in Pytorch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Before we get to a worked example and an exercise, a few quick notes
# about how to use embeddings in Pytorch and in deep learning programming
# in general. Similar to how we defined a unique index for each word when
# making one-hot vectors, we also need to define an index for each word
# when using embeddings. These will be keys into a lookup table. That is,
# embeddings are stored as a :math:`|V| \times D` matrix, where :math:`D`
# is the dimensionality of the embeddings, such that the word assigned
# index :math:`i` has its embedding stored in the :math:`i`'th row of the
# matrix. In all of my code, the mapping from words to indices is a
# dictionary named word\_to\_ix.
#
# The module that allows you to use embeddings is torch.nn.Embedding,
# which takes two arguments: the vocabulary size, and the dimensionality
# of the embeddings.
#
# To index into this table, you must use torch.LongTensor (since the
# indices are integers, not floats).
#

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(autograd.Variable(lookup_tensor))
print(hello_embed)


######################################################################
# An Example: N-Gram Language Modeling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Recall that in an n-gram language model, given a sequence of words
# :math:`w`, we want to compute
#
# .. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
#
# Where :math:`w_i` is the ith word of the sequence.
#
# In this example, we will compute the loss function on some training
# examples and update the parameters with backpropagation.
#

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients.  Before passing in a new instance,
        # you need to zero out the gradients from the old instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!


######################################################################
# Exercise: Computing Word Embeddings: Continuous Bag-of-Words
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep
# learning. It is a model that tries to predict words given the context of
# a few words before and a few words after the target word. This is
# distinct from language modeling, since CBOW is not sequential and does
# not have to be probabilistic. Typcially, CBOW is used to quickly train
# word embeddings, and these embeddings are used to initialize the
# embeddings of some more complicated model. Usually, this is referred to
# as *pretraining embeddings*. It almost always helps performance a couple
# of percent.
#
# The CBOW model is as follows. Given a target word :math:`w_i` and an
# :math:`N` context window on each side, :math:`w_{i-1}, \dots, w_{i-N}`
# and :math:`w_{i+1}, \dots, w_{i+N}`, referring to all context words
# collectively as :math:`C`, CBOW tries to minimize
#
# .. math::  -\log p(w_i | C) = \log \text{Softmax}(A(\sum_{w \in C} q_w) + b)
#
# where :math:`q_w` is the embedding for word :math:`w`.
#
# Implement this model in Pytorch by filling in the class below. Some
# tips: 
# 
# * Think about which parameters you need to define. 
# * Make sure you know what shape each operation expects. Use .view() if you need to
#   reshape.
#

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
word_to_ix = {word: i for i, word in enumerate(raw_text)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

make_context_vector(data[0][0], word_to_ix)  # example


######################################################################
# 7. Sequence Models and Long-Short Term Memory Networks
# ======================================================
#


######################################################################
# At this point, we have seen various feed-forward networks. That is,
# there is no state maintained by the network at all. This might not be
# the behavior we want. Sequence models are central to NLP: they are
# models where there is some sort of dependence through time between your
# inputs. The classical example of a sequence model is the Hidden Markov
# Model for part-of-speech tagging. Another example is the conditional
# random field.
#
# A recurrent neural network is a network that maintains some kind of
# state. For example, its output could be used as part of the next input,
# so that information can propogate along as the network passes over the
# sequence. In the case of an LSTM, for each element in the sequence,
# there is a corresponding *hidden state* :math:`h_t`, which in principle
# can contain information from arbitrary points earlier in the sequence.
# We can use the hidden state to predict words in a language model,
# part-of-speech tags, and a myriad of other things.
#


######################################################################
# LSTM's in Pytorch
# ~~~~~~~~~~~~~~~~~
#
# Before getting to the example, note a few things. Pytorch's LSTM expects
# all of its inputs to be 3D tensors. The semantics of the axes of these
# tensors is important. The first axis is the sequence itself, the second
# indexes instances in the mini-batch, and the third indexes elements of
# the input. We haven't discussed mini-batching, so lets just ignore that
# and assume we will always have just 1 dimension on the second axis. If
# we want to run the sequence model over the sentence "The cow jumped",
# our input should look like
#
# .. math::
#
#
#    \begin{bmatrix}
#    \overbrace{q_\text{The}}^\text{row vector} \\
#    q_\text{cow} \\
#    q_\text{jumped}
#    \end{bmatrix}
#
# Except remember there is an additional 2nd dimension with size 1.
#
# In addition, you could go through the sequence one at a time, in which
# case the 1st axis will have size 1 also.
#
# Let's see a quick example.
#

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [autograd.Variable(torch.randn((1, 3)))
          for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
          autograd.Variable(torch.randn((1, 1, 3))))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout 
# the sequence. the second is just the most recent hidden state 
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropogate, 
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(
    torch.randn((1, 1, 3))))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)


######################################################################
# Example: An LSTM for Part-of-Speech Tagging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# In this section, we will use an LSTM to get part of speech tags. We will
# not use Viterbi or Forward-Backward or anything like that, but as a
# (challenging) exercise to the reader, think about how Viterbi could be
# used after you have seen what is going on.
#
# The model is as follows: let our input sentence be
# :math:`w_1, \dots, w_M`, where :math:`w_i \in V`, our vocab. Also, let
# :math:`T` be our tag set, and :math:`y_i` the tag of word :math:`w_i`.
# Denote our prediction of the tag of word :math:`w_i` by
# :math:`\hat{y}_i`.
#
# This is a structure prediction, model, where our output is a sequence
# :math:`\hat{y}_1, \dots, \hat{y}_M`, where :math:`\hat{y}_i \in T`.
#
# To do the prediction, pass an LSTM over the sentence. Denote the hidden
# state at timestep :math:`i` as :math:`h_i`. Also, assign each tag a
# unique index (like how we had word\_to\_ix in the word embeddings
# section). Then our prediction rule for :math:`\hat{y}_i` is
#
# .. math::  \hat{y}_i = \text{argmax}_j \  (\log \text{Softmax}(Ah_i + b))_j
#
# That is, take the log softmax of the affine map of the hidden state,
# and the predicted tag is the tag that has the maximum value in this
# vector. Note this implies immediately that the dimensionality of the
# target space of :math:`A` is :math:`|T|`.
#

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM, detaching it from its
        # history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into Variables
        # of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j for word i.
# The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
print(tag_scores)


######################################################################
# Exercise: Augmenting the LSTM part-of-speech tagger with character-level features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the example above, each word had an embedding, which served as the
# inputs to our sequence model. Let's augment the word embeddings with a
# representation derived from the characters of the word. We expect that
# this should help significantly, since character-level information like
# affixes have a large bearing on part-of-speech. For example, words with
# the affix *-ly* are almost always tagged as adverbs in English.
#
# Do do this, let :math:`c_w` be the character-level representation of
# word :math:`w`. Let :math:`x_w` be the word embedding as before. Then
# the input to our sequence model is the concatenation of :math:`x_w` and
# :math:`c_w`. So if :math:`x_w` has dimension 5, and :math:`c_w`
# dimension 3, then our LSTM should accept an input of dimension 8.
#
# To get the character level representation, do an LSTM over the
# characters of a word, and let :math:`c_w` be the final hidden state of
# this LSTM. Hints: 
# * There are going to be two LSTM's in your new model.
#   The original one that outputs POS tag scores, and the new one that
#   outputs a character-level representation of each word.
# * To do a sequence model over characters, you will have to embed characters.
#   The character embeddings will be the input to the character LSTM.
#


######################################################################
# 8. Advanced: Making Dynamic Decisions and the Bi-LSTM CRF
# =========================================================
#


######################################################################
# Dyanmic versus Static Deep Learning Toolkits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Pytorch is a *dynamic* neural network kit. Another example of a dynamic
# kit is `Dynet <https://github.com/clab/dynet>`__ (I mention this because
# working with Pytorch and Dynet is similar. If you see an example in
# Dynet, it will probably help you implement it in Pytorch). The opposite
# is the *static* tool kit, which includes Theano, Keras, TensorFlow, etc.
# The core difference is the following: 
#
# * In a static toolkit, you define
#   a computation graph once, compile it, and then stream instances to it.
# * In a dynamic toolkit, you define a computation graph *for each
#   instance*. It is never compiled and is executed on-the-fly
#
# Without a lot of experience, it is difficult to appreciate the
# difference. One example is to suppose we want to build a deep
# constituent parser. Suppose our model involves roughly the following
# steps: 
# 
# * We build the tree bottom up 
# * Tag the root nodes (the words of the sentence) 
# * From there, use a neural network and the embeddings
# 
# of the words to find combinations that form constituents. Whenever you
# form a new constituent, use some sort of technique to get an embedding
# of the constituent. In this case, our network architecture will depend
# completely on the input sentence. In the sentence "The green cat
# scratched the wall", at some point in the model, we will want to combine
# the span :math:`(i,j,r) = (1, 3, \text{NP})` (that is, an NP constituent
# spans word 1 to word 3, in this case "The green cat").
#
# However, another sentence might be "Somewhere, the big fat cat scratched
# the wall". In this sentence, we will want to form the constituent
# :math:`(2, 4, NP)` at some point. The constituents we will want to form
# will depend on the instance. If we just compile the computation graph
# once, as in a static toolkit, it will be exceptionally difficult or
# impossible to program this logic. In a dynamic toolkit though, there
# isn't just 1 pre-defined computation graph. There can be a new
# computation graph for each instance, so this problem goes away.
#
# Dynamic toolkits also have the advantage of being easier to debug and
# the code more closely resembling the host language (by that I mean that
# Pytorch and Dynet look more like actual Python code than Keras or
# Theano).
#


######################################################################
# Bi-LSTM Conditional Random Field Discussion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For this section, we will see a full, complicated example of a Bi-LSTM
# Conditional Random Field for named-entity recognition. The LSTM tagger
# above is typically sufficient for part-of-speech tagging, but a sequence
# model like the CRF is really essential for strong performance on NER.
# Familiarity with CRF's is assumed. Although this name sounds scary, all
# the model is is a CRF but where an LSTM provides the features. This is
# an advanced model though, far more complicated than any earlier model in
# this tutorial. If you want to skip it, that is fine. To see if you're
# ready, see if you can:
#
# -  Write the recurrence for the viterbi variable at step i for tag k.
# -  Modify the above recurrence to compute the forward variables instead.
# -  Modify again the above recurrence to compute the forward variables in
#    log-space (hint: log-sum-exp)
#
# If you can do those three things, you should be able to understand the
# code below. Recall that the CRF computes a conditional probability. Let
# :math:`y` be a tag sequence and :math:`x` an input sequence of words.
# Then we compute
#
# .. math::  P(y|x) = \frac{\exp{(\text{Score}(x, y)})}{\sum_{y'} \exp{(\text{Score}(x, y')})}
#
# Where the score is determined by defining some log potentials
# :math:`\log \psi_i(x,y)` such that
#
# .. math::  \text{Score}(x,y) = \sum_i \log \psi_i(x,y)
#
# To make the partition function tractable, the potentials must look only
# at local features.
#
# In the Bi-LSTM CRF, we define two kinds of potentials: emission and
# transition. The emission potential for the word at index :math:`i` comes
# from the hidden state of the Bi-LSTM at timestep :math:`i`. The
# transition scores are stored in a :math:`|T|x|T|` matrix
# :math:`\textbf{P}`, where :math:`T` is the tag set. In my
# implementation, :math:`\textbf{P}_{j,k}` is the score of transitioning
# to tag :math:`j` from tag :math:`k`. So:
#
# .. math::  \text{Score}(x,y) = \sum_i \log \psi_\text{EMIT}(y_i \rightarrow x_i) + \log \psi_\text{TRANS}(y_{i-1} \rightarrow y_i)
#
# .. math::  = \sum_i h_i[y_i] + \textbf{P}_{y_i, y_{i-1}}
#
# where in this second expression, we think of the tags as being assigned
# unique non-negative indices.
#
# If the above discussion was too brief, you can check out
# `this <http://www.cs.columbia.edu/%7Emcollins/crf.pdf>`__ write up from
# Michael Collins on CRFs.
#
# Implementation Notes
# ~~~~~~~~~~~~~~~~~~~~
#
# The example below implements the forward algorithm in log space to
# compute the partition function, and the viterbi algorithm to decode.
# Backpropagation will compute the gradients automatically for us. We
# don't have to do anything by hand.
#
# The implementation is not optimized. If you understand what is going on,
# you'll probably quickly see that iterating over the next tag in the
# forward algorithm could probably be done in one big operation. I wanted
# to code to be more readable. If you want to make the relevant change,
# you could probably use this tagger for real tasks.
#

# Helper functions to make the code more readable.
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# Compute log sum exp in a numerically stable way for the forward algorithm


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag)
                # before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        self.hidden = self.init_hidden()
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        self.hidden = self.init_hidden()
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into Variables
        # of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # Step 3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        neg_log_likelihood.backward()
        optimizer.step()

# Check predictions after training
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
print(model(precheck_sent))
# We got it!


######################################################################
# Exercise: A new loss function for discriminative tagging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# It wasn't really necessary for us to create a computation graph when
# doing decoding, since we do not backpropagate from the viterbi path
# score. Since we have it anyway, try training the tagger where the loss
# function is the difference between the Viterbi path score and the score
# of the gold-standard path. It should be clear that this function is
# non-negative and 0 when the predicted tag sequence is the correct tag
# sequence. This is essentially *structured perceptron*.
#
# This modification should be short, since Viterbi and score\_sentence are
# already implemented. This is an example of the shape of the computation
# graph *depending on the training instance*. Although I haven't tried
# implementing this in a static toolkit, I imagine that it is possible but
# much less straightforward.
#
# Pick up some real data and do a comparison!
#
