# -*- coding: utf-8 -*-
"""
Introduction to the Hybrid Frontend
===================================
**Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_

In this tutorial, we will experiment with some of the key features of
the PyTorch hybrid frontend.

This flexible programming model enables the co-existance and seamless
interaction between an imperative eager interface designed for
experimentation and easy debugging, and a declarative graph mode in
which models can be optimized for performance and exported to
large-scale production environments.

"""

######################################################################
# Eager Mode
# ----------
#
# PyTorch's eager mode is the familiar PyTorch interface. Eager mode is
# easy because it is simply idiomatic Python. Because of the imperative
# nature of the mode, we can use our favorite Python debugging tools such
# as ``pdb`` and ``print`` statements. This mode is designed for rapid
# model prototyping and ambitious research efforts.
#
# However, it is difficult to optimize models represented in eager mode
# for large-scale production and deployment.
#
# Graph Mode
# ----------
#
# The graph mode creates a deferred model representation, which is a
# common methodology in many deep learning frameworks. In graph mode, you
# write a model definition in Python, which is run later on in the Caffe2
# C++ runtime environment. Having a deferred representation has numerous
# benefits, including:
#
#   - *Model simplicity*: Model representation is simpler and more restricted
#   - *Optimization*: We can apply various optimization techniques to create
#     more efficient deployable models
#   - *Improved performance*: We can leverage parallel and out-of-order
#     execution, and target highly optimized hardware architectures
#   - *Easy export*: Exporting a serialized model is a breeze with ONNX
#
# .. figure:: /_static/img/hybrid_frontend/pytorch_workflow_small.jpg
#    :alt: workflow




######################################################################
# JIT Compiler
# ------------
#
# To accomodate to as many workflows as possible, PyTorch not only
# provides both an eager mode and a graph mode, but a transition mechanism
# between them: ``torch.jit``. This enables the use of the newest
# state-of-the-art models from research environments to be deployed and
# exported in demanding production environments without massive code
# re-writes.
#
# To learn more see `this
# letter <https://pytorch.org/2018/05/02/road-to-1.0.html>`__ from the
# PyTorch team.
#
# There are two separate modes for performing this transition.
#
# Tracing Mode
# ~~~~~~~~~~~~
#
# Tracing mode is a non-invasive tool which traces your Python module or
# function to create an internal representation that is compiled to graph
# mode. The ``torch.jit.trace`` function takes your module or function and
# an example data input, and traces the computational steps that the data
# encounters as it progresses through the model.
#
# An advantage of this approach is that it is an unobtrusive process that
# will work regardless of how your Python code is structured. However,
# because of the way the tracer works, data-dependent control flows (if
# statements, loops) cannot be fully captured. Rather, only the control
# sequence traced using the example input will be recorded.
#
# Script Mode
# ~~~~~~~~~~~
#
# To compile models with data-dependent control flow elements, such as
# RNNs, we simply add a ``@torch.jit.script`` decorator to our function.
# This annotation lets PyTorch know that the function may contain
# data-dependent control flows. Now, our loops, conditionals, and
# iterative data mutations are handled explicitly.
#
# The one constraint for users compiling with script mode is that it
# currently only supports a restricted subset of Python. As of now,
# features such as generators, defs, and Python data structures are not
# supported. To remedy this, you can invoke traced modules from script
# modules (and vice-versa), and you can call pure Python functions and
# modules from script modules. However, the operations done in the pure
# Python functions will not be compiled, and will run as-is.
#
# Regardless of which compilation mode you need, the end result is a
# Python-free model representation which can be optimized and exported.
#


######################################################################
# k-Nearest Neighbors Example
# ---------------------------
#
# To showcase the basics of the hybrid frontend, we will implement a basic
# classification algorithm using a mix of both compilation modes and raw
# Python.
#
# Dataset
# ~~~~~~~
#
# We will use the famous `Iris
# dataset <https://archive.ics.uci.edu/ml/datasets/iris>`__ as our toy
# problem.
#
# -  3 classes of flowers (Iris Setosa, Iris Versicolour, Iris Virginica)
# -  150 total samples (50 of each class)
# -  4 features (Sepal length, Sepal width, Petal length, Petal width)
#
# .. figure:: /_static/img/hybrid_frontend/iris_pic.jpg
#    :alt: iris
#
# Algorithm
# ~~~~~~~~~
#
# k-Nearest Neighbors is a relatively simple classification algorithm. It
# is instance-based and lazy, meaning that it uses all training examples
# to model the training data, and delays all computation (model fitting)
# until inference time.
#
# .. figure:: /_static/img/hybrid_frontend/220px-KnnClassification.png
#    :alt: knn
#
#    k-NN visualization
#
# The idea is to find the **k** closest training samples to a given test
# instance, and predict the class that has the highest representation in
# this group.
#
# For example, if k=3, and the k closest training samples are class {2, 1,
# 2}, the prediction for the test instance is 2. In the event of a voting
# draw, we reduce k and vote again.
#


######################################################################
# On to the Code!
# ---------------
#
# Enough talking, let's get to the code!
#
# We'll start by importing some necessities.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import csv
import random
import operator
import collections
import os


######################################################################
# Handle data
# ~~~~~~~~~~~
#
# The next step is to download the Iris dataset into ``data/iris.data``.
# To get an idea of what the file looks like, we'll print a few lines.
#
# .. Note ::
#    Download the data from
#    `here <https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data>`_
#    and extract it in a ``data`` directory under the current directory.
#

filename = "data/iris.data"

# Print 10 random lines
try:
    datafile = open(filename, 'r')
except IOError:
    print("Cannot open data file: {}. Have you downloaded it yet?".format(filename))
    exit()
lines = datafile.readlines()
# Last line in file is empty, we'll deal with this later
lines = lines[:-1]
random.shuffle(lines)
for line in lines[:10]:
    print(line.strip())
datafile.close()

######################################################################
# Next, we'll create our dataset, which is a list of lists containing each
# row of the data file (shape=(150,5)).
#
# In the next step, we will convert this to a torch tensor, so we have to
# convert the string flower species names to float type so that the types
# of all elements are the same. To do this, we define a simple mapping in
# the ``class_labels`` dictionary.
#

# Declare mapping between string class labels and a numeric representation
class_labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

# Takes a filename with comma separated contents, returns dataset list
def create_dataset(filepath):
    with open(filepath, 'r') as datafile:
        # Create dataset list
        lines = csv.reader(datafile)
        dataset = list(lines)
        # Remove empty lines
        dataset = [x for x in dataset if x]
        # Convert string label to numeric label
        for row in dataset:
            for i in range(4):
                row[i] = float(row[i])
            row[4] = float(class_labels[row[4]])
        return dataset


# Load dataset
dataset = create_dataset("data/iris.data")


######################################################################
# Now we will declare our ``train_ratio`` and split the dataset. Notice
# that we cast the ``train_set`` and ``test_set`` splits to *torch.tensor*
# before returning.
#

# Ratio of samples used for training
train_ratio = .70

# Takes dataset list and a train percent value for splitting data
def split_dataset(dataset, train_ratio):
    # Shuffle data
    random.shuffle(dataset)
    # Calculate number of files in train and test set
    train_len = int(len(dataset) * train_ratio)
    test_len = int(len(dataset) - train_len)
    # Split train and test sets
    train_set = dataset[:train_len]
    test_set = dataset[-test_len:]
    # Convert splits to torch tensor and return
    return torch.tensor(train_set), torch.tensor(test_set)


# Split dataset
train_set, test_set = split_dataset(dataset, train_ratio)


######################################################################
# Implement k-NN functions
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The next step is to implement the helper functions that we will use to
# carry out the k-NN algorithm. Because we use torch tensors to operate
# on, we can use the PyTorch JIT compiler to compile our functions to a
# graph representation.
#
# getDistance
# ^^^^^^^^^^^
#
# The first function that we'll implement is ``getDistance``, which is how
# we will calculate the distance between a ``test_instance`` and a
# ``train_instance``. Because our features are all real numbers, we will
# use Euclidean distance:
#
# .. math:: distance = \sqrt{(train[0] - test[0])^2 + (train[1] - test[1])^2 + \ldots +  (train[n] - test[n])^2}
#
# --------------
#
# **Trace mode**
#
# Because this function involves sequential torch tensor operations, with
# no data-dependent control flows, we can use the **trace** function to
# compile it to a graph representation.
#
# Notice that to trace a function, you must first declare the traced
# version of the function (``getDistance_traced``) using the
# ``torch.jit.trace`` function. Notice that the ``trace`` function is a
# `Curried function <https://en.wikipedia.org/wiki/Currying>`__. For us
# this means that calling it requires two arguments, each surrounded by
# its own set of parenthesis. In the example below, we show that the
# outputs of the non-compiled and compiled functions are identical, and
# use the ``.graph`` attribute to show the graph in a human readable
# format.
#

# Takes a train_instance and test_instance, and returns a scalar torch
#  tensor containing the distance value
def getDistance(train_instance, test_instance):
    # Element wise subtraction; disregard last element (class)
    #diff = torch.sub(train_instance[:4], test_instance[:4])
    diff = train_instance[:4] - test_instance[:4]
    # Square each element of the distance tensor
    #sqrs = torch.pow(diff, torch.tensor(2.0))
    sqrs = diff ** torch.tensor(2.0)
    # Sum all elements in sqrs tensor
    sumsqrs = torch.sum(sqrs)
    # Take square root of sumsqrs to obtain distance
    distance = torch.sqrt(sumsqrs)
    return distance


########## Example ##########
# Non-compiled
print("Euclidean distance between train_set[0] and train_set[1]:")
print("train_set[0]:", train_set[0])
print("train_set[1]:", train_set[1])
distance = getDistance(train_set[0], train_set[1])
print("Non-compiled output:", distance)

# Compiled
getDistance_traced = trace(train_set[0], train_set[1])(getDistance)
distance_traced = getDistance_traced(train_set[0], train_set[1])
print("Compiled output:", distance_traced)
print("getDistance_traced.graph:", getDistance_traced.graph)


######################################################################
# getSortedLabels
# ^^^^^^^^^^^^^^^
#
# The ``getSortedLabels`` function takes two parallel tensors
# (``distances`` & ``labels``) of length *train\_instances* which contain
# the distances between train instances and a given test instance, and the
# corresponding class labels. The ``distances`` are sorted first, and we
# reorder the ``labels`` tensor according to the index changes.
#
# --------------
#
# **Trace mode**
#
# Once again, since this function is simply made up of sequential torch
# operations, we can use **trace** to compile it. Notice that we must use
# a scalar tensor for ``k``.
#

# Takes parallel distances and labels tensors and returns the sorted labels
def getSortedLabels(distances, labels, k):
    # Sort distances
    sorted_dists, indices = torch.sort(distances)
    # Reorder labels according to sorted indices
    sorted_labels = torch.gather(labels, 0, indices)
    # Return top-k sorted_labels
    return sorted_labels[:k]


########## Example ##########
# Non-compiled
print("Sorted labels example:")
ds = torch.tensor([6, 8, 3, 5, 9])
ls = torch.tensor([0, 0, 2, 1, 2])
k = torch.tensor(3)
sorted_labels = getSortedLabels(ds, ls, k)
print("Non-compiled output:", sorted_labels)

# Compiled
getSortedLabels_traced = trace(ds, ls, k)(getSortedLabels)
sorted_labels_traced = getSortedLabels_traced(ds, ls, k)
print("Compiled output:", sorted_labels_traced)
print("getSortedLabels_traced.graph:", getSortedLabels_traced.graph)


######################################################################
# getNeighbors
# ^^^^^^^^^^^^
#
# The ``getNeighbors`` function takes a ``train_set``, a
# ``test_instance``, an empty ``distances`` tensor, an empty ``labels``
# tensor, and a ``k`` value. The ``distances`` and ``labels`` tensors must
# have same length as ``train_set``. It returns a list of the sorted
# *k-nearest* neighbors (as class labels) to the ``test_instance``. It
# does this by iterating over every instance in the ``train_set``,
# recording the distance between the train instance and the
# ``test_instance``, and using our ``getSortedLabels`` function to sort
# the corresbonding labels based on these distances.
#
# --------------
#
# **Script mode**
#
# Notice that this function contains a data-dependent control flow (our
# *for* loop depends on the data size). Because of this, we must compile
# with **script** mode. We indicate the use of this mode by adding a
# ``@script`` decorator. Note that we can call our traced
# ``getDistance_traced``, and ``getSortedLabels_traced`` functions from
# our script function.
#
# Finally, you may be wondering why we use the ``addEntries`` function to
# add to our ``distances`` and ``labels`` tensors instead of simply:
#
# ::
#
#     distances[i] = dist
#     labels[i] = label
#
#
# Remember that only a subset of Python is supported in **script** mode.
# One requirement is that the only expressions allowed on the left side of
# an assignment operator (=) are variable names and starred expressions.
# So, one way to circumvent this is to implement pure Python helper
# functions such as ``addEntries``.
#
# Also, notice in the graph output that the loop is captured
# (``prim::Loop``). The **script** mode enables us to convert the Python
# AST directly to graph mode, rather than tracing an example input through
# the control flow and simply capturing the unrolled loop.
#

# Add elements to distances and labels tensors at index i
def addEntries(i, d, l, distances, labels):
    distances[i] = torch.tensor(d)
    labels[i] = torch.tensor(l)
    return distances, labels

# Returns k-nearest neighbors of a train_set to a test_instance
@script
def getNeighbors(train_set, test_instance, distances, labels, k):
    # Data-dependent control flow
    for i in range(train_set.size(0)):
        train_instance = train_set[i]
        # Calculate Euclidean distance using traced function
        dist = getDistance_traced(train_instance, test_instance)
        # Extract label
        label = train_instance[-1]
        # Call Python function to add entries at index i
        distances, labels = addEntries(i, dist, label, distances, labels)
    # Call traced function to return a labels tensor sorted by distance to the test_instance
    sorted_labels = getSortedLabels_traced(distances, labels, k)
    # Take top k
    neighbors = sorted_labels[:k]
    return neighbors


########## Example ##########
print("getNeighbors example:")
small_train_set = train_set[:10]
k = torch.tensor(3)
distances = torch.empty([small_train_set.size(0)])
labels = torch.empty([small_train_set.size(0)])
neighbors = getNeighbors(small_train_set, train_set[10], distances, labels, k)
print("output:", neighbors)
print("getNeighbors.graph:", getNeighbors.graph)


######################################################################
# getPrediction
# ^^^^^^^^^^^^^
#
# Our final function takes the top k neighbors tensor and returns a final
# prediction. The class that has the most representation in the neighbors
# tensor is the one chosen for the final classification. In the case of a
# draw, the element with the lowest index (the closest neighbor) is
# chosen.
#

# Pure python
def getPrediction(neighbors):
    neighbors = neighbors.numpy()
    return float(max(neighbors, key=collections.Counter(neighbors).get))


########## Example ##########
print("getPrediction example:")
n1 = torch.tensor([2.0, 1.0, 2.0])
n2 = torch.tensor([1.0, 2.0, 0.0])
n3 = torch.tensor([0.0, 1.0, 1.0])
print("n1:", n1, "\tpred:", getPrediction(n1))
print("n2:", n2, "\tpred:", getPrediction(n2))
print("n3:", n3, "\tpred:", getPrediction(n3))


######################################################################
# Run Testing
# ~~~~~~~~~~~
#
# Now it is time to use our functions to test the algorithm on our test
# split!
#
# With the default values of ``k`` = 3 and ``train_ratio`` = .70, you can
# expect an accuracy in the mid to high 90s. Not bad for an algorithm this
# simple!
#
# Feel free to tinker with these parameters and see how the results
# change.
#

# Initialize k
k = torch.tensor(3)

# Initialize counts for accuracy
total_count = 0
correct_count = 0

# Iterate over every instance in the test_set
for test_instance in test_set:
    total_count += 1
    # Record ground truth label
    label = float(test_instance[-1])

    # Initialize distances and labels parallel tensors
    distances = torch.empty([train_set.size(0)])
    labels = torch.empty([train_set.size(0)])
    # Get neighbors
    neighbors = getNeighbors(train_set, test_instance, distances, labels, k)
    # Obtain final prediction
    pred = getPrediction(neighbors)
    # Check if the prediction matches the ground truth label
    if label == pred:
        correct_count += 1

print("Accuracy: {}/{} = {:.2f}%".format(correct_count, total_count, (correct_count/total_count)*100.0))
