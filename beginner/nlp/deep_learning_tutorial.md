Note

Go to the end
to download the full example code.

# Deep Learning with PyTorch

## Deep Learning Building Blocks: Affine maps, non-linearities and objectives

Deep learning consists of composing linearities with non-linearities in
clever ways. The introduction of non-linearities allows for powerful
models. In this section, we will play with these core components, make
up an objective function, and see how the model is trained.

### Affine Maps

One of the core workhorses of deep learning is the affine map, which is
a function \(f(x)\) where

\[f(x) = Ax + b

\]

for a matrix \(A\) and vectors \(x, b\). The parameters to be
learned here are \(A\) and \(b\). Often, \(b\) is refered to
as the *bias* term.

PyTorch and most other deep learning frameworks do things a little
differently than traditional linear algebra. It maps the rows of the
input instead of the columns. That is, the \(i\)'th row of the
output below is the mapping of the \(i\)'th row of the input under
\(A\), plus the bias term. Look at the example below.

```
# Author: Robert Guthrie
```

```
# data is 2x5. A maps from 5 to 3... can we map "data" under A?
```

### Non-Linearities

First, note the following fact, which will explain why we need
non-linearities in the first place. Suppose we have two affine maps
\(f(x) = Ax + b\) and \(g(x) = Cx + d\). What is
\(f(g(x))\)?

\[f(g(x)) = A(Cx + d) + b = ACx + (Ad + b)

\]

\(AC\) is a matrix and \(Ad + b\) is a vector, so we see that
composing affine maps gives you an affine map.

From this, you can see that if you wanted your neural network to be long
chains of affine compositions, that this adds no new power to your model
than just doing a single affine map.

If we introduce non-linearities in between the affine layers, this is no
longer the case, and we can build much more powerful models.

There are a few core non-linearities.
\(\tanh(x), \sigma(x), \text{ReLU}(x)\) are the most common. You are
probably wondering: "why these functions? I can think of plenty of other
non-linearities." The reason for this is that they have gradients that
are easy to compute, and computing gradients is essential for learning.
For example

\[\frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))

\]

A quick note: although you may have learned some neural networks in your
intro to AI class where \(\sigma(x)\) was the default non-linearity,
typically people shy away from it in practice. This is because the
gradient *vanishes* very quickly as the absolute value of the argument
grows. Small gradients means it is hard to learn. Most people default to
tanh or ReLU.

```
# In pytorch, most non-linearities are in torch.functional (we have it imported as F)
# Note that non-linearites typically don't have parameters like affine maps do.
# That is, they don't have weights that are updated during training.
```

### Softmax and Probabilities

The function \(\text{Softmax}(x)\) is also just a non-linearity, but
it is special in that it usually is the last operation done in a
network. This is because it takes in a vector of real numbers and
returns a probability distribution. Its definition is as follows. Let
\(x\) be a vector of real numbers (positive, negative, whatever,
there are no constraints). Then the i'th component of
\(\text{Softmax}(x)\) is

\[\frac{\exp(x_i)}{\sum_j \exp(x_j)}

\]

It should be clear that the output is a probability distribution: each
element is non-negative and the sum over all components is 1.

You could also think of it as just applying an element-wise
exponentiation operator to the input to make everything non-negative and
then dividing by the normalization constant.

```
# Softmax is also in torch.nn.functional
```

### Objective Functions

The objective function is the function that your network is being
trained to minimize (in which case it is often called a *loss function*
or *cost function*). This proceeds by first choosing a training
instance, running it through your neural network, and then computing the
loss of the output. The parameters of the model are then updated by
taking the derivative of the loss function. Intuitively, if your model
is completely confident in its answer, and its answer is wrong, your
loss will be high. If it is very confident in its answer, and its answer
is correct, the loss will be low.

The idea behind minimizing the loss function on your training examples
is that your network will hopefully generalize well and have small loss
on unseen examples in your dev set, test set, or in production. An
example loss function is the *negative log likelihood loss*, which is a
very common objective for multi-class classification. For supervised
multi-class classification, this means training the network to minimize
the negative log probability of the correct output (or equivalently,
maximize the log probability of the correct output).

## Optimization and Training

So what we can compute a loss function for an instance? What do we do
with that? We saw earlier that Tensors know how to compute gradients
with respect to the things that were used to compute it. Well,
since our loss is an Tensor, we can compute gradients with
respect to all of the parameters used to compute it! Then we can perform
standard gradient updates. Let \(\theta\) be our parameters,
\(L(\theta)\) the loss function, and \(\eta\) a positive
learning rate. Then:

\[\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta L(\theta)

\]

There are a huge collection of algorithms and active research in
attempting to do something more than just this vanilla gradient update.
Many attempt to vary the learning rate based on what is happening at
train time. You don't need to worry about what specifically these
algorithms are doing unless you are really interested. Torch provides
many in the torch.optim package, and they are all completely
transparent. Using the simplest gradient update is the same as the more
complicated algorithms. Trying different update algorithms and different
parameters for the update algorithms (like different initial learning
rates) is important in optimizing your network's performance. Often,
just replacing vanilla SGD with an optimizer like Adam or RMSProp will
boost performance noticably.

## Creating Network Components in PyTorch

Before we move on to our focus on NLP, lets do an annotated example of
building a network in PyTorch using only affine maps and
non-linearities. We will also see how to compute a loss function, using
PyTorch's built in negative log likelihood, and update parameters by
backpropagation.

All network components should inherit from nn.Module and override the
forward() method. That is about it, as far as the boilerplate is
concerned. Inheriting from nn.Module provides functionality to your
component. For example, it makes it keep track of its trainable
parameters, you can swap it between CPU and GPU with the `.to(device)`
method, where device can be a CPU device `torch.device("cpu")` or CUDA
device `torch.device("cuda:0")`.

Let's write an annotated example of a network that takes in a sparse
bag-of-words representation and outputs a probability distribution over
two labels: "English" and "Spanish". This model is just logistic
regression.

### Example: Logistic Regression Bag-of-Words classifier

Our model will map a sparse BoW representation to log probabilities over
labels. We assign each word in the vocab an index. For example, say our
entire vocab is two words "hello" and "world", with indices 0 and 1
respectively. The BoW vector for the sentence "hello hello hello hello"
is

\[\left[ 4, 0 \right]

\]

For "hello world world hello", it is

\[\left[ 2, 2 \right]

\]

etc. In general, it is

\[\left[ \text{Count}(\text{hello}), \text{Count}(\text{world}) \right]

\]

Denote this BOW vector as \(x\). The output of our network is:

\[\log \text{Softmax}(Ax + b)

\]

That is, we pass the input through an affine map and then do log
softmax.

```
# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector

# the model knows its parameters. The first output below is A, the second is b.
# Whenever you assign a component to a class variable in the __init__ function
# of a module, which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the PyTorch devs, your module
# (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters

# To run the model, pass in a BoW vector
# Here we don't need to train, so the code is wrapped in torch.no_grad()
```

Which of the above values corresponds to the log probability of ENGLISH,
and which to SPANISH? We never defined it, but we need to if we want to
train the thing.

So lets train! To do this, we pass instances through to get log
probabilities, compute a loss function, compute the gradient of the loss
function, and then update the parameters with a gradient step. Loss
functions are provided by Torch in the nn package. nn.NLLLoss() is the
negative log likelihood loss we want. It also defines optimization
functions in torch.optim. Here, we will just use SGD.

Note that the *input* to NLLLoss is a vector of log probabilities, and a
target label. It doesn't compute the log probabilities for us. This is
why the last layer of our network is log softmax. The loss function
nn.CrossEntropyLoss() is the same as NLLLoss(), except it does the log
softmax for you.

```
# Run on test data before we train, just to see a before-and-after

# Print the matrix column corresponding to "creo"

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances. Usually, somewhere between 5 and 30 epochs is reasonable.

# Index corresponding to Spanish goes up, English goes down!
```

We got the right answer! You can see that the log probability for
Spanish is much higher in the first example, and the log probability for
English is much higher in the second for the test data, as it should be.

Now you see how to make a PyTorch component, pass some data through it
and do gradient updates. We are ready to dig deeper into what deep NLP
has to offer.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: deep_learning_tutorial.ipynb`](../../_downloads/dd1c511de656ab48216de2866264b28f/deep_learning_tutorial.ipynb)

[`Download Python source code: deep_learning_tutorial.py`](../../_downloads/f1703ff94c4a6544f91c89b37c3a1fcf/deep_learning_tutorial.py)

[`Download zipped: deep_learning_tutorial.zip`](../../_downloads/388445308536355abc2e460d51dfa282/deep_learning_tutorial.zip)