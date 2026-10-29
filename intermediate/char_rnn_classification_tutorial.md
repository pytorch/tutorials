Note

Go to the end
to download the full example code.

# NLP From Scratch: Classifying Names with a Character-Level RNN

**Author**: [Sean Robertson](https://github.com/spro)

This tutorials is part of a three-part series:

- [NLP From Scratch: Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [NLP From Scratch: Generating Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

We will be building and training a basic character-level Recurrent Neural
Network (RNN) to classify words. This tutorial, along with two other
Natural Language Processing (NLP) "from scratch" tutorials
[NLP From Scratch: Generating Names with a Character-Level RNN](char_rnn_generation_tutorial.html) and
[NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](seq2seq_translation_tutorial.html), show how to
preprocess data to model NLP. In particular, these tutorials show how
preprocessing to model NLP works at a low level.

A character-level RNN reads words as a series of characters -
outputting a prediction and "hidden state" at each step, feeding its
previous hidden state into each next step. We take the final prediction
to be the output, i.e. which class the word belongs to.

Specifically, we'll train on a few thousand surnames from 18 languages
of origin, and predict which language a name is from based on the
spelling.

## Recommended Preparation

Before starting this tutorial it is recommended that you have installed PyTorch,
and have a basic understanding of Python programming language and Tensors:

- [https://pytorch.org/](https://pytorch.org/) For installation instructions
- [Deep Learning with PyTorch: A 60 Minute Blitz](../beginner/deep_learning_60min_blitz.html) to get started with PyTorch in general
and learn the basics of Tensors
- [Learning PyTorch with Examples](../beginner/pytorch_with_examples.html) for a wide and deep overview
- [PyTorch for Former Torch Users](../beginner/former_torchies_tutorial.html) if you are former Lua Torch user

It would also be useful to know about RNNs and how they work:

- [The Unreasonable Effectiveness of Recurrent Neural
Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
shows a bunch of real life examples
- [Understanding LSTM
Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
is about LSTMs specifically but also informative about RNNs in
general

## Preparing Torch

Set up torch to default to the right device use GPU acceleration depending on your hardware (CPU or CUDA).

```
# Check if CUDA is available
```

## Preparing the Data

Download the data from [here](https://download.pytorch.org/tutorial/data.zip)
and extract it to the current directory.

Included in the `data/names` directory are 18 text files named as
`[Language].txt`. Each file contains a bunch of names, one name per
line, mostly romanized (but we still need to convert from Unicode to
ASCII).

The first step is to define and clean our data. Initially, we need to convert Unicode to plain ASCII to
limit the RNN input layers. This is accomplished by converting Unicode strings to ASCII and allowing only a small set of allowed characters.

```
# We can use "_" to represent an out-of-vocabulary character, that is, any character we are not handling in our model

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
```

Here's an example of converting a unicode alphabet name to plain ASCII. This simplifies the input layer

## Turning Names into Tensors

Now that we have all the names organized, we need to turn them into
Tensors to make any use of them.

To represent a single letter, we use a "one-hot vector" of size
`<1 x n_letters>`. A one-hot vector is filled with 0s except for a 1
at index of the current letter, e.g. `"b" = <0 1 0 0 0 ...>`.

To make a word we join a bunch of those into a 2D matrix
`<line_length x 1 x n_letters>`.

That extra 1 dimension is because PyTorch assumes everything is in
batches - we're just using a batch size of 1 here.

```
# Find letter index from all_letters, e.g. "a" = 0

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
```

Here are some examples of how to use `lineToTensor()` for a single and multiple character string.

Congratulations, you have built the foundational tensor objects for this learning task! You can use a similar approach
for other RNN tasks with text.

Next, we need to combine all our examples into a dataset so we can train, test and validate our models. For this,
we will use the [Dataset and DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) classes
to hold our dataset. Each Dataset needs to implement three functions: `__init__`, `__len__`, and `__getitem__`.

Here we can load our example data into the `NamesDataset`

Using the dataset object allows us to easily split the data into train and test sets. Here we create a 85/15

split but the `torch.utils.data` has more useful utilities. Here we specify a generator since we need to use the

same device as PyTorch defaults to above.

Now we have a basic dataset containing **20074** examples where each example is a pairing of label and name. We have also
split the dataset into training and testing so we can validate the model that we build.

## Creating the Network

Before autograd, creating a recurrent neural network in Torch involved
cloning the parameters of a layer over several timesteps. The layers
held hidden state and gradients which are now entirely handled by the
graph itself. This means you can implement a RNN in a very "pure" way,
as regular feed-forward layers.

This CharRNN class implements an RNN with three components.
First, we use the [nn.RNN implementation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html).
Next, we define a layer that maps the RNN hidden layers to our output. And finally, we apply a `softmax` function. Using `nn.RNN`
leads to a significant improvement in performance, such as cuDNN-accelerated kernels, versus implementing
each layer as a `nn.Linear`. It also simplifies the implementation in `forward()`.

We can then create an RNN with 58 input nodes, 128 hidden nodes, and 18 outputs:

After that we can pass our Tensor to the RNN to obtain a predicted output. Subsequently,
we use a helper function, `label_from_output`, to derive a text label for the class.

## Training

### Training the Network

Now all it takes to train this network is show it a bunch of examples,
have it make guesses, and tell it if it's wrong.

We do this by defining a `train()` function which trains the model on a given dataset using minibatches. RNNs
RNNs are trained similarly to other networks; therefore, for completeness, we include a batched training method here.
The loop (`for i in batch`) computes the losses for each of the items in the batch before adjusting the
weights. This operation is repeated until the number of epochs is reached.

We can now train a dataset with minibatches for a specified number of epochs. The number of epochs for this
example is reduced to speed up the build. You can get better results with different parameters.

### Plotting the Results

Plotting the historical loss from `all_losses` shows the network
learning:

## Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every actual language (rows)
which language the network guesses (columns). To calculate the confusion
matrix a bunch of samples are run through the network with
`evaluate()`, which is the same as `train()` minus the backprop.

You can pick out bright spots off the main axis that show which
languages it guesses incorrectly, e.g. Chinese for Korean, and Spanish
for Italian. It seems to do very well with Greek, and very poorly with
English (perhaps because of overlap with other languages).

## Exercises

- Get better results with a bigger and/or better shaped network

- Adjust the hyperparameters to enhance performance, such as changing the number of epochs, batch size, and learning rate
- Try the `nn.LSTM` and `nn.GRU` layers
- Modify the size of the layers, such as increasing or decreasing the number of hidden nodes or adding additional linear layers
- Combine multiple of these RNNs as a higher level network
- Try with a different dataset of line -> label, for example:

- Any word -> language
- First name -> gender
- Character name -> writer
- Page title -> blog or subreddit

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: char_rnn_classification_tutorial.ipynb`](../_downloads/13b143c2380f4768d9432d808ad50799/char_rnn_classification_tutorial.ipynb)

[`Download Python source code: char_rnn_classification_tutorial.py`](../_downloads/37c8905519d3fd3f437b783a48d06eac/char_rnn_classification_tutorial.py)

[`Download zipped: char_rnn_classification_tutorial.zip`](../_downloads/9e1d6a0463b9cd1a32b6e306664a0744/char_rnn_classification_tutorial.zip)