Note

Go to the end
to download the full example code.

# NLP From Scratch: Generating Names with a Character-Level RNN

**Author**: [Sean Robertson](https://github.com/spro)

This tutorials is part of a three-part series:

- [NLP From Scratch: Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [NLP From Scratch: Generating Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

This is our second of three tutorials on "NLP From Scratch".
In the [first tutorial](/tutorials/intermediate/char_rnn_classification_tutorial)
we used a RNN to classify names into their language of origin. This time
we'll turn around and generate names from languages.

```
> python sample.py Russian RUS
Rovakov
Uantov
Shavakov

> python sample.py German GER
Gerren
Ereng
Rosher

> python sample.py Spanish SPA
Salla
Parer
Allan

> python sample.py Chinese CHI
Chan
Hang
Iun
```

We are still hand-crafting a small RNN with a few linear layers. The big
difference is instead of predicting a category after reading in all the
letters of a name, we input a category and output one letter at a time.
Recurrently predicting characters to form language (this could also be
done with words or other higher order constructs) is often referred to
as a "language model".

**Recommended Reading:**

I assume you have at least installed PyTorch, know Python, and
understand Tensors:

- [https://pytorch.org/](https://pytorch.org/) For installation instructions
- [Deep Learning with PyTorch: A 60 Minute Blitz](../beginner/deep_learning_60min_blitz.html) to get started with PyTorch in general
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

I also suggest the previous tutorial, [NLP From Scratch: Classifying Names with a Character-Level RNN](char_rnn_classification_tutorial.html)

## Preparing the Data

Note

Download the data from
[here](https://download.pytorch.org/tutorial/data.zip)
and extract it to the current directory.

See the last tutorial for more detail of this process. In short, there
are a bunch of plain text files `data/names/[Language].txt` with a
name per line. We split lines into an array, convert Unicode to ASCII,
and end up with a dictionary `{language: [names ...]}`.

```
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427

# Read a file and split into lines

# Build the category_lines dictionary, a list of lines per category
```

## Creating the Network

This network extends the last tutorial's RNN
with an extra argument for the category tensor, which is concatenated
along with the others. The category tensor is a one-hot vector just like
the letter input.

We will interpret the output as the probability of the next letter. When
sampling, the most likely output letter is used as the next input
letter.

I added a second linear layer `o2o` (after combining hidden and
output) to give it more muscle to work with. There's also a dropout
layer, which [randomly zeros parts of its
input](https://arxiv.org/abs/1207.0580) with a given probability
(here 0.1) and is usually used to fuzz inputs to prevent overfitting.
Here we're using it towards the end of the network to purposely add some
chaos and increase sampling variety.

![](https://i.imgur.com/jzVrf7f.png)

## Training

### Preparing for Training

First of all, helper functions to get random pairs of (category, line):

```
# Random item from a list

# Get a random category and random line from that category
```

For each timestep (that is, for each letter in a training word) the
inputs of the network will be
`(category, current letter, hidden state)` and the outputs will be
`(next letter, next hidden state)`. So for each training set, we'll
need the category, a set of input letters, and a set of output/target
letters.

Since we are predicting the next letter from the current letter for each
timestep, the letter pairs are groups of consecutive letters from the
line - e.g. for `"ABCD<EOS>"` we would create ("A", "B"), ("B", "C"),
("C", "D"), ("D", "EOS").

![](https://i.imgur.com/JH58tXY.png)

The category tensor is a [one-hot
tensor](https://en.wikipedia.org/wiki/One-hot) of size
`<1 x n_categories>`. When training we feed it to the network at every
timestep - this is a design choice, it could have been included as part
of initial hidden state or some other strategy.

```
# One-hot vector for category

# One-hot matrix of first to last letters (not including EOS) for input

# ``LongTensor`` of second letter to end (EOS) for target
```

For convenience during training we'll make a `randomTrainingExample`
function that fetches a random (category, line) pair and turns them into
the required (category, input, target) tensors.

```
# Make category, input, and target tensors from a random category, line pair
```

### Training the Network

In contrast to classification, where only the last output is used, we
are making a prediction at every step, so we are calculating loss at
every step.

The magic of autograd allows you to simply sum these losses at each step
and call backward at the end.

To keep track of how long training takes I am adding a
`timeSince(timestamp)` function which returns a human readable string:

Training is business as usual - call train a bunch of times and wait a
few minutes, printing the current time and loss every `print_every`
examples, and keeping store of an average loss per `plot_every` examples
in `all_losses` for plotting later.

### Plotting the Losses

Plotting the historical loss from all_losses shows the network
learning:

## Sampling the Network

To sample we give the network a letter and ask what the next one is,
feed that in as the next letter, and repeat until the EOS token.

- Create tensors for input category, starting letter, and empty hidden
state
- Create a string `output_name` with the starting letter
- Up to a maximum output length,

- Feed the current letter to the network
- Get the next letter from highest output, and next hidden state
- If the letter is EOS, stop here
- If a regular letter, add to `output_name` and continue
- Return the final name

Note

Rather than having to give it a starting letter, another
strategy would have been to include a "start of string" token in
training and have the network choose its own starting letter.

```
# Sample from a category and starting letter

# Get multiple samples from one category and multiple starting letters
```

## Exercises

- Try with a different dataset of category -> line, for example:

- Fictional series -> Character name
- Part of speech -> Word
- Country -> City
- Use a "start of sentence" token so that sampling can be done without
choosing a start letter
- Get better results with a bigger and/or better shaped network

- Try the `nn.LSTM` and `nn.GRU` layers
- Combine multiple of these RNNs as a higher level network

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: char_rnn_generation_tutorial.ipynb`](../_downloads/a75cfadf4fa84dd594874d4c53b62820/char_rnn_generation_tutorial.ipynb)

[`Download Python source code: char_rnn_generation_tutorial.py`](../_downloads/322506af160d5e2056afd75de1fd34ee/char_rnn_generation_tutorial.py)

[`Download zipped: char_rnn_generation_tutorial.zip`](../_downloads/3af0543b3600b020bfdbf10ab130c2f8/char_rnn_generation_tutorial.zip)