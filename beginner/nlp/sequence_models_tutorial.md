Note

Go to the end
to download the full example code.

# Sequence Models and Long Short-Term Memory Networks

At this point, we have seen various feed-forward networks. That is,
there is no state maintained by the network at all. This might not be
the behavior we want. Sequence models are central to NLP: they are
models where there is some sort of dependence through time between your
inputs. The classical example of a sequence model is the Hidden Markov
Model for part-of-speech tagging. Another example is the conditional
random field.

A recurrent neural network is a network that maintains some kind of
state. For example, its output could be used as part of the next input,
so that information can propagate along as the network passes over the
sequence. In the case of an LSTM, for each element in the sequence,
there is a corresponding *hidden state* \(h_t\), which in principle
can contain information from arbitrary points earlier in the sequence.
We can use the hidden state to predict words in a language model,
part-of-speech tags, and a myriad of other things.

## LSTMs in Pytorch

Before getting to the example, note a few things. Pytorch's LSTM expects
all of its inputs to be 3D tensors. The semantics of the axes of these
tensors is important. The first axis is the sequence itself, the second
indexes instances in the mini-batch, and the third indexes elements of
the input. We haven't discussed mini-batching, so let's just ignore that
and assume we will always have just 1 dimension on the second axis. If
we want to run the sequence model over the sentence "The cow jumped",
our input should look like

\[\begin{bmatrix}
\overbrace{q_\text{The}}^\text{row vector} \\
q_\text{cow} \\
q_\text{jumped}
\end{bmatrix}\]

Except remember there is an additional 2nd dimension with size 1.

In addition, you could go through the sequence one at a time, in which
case the 1st axis will have size 1 also.

Let's see a quick example.

```
# Author: Robert Guthrie
```

```
# initialize the hidden state.

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument to the lstm at a later time
# Add the extra 2nd dimension
```

## Example: An LSTM for Part-of-Speech Tagging

In this section, we will use an LSTM to get part of speech tags. We will
not use Viterbi or Forward-Backward or anything like that, but as a
(challenging) exercise to the reader, think about how Viterbi could be
used after you have seen what is going on. In this example, we also refer
to embeddings. If you are unfamiliar with embeddings, you can read up
about them [here](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html).

The model is as follows: let our input sentence be
\(w_1, \dots, w_M\), where \(w_i \in V\), our vocab. Also, let
\(T\) be our tag set, and \(y_i\) the tag of word \(w_i\).
Denote our prediction of the tag of word \(w_i\) by
\(\hat{y}_i\).

This is a structure prediction, model, where our output is a sequence
\(\hat{y}_1, \dots, \hat{y}_M\), where \(\hat{y}_i \in T\).

To do the prediction, pass an LSTM over the sentence. Denote the hidden
state at timestep \(i\) as \(h_i\). Also, assign each tag a
unique index (like how we had word_to_ix in the word embeddings
section). Then our prediction rule for \(\hat{y}_i\) is

\[\hat{y}_i = \text{argmax}_j \ (\log \text{Softmax}(Ah_i + b))_j

\]

That is, take the log softmax of the affine map of the hidden state,
and the predicted tag is the tag that has the maximum value in this
vector. Note this implies immediately that the dimensionality of the
target space of \(A\) is \(|T|\).

Prepare data:

```
# For each words-list (sentence) and tags-list in each tuple of training_data

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
```

Create the model:

Train the model:

```
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()

# See what the scores are after training
```

## Exercise: Augmenting the LSTM part-of-speech tagger with character-level features

In the example above, each word had an embedding, which served as the
inputs to our sequence model. Let's augment the word embeddings with a
representation derived from the characters of the word. We expect that
this should help significantly, since character-level information like
affixes have a large bearing on part-of-speech. For example, words with
the affix *-ly* are almost always tagged as adverbs in English.

To do this, let \(c_w\) be the character-level representation of
word \(w\). Let \(x_w\) be the word embedding as before. Then
the input to our sequence model is the concatenation of \(x_w\) and
\(c_w\). So if \(x_w\) has dimension 5, and \(c_w\)
dimension 3, then our LSTM should accept an input of dimension 8.

To get the character level representation, do an LSTM over the
characters of a word, and let \(c_w\) be the final hidden state of
this LSTM. Hints:

- There are going to be two LSTM's in your new model.
The original one that outputs POS tag scores, and the new one that
outputs a character-level representation of each word.
- To do a sequence model over characters, you will have to embed characters.
The character embeddings will be the input to the character LSTM.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: sequence_models_tutorial.ipynb`](../../_downloads/5edaebfc06ec3968b8c1da100da2253d/sequence_models_tutorial.ipynb)

[`Download Python source code: sequence_models_tutorial.py`](../../_downloads/00262d1f472765b291ae3a1a5bf86bae/sequence_models_tutorial.py)

[`Download zipped: sequence_models_tutorial.zip`](../../_downloads/1bc7594c183f27776c2eaa0578b0dad4/sequence_models_tutorial.zip)