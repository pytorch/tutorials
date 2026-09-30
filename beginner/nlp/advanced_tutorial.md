Note

Go to the end
to download the full example code.

# Advanced: Making Dynamic Decisions and the Bi-LSTM CRF

## Dynamic versus Static Deep Learning Toolkits

Pytorch is a *dynamic* neural network kit. Another example of a dynamic
kit is [Dynet](https://github.com/clab/dynet) (I mention this because
working with Pytorch and Dynet is similar. If you see an example in
Dynet, it will probably help you implement it in Pytorch). The opposite
is the *static* tool kit, which includes Theano, Keras, TensorFlow, etc.
The core difference is the following:

- In a static toolkit, you define
a computation graph once, compile it, and then stream instances to it.
- In a dynamic toolkit, you define a computation graph *for each
instance*. It is never compiled and is executed on-the-fly

Without a lot of experience, it is difficult to appreciate the
difference. One example is to suppose we want to build a deep
constituent parser. Suppose our model involves roughly the following
steps:

- We build the tree bottom up
- Tag the root nodes (the words of the sentence)
- From there, use a neural network and the embeddings
of the words to find combinations that form constituents. Whenever you
form a new constituent, use some sort of technique to get an embedding
of the constituent. In this case, our network architecture will depend
completely on the input sentence. In the sentence "The green cat
scratched the wall", at some point in the model, we will want to combine
the span \((i,j,r) = (1, 3, \text{NP})\) (that is, an NP constituent
spans word 1 to word 3, in this case "The green cat").

However, another sentence might be "Somewhere, the big fat cat scratched
the wall". In this sentence, we will want to form the constituent
\((2, 4, NP)\) at some point. The constituents we will want to form
will depend on the instance. If we just compile the computation graph
once, as in a static toolkit, it will be exceptionally difficult or
impossible to program this logic. In a dynamic toolkit though, there
isn't just 1 pre-defined computation graph. There can be a new
computation graph for each instance, so this problem goes away.

Dynamic toolkits also have the advantage of being easier to debug and
the code more closely resembling the host language (by that I mean that
Pytorch and Dynet look more like actual Python code than Keras or
Theano).

## Bi-LSTM Conditional Random Field Discussion

For this section, we will see a full, complicated example of a Bi-LSTM
Conditional Random Field for named-entity recognition. The LSTM tagger
above is typically sufficient for part-of-speech tagging, but a sequence
model like the CRF is really essential for strong performance on NER.
Familiarity with CRF's is assumed. Although this name sounds scary, all
the model is a CRF but where an LSTM provides the features. This is
an advanced model though, far more complicated than any earlier model in
this tutorial. If you want to skip it, that is fine. To see if you're
ready, see if you can:

- Write the recurrence for the viterbi variable at step i for tag k.
- Modify the above recurrence to compute the forward variables instead.
- Modify again the above recurrence to compute the forward variables in
log-space (hint: log-sum-exp)

If you can do those three things, you should be able to understand the
code below. Recall that the CRF computes a conditional probability. Let
\(y\) be a tag sequence and \(x\) an input sequence of words.
Then we compute

\[P(y|x) = \frac{\exp{(\text{Score}(x, y)})}{\sum_{y'} \exp{(\text{Score}(x, y')})}

\]

Where the score is determined by defining some log potentials
\(\log \psi_i(x,y)\) such that

\[\text{Score}(x,y) = \sum_i \log \psi_i(x,y)

\]

To make the partition function tractable, the potentials must look only
at local features.

In the Bi-LSTM CRF, we define two kinds of potentials: emission and
transition. The emission potential for the word at index \(i\) comes
from the hidden state of the Bi-LSTM at timestep \(i\). The
transition scores are stored in a \(|T|x|T|\) matrix
\(\textbf{P}\), where \(T\) is the tag set. In my
implementation, \(\textbf{P}_{j,k}\) is the score of transitioning
to tag \(j\) from tag \(k\). So:

\[\text{Score}(x,y) = \sum_i \log \psi_\text{EMIT}(y_i \rightarrow x_i) + \log \psi_\text{TRANS}(y_{i-1} \rightarrow y_i)

\]

\[= \sum_i h_i[y_i] + \textbf{P}_{y_i, y_{i-1}}

\]

where in this second expression, we think of the tags as being assigned
unique non-negative indices.

If the above discussion was too brief, you can check out
[this](http://www.cs.columbia.edu/%7Emcollins/crf.pdf) write up from
Michael Collins on CRFs.

## Implementation Notes

The example below implements the forward algorithm in log space to
compute the partition function, and the viterbi algorithm to decode.
Backpropagation will compute the gradients automatically for us. We
don't have to do anything by hand.

The implementation is not optimized. If you understand what is going on,
you'll probably quickly see that iterating over the next tag in the
forward algorithm could probably be done in one big operation. I wanted
to code to be more readable. If you want to make the relevant change,
you could probably use this tagger for real tasks.

```
# Author: Robert Guthrie
```

Helper functions to make the code more readable.

```
# Compute log sum exp in a numerically stable way for the forward algorithm
```

Create model

Run training

```
# Make up some training data

# Check predictions before training

# Make sure prepare_sequence from earlier in the LSTM section is loaded

# Check predictions after training

# We got it!
```

## Exercise: A new loss function for discriminative tagging

It wasn't really necessary for us to create a computation graph when
doing decoding, since we do not backpropagate from the viterbi path
score. Since we have it anyway, try training the tagger where the loss
function is the difference between the Viterbi path score and the score
of the gold-standard path. It should be clear that this function is
non-negative and 0 when the predicted tag sequence is the correct tag
sequence. This is essentially *structured perceptron*.

This modification should be short, since Viterbi and score_sentence are
already implemented. This is an example of the shape of the computation
graph *depending on the training instance*. Although I haven't tried
implementing this in a static toolkit, I imagine that it is possible but
much less straightforward.

Pick up some real data and do a comparison!

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: advanced_tutorial.ipynb`](../../_downloads/99fdc80b1d3c74fcdb1432b4e7df20f7/advanced_tutorial.ipynb)

[`Download Python source code: advanced_tutorial.py`](../../_downloads/2a7469f7250c4f405b1c4f885994ea69/advanced_tutorial.py)

[`Download zipped: advanced_tutorial.zip`](../../_downloads/9f13e061d5639e943dba7c700afae213/advanced_tutorial.zip)