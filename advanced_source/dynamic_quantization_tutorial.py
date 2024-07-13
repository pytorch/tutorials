"""
(beta) Dynamic Quantization on an LSTM Word Language Model
==================================================================

**Author**: `James Reed <https://github.com/jamesr66a>`_

**Edited by**: `Seth Weidman <https://github.com/SethHWeidman/>`_

Introduction
------------

Quantization involves converting the weights and activations of your model from float
to int, which can result in smaller model size and faster inference with only a small
hit to accuracy.

In this tutorial, we will apply the easiest form of quantization -
`dynamic quantization <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`_ -
to an LSTM-based next word-prediction model, closely following the
`word language model <https://github.com/pytorch/examples/tree/master/word_language_model>`_
from the PyTorch examples.
"""

# imports
import os
from io import open
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################
# 1. Define the model
# -------------------
#
# Here we define the LSTM model architecture, following the
# `model <https://github.com/pytorch/examples/blob/master/word_language_model/model.py>`_
# from the word language model example.

class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

######################################################################
# 2. Load in the text data
# ------------------------
#
# Next, we load the
# `Wikitext-2 dataset <https://www.google.com/search?q=wikitext+2+data>`_ into a `Corpus`,
# again following the
# `preprocessing <https://github.com/pytorch/examples/blob/master/word_language_model/data.py>`_
# from the word language model example.

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

model_data_filepath = 'data/'

corpus = Corpus(model_data_filepath + 'wikitext-2')

######################################################################
# 3. Load the pretrained model
# -----------------------------
#
# This is a tutorial on dynamic quantization, a quantization technique
# that is applied after a model has been trained. Therefore, we'll simply load some
# pretrained weights into this model architecture; these weights were obtained
# by training for five epochs using the default settings in the word language model
# example.

ntokens = len(corpus.dictionary)

model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

model.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu')
        )
    )

model.eval()
print(model)

######################################################################
# Now let's generate some text to ensure that the pretrained model is working
# properly - similarly to before, we follow
# `here <https://github.com/pytorch/examples/blob/master/word_language_model/generate.py>`_

input_ = torch.randint(ntokens, (1, 1), dtype=torch.long)
hidden = model.init_hidden(1)
temperature = 1.0
num_words = 1000

with open(model_data_filepath + 'out.txt', 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(num_words):
            output, hidden = model(input_, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(str(word.encode('utf-8')) + ('\n' if i % 20 == 19 else ' '))

            if i % 100 == 0:
                print('| Generated {}/{} words'.format(i, 1000))

with open(model_data_filepath + 'out.txt', 'r') as outf:
    all_output = outf.read()
    print(all_output)

######################################################################
# It's no GPT-2, but it looks like the model has started to learn the structure of
# language!
#
# We're almost ready to demonstrate dynamic quantization. We just need to define a few more
# helper functions:

bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

# create test data set
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into ``bsz`` parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the ``bsz`` batches.
    return data.view(bsz, -1).t().contiguous()

test_data = batchify(corpus.test, eval_batch_size)

# Evaluation functions
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history."""

  if isinstance(h, torch.Tensor):
      return h.detach()
  else:
      return tuple(repackage_hidden(v) for v in h)

def evaluate(model_, data_source):
    # Turn on evaluation mode which disables dropout.
    model_.eval()
    total_loss = 0.
    hidden = model_.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model_(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

######################################################################
# 4. Test dynamic quantization
# ----------------------------
#
# Finally, we can call ``torch.quantization.quantize_dynamic`` on the model!
# Specifically,
#
# - We specify that we want the ``nn.LSTM`` and ``nn.Linear`` modules in our
#   model to be quantized
# - We specify that we want weights to be converted to ``int8`` values

import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)

######################################################################
# The model looks the same; how has this benefited us? First, we see a
# significant reduction in model size:

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)

######################################################################
# Second, we see faster inference time, with no difference in evaluation loss:
#
# Note: we set the number of threads to one for single threaded comparison, since quantized
# models run single threaded.

torch.set_num_threads(1)

def time_model_evaluation(model, test_data):
    s = time.time()
    loss = evaluate(model, test_data)
    elapsed = time.time() - s
    print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

time_model_evaluation(model, test_data)
time_model_evaluation(quantized_model, test_data)

######################################################################
# Running this locally on a MacBook Pro, without quantization, inference takes about 200 seconds,
# and with quantization it takes just about 100 seconds.
#
# Conclusion
# ----------
#
# Dynamic quantization can be an easy way to reduce model size while only
# having a limited effect on accuracy.
#
# Thanks for reading! As always, we welcome any feedback, so please create an issue
# `here <https://github.com/pytorch/pytorch/issues>`_ if you have any.
