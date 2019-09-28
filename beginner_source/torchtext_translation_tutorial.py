"""
Language Translation with TorchText
===================================

This tutorial shows how to use several convenience classes of ``torchtext`` to preprocess
data from a well-known dataset containing sentences in both English and German and use it to
train a sequence-to-sequence model with attention that can translate German sentences
into English.

It is based off of
`this tutorial <https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>`__
from PyTorch community member `Ben Trevett <https://github.com/bentrevett>`__
and was created by `Seth Weidman <https://github.com/SethHWeidman/>`__ with Ben's permission.

By the end of this tutorial, you will be able to:

- Preprocess sentences into a commonly-used format for NLP modeling using the following ``torchtext`` convenience classes:
    - `TranslationDataset <https://torchtext.readthedocs.io/en/latest/datasets.html#torchtext.datasets.TranslationDataset>`__
    - `Field <https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field>`__
    - `BucketIterator <https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator>`__
"""

######################################################################
# `Field` and `TranslationDataset`
# ----------------
# ``torchtext`` has utilities for creating datasets that can be easily
# iterated through for the purposes of creating a language translation
# model. One key class is a
# `Field <https://github.com/pytorch/text/blob/master/torchtext/data/field.py#L64>`__,
# which specifies the way each sentence should be preprocessed, and another is the
# `TranslationDataset` ; ``torchtext``
# has several such datasets; in this tutorial we'll use the
# `Multi30k dataset <https://github.com/multi30k/dataset>`__, which contains about
# 30,000 sentences (averaging about 13 words in length) in both English and German.
#
# Note: the tokenization in this tutorial requires `Spacy <https://spacy.io>`__
# We use Spacy because it provides strong support for tokenization in languages
# other than English. ``torchtext`` provides a ``basic_english`` tokenizer
# and supports other tokenizers for English (e.g.
# `Moses <https://bitbucket.org/luismsgomes/mosestokenizer/src/default/>`__)
# but for language translation - where multiple languages are required -
# Spacy is your best bet.
#
# To run this tutorial, first install ``spacy`` using ``pip`` or ``conda``.
# Next, download the raw data for the English and German Spacy tokenizers:
#
# ::
#
#    python -m spacy download en
#    python -m spacy download de
#
# With Spacy installed, the following code will tokenize each of the sentences
# in the ``TranslationDataset`` based on the tokenizer defined in the ``Field``

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

######################################################################
# Now that we've defined ``train_data``, we can see an extremely useful
# feature of ``torchtext``'s ``Field``: the ``build_vocab`` method
# now allows us to create the vocabulary associated with each language

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

######################################################################
# Once these lines of code have been run, ``SRC.vocab.stoi`` will  be a
# dictionary with the tokens in the vocabulary as keys and their
# corresponding indices as values; ``SRC.vocab.itos`` will be the same
# dictionary with the keys and values swapped. We won't make extensive
# use of this fact in this tutorial, but this will likely be useful in
# other NLP tasks you'll encounter.

######################################################################
# ``BucketIterator``
# ----------------
# The last ``torchtext`` specific feature we'll use is the ``BucketIterator``,
# which is easy to use since it takes a ``TranslationDataset`` as its
# first argument. Specifically, as the docs say:
# Defines an iterator that batches examples of similar lengths together.
# Minimizes amount of padding needed while producing freshly shuffled
# batches for each new epoch. See pool for the bucketing procedure used.

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

######################################################################
# These iterators can be called just like ``DataLoader``s; below, in
# the ``train`` and ``evaluate`` functions, they are called simply with:
#
# ::
#
#    for i, batch in enumerate(iterator):
#
# Each ``batch`` then has ``src`` and ``trg`` attributes:
#
# ::
#
#    src = batch.src
#    trg = batch.trg

######################################################################
# Defining our ``nn.Module`` and ``Optimizer``
# ----------------
# That's mostly it from a ``torchtext`` perspecive: with the dataset built
# and the iterator defined, the rest of this tutorial simply defines our
# model as an ``nn.Module``, along with an ``Optimizer``, and then trains it.
#
# Our model specifically, follows the architecture described
# `here <https://arxiv.org/abs/1409.0473>`__ (you can find a
# significantly more commented version
# `here <https://github.com/SethHWeidman/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>`__).
#
# Note: this model is just an example model that can be used for language
# translation; we choose it because it is a standard model for the task,
# not because it is the recommended model to use for translation. As you're
# likely aware, state-of-the-art models are currently based on Transformers;
# you can see PyTorch's capabilities for implementing Transformer layers
# `here <https://pytorch.org/docs/stable/nn.html#transformer-layers>`__; and
# in particular, the "attention" used in the model below is different from
# the multi-headed self-attention present in a transformer model.


import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters())


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

######################################################################
# Note: when scoring the performance of a language translation model in
# particular, we have to tell the ``nn.CrossEntropyLoss`` function to
# ignore the indices where the target is simply padding.

PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

######################################################################
# Finally, we can train and evaluate this model:

import math
import time


def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

######################################################################
# Next steps
# --------------
#
# - Check out the rest of Ben Trevett's tutorials using ``torchtext``
#   `here <https://github.com/bentrevett/>`__
# - Stay tuned for a tutorial using other ``torchtext`` features along
#   with ``nn.Transformer`` for language modeling via next word prediction!
#
