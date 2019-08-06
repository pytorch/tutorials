.. image:: https://travis-ci.org/pytorch/text.svg?branch=master
    :target: https://travis-ci.org/pytorch/text

.. image:: https://codecov.io/gh/pytorch/text/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pytorch/text

.. image:: http://readthedocs.org/projects/torchtext/badge/?version=latest
    :target: http://torchtext.readthedocs.io/en/latest/?badge=latest

torchtext
+++++++++

This repository consists of:

* `torchtext.data <#data>`_: Generic data loaders, abstractions, and iterators for text (including vocabulary and word vectors)
* `torchtext.datasets <#datasets>`_: Pre-built loaders for common NLP datasets

Installation
============


Make sure you have Python 2.7 or 3.5+ and PyTorch 0.4.0 or newer. You can then install torchtext using pip::

    pip install torchtext
    
For PyTorch versions before 0.4.0, please use `pip install torchtext==0.2.3`.

Optional requirements
---------------------

If you want to use English tokenizer from `SpaCy <http://spacy.io/>`_, you need to install SpaCy and download its English model::

    pip install spacy
    python -m spacy download en

Alternatively, you might want to use the `Moses <http://www.statmt.org/moses/>`_ tokenizer port in `SacreMoses <https://github.com/alvations/sacremoses>`_ (split from `NLTK <http://nltk.org/>`_). You have to install SacreMoses::

    pip install sacremoses

Documentation
=============

Find the documentation `here <https://torchtext.readthedocs.io/en/latest/index.html>`_.

Data
====

The data module provides the following:

* Ability to describe declaratively how to load a custom NLP dataset that's in a "normal" format:

  .. code-block:: python

      >>> pos = data.TabularDataset(
      ...    path='data/pos/pos_wsj_train.tsv', format='tsv',
      ...    fields=[('text', data.Field()),
      ...            ('labels', data.Field())])
      ...
      >>> sentiment = data.TabularDataset(
      ...    path='data/sentiment/train.json', format='json',
      ...    fields={'sentence_tokenized': ('text', data.Field(sequential=True)),
      ...            'sentiment_gold': ('labels', data.Field(sequential=False))})

* Ability to define a preprocessing pipeline:

  .. code-block:: python

      >>> src = data.Field(tokenize=my_custom_tokenizer)
      >>> trg = data.Field(tokenize=my_custom_tokenizer)
      >>> mt_train = datasets.TranslationDataset(
      ...     path='data/mt/wmt16-ende.train', exts=('.en', '.de'),
      ...     fields=(src, trg))

* Batching, padding, and numericalizing (including building a vocabulary object):

  .. code-block:: python

      >>> # continuing from above
      >>> mt_dev = datasets.TranslationDataset(
      ...     path='data/mt/newstest2014', exts=('.en', '.de'),
      ...     fields=(src, trg))
      >>> src.build_vocab(mt_train, max_size=80000)
      >>> trg.build_vocab(mt_train, max_size=40000)
      >>> # mt_dev shares the fields, so it shares their vocab objects
      >>>
      >>> train_iter = data.BucketIterator(
      ...     dataset=mt_train, batch_size=32,
      ...     sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
      >>> # usage
      >>> next(iter(train_iter))
      <data.Batch(batch_size=32, src=[LongTensor (32, 25)], trg=[LongTensor (32, 28)])>

* Wrapper for dataset splits (train, validation, test):

  .. code-block:: python

      >>> TEXT = data.Field()
      >>> LABELS = data.Field()
      >>>
      >>> train, val, test = data.TabularDataset.splits(
      ...     path='/data/pos_wsj/pos_wsj', train='_train.tsv',
      ...     validation='_dev.tsv', test='_test.tsv', format='tsv',
      ...     fields=[('text', TEXT), ('labels', LABELS)])
      >>>
      >>> train_iter, val_iter, test_iter = data.BucketIterator.splits(
      ...     (train, val, test), batch_sizes=(16, 256, 256),
      >>>     sort_key=lambda x: len(x.text), device=0)
      >>>
      >>> TEXT.build_vocab(train)
      >>> LABELS.build_vocab(train)

Datasets
========

The datasets module currently contains:

* Sentiment analysis: SST and IMDb
* Question classification: TREC
* Entailment: SNLI, MultiNLI
* Language modeling: abstract class + WikiText-2, WikiText103, PennTreebank
* Machine translation: abstract class + Multi30k, IWSLT, WMT14
* Sequence tagging (e.g. POS/NER): abstract class + UDPOS, CoNLL2000Chunking
* Question answering: 20 QA bAbI tasks
* Text classification: AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, YelpReviewFull, YahooAnswers, AmazonReviewPolarity, AmazonReviewFull

Others are planned or a work in progress:

* Question answering: SQuAD

See the ``test`` directory for examples of dataset usage.
