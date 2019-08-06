torchtext.datasets
====================

.. currentmodule:: torchtext.datasets

All datasets are subclasses of :class:`torchtext.data.Dataset`, which
inherits from :class:`torch.utils.data.Dataset` i.e, they have ``split`` and
``iters`` methods implemented.

General use cases are as follows:

Approach 1, ``splits``: ::

    # set up fields
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)

    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=3, device=0)

Approach 2, ``iters``: ::

    # use default configurations
    train_iter, test_iter = datasets.IMDB.iters(batch_size=4)

The following datasets are available:

.. contents:: Datasets
    :local:


Sentiment Analysis
^^^^^^^^^^^^^^^^^^

SST
~~~

.. autoclass:: SST
  :members: splits, iters

IMDb
~~~~

.. autoclass:: IMDB
  :members: splits, iters


TextClassificationDataset
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TextClassificationDataset
  :members: __init__

AG_NEWS
~~~~~~~

AG_NEWS dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: AG_NEWS
  :members: __init__

SogouNews
~~~~~~~~~

SogouNews dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: SogouNews
  :members: __init__

DBpedia
~~~~~~~

DBpedia dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: DBpedia
  :members: __init__

YelpReviewPolarity
~~~~~~~~~~~~~~~~~~

YelpReviewPolarity dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: YelpReviewPolarity
  :members: __init__

YelpReviewFull
~~~~~~~~~~~~~~

YelpReviewFull dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: YelpReviewFull
  :members: __init__

YahooAnswers
~~~~~~~~~~~~

YahooAnswers dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: YahooAnswers
  :members: __init__

AmazonReviewPolarity
~~~~~~~~~~~~~~~~~~~~

AmazonReviewPolarity dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: AmazonReviewPolarity
  :members: __init__

AmazonReviewFull
~~~~~~~~~~~~~~~~

AmazonReviewFull dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: AmazonReviewFull
  :members: __init__


Question Classification
^^^^^^^^^^^^^^^^^^^^^^^

TREC
~~~~

.. autoclass:: TREC
  :members: splits, iters

Entailment
^^^^^^^^^^

SNLI
~~~~

.. autoclass:: SNLI
  :members: splits, iters


MultiNLI
~~~~~~~~

.. autoclass:: MultiNLI
  :members: splits, iters


Language Modeling
^^^^^^^^^^^^^^^^^

Language modeling datasets are subclasses of ``LanguageModelingDataset`` class.

.. autoclass:: LanguageModelingDataset
  :members: __init__


WikiText-2
~~~~~~~~~~

.. autoclass:: WikiText2
  :members: splits, iters


WikiText103
~~~~~~~~~~~

.. autoclass:: WikiText103
  :members: splits, iters


PennTreebank
~~~~~~~~~~~~

.. autoclass:: PennTreebank
  :members: splits, iters


Machine Translation
^^^^^^^^^^^^^^^^^^^

Machine translation datasets are subclasses of ``TranslationDataset`` class.

.. autoclass:: TranslationDataset
  :members: __init__


Multi30k
~~~~~~~~

.. autoclass:: Multi30k
  :members: splits

IWSLT
~~~~~

.. autoclass:: IWSLT
  :members: splits

WMT14
~~~~~

.. autoclass:: WMT14
  :members: splits


Sequence Tagging
^^^^^^^^^^^^^^^^

Sequence tagging datasets are subclasses of ``SequenceTaggingDataset`` class.

.. autoclass:: SequenceTaggingDataset
  :members: __init__


UDPOS
~~~~~

.. autoclass:: UDPOS
  :members: splits

CoNLL2000Chunking
~~~~~~~~~~~~~~~~~

.. autoclass:: CoNLL2000Chunking
  :members: splits

Question Answering
^^^^^^^^^^^^^^^^^^

BABI20
~~~~~~

.. autoclass:: BABI20
  :members: __init__, splits, iters
