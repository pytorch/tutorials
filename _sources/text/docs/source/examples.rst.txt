.. role:: hidden
    :class: hidden-section

Examples
=========

1. Ability to describe declaratively how to load a custom NLP dataset that's in a "normal" format:

.. code-block:: python

    pos = data.TabularDataset(
    path='data/pos/pos_wsj_train.tsv', format='tsv',
    fields=[('text', data.Field()),
            ('labels', data.Field())])

    sentiment = data.TabularDataset(
        path='data/sentiment/train.json', format='json',
        fields={'sentence_tokenized': ('text', data.Field(sequential=True)),
                 'sentiment_gold': ('labels', data.Field(sequential=False))})

2. Ability to parse nested keys for loading a JSON dataset

2.1 sample.json

.. code-block:: json

    {"foods": {
        "fruits": ["Apple", "Banana"], 
        "vegetables": [{"name": "lettuce"}, {"name": "marrow"}]
        }
    }

2.2 pass in nested keys to parse nested data directly

.. code-block:: python

    In [1]: from torchtext import data
    In [2]: fields = {'foods.vegetables.name': ('vegs', data.Field())}
    In [3]: dataset = data.TabularDataset(path='sample.json', format='json', fields=fields)
    In [4]: dataset.examples[0].vegs
    Out[4]: ['lettuce', 'marrow']

3. Ability to define a preprocessing pipeline:

.. code-block:: python

    src = data.Field(tokenize=my_custom_tokenizer)
    trg = data.Field(tokenize=my_custom_tokenizer)
    mt_train = datasets.TranslationDataset(
        path='data/mt/wmt16-ende.train', exts=('.en', '.de'),
        fields=(src, trg))

4. Batching, padding, and numericalizing (including building vocabulary object):

.. code-block:: python

    # continuing from above
    mt_dev = data.TranslationDataset(
        path='data/mt/newstest2014', exts=('.en', '.de'),
        fields=(src, trg))
    src.build_vocab(mt_train, max_size=80000)
    trg.build_vocab(mt_train, max_size=40000)
    # mt_dev shares the fields, so it shares their vocab objects

    train_iter = data.BucketIterator(
        dataset=mt_train, batch_size=32,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
    # usage
    >>>next(iter(train_iter))
    <data.Batch(batch_size=32, src=[LongTensor (32, 25)], trg=[LongTensor (32, 28)])>

5. Wrapper for dataset splits (train, validation, test):

.. code-block:: python

    TEXT = data.Field()
    LABELS = data.Field()

    train, val, test = data.TabularDataset.splits(
        path='/data/pos_wsj/pos_wsj', train='_train.tsv',
        validation='_dev.tsv', test='_test.tsv', format='tsv',
        fields=[('text', TEXT), ('labels', LABELS)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(16, 256, 256),
        sort_key=lambda x: len(x.text), device=0)

    TEXT.build_vocab(train)
    LABELS.build_vocab(train)
