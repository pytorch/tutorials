.. role:: hidden
    :class: hidden-section

torchtext.data
=================

The data module provides the following:

- Ability to define a preprocessing pipeline
- Batching, padding, and numericalizing (including building a vocabulary object)
- Wrapper for dataset splits (train, validation, test)
- Loader for a custom NLP dataset


.. automodule:: torchtext.data
.. currentmodule:: torchtext.data

Dataset, Batch, and Example
---------------------------

:hidden:`Dataset`
~~~~~~~~~~~~~~~~~~

.. autoclass:: Dataset
    :members:
    :special-members: __init__

:hidden:`TabularDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TabularDataset
    :members:
    :special-members: __init__

:hidden:`Batch`
~~~~~~~~~~~~~~~

.. autoclass:: Batch
    :members:
    :special-members: __init__

:hidden:`Example`
~~~~~~~~~~~~~~~~~

.. autoclass:: Example
    :members:
    :undoc-members:
    :special-members: __init__

Fields
--------------------

:hidden:`RawField`
~~~~~~~~~~~~~~~~~~

.. autoclass:: RawField
    :members:
    :special-members: __init__

:hidden:`Field`
~~~~~~~~~~~~~~~

.. autoclass:: Field
    :members:
    :special-members: __init__

:hidden:`ReversibleField`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ReversibleField
    :members:
    :special-members: __init__

:hidden:`SubwordField`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SubwordField
    :members:
    :special-members: __init__

:hidden:`NestedField`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NestedField
    :members:
    :special-members: __init__

Iterators
--------------------

:hidden:`Iterator`
~~~~~~~~~~~~~~~~~~

.. autoclass:: Iterator
    :members:
    :special-members: __init__

:hidden:`BucketIterator`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BucketIterator
    :members:
    :special-members: __init__

:hidden:`BPTTIterator`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BPTTIterator
    :members:
    :special-members: __init__

Pipeline
--------------------

:hidden:`Pipeline`
~~~~~~~~~~~~~~~~~~

.. autoclass:: Pipeline
    :members:
    :special-members: __init__

Functions
---------------

:hidden:`batch`
~~~~~~~~~~~~~~~

.. autofunction:: batch

:hidden:`pool`
~~~~~~~~~~~~~~

.. autofunction:: pool

:hidden:`get_tokenizer`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_tokenizer

:hidden:`interleave_keys`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: interleave_keys
