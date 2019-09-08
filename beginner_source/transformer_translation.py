"""
Language Translation with Transformers
============================

This tutorial shows how to use several features of ``torchtext`` to preprocess 
a common language dataset used for benchmarks and use it to train a 
Transformer model to do language translation. 

It is based off of 
`this tutorial <https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb>`__ 
from PyTorch community member `Ben Trevett <https://github.com/bentrevett>`__.

By the end of this tutorial, you will be able to:

- Preprocess sentences into a commonly-used format for NLP modeling using ``torchtext`` features
    - `TranslationDataset <https://torchtext.readthedocs.io/en/latest/datasets.html#torchtext.datasets.TranslationDataset>`__
    - `Field <https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field>`__
    - `BucketIterator <https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator>`__        
- Use `nn.Transformer <https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator>`__ as part of a language translation model
    
    
    
"""

# TODO