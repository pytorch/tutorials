"""
(beta) Quadric Layers
==================================================================

**Author**: `Dirk Roeckmann <https://github.com/diro5t>`_

Introduction
------------

Quadric layers are drop-in layers for linear layers which introduce quadratic functions (quadric). 
Hereby second-order decision boundaries are achieved, which can in many applications

In this tutorial, we will apply the easiest form of quantization -
`dynamic quantization <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`_ -
to an LSTM-based next word-prediction model, closely following the
`word language model <https://github.com/pytorch/examples/tree/master/word_language_model>`_
from the PyTorch examples.
"""
