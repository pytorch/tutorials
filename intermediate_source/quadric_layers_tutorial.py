"""
(beta) Quadric Layers
==================================================================

**Author**: `Dirk Roeckmann <https://github.com/diro5t>`_

Introduction
------------

Quadric layers introduce quadratic functions with second-order decision boundaries (quadric hypersurfaces)
and can be used as 100% drop-in layers for linear layers (torch.nn.Linear) and present a high-level means
to reduce overall model size.

In comparison to linear layers with n weights and 1 bias (if needed) per neuron, a quadric neuron consists of
2n weights (n quadratic weights and n linear weights) and 1 bias (if needed).
Although this means a doubling in weights per neuron, the more powerful decision boundaries per neuron lead 
in many applications to siginificantly less neurons per layer or even less layers and in total to less model parameters.

In this tutorial, I will present a trivial example - binary classification of circular data using a quadric layer 
with just one neuron.
`dynamic quantization <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`_ -
to an LSTM-based next word-prediction model, closely following the
`word language model <https://github.com/pytorch/examples/tree/master/word_language_model>`_
from the PyTorch examples.
"""
