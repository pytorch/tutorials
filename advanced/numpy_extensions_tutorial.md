Note

Go to the end
to download the full example code.

# Creating Extensions Using NumPy and SciPy

**Author**: [Adam Paszke](https://github.com/apaszke)

**Updated by**: [Adam Dziedzic](https://github.com/adam-dziedzic)

In this tutorial, we shall go through two tasks:

1. Create a neural network layer with no parameters.

> - This calls into **numpy** as part of its implementation
2. Create a neural network layer that has learnable weights

> - This calls into **SciPy** as part of its implementation

## Parameter-less example

This layer doesn't particularly do anything useful or mathematically
correct.

It is aptly named `BadFFTFunction`

**Layer Implementation**

```
# since this layer does not have any parameters, we can
# simply declare this as a function, rather than as an ``nn.Module`` class
```

**Example usage of the created layer:**

## Parametrized example

In deep learning literature, this layer is confusingly referred
to as convolution while the actual operation is cross-correlation
(the only difference is that filter is flipped for convolution,
which is not the case for cross-correlation).

Implementation of a layer with learnable weights, where cross-correlation
has a filter (kernel) that represents weights.

The backward pass computes the gradient `wrt` the input and the gradient `wrt` the filter.

**Example usage:**

**Check the gradients:**

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: numpy_extensions_tutorial.ipynb`](../_downloads/52d4aaa33601a2b3990ace6aa45546ce/numpy_extensions_tutorial.ipynb)

[`Download Python source code: numpy_extensions_tutorial.py`](../_downloads/ee55f2537c08fa13e041e479675a6c2c/numpy_extensions_tutorial.py)

[`Download zipped: numpy_extensions_tutorial.zip`](../_downloads/7ce1d46c486fec7f0114e19f48899f48/numpy_extensions_tutorial.zip)