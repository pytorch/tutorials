Note

Go to the end
to download the full example code.

# Model ensembling

This tutorial illustrates how to vectorize model ensembling using `torch.vmap`.

## What is model ensembling?

Model ensembling combines the predictions from multiple models together.
Traditionally this is done by running each model on some inputs separately
and then combining the predictions. However, if you're running models with
the same architecture, then it may be possible to combine them together
using `torch.vmap`. `vmap` is a function transform that maps functions across
dimensions of the input tensors. One of its use cases is eliminating
for-loops and speeding them up through vectorization.

Let's demonstrate how to do this using an ensemble of simple MLPs.

Note

This tutorial requires PyTorch 2.0.0 or later.

```
# Here's a simple MLP
```

Let's generate a batch of dummy data and pretend that we're working with
an MNIST dataset. Thus, the dummy images are 28 by 28, and we have a
minibatch of size 64. Furthermore, lets say we want to combine the predictions
from 10 different models.

We have a couple of options for generating predictions. Maybe we want to
give each model a different randomized minibatch of data. Alternatively,
maybe we want to run the same minibatch of data through each model (e.g.
if we were testing the effect of different model initializations).

Option 1: different minibatch for each model

Option 2: Same minibatch

## Using `vmap` to vectorize the ensemble

Let's use `vmap` to speed up the for-loop. We must first prepare the models
for use with `vmap`.

First, let's combine the states of the model together by stacking each
parameter. For example, `model[i].fc1.weight` has shape `[784, 128]`; we are
going to stack the `.fc1.weight` of each of the 10 models to produce a big
weight of shape `[10, 784, 128]`.

PyTorch offers the `torch.func.stack_module_state` convenience function to do
this.

Next, we need to define a function to `vmap` over. The function should,
given parameters and buffers and inputs, run the model using those
parameters, buffers, and inputs. We'll use `torch.func.functional_call`
to help out:

```
# Construct a "stateless" version of one of the models. It is "stateless" in
# the sense that the parameters are meta Tensors and do not have storage.
```

Option 1: get predictions using a different minibatch for each model.

By default, `vmap` maps a function across the first dimension of all inputs to
the passed-in function. After using `stack_module_state`, each of
the `params` and buffers have an additional dimension of size 'num_models' at
the front, and minibatches has a dimension of size 'num_models'.

```
# verify the ``vmap`` predictions match the
```

Option 2: get predictions using the same minibatch of data.

`vmap` has an `in_dims` argument that specifies which dimensions to map over.
By using `None`, we tell `vmap` we want the same minibatch to apply for all of
the 10 models.

A quick note: there are limitations around what types of functions can be
transformed by `vmap`. The best functions to transform are ones that are pure
functions: a function where the outputs are only determined by the inputs
that have no side effects (e.g. mutation). `vmap` is unable to handle mutation
of arbitrary Python data structures, but it is able to handle many in-place
PyTorch operations.

## Performance

Curious about performance numbers? Here's how the numbers look.

There's a large speedup using `vmap`!

In general, vectorization with `vmap` should be faster than running a function
in a for-loop and competitive with manual batching. There are some exceptions
though, like if we haven't implemented the `vmap` rule for a particular
operation or if the underlying kernels weren't optimized for older hardware
(GPUs). If you see any of these cases, please let us know by opening an issue
on GitHub.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: ensembling.ipynb`](../_downloads/1342193c7104875f1847417466d1417c/ensembling.ipynb)

[`Download Python source code: ensembling.py`](../_downloads/626f23350a6d0b457ded1932a69ec7eb/ensembling.py)

[`Download zipped: ensembling.zip`](../_downloads/cadffb3b32c723f1ec5de36af2cd8782/ensembling.zip)