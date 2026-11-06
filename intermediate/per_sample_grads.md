Note

Go to the end
to download the full example code.

# Per-sample-gradients

## What is it?

Per-sample-gradient computation is computing the gradient for each and every
sample in a batch of data. It is a useful quantity in differential privacy,
meta-learning, and optimization research.

Note

This tutorial requires PyTorch 2.0.0 or later.

```
# Here's a simple CNN and loss function:
```

Let's generate a batch of dummy data and pretend that we're working with an MNIST dataset.
The dummy images are 28 by 28 and we use a minibatch of size 64.

In regular model training, one would forward the minibatch through the model,
and then call .backward() to compute gradients. This would generate an
'average' gradient of the entire mini-batch:

In contrast to the above approach, per-sample-gradient computation is
equivalent to:

- for each individual sample of the data, perform a forward and a backward
pass to get an individual (per-sample) gradient.

`sample_grads[0]` is the per-sample-grad for model.conv1.weight.
`model.conv1.weight.shape` is `[32, 1, 3, 3]`; notice how there is one
gradient, per sample, in the batch for a total of 64.

## Per-sample-grads, *the efficient way*, using function transforms

We can compute per-sample-gradients efficiently by using function transforms.

The `torch.func` function transform API transforms over functions.
Our strategy is to define a function that computes the loss and then apply
transforms to construct a function that computes per-sample-gradients.

We'll use the `torch.func.functional_call` function to treat an `nn.Module`
like a function.

First, let's extract the state from `model` into two dictionaries,
parameters and buffers. We'll be detaching them because we won't use
regular PyTorch autograd (e.g. Tensor.backward(), torch.autograd.grad).

Next, let's define a function to compute the loss of the model given a
single input rather than a batch of inputs. It is important that this
function accepts the parameters, the input, and the target, because we will
be transforming over them.

Note - because the model was originally written to handle batches, we'll
use `torch.unsqueeze` to add a batch dimension.

Now, let's use the `grad` transform to create a new function that computes
the gradient with respect to the first argument of `compute_loss`
(i.e. the `params`).

The `ft_compute_grad` function computes the gradient for a single
(sample, target) pair. We can use `vmap` to get it to compute the gradient
over an entire batch of samples and targets. Note that
`in_dims=(None, None, 0, 0)` because we wish to map `ft_compute_grad` over
the 0th dimension of the data and targets, and use the same `params` and
buffers for each.

Finally, let's used our transformed function to compute per-sample-gradients:

we can double check that the results using `grad` and `vmap` match the
results of hand processing each one individually:

A quick note: there are limitations around what types of functions can be
transformed by `vmap`. The best functions to transform are ones that are pure
functions: a function where the outputs are only determined by the inputs,
and that have no side effects (e.g. mutation). `vmap` is unable to handle
mutation of arbitrary Python data structures, but it is able to handle many
in-place PyTorch operations.

## Performance comparison

Curious about how the performance of `vmap` compares?

Currently the best results are obtained on newer GPU's such as the A100
(Ampere) where we've seen up to 25x speedups on this example, but here are
some results on our build machines:

There are other optimized solutions (like in [pytorch/opacus](https://github.com/pytorch/opacus))
to computing per-sample-gradients in PyTorch that also perform better than
the naive method. But it's cool that composing `vmap` and `grad` give us a
nice speedup.

In general, vectorization with `vmap` should be faster than running a function
in a for-loop and competitive with manual batching. There are some exceptions
though, like if we haven't implemented the `vmap` rule for a particular
operation or if the underlying kernels weren't optimized for older hardware
(GPUs). If you see any of these cases, please let us know by opening an issue
at on GitHub.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: per_sample_grads.ipynb`](../_downloads/df89b8f78d7ed3520a0f632afae4a5b9/per_sample_grads.ipynb)

[`Download Python source code: per_sample_grads.py`](../_downloads/bb0e78bec4d7a6e9b86b2e285cd06671/per_sample_grads.py)

[`Download zipped: per_sample_grads.zip`](../_downloads/ad57f2ac72983468f235e389bb95119c/per_sample_grads.zip)