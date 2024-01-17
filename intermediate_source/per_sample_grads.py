# -*- coding: utf-8 -*-
"""
Per-sample-gradients
====================

What is it?
-----------

Per-sample-gradient computation is computing the gradient for each and every
sample in a batch of data. It is a useful quantity in differential privacy,
meta-learning, and optimization research.

.. note::

   This tutorial requires PyTorch 2.0.0 or later.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# Here's a simple CNN and loss function:

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output

def loss_fn(predictions, targets):
    return F.nll_loss(predictions, targets)


######################################################################
# Let’s generate a batch of dummy data and pretend that we’re working with an MNIST dataset.
# The dummy images are 28 by 28 and we use a minibatch of size 64.

device = 'cuda'

num_models = 10
batch_size = 64
data = torch.randn(batch_size, 1, 28, 28, device=device)

targets = torch.randint(10, (64,), device=device)

######################################################################
# In regular model training, one would forward the minibatch through the model,
# and then call .backward() to compute gradients.  This would generate an
# 'average' gradient of the entire mini-batch:

model = SimpleCNN().to(device=device)
predictions = model(data)  # move the entire mini-batch through the model

loss = loss_fn(predictions, targets)
loss.backward()  # back propagate the 'average' gradient of this mini-batch

######################################################################
# In contrast to the above approach, per-sample-gradient computation is
# equivalent to:
#
# - for each individual sample of the data, perform a forward and a backward
#   pass to get an individual (per-sample) gradient.

def compute_grad(sample, target):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_fn(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(data, targets):
    """ manually process each sample with per sample gradient """
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

per_sample_grads = compute_sample_grads(data, targets)

######################################################################
# ``sample_grads[0]`` is the per-sample-grad for model.conv1.weight.
# ``model.conv1.weight.shape`` is ``[32, 1, 3, 3]``; notice how there is one
# gradient, per sample, in the batch for a total of 64.

print(per_sample_grads[0].shape)

######################################################################
# Per-sample-grads, *the efficient way*, using function transforms
# ----------------------------------------------------------------
# We can compute per-sample-gradients efficiently by using function transforms.
#
# The ``torch.func`` function transform API transforms over functions.
# Our strategy is to define a function that computes the loss and then apply
# transforms to construct a function that computes per-sample-gradients.
#
# We'll use the ``torch.func.functional_call`` function to treat an ``nn.Module``
# like a function.
#
# First, let’s extract the state from ``model`` into two dictionaries,
# parameters and buffers. We'll be detaching them because we won't use
# regular PyTorch autograd (e.g. Tensor.backward(), torch.autograd.grad).

from torch.func import functional_call, vmap, grad

params = {k: v.detach() for k, v in model.named_parameters()}
buffers = {k: v.detach() for k, v in model.named_buffers()}

######################################################################
# Next, let's define a function to compute the loss of the model given a
# single input rather than a batch of inputs. It is important that this
# function accepts the parameters, the input, and the target, because we will
# be transforming over them.
#
# Note - because the model was originally written to handle batches, we’ll
# use ``torch.unsqueeze`` to add a batch dimension.

def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = functional_call(model, (params, buffers), (batch,))
    loss = loss_fn(predictions, targets)
    return loss

######################################################################
# Now, let’s use the ``grad`` transform to create a new function that computes
# the gradient with respect to the first argument of ``compute_loss``
# (i.e. the ``params``).

ft_compute_grad = grad(compute_loss)

######################################################################
# The ``ft_compute_grad`` function computes the gradient for a single
# (sample, target) pair. We can use ``vmap`` to get it to compute the gradient
# over an entire batch of samples and targets. Note that
# ``in_dims=(None, None, 0, 0)`` because we wish to map ``ft_compute_grad`` over
# the 0th dimension of the data and targets, and use the same ``params`` and
# buffers for each.

ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

######################################################################
# Finally, let's used our transformed function to compute per-sample-gradients:

ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)

######################################################################
# we can double check that the results using ``grad`` and ``vmap`` match the
# results of hand processing each one individually:

for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads.values()):
    assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=3e-3, rtol=1e-5)

######################################################################
# A quick note: there are limitations around what types of functions can be
# transformed by ``vmap``. The best functions to transform are ones that are pure
# functions: a function where the outputs are only determined by the inputs,
# and that have no side effects (e.g. mutation). ``vmap`` is unable to handle
# mutation of arbitrary Python data structures, but it is able to handle many
# in-place PyTorch operations.
#
# Performance comparison
# ----------------------
#
# Curious about how the performance of ``vmap`` compares?
#
# Currently the best results are obtained on newer GPU's such as the A100
# (Ampere) where we've seen up to 25x speedups on this example, but here are
# some results on our build machines:

def get_perf(first, first_descriptor, second, second_descriptor):
    """takes torch.benchmark objects and compares delta of second vs first."""
    second_res = second.times[0]
    first_res = first.times[0]

    gain = (first_res-second_res)/first_res
    if gain < 0: gain *=-1 
    final_gain = gain*100

    print(f"Performance delta: {final_gain:.4f} percent improvement with {first_descriptor} ")

from torch.utils.benchmark import Timer

without_vmap = Timer(stmt="compute_sample_grads(data, targets)", globals=globals())
with_vmap = Timer(stmt="ft_compute_sample_grad(params, buffers, data, targets)",globals=globals())
no_vmap_timing = without_vmap.timeit(100)
with_vmap_timing = with_vmap.timeit(100)

print(f'Per-sample-grads without vmap {no_vmap_timing}')
print(f'Per-sample-grads with vmap {with_vmap_timing}')

get_perf(with_vmap_timing, "vmap", no_vmap_timing, "no vmap")

######################################################################
# There are other optimized solutions (like in https://github.com/pytorch/opacus)
# to computing per-sample-gradients in PyTorch that also perform better than
# the naive method. But it’s cool that composing ``vmap`` and ``grad`` give us a
# nice speedup.
#
# In general, vectorization with ``vmap`` should be faster than running a function
# in a for-loop and competitive with manual batching. There are some exceptions
# though, like if we haven’t implemented the ``vmap`` rule for a particular
# operation or if the underlying kernels weren’t optimized for older hardware
# (GPUs). If you see any of these cases, please let us know by opening an issue
# at on GitHub.
