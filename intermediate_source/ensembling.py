# -*- coding: utf-8 -*-
"""
Model ensembling
================

This tutorial illustrates how to vectorize model ensembling using ``torch.vmap``.

What is model ensembling?
-------------------------
Model ensembling combines the predictions from multiple models together.
Traditionally this is done by running each model on some inputs separately
and then combining the predictions. However, if you're running models with
the same architecture, then it may be possible to combine them together
using ``torch.vmap``. ``vmap`` is a function transform that maps functions across
dimensions of the input tensors. One of its use cases is eliminating
for-loops and speeding them up through vectorization.

Let's demonstrate how to do this using an ensemble of simple MLPs.

.. note::

   This tutorial requires PyTorch 2.0.0 or later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# Here's a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

######################################################################
# Let’s generate a batch of dummy data and pretend that we’re working with
# an MNIST dataset. Thus, the dummy images are 28 by 28, and we have a
# minibatch of size 64. Furthermore, lets say we want to combine the predictions
# from 10 different models.

device = 'cuda'
num_models = 10

data = torch.randn(100, 64, 1, 28, 28, device=device)
targets = torch.randint(10, (6400,), device=device)

models = [SimpleMLP().to(device) for _ in range(num_models)]

######################################################################
# We have a couple of options for generating predictions. Maybe we want to
# give each model a different randomized minibatch of data. Alternatively,
# maybe we want to run the same minibatch of data through each model (e.g.
# if we were testing the effect of different model initializations).

######################################################################
# Option 1: different minibatch for each model

minibatches = data[:num_models]
predictions_diff_minibatch_loop = [model(minibatch) for model, minibatch in zip(models, minibatches)]

######################################################################
# Option 2: Same minibatch

minibatch = data[0]
predictions2 = [model(minibatch) for model in models]

######################################################################
# Using ``vmap`` to vectorize the ensemble
# ----------------------------------------
#
# Let's use ``vmap`` to speed up the for-loop. We must first prepare the models
# for use with ``vmap``.
#
# First, let’s combine the states of the model together by stacking each
# parameter. For example, ``model[i].fc1.weight`` has shape ``[784, 128]``; we are
# going to stack the ``.fc1.weight`` of each of the 10 models to produce a big
# weight of shape ``[10, 784, 128]``.
#
# PyTorch offers the ``torch.func.stack_module_state`` convenience function to do
# this.
from torch.func import stack_module_state

params, buffers = stack_module_state(models)

######################################################################
# Next, we need to define a function to ``vmap`` over. The function should,
# given parameters and buffers and inputs, run the model using those
# parameters, buffers, and inputs. We'll use ``torch.func.functional_call``
# to help out:

from torch.func import functional_call
import copy

# Construct a "stateless" version of one of the models. It is "stateless" in
# the sense that the parameters are meta Tensors and do not have storage.
base_model = copy.deepcopy(models[0])
base_model = base_model.to('meta')

def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))

######################################################################
# Option 1: get predictions using a different minibatch for each model.
#
# By default, ``vmap`` maps a function across the first dimension of all inputs to
# the passed-in function. After using ``stack_module_state``, each of
# the ``params`` and buffers have an additional dimension of size 'num_models' at
# the front, and minibatches has a dimension of size 'num_models'.

print([p.size(0) for p in params.values()]) # show the leading 'num_models' dimension

assert minibatches.shape == (num_models, 64, 1, 28, 28) # verify minibatch has leading dimension of size 'num_models'

from torch import vmap

predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)

# verify the ``vmap`` predictions match the
assert torch.allclose(predictions1_vmap, torch.stack(predictions_diff_minibatch_loop), atol=1e-3, rtol=1e-5)

######################################################################
# Option 2: get predictions using the same minibatch of data.
#
# ``vmap`` has an ``in_dims`` argument that specifies which dimensions to map over.
# By using ``None``, we tell ``vmap`` we want the same minibatch to apply for all of
# the 10 models.

predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, minibatch)

assert torch.allclose(predictions2_vmap, torch.stack(predictions2), atol=1e-3, rtol=1e-5)

######################################################################
# A quick note: there are limitations around what types of functions can be
# transformed by ``vmap``. The best functions to transform are ones that are pure
# functions: a function where the outputs are only determined by the inputs
# that have no side effects (e.g. mutation). ``vmap`` is unable to handle mutation
# of arbitrary Python data structures, but it is able to handle many in-place
# PyTorch operations.

######################################################################
# Performance
# -----------
# Curious about performance numbers? Here's how the numbers look.

from torch.utils.benchmark import Timer
without_vmap = Timer(
    stmt="[model(minibatch) for model, minibatch in zip(models, minibatches)]",
    globals=globals())
with_vmap = Timer(
    stmt="vmap(fmodel)(params, buffers, minibatches)",
    globals=globals())
print(f'Predictions without vmap {without_vmap.timeit(100)}')
print(f'Predictions with vmap {with_vmap.timeit(100)}')

######################################################################
# There's a large speedup using ``vmap``!
#
# In general, vectorization with ``vmap`` should be faster than running a function
# in a for-loop and competitive with manual batching. There are some exceptions
# though, like if we haven’t implemented the ``vmap`` rule for a particular
# operation or if the underlying kernels weren’t optimized for older hardware
# (GPUs). If you see any of these cases, please let us know by opening an issue
# on GitHub.
