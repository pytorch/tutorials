# -*- coding: utf-8 -*-
"""
nn package
==========

We’ve redesigned the nn package, so that it’s fully integrated with
autograd. Let's review the changes.

**Replace containers with autograd:**

    You no longer have to use Containers like ``ConcatTable``, or modules like
    ``CAddTable``, or use and debug with nngraph. We will seamlessly use
    autograd to define our neural networks. For example,

    * ``output = nn.CAddTable():forward({input1, input2})`` simply becomes
      ``output = input1 + input2``
    * ``output = nn.MulConstant(0.5):forward(input)`` simply becomes
      ``output = input * 0.5``

**State is no longer held in the module, but in the network graph:**

    Using recurrent networks should be simpler because of this reason. If
    you want to create a recurrent network, simply use the same Linear layer
    multiple times, without having to think about sharing weights.

    .. figure:: /_static/img/torch-nn-vs-pytorch-nn.png
       :alt: torch-nn-vs-pytorch-nn

       torch-nn-vs-pytorch-nn

**Simplified debugging:**

    Debugging is intuitive using Python’s pdb debugger, and **the debugger
    and stack traces stop at exactly where an error occurred.** What you see
    is what you get.

Example 1: ConvNet
------------------

Let’s see how to create a small ConvNet.

All of your networks are derived from the base class ``nn.Module``:

-  In the constructor, you declare all the layers you want to use.
-  In the forward function, you define how your model is going to be
   run, from input to output
"""













































###############################################################
# Let's use the defined ConvNet now.
# You create an instance of the class first.





########################################################################
# .. note::
#
#     ``torch.nn`` only supports mini-batches The entire ``torch.nn``
#     package only supports inputs that are a mini-batch of samples, and not
#     a single sample.
#
#     For example, ``nn.Conv2d`` will take in a 4D Tensor of
#     ``nSamples x nChannels x Height x Width``.
#
#     If you have a single sample, just use ``input.unsqueeze(0)`` to add
#     a fake batch dimension.
#
# Create a mini-batch containing a single sample of random data and send the
# sample through the ConvNet.





########################################################################
# Define a dummy target label and compute error using a loss function.








########################################################################
# The output of the ConvNet ``out`` is a ``Tensor``. We compute the loss
# using that, and that results in ``err`` which is also a ``Tensor``.
# Calling ``.backward`` on ``err`` hence will propagate gradients all the
# way through the ConvNet to it’s weights
#
# Let's access individual layer weights and gradients:



########################################################################



########################################################################
# Forward and Backward Function Hooks
# -----------------------------------
#
# We’ve inspected the weights and the gradients. But how about inspecting
# / modifying the output and grad\_output of a layer?
#
# We introduce **hooks** for this purpose.
#
# You can register a function on a ``Module`` or a ``Tensor``.
# The hook can be a forward hook or a backward hook.
# The forward hook will be executed when a forward call is executed.
# The backward hook will be executed in the backward phase.
# Let’s look at an example.
#
# We register a forward hook on conv2 and print some information




















########################################################################
#
# We register a backward hook on conv2 and print some information






















########################################################################
# A full and working MNIST example is located here
# https://github.com/pytorch/examples/tree/master/mnist
#
# Example 2: Recurrent Net
# ------------------------
#
# Next, let’s look at building recurrent nets with PyTorch.
#
# Since the state of the network is held in the graph and not in the
# layers, you can simply create an nn.Linear and reuse it over and over
# again for the recurrence.























########################################################################
#
# A more complete Language Modeling example using LSTMs and Penn Tree-bank
# is located
# `here <https://github.com/pytorch/examples/tree/master/word\_language\_model>`_
#
# PyTorch by default has seamless CuDNN integration for ConvNets and
# Recurrent Nets






# Create some fake data












# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%