"""
Dynamic Quantization
====================

In this recipe you will see how to take advantage of Dynamic
Quantization to accelerate inference on an LSTM-style recurrent neural
network. This reduces the size of the model weights and speeds up model
execution.

Introduction
-------------

There are a number of trade-offs that can be made when designing neural
networks. During model development and training you can alter the
number of layers and number of parameters in a recurrent neural network
and trade-off accuracy against model size and/or model latency or
throughput. Such changes can take lot of time and compute resources
because you are iterating over the model training. Quantization gives
you a way to make a similar trade off between performance and model
accuracy with a known model after training is completed.

You can give it a try in a single session and you will certainly reduce
your model size significantly and may get a significant latency
reduction without losing a lot of accuracy.

What is dynamic quantization?
-----------------------------

Quantizing a network means converting it to use a reduced precision
integer representation for the weights and/or activations. This saves on
model size and allows the use of higher throughput math operations on
your CPU or GPU.

When converting from floating point to integer values you are
essentially multiplying the floating point value by some scale factor
and rounding the result to a whole number. The various quantization
approaches differ in the way they approach determining that scale
factor.

The key idea with dynamic quantization as described here is that we are
going to determine the scale factor for activations dynamically based on
the data range observed at runtime. This ensures that the scale factor
is "tuned" so that as much signal as possible about each observed
dataset is preserved.

The model parameters on the other hand are known during model conversion
and they are converted ahead of time and stored in INT8 form.

Arithmetic in the quantized model is done using vectorized INT8
instructions. Accumulation is typically done with INT16 or INT32 to
avoid overflow. This higher precision value is scaled back to INT8 if
the next layer is quantized or converted to FP32 for output.

Dynamic quantization is relatively free of tuning parameters which makes
it well suited to be added into production pipelines as a standard part
of converting LSTM models to deployment.



.. note::
   Limitations on the approach taken here


   This recipe provides a quick introduction to the dynamic quantization
   features in PyTorch and the workflow for using it. Our focus is on
   explaining the specific functions used to convert the model. We will
   make a number of significant simplifications in the interest of brevity
   and clarity


1. You will start with a minimal LSTM network
2. You are simply going to initialize the network with a random hidden
   state
3. You are going to test the network with random inputs
4. You are not going to train the network in this tutorial
5. You will see that the quantized form of this network is smaller and
   runs faster than the floating point network we started with
6. You will see that the output values are generally in the same
   ballpark as the output of the FP32 network, but we are not
   demonstrating here the expected accuracy loss on a real trained
   network

You will see how dynamic quantization is done and be able to see
suggestive reductions in memory use and latency times. Providing a
demonstration that the technique can preserve high levels of model
accuracy on a trained LSTM is left to a more advanced tutorial. If you
want to move right away to that more rigorous treatment please proceed
to the `advanced dynamic quantization
tutorial <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`__.

Steps
-------------

This recipe has 5 steps.

1. Set Up - Here you define a very simple LSTM, import modules, and establish
   some random input tensors.

2. Do the Quantization - Here you instantiate a floating point model and then create quantized
   version of it.

3. Look at Model Size - Here you show that the model size gets smaller.

4. Look at Latency - Here you run the two models and compare model runtime (latency).

5. Look at Accuracy - Here you run the two models and compare outputs.


1: Set Up
~~~~~~~~~~~~~~~
This is a straightforward bit of code to set up for the rest of the
recipe.

The unique module we are importing here is torch.quantization which
includes PyTorch's quantized operators and conversion functions. We also
define a very simple LSTM model and set up some inputs.

"""

# import the modules used here in this recipe
import torch
import torch.quantization
import torch.nn as nn
import copy
import os
import time

# define a very, very simple LSTM for demonstration purposes
# in this case, we are wrapping ``nn.LSTM``, one layer, no preprocessing or postprocessing
# inspired by
# `Sequence Models and Long Short-Term Memory Networks tutorial <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html`_, by Robert Guthrie
# and `Dynamic Quanitzation tutorial <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`__.
class lstm_for_demonstration(nn.Module):
  """Elementary Long Short Term Memory style model which simply wraps ``nn.LSTM``
     Not to be used for anything other than demonstration.
  """
  def __init__(self,in_dim,out_dim,depth):
     super(lstm_for_demonstration,self).__init__()
     self.lstm = nn.LSTM(in_dim,out_dim,depth)

  def forward(self,inputs,hidden):
     out,hidden = self.lstm(inputs,hidden)
     return out, hidden


torch.manual_seed(29592)  # set the seed for reproducibility

#shape parameters
model_dimension=8
sequence_length=20
batch_size=1
lstm_depth=1

# random data for input
inputs = torch.randn(sequence_length,batch_size,model_dimension)
# hidden is actually is a tuple of the initial hidden state and the initial cell state
hidden = (torch.randn(lstm_depth,batch_size,model_dimension), torch.randn(lstm_depth,batch_size,model_dimension))


######################################################################
# 2: Do the Quantization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now we get to the fun part. First we create an instance of the model
# called ``float\_lstm`` then we are going to quantize it. We're going to use
# the `torch.quantization.quantize_dynamic <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`__ function, which takes the model, then a list of the submodules
# which we want to
# have quantized if they appear, then the datatype we are targeting. This
# function returns a quantized version of the original model as a new
# module.
#
# That's all it takes.
#

 # here is our floating point instance
float_lstm = lstm_for_demonstration(model_dimension, model_dimension,lstm_depth)

# this is the call that does the work
quantized_lstm = torch.quantization.quantize_dynamic(
    float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# show the changes that were made
print('Here is the floating point version of this module:')
print(float_lstm)
print('')
print('and now the quantized version:')
print(quantized_lstm)


######################################################################
# 3. Look at Model Size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We've quantized the model. What does that get us? Well the first
# benefit is that we've replaced the FP32 model parameters with INT8
# values (and some recorded scale factors). This means about 75% less data
# to store and move around. With the default values the reduction shown
# below will be less than 75% but if you increase the model size above
# (for example you can set model dimension to something like 80) this will
# converge towards 4x smaller as the stored model size dominated more and
# more by the parameter values.
#

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
f=print_size_of_model(float_lstm,"fp32")
q=print_size_of_model(quantized_lstm,"int8")
print("{0:.2f} times smaller".format(f/q))


######################################################################
# 4. Look at Latency
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The second benefit is that the quantized model will typically run
# faster. This is due to a combinations of effects including at least:
#
# 1. Less time spent moving parameter data in
# 2. Faster INT8 operations
#
# As you will see the quantized version of this super-simple network runs
# faster. This will generally be true of more complex networks but as they
# say "your mileage may vary" depending on a number of factors including
# the structure of the model and the hardware you are running on.
#

# compare the performance
print("Floating point FP32")

#####################################################################
# .. code-block:: python
#
#    %timeit float_lstm.forward(inputs, hidden)

print("Quantized INT8")

######################################################################
# .. code-block:: python
#
#    %timeit quantized_lstm.forward(inputs,hidden)


######################################################################
# 5: Look at Accuracy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We are not going to do a careful look at accuracy here because we are
# working with a randomly initialized network rather than a properly
# trained one. However, I think it is worth quickly showing that the
# quantized network does produce output tensors that are "in the same
# ballpark" as the original one.
#
# For a more detailed analysis please see the more advanced tutorials
# referenced at the end of this recipe.
#

# run the float model
out1, hidden1 = float_lstm(inputs, hidden)
mag1 = torch.mean(abs(out1)).item()
print('mean absolute value of output tensor values in the FP32 model is {0:.5f} '.format(mag1))

# run the quantized model
out2, hidden2 = quantized_lstm(inputs, hidden)
mag2 = torch.mean(abs(out2)).item()
print('mean absolute value of output tensor values in the INT8 model is {0:.5f}'.format(mag2))

# compare them
mag3 = torch.mean(abs(out1-out2)).item()
print('mean absolute value of the difference between the output tensors is {0:.5f} or {1:.2f} percent'.format(mag3,mag3/mag1*100))


######################################################################
# Learn More
# ------------
# We've explained what dynamic quantization is, what benefits it brings,
# and you have used the ``torch.quantization.quantize_dynamic()`` function
# to quickly quantize a simple LSTM model.
#
# This was a fast and high level treatment of this material; for more
# detail please continue learning with `(beta) Dynamic Quantization on an LSTM Word Language Model Tutorial <https://pytorch.org/tutorials/advanced/dynamic\_quantization\_tutorial.html>`_.
#
#
# Additional Resources
# --------------------
#
# * `Quantization API Documentaion <https://pytorch.org/docs/stable/quantization.html>`_
# * `(beta) Dynamic Quantization on BERT <https://pytorch.org/tutorials/intermediate/dynamic\_quantization\_bert\_tutorial.html>`_
# * `(beta) Dynamic Quantization on an LSTM Word Language Model <https://pytorch.org/tutorials/advanced/dynamic\_quantization\_tutorial.html>`_
# * `Introduction to Quantization on PyTorch <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/>`_
#
