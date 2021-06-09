"""
PyTorch Profiler
====================================
This recipe explains how to use PyTorch profiler and measure the time and
memory consumption of the model's operators.

Introduction
------------
PyTorch includes a simple profiler API that is useful when user needs
to determine the most expensive operators in the model.

In this recipe, we will use a simple Resnet model to demonstrate how to
use profiler to analyze model performance.

Setup
-----
To install ``torch`` and ``torchvision`` use the following command:

::

   pip install torch torchvision


"""


######################################################################
# Steps
# -----
#
# 1. Import all necessary libraries
# 2. Instantiate a simple Resnet model
# 3. Use profiler to analyze execution time
# 4. Use profiler to analyze memory consumption
# 5. Using tracing functionality
#
# 1. Import all necessary libraries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this recipe we will use ``torch``, ``torchvision.models``
# and ``profiler`` modules:
#

import torch
import torchvision.models as models
import torch.autograd.profiler as profiler


######################################################################
# 2. Instantiate a simple Resnet model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's create an instance of a Resnet model and prepare an input
# for it:
#

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

######################################################################
# 3. Use profiler to analyze execution time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PyTorch profiler is enabled through the context manager and accepts
# a number of parameters, some of the most useful are:
#
# - ``record_shapes`` - whether to record shapes of the operator inputs;
# - ``profile_memory`` - whether to report amount of memory consumed by
#   model's Tensors;
# - ``use_cuda`` - whether to measure execution time of CUDA kernels.
#
# Let's see how we can use profiler to analyze the execution time:

with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        model(inputs)

######################################################################
# Note that we can use ``record_function`` context manager to label
# arbitrary code ranges with user provided names
# (``model_inference`` is used as a label in the example above).
# Profiler allows one to check which operators were called during the
# execution of a code range wrapped with a profiler context manager.
# If multiple profiler ranges are active at the same time (e.g. in
# parallel PyTorch threads), each profiling context manager tracks only
# the operators of its corresponding range.
# Profiler also automatically profiles the async tasks launched
# with ``torch.jit._fork`` and (in case of a backward pass)
# the backward pass operators launched with ``backward()`` call.
#
# Let's print out the stats for the execution above:

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

######################################################################
# The output will look like (omitting some columns):

# -------------------------  --------------  ----------  ------------  ---------
# Name                       Self CPU total   CPU total  CPU time avg  # Calls
# -------------------------  --------------  ----------  ------------  ---------
# model_inference            3.541ms         69.571ms    69.571ms      1
# conv2d                     69.122us        40.556ms    2.028ms       20
# convolution                79.100us        40.487ms    2.024ms       20
# _convolution               349.533us       40.408ms    2.020ms       20
# mkldnn_convolution         39.822ms        39.988ms    1.999ms       20
# batch_norm                 105.559us       15.523ms    776.134us     20
# _batch_norm_impl_index     103.697us       15.417ms    770.856us     20
# native_batch_norm          9.387ms         15.249ms    762.471us     20
# max_pool2d                 29.400us        7.200ms     7.200ms       1
# max_pool2d_with_indices    7.154ms         7.170ms     7.170ms       1
# -------------------------  --------------  ----------  ------------  ---------

######################################################################
# Here we see that, as expected, most of the time is spent in convolution (and specifically in ``mkldnn_convolution``
# for PyTorch compiled with MKL-DNN support).
# Note the difference between self cpu time and cpu time - operators can call other operators, self cpu time exludes time
# spent in children operator calls, while total cpu time includes it.
#
# To get a finer granularity of results and include operator input shapes, pass ``group_by_input_shape=True``:

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

# (omitting some columns)
# -------------------------  -----------  --------  -------------------------------------
# Name                       CPU total    # Calls         Input Shapes
# -------------------------  -----------  --------  -------------------------------------
# model_inference            69.571ms     1         []
# conv2d                     9.019ms      4         [[5, 64, 56, 56], [64, 64, 3, 3], []]
# convolution                9.006ms      4         [[5, 64, 56, 56], [64, 64, 3, 3], []]
# _convolution               8.982ms      4         [[5, 64, 56, 56], [64, 64, 3, 3], []]
# mkldnn_convolution         8.894ms      4         [[5, 64, 56, 56], [64, 64, 3, 3], []]
# max_pool2d                 7.200ms      1         [[5, 64, 112, 112]]
# conv2d                     7.189ms      3         [[5, 512, 7, 7], [512, 512, 3, 3], []]
# convolution                7.180ms      3         [[5, 512, 7, 7], [512, 512, 3, 3], []]
# _convolution               7.171ms      3         [[5, 512, 7, 7], [512, 512, 3, 3], []]
# max_pool2d_with_indices    7.170ms      1         [[5, 64, 112, 112]]
# -------------------------  -----------  --------  --------------------------------------


######################################################################
# 4. Use profiler to analyze memory consumption
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PyTorch profiler can also show the amount of memory (used by the model's tensors)
# that was allocated (or released) during the execution of the model's operators.
# In the output below, 'self' memory corresponds to the memory allocated (released)
# by the operator, excluding the children calls to the other operators.
# To enable memory profiling functionality pass ``profile_memory=True``.

with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# (omitting some columns)
# ---------------------------  ---------------  ---------------  ---------------
# Name                         CPU Mem          Self CPU Mem     Number of Calls
# ---------------------------  ---------------  ---------------  ---------------
# empty                        94.79 Mb         94.79 Mb         123
# resize_                      11.48 Mb         11.48 Mb         2
# addmm                        19.53 Kb         19.53 Kb         1
# empty_strided                4 b              4 b              1
# conv2d                       47.37 Mb         0 b              20
# ---------------------------  ---------------  ---------------  ---------------

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

# (omitting some columns)
# ---------------------------  ---------------  ---------------  ---------------
# Name                         CPU Mem          Self CPU Mem     Number of Calls
# ---------------------------  ---------------  ---------------  ---------------
# empty                        94.79 Mb         94.79 Mb         123
# batch_norm                   47.41 Mb         0 b              20
# _batch_norm_impl_index       47.41 Mb         0 b              20
# native_batch_norm            47.41 Mb         0 b              20
# conv2d                       47.37 Mb         0 b              20
# convolution                  47.37 Mb         0 b              20
# _convolution                 47.37 Mb         0 b              20
# mkldnn_convolution           47.37 Mb         0 b              20
# empty_like                   47.37 Mb         0 b              20
# max_pool2d                   11.48 Mb         0 b              1
# max_pool2d_with_indices      11.48 Mb         0 b              1
# resize_                      11.48 Mb         11.48 Mb         2
# addmm                        19.53 Kb         19.53 Kb         1
# adaptive_avg_pool2d          10.00 Kb         0 b              1
# mean                         10.00 Kb         0 b              1
# ---------------------------  ---------------  ---------------  ---------------

######################################################################
# 5. Using tracing functionality
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Profiling results can be outputted as a .json trace file:

with profiler.profile() as prof:
    with profiler.record_function("model_inference"):
        model(inputs)

prof.export_chrome_trace("trace.json")

######################################################################
# User can examine the sequence of profiled operators after loading the trace file
# in Chrome (``chrome://tracing``):
#
# .. image:: ../../_static/img/trace_img.png
#    :scale: 25 %

######################################################################
# Learn More
# ----------
#
# Take a look at the following recipes/tutorials to continue your learning:
#
# -  `PyTorch Benchmark <https://pytorch.org/tutorials/recipes/recipes/benchmark.html>`_
# -  `Visualizing models, data, and training with TensorBoard <https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html>`_ tutorial
#
