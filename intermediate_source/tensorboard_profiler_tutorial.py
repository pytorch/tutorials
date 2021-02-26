"""
PyTorch TensorBoard Profiler
====================================
This recipe explains how to use PyTorch TensorBoard Profiler
and measure the performance bottleneck of the model.

.. note::
    PyTorch 1.8 introduces the new API that will replace the older profiler API
    in the future releases. Check the new API at `this page <https://pytorch.org/docs/master/profiler.html>`__.

Introduction
------------
PyTorch 1.8 includes an updated profiler API that could help user
record both the operators running on CPU side and the CUDA kernels running on GPU side.
Given the profiling information,
we can use this TensorBoard Plugin to visualize it and analyze the performance bottleneck.

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
# 1. Prepare the data and model
# 2. Use profiler to record execution events
# 3. Run the profiler
# 4. Use TensorBoard to view and analyze performance
# 5. Improve performance with the help of profiler
#
# 1. Prepare the data and model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Firstly, let’s import all necessary libraries:
#

import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

######################################################################
# Then prepare the input data. For this tutorial, we use the CIFAR10 dataset.
# We transform it to desired format and use DataLoader to load each batch.

transform = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True) # num_workers=4

######################################################################
# Let’s create an instance of a Resnet model, an instance of loss, and an instance of optimizer.
# To run on GPU, we put model and loss to GPU device.

device = torch.device("cuda:0")
model = torchvision.models.resnet18(pretrained=True).cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()


######################################################################
# We define the training step for each batch of input data.

def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


######################################################################
# 2. Use profiler to record execution events
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The profiler is enabled through the context manager and accepts a number of parameters,
# some of the most useful are:
#
# - ``schedule`` - callable that takes step (int) as a single parameter
#   and returns the profiler action to perform at each step;
#   In this example with wait=1, warmup=1, active=5,
#   profiler will skip the first step/iteration,
#   start warming up on the second,
#   record the following five iterations,
#   after which the trace will become available and on_trace_ready (when set) is called;
#   The cycle repeats starting with the next step until the loop exits.
#   During ``wait`` steps, the profiler does not work.
#   During ``warmup`` steps, the profiler starts profiling as warmup but does not record any events.
#   This is for reducing the profiling overhead.
#   The overhead at the beginning of profiling is high and easy to bring skew to the profiling result.
#   During ``active`` steps, the profiler works and record events.
# - ``on_trace_ready`` - callable that is called at the end of each cycle;
#   In this example we use ``torch.profiler.tensorboard_trace_handler`` to generate result files for TensorBoard.
#   After profiling, result files can be generated in the ``./log/resnet18`` directory,
#   which could be specified to open and analyzed in TensorBoard.
# - ``record_shapes`` - whether to record shapes of the operator inputs.

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        if step >= 7:
            break
        train(batch_data)
        prof.step()


######################################################################
# 3. Run the profiler
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Run the above code. The profiling result will be saved under ``./log`` directory.


######################################################################
# 4. Use TensorBoard to view and analyze performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This requires the latest versions of PyTorch TensorBoard Profiler.
#
# ::
#
#     pip install torch_tb_profiler
#

######################################################################
# Launch the TensorBoard Profiler.
#
# ::
#
#     tensorboard --logdir=./log
#

######################################################################
# Open the TensorBoard profile URL in Google Chrome browser or Microsoft Edge browser.
#
# ::
#
#     http://localhost:6006/#torch_profiler
#

######################################################################
# The profiler’s front page is as below.
#
# .. image:: ../../_static/img/profiler_overview1.png
#    :scale: 25 %
#
# This overview shows a high-level summary of performance.
#
# The "Step Time Breakdown" break the time spent on each step into multiple categories.
# In this example, you can see the ``DataLoader`` costs a lot of time.
#
# The bottom "Performance Recommendation" leverages the profiling result
# to automatically highlight likely bottlenecks,
# and gives you actionable optimization suggestions.
#
# You can change the view page in left "Views" dropdown list.
#
# .. image:: ../../_static/img/profiler_views_list.png
#    :alt:
#
# The operator view displays the performance of every PyTorch operator
# that is executed either on the host or device.
#
# The GPU kernel view shows all kernels’ time spent on GPU.
#
# The trace view shows timeline of profiled operators and GPU kernels.
# You can select it to see detail as below.
#
# .. image:: ../../_static/img/profiler_trace_view1.png
#    :scale: 25 %
#
# You can move the graph and zoom in/out with the help of right side toolbar.
#
# In this example, we can see the event prefixed with ``enumerate(DataLoader)`` costs a lot of time.
# And during most of this period, the GPU is idle.
# Because this function is loading data and transforming data on host side,
# during which the GPU resource is wasted.


######################################################################
# 5. Improve performance with the help of profiler
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The PyTorch DataLoader uses single process by default.
# User could enable multi-process data loading by setting the parameter ``num_workers``.
# `Here <https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading>`_ is more details.
#
# In this example, we can set ``num_workers`` as below,
# pass a different name such as ``./log/resnet18_4workers`` to tensorboard_trace_handler, and run it again.
#
# ::
#
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
#

######################################################################
# Then let’s choose the just profiled run in left "Runs" dropdown list.
#
# .. image:: ../../_static/img/profiler_overview2.png
#    :scale: 25 %
#
# From the above view, we can find the step time is reduced,
# and the time reduction of ``DataLoader`` mainly contributes.
#
# .. image:: ../../_static/img/profiler_trace_view2.png
#    :scale: 25 %
#
# From the above view, we can find the event of ``enumerate(DataLoader)`` is shortened,
# and the GPU utilization is increased.

######################################################################
# Learn More
# ----------
#
# Take a look at the following recipes/tutorials to continue your learning:
#
# -  `Pytorch TensorBoard Profiler github <https://github.com/pytorch/kineto/tree/master/tb_plugin>`_
# -  `Pytorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_
# -  `Profiling Your Pytorch Module <https://pytorch.org/tutorials/beginner/profiler.html>`_ tutorial
