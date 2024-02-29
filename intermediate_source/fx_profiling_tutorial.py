# -*- coding: utf-8 -*-
"""
(beta) Building a Simple CPU Performance Profiler with FX
*********************************************************
**Author**: `James Reed <https://github.com/jamesr66a>`_

In this tutorial, we are going to use FX to do the following:

1) Capture PyTorch Python code in a way that we can inspect and gather
   statistics about the structure and execution of the code
2) Build out a small class that will serve as a simple performance "profiler",
   collecting runtime statistics about each part of the model from actual
   runs.

"""

######################################################################
# For this tutorial, we are going to use the torchvision ResNet18 model
# for demonstration purposes.

import torch
import torch.fx
import torchvision.models as models

rn18 = models.resnet18()
rn18.eval()

######################################################################
# Now that we have our model, we want to inspect deeper into its
# performance. That is, for the following invocation, which parts
# of the model are taking the longest?
input = torch.randn(5, 3, 224, 224)
output = rn18(input)

######################################################################
# A common way of answering that question is to go through the program
# source, add code that collects timestamps at various points in the
# program, and compare the difference between those timestamps to see
# how long the regions between the timestamps take.
#
# That technique is certainly applicable to PyTorch code, however it
# would be nicer if we didn't have to copy over model code and edit it,
# especially code we haven't written (like this torchvision model).
# Instead, we are going to use FX to automate this "instrumentation"
# process without needing to modify any source.

######################################################################
# First, let's get some imports out of the way (we will be using all
# of these later in the code).

import statistics, tabulate, time
from typing import Any, Dict, List
from torch.fx import Interpreter

######################################################################
# .. note::
#     ``tabulate`` is an external library that is not a dependency of PyTorch.
#     We will be using it to more easily visualize performance data. Please
#     make sure you've installed it from your favorite Python package source.

######################################################################
# Capturing the Model with Symbolic Tracing
# -----------------------------------------
# Next, we are going to use FX's symbolic tracing mechanism to capture
# the definition of our model in a data structure we can manipulate
# and examine.

traced_rn18 = torch.fx.symbolic_trace(rn18)
print(traced_rn18.graph)

######################################################################
# This gives us a Graph representation of the ResNet18 model. A Graph
# consists of a series of Nodes connected to each other. Each Node
# represents a call-site in the Python code (whether to a function,
# a module, or a method) and the edges (represented as ``args`` and ``kwargs``
# on each node) represent the values passed between these call-sites. More
# information about the Graph representation and the rest of FX's APIs ca
# be found at the FX documentation https://pytorch.org/docs/master/fx.html.


######################################################################
# Creating a Profiling Interpreter
# --------------------------------
# Next, we are going to create a class that inherits from ``torch.fx.Interpreter``.
# Though the ``GraphModule`` that ``symbolic_trace`` produces compiles Python code
# that is run when you call a ``GraphModule``, an alternative way to run a
# ``GraphModule`` is by executing each ``Node`` in the ``Graph`` one by one. That is
# the functionality that ``Interpreter`` provides: It interprets the graph node-
# by-node.
#
# By inheriting from ``Interpreter``, we can override various functionality and
# install the profiling behavior we want. The goal is to have an object to which
# we can pass a model, invoke the model 1 or more times, then get statistics about
# how long the model and each part of the model took during those runs.
#
# Let's define our ``ProfilingInterpreter`` class:

class ProfilingInterpreter(Interpreter):
    def __init__(self, mod : torch.nn.Module):
        # Rather than have the user symbolically trace their model,
        # we're going to do it in the constructor. As a result, the
        # user can pass in any ``Module`` without having to worry about
        # symbolic tracing APIs
        gm = torch.fx.symbolic_trace(mod)
        super().__init__(gm)

        # We are going to store away two things here:
        #
        # 1. A list of total runtimes for ``mod``. In other words, we are
        #    storing away the time ``mod(...)`` took each time this
        #    interpreter is called.
        self.total_runtime_sec : List[float] = []
        # 2. A map from ``Node`` to a list of times (in seconds) that
        #    node took to run. This can be seen as similar to (1) but
        #    for specific sub-parts of the model.
        self.runtimes_sec : Dict[torch.fx.Node, List[float]] = {}

    ######################################################################
    # Next, let's override our first method: ``run()``. ``Interpreter``'s ``run``
    # method is the top-level entry point for execution of the model. We will
    # want to intercept this so that we can record the total runtime of the
    # model.

    def run(self, *args) -> Any:
        # Record the time we started running the model
        t_start = time.time()
        # Run the model by delegating back into Interpreter.run()
        return_val = super().run(*args)
        # Record the time we finished running the model
        t_end = time.time()
        # Store the total elapsed time this model execution took in the
        # ``ProfilingInterpreter``
        self.total_runtime_sec.append(t_end - t_start)
        return return_val

    ######################################################################
    # Now, let's override ``run_node``. ``Interpreter`` calls ``run_node`` each
    # time it executes a single node. We will intercept this so that we
    # can measure and record the time taken for each individual call in
    # the model.

    def run_node(self, n : torch.fx.Node) -> Any:
        # Record the time we started running the op
        t_start = time.time()
        # Run the op by delegating back into Interpreter.run_node()
        return_val = super().run_node(n)
        # Record the time we finished running the op
        t_end = time.time()
        # If we don't have an entry for this node in our runtimes_sec
        # data structure, add one with an empty list value.
        self.runtimes_sec.setdefault(n, [])
        # Record the total elapsed time for this single invocation
        # in the runtimes_sec data structure
        self.runtimes_sec[n].append(t_end - t_start)
        return return_val

    ######################################################################
    # Finally, we are going to define a method (one which doesn't override
    # any ``Interpreter`` method) that provides us a nice, organized view of
    # the data we have collected.

    def summary(self, should_sort : bool = False) -> str:
        # Build up a list of summary information for each node
        node_summaries : List[List[Any]] = []
        # Calculate the mean runtime for the whole network. Because the
        # network may have been called multiple times during profiling,
        # we need to summarize the runtimes. We choose to use the
        # arithmetic mean for this.
        mean_total_runtime = statistics.mean(self.total_runtime_sec)

        # For each node, record summary statistics
        for node, runtimes in self.runtimes_sec.items():
            # Similarly, compute the mean runtime for ``node``
            mean_runtime = statistics.mean(runtimes)
            # For easier understanding, we also compute the percentage
            # time each node took with respect to the whole network.
            pct_total = mean_runtime / mean_total_runtime * 100
            # Record the node's type, name of the node, mean runtime, and
            # percent runtime.
            node_summaries.append(
                [node.op, str(node), mean_runtime, pct_total])

        # One of the most important questions to answer when doing performance
        # profiling is "Which op(s) took the longest?". We can make this easy
        # to see by providing sorting functionality in our summary view
        if should_sort:
            node_summaries.sort(key=lambda s: s[2], reverse=True)

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers : List[str] = [
            'Op type', 'Op', 'Average runtime (s)', 'Pct total runtime'
        ]
        return tabulate.tabulate(node_summaries, headers=headers)

######################################################################
# .. note::
#       We use Python's ``time.time`` function to pull wall clock
#       timestamps and compare them. This is not the most accurate
#       way to measure performance, and will only give us a first-
#       order approximation. We use this simple technique only for the
#       purpose of demonstration in this tutorial.

######################################################################
# Investigating the Performance of ResNet18
# -----------------------------------------
# We can now use ``ProfilingInterpreter`` to inspect the performance
# characteristics of our ResNet18 model;

interp = ProfilingInterpreter(rn18)
interp.run(input)
print(interp.summary(True))

######################################################################
# There are two things we should call out here:
#
# * ``MaxPool2d`` takes up the most time. This is a known issue:
#   https://github.com/pytorch/pytorch/issues/51393
# * BatchNorm2d also takes up significant time. We can continue this
#   line of thinking and optimize this in the Conv-BN Fusion with FX
#   `tutorial <https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html>`_. 
#
#
# Conclusion
# ----------
# As we can see, using FX we can easily capture PyTorch programs (even
# ones we don't have the source code for!) in a machine-interpretable
# format and use that for analysis, such as the performance analysis
# we've done here. FX opens up an exciting world of possibilities for
# working with PyTorch programs.
#
# Finally, since FX is still in beta, we would be happy to hear any
# feedback you have about using it. Please feel free to use the
# PyTorch Forums (https://discuss.pytorch.org/) and the issue tracker
# (https://github.com/pytorch/pytorch/issues) to provide any feedback
# you might have.
