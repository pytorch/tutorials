# -*- coding: utf-8 -*-
"""
Learning Hybrid Frontend Syntax Through Example
===============================================
**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`_

This document is meant to highlight the syntax of the Hybrid Frontend
through a non-code intensive example. The Hybrid Frontend is one of the
new shiny features of Pytorch 1.0 and provides an avenue for developers
to transition their models from **eager** to **graph** mode. PyTorch
users are very familiar with eager mode as it provides the ease-of-use
and flexibility that we all enjoy as researchers. Caffe2 users are more
aquainted with graph mode which has the benefits of speed, optimization
opportunities, and functionality in C++ runtime environments. The hybrid
frontend bridges the gap between the the two modes by allowing
researchers to develop and refine their models in eager mode (i.e.
PyTorch), then gradually transition the proven model to graph mode for
production, when speed and resouce consumption become critical.

Hybrid Frontend Information
---------------------------

The process for transitioning a model to graph mode is as follows.
First, the developer constructs, trains, and tests the model in eager
mode. Then they incrementally **trace** and **script** each
function/module of the model with the Just-In-Time (JIT) compiler, at
each step verifying that the output is correct. Finally, when each of
the components of the top-level model have been traced and scripted, the
model itself is traced. At which point the model has been transitioned
to graph mode, and has a complete python-free representation. With this
representation, the model runtime can take advantage of high-performance
Caffe2 operators and graph based optimizations.

Before we continue, it is important to understand the idea of tracing
and scripting, and why they are separate. The goal of **trace** and
**script** is the same, and that is to create a graph representation of
the operations taking place in a given function. The discrepency comes
from the flexibility of eager mode that allows for **data-dependent
control flows** within the model architecture. When a function does NOT
have a data-dependent control flow, it may be *traced* with
``torch.jit.trace``. However, when the function *has* a data-dependent
control flow it must be *scripted* with ``torch.jit.script``. We will
leave the details of the interworkings of the hybrid frontend for
another document, but the code example below will show the syntax of how
to trace and script different pure python functions and torch Modules.
Hopefully, you will find that using the hybrid frontend is non-intrusive
as it mostly involves adding decorators to the existing function and
class definitions.

Motivating Example
------------------

In this example we will implement a strange math function that may be
logically broken up into four parts that do, and do not contain
data-dependent control flows. The purpose here is to show a non-code
intensive example where the use of the JIT is highlighted. This example
is a stand-in representation of a useful model, whose implementation has
been divided into various pure python functions and modules.

The function we seek to implement, :math:`Y(x)`, is defined for
:math:`x \epsilon \mathbb{N}` as

.. math::

    z(x) = \Biggl \lfloor \\frac{\sqrt{\prod_{i=1}^{|2 x|}i}}{5} \Biggr \\rfloor

.. math::

    Y(x) = \\begin{cases}
      \\frac{z(x)}{2}  &  \\text{if } z(x)\%2 == 0, \\\\
      z(x)             &  \\text{otherwise}
    \end{cases}

.. math::

    \\begin{array}{| r  | r |} \hline
    x &1 &2 &3 &4 &5 &6 &7 \\\\ \hline
    Y(x) &0 &0 &-5 &20 &190 &-4377 &-59051 \\\\ \hline
    \end{array}

As mentioned, the computation is split into four parts. Part one is the
simple tensor calculation of :math:`|2x|`, which can be traced. Part two
is the iterative product calculation that represents a data dependent
control flow to be scripted (the number of loop iteration depends on the
input at runtime). Part three is a trace-able
:math:`\lfloor \sqrt{a/5} \\rfloor` calculation. Finally, part 4 handles
the output cases depending on the value of :math:`z(x)` and must be
scripted due to the data dependency. Now, let's see how this looks in
code.

Part 1 - Tracing a pure python function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can implement part one as a pure python function as below. Notice, to
trace this function we call ``torch.jit.trace`` and pass in the function
to be traced. Since the trace requires a dummy input of the expected
runtime type and shape, we also include the ``torch.rand`` to generate a
single valued torch tensor.

"""

import torch

def fn(x):
    return torch.abs(2*x)

# This is how you define a traced function
# Pass in both the function to be traced and an example input to ``torch.jit.trace``
traced_fn = torch.jit.trace(fn, torch.rand(()))

######################################################################
# Part 2 - Scripting a pure python function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can also implement part 2 as a pure python function where we
# iteratively compute the product. Since the number of iterations depends
# on the value of the input, we have a data dependent control flow, so the
# function must be scripted. We can script python functions simply with
# the ``@torch.jit.script`` decorator.
#

# This is how you define a script function
# Apply this decorator directly to the function
@torch.jit.script
def script_fn(x):
    z = torch.ones([1], dtype=torch.int64)
    for i in range(int(x)):
        z = z * (i + 1)
    return z


######################################################################
# Part 3 - Tracing a nn.Module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we will implement part 3 of the computation within the forward
# function of a ``torch.nn.Module``. This module may be traced, but rather
# than adding a decorator here, we will handle the tracing where the
# Module is constructed. Thus, the class definition is not changed at all.
#

# This is a normal module that can be traced.
class TracedModule(torch.nn.Module):
    def forward(self, x):
        x = x.type(torch.float32)
        return torch.floor(torch.sqrt(x) / 5.)


######################################################################
# Part 4 - Scripting a nn.Module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the final part of the computation we have a ``torch.nn.Module`` that
# must be scripted. To accomodate this, we inherit from
# ``torch.jit.ScriptModule`` and add the ``@torch.jit.script_method``
# decorator to the forward function.
#

# This is how you define a scripted module.
# The module should inherit from ScriptModule and the forward should have the
#   script_method decorator applied to it.
class ScriptModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x):
        r = -x
        if int(torch.fmod(x, 2.0)) == 0.0:
            r = x / 2.0
        return r


######################################################################
# Top-Level Module
# ~~~~~~~~~~~~~~~~
#
# Now we will put together the pieces of the computation via a top level
# module called ``Net``. In the constructor, we will instantiate the
# ``TracedModule`` and ``ScriptModule`` as attributes. This must be done
# because we ultimately want to trace/script the top level module, and
# having the traced/scripted modules as attributes allows the Net to
# inherit the required submodules' parameters. Notice, this is where we
# actually trace the ``TracedModule`` by calling ``torch.jit.trace()`` and
# providing the necessary dummy input. Also notice that the
# ``ScriptModule`` is constructed as normal because we handled the
# scripting in the class definition.
#
# Here we can also print the graphs created for each individual part of
# the computation. The printed graphs allows us to see how the JIT
# ultimately interpreted the functions as graph computations.
#
# Finally, we define the ``forward`` function for the Net module where we
# run the input data ``x`` through the four parts of the computation.
# There is no strange syntax here and we call the traced and scripted
# modules and functions as expected.
#

# This is a demonstration net that calls all of the different types of
# methods and functions
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Modules must be attributes on the Module because if you want to trace
        # or script this Module, we must be able to inherit the submodules'
        # params.
        self.traced_module = torch.jit.trace(TracedModule(), torch.rand(()))
        self.script_module = ScriptModule()

        print('traced_fn graph', traced_fn.graph)
        print('script_fn graph', script_fn.graph)
        print('TracedModule graph', self.traced_module.__getattr__('forward').graph)
        print('ScriptModule graph', self.script_module.__getattr__('forward').graph)

    def forward(self, x):
        # Call a traced function
        x = traced_fn(x)

        # Call a script function
        x = script_fn(x)

        # Call a traced submodule
        x = self.traced_module(x)

        # Call a scripted submodule
        x = self.script_module(x)

        return x


######################################################################
# Running the Model
# ~~~~~~~~~~~~~~~~~
#
# All that's left to do is construct the Net and compute the output
# through the forward function. Here, we use :math:`x=5` as the test input
# value and expect :math:`Y(x)=190.` Also, check out the graphs that were
# printed during the construction of the Net.
#

# Instantiate this net and run it
n = Net()
print(n(torch.tensor([5]))) # 190.


######################################################################
# Tracing the Top-Level Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The last part of the example is to trace the top-level module, ``Net``.
# As mentioned previously, since the traced/scripted modules are
# attributes of Net, we are able to trace ``Net`` as it inherits the
# parameters of the traced/scripted submodules. Note, the syntax for
# tracing Net is identical to the syntax for tracing ``TracedModule``.
# Also, check out the graph that is created.
#

n_traced = torch.jit.trace(n, torch.tensor([5]))
print(n_traced(torch.tensor([5])))
print('n_traced graph', n_traced.graph)


######################################################################
# Hopefully, this document can serve as an introduction to the hybrid
# frontend as well as a syntax reference guide for more experienced users.
# Also, there are a few things to keep in mind when using the hybrid
# frontend. There is a constraint that traced/scripted methods must be
# written in a restricted subset of python, as features like generators,
# defs, and Python data structures are not supported. As a workaround, the
# scripting model *is* designed to work with both traced and non-traced
# code which means you can call non-traced code from traced functions.
# However, such a model may not be exported to ONNX.
#
