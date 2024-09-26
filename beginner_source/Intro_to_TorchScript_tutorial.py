"""
Introduction to TorchScript
===========================

**Authors:** James Reed (jamesreed@fb.com), Michael Suo (suo@fb.com), rev2

This tutorial is an introduction to TorchScript, an intermediate
representation of a PyTorch model (subclass of ``nn.Module``) that
can then be run in a high-performance environment such as C++.

In this tutorial we will cover:

1. The basics of model authoring in PyTorch, including:

-  Modules
-  Defining ``forward`` functions
-  Composing modules into a hierarchy of modules

2. Specific methods for converting PyTorch modules to TorchScript, our
   high-performance deployment runtime

-  Tracing an existing module
-  Using scripting to directly compile a module
-  How to compose both approaches
-  Saving and loading TorchScript modules

We hope that after you complete this tutorial, you will proceed to go through
`the follow-on tutorial <https://pytorch.org/tutorials/advanced/cpp_export.html>`_
which will walk you through an example of actually calling a TorchScript
model from C++.

"""

import torch  # This is all you need to use both PyTorch and TorchScript!
print(torch.__version__)
torch.manual_seed(191009)  # set the seed for reproducibility


######################################################################
# Basics of PyTorch Model Authoring
# ---------------------------------
#
# Let’s start out by defining a simple ``Module``. A ``Module`` is the
# basic unit of composition in PyTorch. It contains:
#
# 1. A constructor, which prepares the module for invocation
# 2. A set of ``Parameters`` and sub-\ ``Modules``. These are initialized
#    by the constructor and can be used by the module during invocation.
# 3. A ``forward`` function. This is the code that is run when the module
#    is invoked.
#
# Let’s examine a small example:
#

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h

my_cell = MyCell()
x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))


######################################################################
# So we’ve:
#
# 1. Created a class that subclasses ``torch.nn.Module``.
# 2. Defined a constructor. The constructor doesn’t do much, just calls
#    the constructor for ``super``.
# 3. Defined a ``forward`` function, which takes two inputs and returns
#    two outputs. The actual contents of the ``forward`` function are not
#    really important, but it’s sort of a fake `RNN
#    cell <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__–that
#    is–it’s a function that is applied on a loop.
#
# We instantiated the module, and made ``x`` and ``h``, which are just 3x4
# matrices of random values. Then we invoked the cell with
# ``my_cell(x, h)``. This in turn calls our ``forward`` function.
#
# Let’s do something a little more interesting:
#

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))


######################################################################
# We’ve redefined our module ``MyCell``, but this time we’ve added a
# ``self.linear`` attribute, and we invoke ``self.linear`` in the forward
# function.
#
# What exactly is happening here? ``torch.nn.Linear`` is a ``Module`` from
# the PyTorch standard library. Just like ``MyCell``, it can be invoked
# using the call syntax. We are building a hierarchy of ``Module``\ s.
#
# ``print`` on a ``Module`` will give a visual representation of the
# ``Module``\ ’s subclass hierarchy. In our example, we can see our
# ``Linear`` subclass and its parameters.
#
# By composing ``Module``\ s in this way, we can succinctly and readably
# author models with reusable components.
#
# You may have noticed ``grad_fn`` on the outputs. This is a detail of
# PyTorch’s method of automatic differentiation, called
# `autograd <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`__.
# In short, this system allows us to compute derivatives through
# potentially complex programs. The design allows for a massive amount of
# flexibility in model authoring.
#
# Now let’s examine said flexibility:
#

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.dg = MyDecisionGate()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))


######################################################################
# We’ve once again redefined our ``MyCell`` class, but here we’ve defined
# ``MyDecisionGate``. This module utilizes **control flow**. Control flow
# consists of things like loops and ``if``-statements.
#
# Many frameworks take the approach of computing symbolic derivatives
# given a full program representation. However, in PyTorch, we use a
# gradient tape. We record operations as they occur, and replay them
# backwards in computing derivatives. In this way, the framework does not
# have to explicitly define derivatives for all constructs in the
# language.
#
# .. figure:: https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif
#    :alt: How autograd works
#
#    How autograd works
#


######################################################################
# Basics of TorchScript
# ---------------------
#
# Now let’s take our running example and see how we can apply TorchScript.
#
# In short, TorchScript provides tools to capture the definition of your
# model, even in light of the flexible and dynamic nature of PyTorch.
# Let’s begin by examining what we call **tracing**.
#
# Tracing ``Modules``
# ~~~~~~~~~~~~~~~~~~~
#

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
traced_cell(x, h)


######################################################################
# We’ve rewinded a bit and taken the second version of our ``MyCell``
# class. As before, we’ve instantiated it, but this time, we’ve called
# ``torch.jit.trace``, passed in the ``Module``, and passed in *example
# inputs* the network might see.
#
# What exactly has this done? It has invoked the ``Module``, recorded the
# operations that occurred when the ``Module`` was run, and created an
# instance of ``torch.jit.ScriptModule`` (of which ``TracedModule`` is an
# instance)
#
# TorchScript records its definitions in an Intermediate Representation
# (or IR), commonly referred to in Deep learning as a *graph*. We can
# examine the graph with the ``.graph`` property:
#

print(traced_cell.graph)


######################################################################
# However, this is a very low-level representation and most of the
# information contained in the graph is not useful for end users. Instead,
# we can use the ``.code`` property to give a Python-syntax interpretation
# of the code:
#

print(traced_cell.code)


######################################################################
# So **why** did we do all this? There are several reasons:
#
# 1. TorchScript code can be invoked in its own interpreter, which is
#    basically a restricted Python interpreter. This interpreter does not
#    acquire the Global Interpreter Lock, and so many requests can be
#    processed on the same instance simultaneously.
# 2. This format allows us to save the whole model to disk and load it
#    into another environment, such as in a server written in a language
#    other than Python
# 3. TorchScript gives us a representation in which we can do compiler
#    optimizations on the code to provide more efficient execution
# 4. TorchScript allows us to interface with many backend/device runtimes
#    that require a broader view of the program than individual operators.
#
# We can see that invoking ``traced_cell`` produces the same results as
# the Python module:
#

print(my_cell(x, h))
print(traced_cell(x, h))


######################################################################
# Using Scripting to Convert Modules
# ----------------------------------
#
# There’s a reason we used version two of our module, and not the one with
# the control-flow-laden submodule. Let’s examine that now:
#

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell.dg.code)
print(traced_cell.code)


######################################################################
# Looking at the ``.code`` output, we can see that the ``if-else`` branch
# is nowhere to be found! Why? Tracing does exactly what we said it would:
# run the code, record the operations *that happen* and construct a
# ``ScriptModule`` that does exactly that. Unfortunately, things like control
# flow are erased.
#
# How can we faithfully represent this module in TorchScript? We provide a
# **script compiler**, which does direct analysis of your Python source
# code to transform it into TorchScript. Let’s convert ``MyDecisionGate``
# using the script compiler:
#

scripted_gate = torch.jit.script(MyDecisionGate())

my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

print(scripted_gate.code)
print(scripted_cell.code)


######################################################################
# Hooray! We’ve now faithfully captured the behavior of our program in
# TorchScript. Let’s now try running the program:
#

# New inputs
x, h = torch.rand(3, 4), torch.rand(3, 4)
print(scripted_cell(x, h))


######################################################################
# Mixing Scripting and Tracing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Some situations call for using tracing rather than scripting (e.g. a
# module has many architectural decisions that are made based on constant
# Python values that we would like to not appear in TorchScript). In this
# case, scripting can be composed with tracing: ``torch.jit.script`` will
# inline the code for a traced module, and tracing will inline the code
# for a scripted module.
#
# An example of the first case:
#

class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)



######################################################################
# And an example of the second case:
#

class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)


######################################################################
# This way, scripting and tracing can be used when the situation calls for
# each of them and used together.
#
# Saving and Loading models
# -------------------------
#
# We provide APIs to save and load TorchScript modules to/from disk in an
# archive format. This format includes code, parameters, attributes, and
# debug information, meaning that the archive is a freestanding
# representation of the model that can be loaded in an entirely separate
# process. Let’s save and load our wrapped RNN module:
#

traced.save('wrapped_rnn.pt')

loaded = torch.jit.load('wrapped_rnn.pt')

print(loaded)
print(loaded.code)


######################################################################
# As you can see, serialization preserves the module hierarchy and the
# code we’ve been examining throughout. The model can also be loaded, for
# example, `into
# C++ <https://pytorch.org/tutorials/advanced/cpp_export.html>`__ for
# python-free execution.
#
# Further Reading
# ~~~~~~~~~~~~~~~
#
# We’ve completed our tutorial! For a more involved demonstration, check
# out the NeurIPS demo for converting machine translation models using
# TorchScript:
# https://colab.research.google.com/drive/1HiICg6jRkBnr5hvK2-VnMi88Vi9pUzEJ
#
