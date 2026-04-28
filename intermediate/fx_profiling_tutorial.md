Note

Go to the end
to download the full example code.

# (beta) Building a Simple CPU Performance Profiler with FX

**Author**: [James Reed](https://github.com/jamesr66a)

In this tutorial, we are going to use FX to do the following:

1. Capture PyTorch Python code in a way that we can inspect and gather
statistics about the structure and execution of the code
2. Build out a small class that will serve as a simple performance "profiler",
collecting runtime statistics about each part of the model from actual
runs.

For this tutorial, we are going to use the torchvision ResNet18 model
for demonstration purposes.

Now that we have our model, we want to inspect deeper into its
performance. That is, for the following invocation, which parts
of the model are taking the longest?

A common way of answering that question is to go through the program
source, add code that collects timestamps at various points in the
program, and compare the difference between those timestamps to see
how long the regions between the timestamps take.

That technique is certainly applicable to PyTorch code, however it
would be nicer if we didn't have to copy over model code and edit it,
especially code we haven't written (like this torchvision model).
Instead, we are going to use FX to automate this "instrumentation"
process without needing to modify any source.

First, let's get some imports out of the way (we will be using all
of these later in the code).

Note

`tabulate` is an external library that is not a dependency of PyTorch.
We will be using it to more easily visualize performance data. Please
make sure you've installed it from your favorite Python package source.

## Capturing the Model with Symbolic Tracing

Next, we are going to use FX's symbolic tracing mechanism to capture
the definition of our model in a data structure we can manipulate
and examine.

This gives us a Graph representation of the ResNet18 model. A Graph
consists of a series of Nodes connected to each other. Each Node
represents a call-site in the Python code (whether to a function,
a module, or a method) and the edges (represented as `args` and `kwargs`
on each node) represent the values passed between these call-sites. More
information about the Graph representation and the rest of FX's APIs ca
be found at the FX documentation [https://pytorch.org/docs/master/fx.html](https://pytorch.org/docs/master/fx.html).

## Creating a Profiling Interpreter

Next, we are going to create a class that inherits from `torch.fx.Interpreter`.
Though the `GraphModule` that `symbolic_trace` produces compiles Python code
that is run when you call a `GraphModule`, an alternative way to run a
`GraphModule` is by executing each `Node` in the `Graph` one by one. That is
the functionality that `Interpreter` provides: It interprets the graph node-
by-node.

By inheriting from `Interpreter`, we can override various functionality and
install the profiling behavior we want. The goal is to have an object to which
we can pass a model, invoke the model 1 or more times, then get statistics about
how long the model and each part of the model took during those runs.

Let's define our `ProfilingInterpreter` class:

Note

We use Python's `time.time` function to pull wall clock
timestamps and compare them. This is not the most accurate
way to measure performance, and will only give us a first-
order approximation. We use this simple technique only for the
purpose of demonstration in this tutorial.

## Investigating the Performance of ResNet18

We can now use `ProfilingInterpreter` to inspect the performance
characteristics of our ResNet18 model;

There are two things we should call out here:

- `MaxPool2d` takes up the most time. This is a known issue:
[pytorch/pytorch#51393](https://github.com/pytorch/pytorch/issues/51393)

## Conclusion

As we can see, using FX we can easily capture PyTorch programs (even
ones we don't have the source code for!) in a machine-interpretable
format and use that for analysis, such as the performance analysis
we've done here. FX opens up an exciting world of possibilities for
working with PyTorch programs.

Finally, since FX is still in beta, we would be happy to hear any
feedback you have about using it. Please feel free to use the
PyTorch Forums ([https://discuss.pytorch.org/](https://discuss.pytorch.org/)) and the issue tracker
([pytorch/pytorch#issues](https://github.com/pytorch/pytorch/issues)) to provide any feedback
you might have.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: fx_profiling_tutorial.ipynb`](../_downloads/945dab6b984b8789385e32187d4a8964/fx_profiling_tutorial.ipynb)

[`Download Python source code: fx_profiling_tutorial.py`](../_downloads/8c575aa36ad9a61584ec0ddf11cbe84d/fx_profiling_tutorial.py)

[`Download zipped: fx_profiling_tutorial.zip`](../_downloads/b6301a206807fcf9ca7e4168fdc08e4a/fx_profiling_tutorial.zip)