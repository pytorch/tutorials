Note

Go to the end
to download the full example code.

# Introduction to `torch.compile`

**Author:** William Wen

`torch.compile` is the new way to speed up your PyTorch code!
`torch.compile` makes PyTorch code run faster by
JIT-compiling PyTorch code into optimized kernels,
while requiring minimal code changes.

`torch.compile` accomplishes this by tracing through
your Python code, looking for PyTorch operations.
Code that is difficult to trace will result a
**graph break**, which are lost optimization opportunities, rather
than errors or silent incorrectness.

`torch.compile` is available in PyTorch 2.0 and later.

This introduction covers basic `torch.compile` usage
and demonstrates the advantages of `torch.compile` over
our previous PyTorch compiler solution,
[TorchScript](https://pytorch.org/docs/stable/jit.html).

For an end-to-end example on a real model, check out our [end-to-end torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_full_example.html).

To troubleshoot issues and to gain a deeper understanding of how to apply `torch.compile` to your code, check out [the torch.compile programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html).

**Contents**

**Required pip dependencies for this tutorial**

- `torch >= 2.0`
- `numpy`
- `scipy`

**System requirements**
- A C++ compiler, such as `g++`
- Python development package (`python-devel`/`python-dev`)

## Basic Usage

We turn on some logging to help us to see what `torch.compile` is doing
under the hood in this tutorial.
The following code will print out the PyTorch ops that `torch.compile` traced.

```

```

`torch.compile` is a decorator that takes an arbitrary Python function.

`torch.compile` is applied recursively, so nested function calls
within the top-level compiled function will also be compiled.

We can also optimize `torch.nn.Module` instances by either calling
its `.compile()` method or by directly `torch.compile`-ing the module.
This is equivalent to `torch.compile`-ing the module's `__call__` method
(which indirectly calls `forward`).

## Demonstrating Speedups

Now let's demonstrate how `torch.compile` speeds up a simple PyTorch example.
For a demonstration on a more complex model, see our [end-to-end torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_full_example.html).

```
# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
```

Notice that `torch.compile` appears to take a lot longer to complete
compared to eager. This is because `torch.compile` takes extra time to compile
the model on the first few executions.
`torch.compile` re-uses compiled code whever possible,
so if we run our optimized model several more times, we should
see a significant improvement compared to eager.

```
# turn off logging for now to prevent spam
```

And indeed, we can see that running our model with `torch.compile`
results in a significant speedup. Speedup mainly comes from reducing Python overhead and
GPU read/writes, and so the observed speedup may vary on factors such as model
architecture and batch size. For example, if a model's architecture is simple
and the amount of data is large, then the bottleneck would be
GPU compute and the observed speedup may be less significant.

To see speedups on a real model, check out our [end-to-end torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_full_example.html).

## Benefits over TorchScript

Why should we use `torch.compile` over TorchScript? Primarily, the
advantage of `torch.compile` lies in its ability to handle
arbitrary Python code with minimal changes to existing code.

Compare to TorchScript, which has a tracing mode (`torch.jit.trace`) and
a scripting mode (`torch.jit.script`). Tracing mode is susceptible to
silent incorrectness, while scripting mode requires significant code changes
and will raise errors on unsupported Python code.

For example, TorchScript tracing silently fails on data-dependent control flow
(the `if x.sum() < 0:` line below)
because only the actual control flow path is traced.
In comparison, `torch.compile` is able to correctly handle it.

```
# Test that `fn1` and `fn2` return the same result, given the same arguments `args`.
```

TorchScript scripting can handle data-dependent control flow,
but it can require major code changes and will raise errors when unsupported Python
is used.

In the example below, we forget TorchScript type annotations and we receive
a TorchScript error because the input type for argument `y`, an `int`,
does not match with the default argument type, `torch.Tensor`.
In comparison, `torch.compile` works without requiring any type annotations.

## Graph Breaks

The graph break is one of the most fundamental concepts within `torch.compile`.
It allows `torch.compile` to handle arbitrary Python code by interrupting
compilation, running the unsupported code, then resuming compilation.
The term "graph break" comes from the fact that `torch.compile` attempts
to capture and optimize the PyTorch operation graph. When unsupported Python code is encountered,
then this graph must be "broken".
Graph breaks result in lost optimization opportunities, which may still be undesirable,
but this is better than silent incorrectness or a hard crash.

Let's look at a data-dependent control flow example to better see how graph breaks work.

The first time we run `bar`, we see that `torch.compile` traced 2 graphs
corresponding to the following code (noting that `b.sum() < 0` is False):

1. `x = a / (torch.abs(a) + 1); b.sum()`
2. `return x * b`

The second time we run `bar`, we take the other branch of the if statement
and we get 1 traced graph corresponding to the code `b = b * -1; return x * b`.
We do not see a graph of `x = a / (torch.abs(a) + 1); b.sum()` outputted the second time
since `torch.compile` cached this graph from the first run and re-used it.

Let's investigate by example how TorchDynamo would step through `bar`.
If `b.sum() < 0`, then TorchDynamo would run graph 1, let
Python determine the result of the conditional, then run
graph 2. On the other hand, if `not b.sum() < 0`, then TorchDynamo
would run graph 1, let Python determine the result of the conditional, then
run graph 3.

We can see all graph breaks by using `torch._logging.set_logs(graph_breaks=True)`.

```
# Reset to clear the torch.compile cache
```

In order to maximize speedup, graph breaks should be limited.
We can force TorchDynamo to raise an error upon the first graph
break encountered by using `fullgraph=True`:

```
# Reset to clear the torch.compile cache
```

In our example above, we can work around this graph break by replacing
the if statement with a `torch.cond`:

In order to serialize graphs or to run graphs on different (i.e. Python-less)
environments, consider using `torch.export` instead (from PyTorch 2.1+).
One important restriction is that `torch.export` does not support graph breaks. Please check
[the torch.export tutorial](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html)
for more details on `torch.export`.

Check out our [section on graph breaks in the torch.compile programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html)
for tips on how to work around graph breaks.

## Troubleshooting

Is `torch.compile` failing to speed up your model? Is compile time unreasonably long?
Is your code recompiling excessively? Are you having difficulties dealing with graph breaks?
Are you looking for tips on how to best use `torch.compile`?
Or maybe you simply want to learn more about the inner workings of `torch.compile`?

Check out [the torch.compile programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html).

## Conclusion

In this tutorial, we introduced `torch.compile` by covering
basic usage, demonstrating speedups over eager mode, comparing to TorchScript,
and briefly describing graph breaks.

For an end-to-end example on a real model, check out our [end-to-end torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_full_example.html).

To troubleshoot issues and to gain a deeper understanding of how to apply `torch.compile` to your code, check out [the torch.compile programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html).

We hope that you will give `torch.compile` a try!

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: torch_compile_tutorial.ipynb`](../_downloads/96ad88eb476f41a5403dcdade086afb8/torch_compile_tutorial.ipynb)

[`Download Python source code: torch_compile_tutorial.py`](../_downloads/6b019e0b5f84b568fcca1120bd28e230/torch_compile_tutorial.py)

[`Download zipped: torch_compile_tutorial.zip`](../_downloads/2802fa0f38eb25e527bdd4d098be787f/torch_compile_tutorial.zip)