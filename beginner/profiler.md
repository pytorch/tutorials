Note

Go to the end
to download the full example code.

# Profiling your PyTorch Module

**Author:** [Suraj Subramanian](https://github.com/subramen)

PyTorch includes a profiler API that is useful to identify the time and
memory costs of various PyTorch operations in your code. Profiler can be
easily integrated in your code, and the results can be printed as a table
or returned in a JSON trace file.

Note

Profiler supports multithreaded models. Profiler runs in the
same thread as the operation but it will also profile child operators
that might run in another thread. Concurrently-running profilers will be
scoped to their own thread to prevent mixing of results.

Note

PyTorch 1.8 introduces the new API that will replace the older profiler API
in the future releases. Check the new API at [this page](https://pytorch.org/docs/master/profiler.html).

Head on over to [this
recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
for a quicker walkthrough of Profiler API usage.

---

## Performance debugging using Profiler

Profiler can be useful to identify performance bottlenecks in your
models. In this example, we build a custom module that performs two
sub-tasks:

- a linear transformation on the input, and
- use the transformation result to get indices on a mask tensor.

We wrap the code for each sub-task in separate labelled context managers using
`profiler.record_function("label")`. In the profiler output, the
aggregate performance metrics of all operations in the sub-task will
show up under its corresponding label.

Note that using Profiler incurs some overhead, and is best used only for investigating
code. Remember to remove it if you are benchmarking runtimes.

## Profile the forward pass

We initialize random input and mask tensors, and the model.

Before we run the profiler, we warm-up CUDA to ensure accurate
performance benchmarking. We wrap the forward pass of our module in the
`profiler.profile` context manager. The `with_stack=True` parameter appends the
file and line number of the operation in the trace.

Warning

`with_stack=True` incurs an additional overhead, and is better suited for investigating code.
Remember to remove it if you are benchmarking performance.

```
# warm-up
```

## Print profiler results

Finally, we print the profiler results. `profiler.key_averages`
aggregates the results by operator name, and optionally by input
shapes and/or stack trace events.
Grouping by input shapes is useful to identify which tensor shapes
are utilized by the model.

Here, we use `group_by_stack_n=5` which aggregates runtimes by the
operation and its traceback (truncated to the most recent 5 events), and
display the events in the order they are registered. The table can also
be sorted by passing a `sort_by` argument (refer to the
[docs](https://pytorch.org/docs/stable/autograd.html#profiler) for
valid sorting keys).

Note

When running profiler in a notebook, you might see entries like `<ipython-input-18-193a910735e8>(13): forward`
instead of filenames in the stacktrace. These correspond to `<notebook-cell>(line number): calling-function`.

```
"""
(Some columns are omitted)

------------- ------------ ------------ ------------ ---------------------------------
 Name Self CPU % Self CPU Self CPU Mem Source Location
------------- ------------ ------------ ------------ ---------------------------------
 MASK INDICES 87.88% 5.212s -953.67 Mb /mnt/xarfuse/.../torch/au
 <ipython-input-...>(10): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/

 aten::copy_ 12.07% 715.848ms 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 LINEAR PASS 0.01% 350.151us -20 b /mnt/xarfuse/.../torch/au
 <ipython-input-...>(7): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/

 aten::addmm 0.00% 293.342us 0 b /mnt/xarfuse/.../torch/nn
 /mnt/xarfuse/.../torch/nn
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(8): forward
 /mnt/xarfuse/.../torch/nn

 aten::mean 0.00% 235.095us 0 b <ipython-input-...>(11): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

----------------------------- ------------ ---------- ----------------------------------
Self CPU time total: 5.931s

"""
```

## Improve memory performance

Note that the most expensive operations - in terms of memory and time -
are at `forward (10)` representing the operations within MASK INDICES. Let's try to
tackle the memory consumption first. We can see that the `.to()`
operation at line 12 consumes 953.67 Mb. This operation copies `mask` to the CPU.
`mask` is initialized with a `torch.double` datatype. Can we reduce the memory footprint by casting
it to `torch.float` instead?

```
# warm-up

"""
(Some columns are omitted)

----------------- ------------ ------------ ------------ --------------------------------
 Name Self CPU % Self CPU Self CPU Mem Source Location
----------------- ------------ ------------ ------------ --------------------------------
 MASK INDICES 93.61% 5.006s -476.84 Mb /mnt/xarfuse/.../torch/au
 <ipython-input-...>(10): forward
 /mnt/xarfuse/ /torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/

 aten::copy_ 6.34% 338.759ms 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 aten::as_strided 0.01% 281.808us 0 b <ipython-input-...>(11): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 aten::addmm 0.01% 275.721us 0 b /mnt/xarfuse/.../torch/nn
 /mnt/xarfuse/.../torch/nn
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(8): forward
 /mnt/xarfuse/.../torch/nn

 aten::_local 0.01% 268.650us 0 b <ipython-input-...>(11): forward
 _scalar_dense /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

----------------- ------------ ------------ ------------ --------------------------------
Self CPU time total: 5.347s

"""
```

The CPU memory footprint for this operation has halved.

## Improve time performance

While the time consumed has also reduced a bit, it's still too high.
Turns out copying a matrix from CUDA to CPU is pretty expensive!
The `aten::copy_` operator in `forward (12)` copies `mask` to CPU
so that it can use the NumPy `argwhere` function. `aten::copy_` at `forward(13)`
copies the array back to CUDA as a tensor. We could eliminate both of these if we use a
`torch` function `nonzero()` here instead.

```
# warm-up

"""
(Some columns are omitted)

-------------- ------------ ------------ ------------ ---------------------------------
 Name Self CPU % Self CPU Self CPU Mem Source Location
-------------- ------------ ------------ ------------ ---------------------------------
 aten::gt 57.17% 129.089ms 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 aten::nonzero 37.38% 84.402ms 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 INDEX SCORE 3.32% 7.491ms -119.21 Mb /mnt/xarfuse/.../torch/au
 <ipython-input-...>(10): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/

aten::as_strided 0.20% 441.587us 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 aten::nonzero
 _numpy 0.18% 395.602us 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/
-------------- ------------ ------------ ------------ ---------------------------------
Self CPU time total: 225.801ms

"""
```

## Further Reading

We have seen how Profiler can be used to investigate time and memory bottlenecks in PyTorch models.
Read more about Profiler here:

- [Profiler Usage Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Profiling RPC-Based Workloads](https://pytorch.org/tutorials/recipes/distributed_rpc_profiling.html)
- [Profiler API Docs](https://pytorch.org/docs/stable/autograd.html?highlight=profiler#profiler)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: profiler.ipynb`](../_downloads/9fc6c90b1bbbfd4201d66c498708f33f/profiler.ipynb)

[`Download Python source code: profiler.py`](../_downloads/1df539a85371bf035ce170fb872b4f7f/profiler.py)

[`Download zipped: profiler.zip`](../_downloads/9ec2c08d66bc2989b2d042cc958fe39e/profiler.zip)