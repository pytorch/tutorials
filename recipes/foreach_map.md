Note

Go to the end
to download the full example code.

# Explicit horizontal fusion with foreach_map and torch.compile

**Author:** [Michael Lazos](https://github.com/mlazos)

Horizontal fusion is a key optimization in ML compilers. In eager,

this is typically expressed using the torch._foreach* ops which parallelizes
operations across a list of tensors. However, supporting all possible permutations
of arguments is quite difficult (e.g. mixtures of scalars and lists). Foreach_map
allows conversion of any pointwise op in `torch` to a horiztonally fused foreach
variant. In this tutorial, we will demonstrate how to implement the Adam optimizer
with `foreach_map` to generate a fully fused kernel.

Note

This recipe describes a prototype feature. Prototype features are typically
at an early stage for feedback and testing and are subject to change.

## Prerequisites

- PyTorch v2.7.0 or later

### Model Setup

For this example, we'll use a simple sequence of linear layers.
We instantiate an independent copy to compare the two optimizer implementations.

```
# exit cleanly if we are on a device that doesn't support ``torch.compile``

# Create simple model

# run forward pass

# run backward to populate the grads for our optimizer below
```

### Helper functions for foreach_map implementation

In this section, we'll begin our implementation of the Adam optimizer.

```
# Helper function to extract optimizer states from a torch.optim.Adam instance

# Functions to update the different optimizer states

# Our full Adam implementation
```

### Setting up and running the compiled kernel

In this section, we'll run our Adam optimizer
and compare the results

Note

`torch.compile` is only supported on CUDA devices that have a compute capability of 7.0 or higher.

```
# warm up the optimizer state dict

# optionally view the output code

# Warmup runs to compile the function

# Benchmark performance
```

### Conclusion

In this tutorial, we successfully implemented a custom fully-fused Adam optimizer using foreach_map.
By leveraging the power of foreach_map and torch.compile, we were able to create an optimized version of the Adam
optimizer that can be used in various machine learning applications. This tutorial provides a comprehensive guide
on how to use foreach_map and torch.compile to optimize machine learning models, and serves as a
valuable resource for developers looking to improve the performance of their models with horizontal fusion.

See also:

- [Compiled optimizer tutorial](https://pytorch.org/tutorials/recipes/compiling_optimizer.html) - an intro into the compiled optimizer.
- [Compiling the optimizer with PT2](https://dev-discuss.pytorch.org/t/compiling-the-optimizer-with-pt2/1669) - deeper technical details on the compiled optimizer.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: foreach_map.ipynb`](../_downloads/162cf335b789dd055d4192f77cb0251c/foreach_map.ipynb)

[`Download Python source code: foreach_map.py`](../_downloads/bcb9aa4fd3968b85310b970dbd86bbc3/foreach_map.py)

[`Download zipped: foreach_map.zip`](../_downloads/faee5eeb51c8f314872395cc1b776677/foreach_map.zip)