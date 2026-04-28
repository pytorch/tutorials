Note

Go to the end
to download the full example code.

# (Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA)

**Author:** [Driss Guessous](https://github.com/drisspg)

## Summary

In this tutorial, we want to highlight a new `torch.nn.functional` function
that can be helpful for implementing transformer architectures. The
function is named `torch.nn.functional.scaled_dot_product_attention`.
For detailed description of the function, see the [PyTorch documentation](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention).
This function has already been incorporated into `torch.nn.MultiheadAttention` and `torch.nn.TransformerEncoderLayer`.

## Overview

At a high level, this PyTorch function calculates the
scaled dot product attention (SDPA) between query, key, and value according to
the definition found in the paper [Attention is all you
need](https://arxiv.org/abs/1706.03762). While this function can
be written in PyTorch using existing functions, a fused implementation can provide
large performance benefits over a naive implementation.

## Fused implementations

For CUDA tensor inputs, the function will dispatch into one of the following
implementations:

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Memory-Efficient Attention](https://github.com/facebookresearch/xformers)
- A PyTorch implementation defined in C++

Note

This tutorial requires PyTorch 2.0.0 or later.

```
# Example Usage:
```

## Explicit Dispatcher Control

While the function will implicitly dispatch to one of the three
implementations, the user can also explicitly control the dispatch via
the use of a context manager. This context manager allows users to
explicitly disable certain implementations. If a user wants to ensure
the function is indeed using the fastest implementation for their
specific inputs, the context manager can be used to sweep through
measuring performance.

```
# Lets define a helpful benchmarking function:

# Lets define the hyper-parameters of our input

# Lets explore the speed of each of the 3 implementations
```

## Hardware dependence

Depending on what machine you ran the above cell on and what hardware is
available, your results might be different.
- If you don't have a GPU and are running on CPU then with FP32 the context manager
will have no effect and all three runs should return similar timings.
- Depending on what compute capability your graphics card supports
flash attention or memory efficient might have failed.

## Causal Self Attention

Below is an example implementation of a multi-headed causal self
attention block inspired by
[Andrej Karpathy NanoGPT](https://github.com/karpathy/nanoGPT) repository.

### `NestedTensor` and Dense tensor support

SDPA supports both `NestedTensor` and Dense tensor inputs. `NestedTensors` handle the case where the input is a batch of variable length sequences
without needing to pad each sequence to the maximum length in the batch. For more information about `NestedTensors` see
[torch.nested](https://pytorch.org/docs/stable/nested.html) and [NestedTensors Tutorial](https://pytorch.org/tutorials/prototype/nestedtensor.html).

```
# Currently the fused implementations don't support ``NestedTensor`` for training
```

## Using SDPA with `torch.compile`

With the release of PyTorch 2.0, a new feature called
`torch.compile()` has been introduced, which can provide
significant performance improvements over eager mode.
Scaled dot product attention is fully composable with `torch.compile()`.
To demonstrate this, let's compile the `CausalSelfAttention` module using
`torch.compile()` and observe the resulting performance improvements.

```
# Let's compile it
```

The exact execution time is dependent on machine, however the results for mine:
The non compiled module runs in 166.616 microseconds
The compiled module runs in 166.726 microseconds
That is not what we were expecting. Let's dig a little deeper.
PyTorch comes with an amazing built-in profiler that you can use to
inspect the performance characteristics of your code.

```
# For even more insights, you can export the trace and use ``chrome://tracing`` to view the results
#
# .. code-block:: python
#
# prof.export_chrome_trace("compiled_causal_attention_trace.json").
```

The previous code snippet generates a report of the top 10 PyTorch functions
that consumed the most GPU execution time, for both the compiled and non-compiled module.
The analysis reveals that the majority of time spent on the GPU is concentrated
on the same set of functions for both modules.
The reason for this here is that `torch.compile` is very good at removing the
framework overhead associated with PyTorch. If your model is launching
large, efficient CUDA kernels, which in this case `CausalSelfAttention`
is, then the overhead of PyTorch can be hidden.

In reality, your module does not normally consist of a singular
`CausalSelfAttention` block. When experimenting with [Andrej Karpathy NanoGPT](https://github.com/karpathy/nanoGPT) repository, compiling
the module took the time per train step from: `6090.49ms` to
`3273.17ms`! This was done on commit: `ae3a8d5` of NanoGPT training on
the Shakespeare dataset.

## Using SDPA with attn_bias subclasses

```
# As of PyTorch 2.3, we have added a new submodule that contains tensor subclasses.
# Designed to be used with ``torch.nn.functional.scaled_dot_product_attention``.
# The module is named ``torch.nn.attention.bias`` and contains the following two
# utilities for generating causal attention variants:
#
# - ``torch.nn.attention.bias.causal_upper_left``
# - ``torch.nn.attention.bias.causal_lower_right``
#
# .. note::
# The current argument ``is_causal`` in ``torch.nn.functional.scaled_dot_product_attention``
# is the same as using ``torch.nn.attention.bias.causal_upper_left``.
#

# As you can see from the previous output, are the same type ``torch.nn.attention.bias.CausalBias``
# and subclass ``torch.Tensor``

# Lets see what these tensors look like

# Upper Left Bias aligns the causal attention mask to the upper left corner of the attention scores matrix.
# This only has an impact when the attention scores matrix is not square, which is common for decoding use cases.
# Another way of thinking about this concept is that when you use upper left bias,
# the 0th token in the query is aligned to the 0th token in the key, while for lower right bias,
# Assuming the attention score matrix is two dimensional, ``attn_score[0][0]`` is the attention score
# between the 0th token in the query and the 0th token in the key.
# For lower right bias, the sequence of q is aligned so that the last token in q is aligned to the last token in k
# (for example, ``attn_score[-1][-1])`` is all True since the last token in q is at the same position as the last token in k
# even if the sequence length of q and k are different.

# These objects are intended to be used with sdpa

# These attention biases should also be compatible with torch.compile
```

## Conclusion

In this tutorial, we have demonstrated the basic usage of
`torch.nn.functional.scaled_dot_product_attention`. We have shown how
the `sdpa_kernel` context manager can be used to assert a certain
implementation is used on GPU. As well, we built a simple
`CausalSelfAttention` module that works with `NestedTensor` and is torch
compilable. In the process we have shown how to the profiling tools can
be used to explore the performance characteristics of a user defined
module.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: scaled_dot_product_attention_tutorial.ipynb`](../_downloads/fc133e4ffc6275f9d1c3a74ddd10e0a2/scaled_dot_product_attention_tutorial.ipynb)

[`Download Python source code: scaled_dot_product_attention_tutorial.py`](../_downloads/e40ced94a143a49f0f8745e10c981139/scaled_dot_product_attention_tutorial.py)

[`Download zipped: scaled_dot_product_attention_tutorial.zip`](../_downloads/ac14cbb6e0ecc257ad2661072fa715c2/scaled_dot_product_attention_tutorial.zip)