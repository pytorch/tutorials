"""
(Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA)
==========================================================================================


**Author:** `Driss Guessous <https://github.com/drisspg>`_
"""

######################################################################
# Summary
# ~~~~~~~~
#
# In this tutorial, we want to highlight a new ``torch.nn.functional`` function
# that can be helpful for implementing transformer architectures. The
# function is named ``torch.nn.functional.scaled_dot_product_attention``.
# For detailed description of the function, see the `PyTorch documentation <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention>`__.
# This function has already been incorporated into ``torch.nn.MultiheadAttention`` and ``torch.nn.TransformerEncoderLayer``.
#
# Overview
# ~~~~~~~~~
# At a high level, this PyTorch function calculates the
# scaled dot product attention (SDPA) between query, key, and value according to
# the definition found in the paper `Attention is all you
# need <https://arxiv.org/abs/1706.03762>`__. While this function can
# be written in PyTorch using existing functions, a fused implementation can provide
# large performance benefits over a naive implementation.
#
# Fused implementations
# ~~~~~~~~~~~~~~~~~~~~~~
#
# For CUDA tensor inputs, the function will dispatch into one of the following
# implementations:
#
# * `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`__
# * `Memory-Efficient Attention <https://github.com/facebookresearch/xformers>`__
# * A PyTorch implementation defined in C++
#
# .. note::
#
#   This tutorial requires PyTorch 2.0.0 or later.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

# Example Usage:
query, key, value = torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device)
F.scaled_dot_product_attention(query, key, value)


######################################################################
# Explicit Dispatcher Control
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# While the function will implicitly dispatch to one of the three
# implementations, the user can also explicitly control the dispatch via
# the use of a context manager. This context manager allows users to
# explicitly disable certain implementations. If a user wants to ensure
# the function is indeed using the fastest implementation for their
# specific inputs, the context manager can be used to sweep through
# measuring performance.
#

# Lets define a helpful benchmarking function:
import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# Lets define the hyper-parameters of our input
batch_size = 32
max_sequence_len = 1024
num_heads = 32
embed_dimension = 32

dtype = torch.float16

query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# Lets explore the speed of each of the 3 implementations
from torch.nn.attention import SDPBackend, sdpa_kernel


with sdpa_kernel(SDPBackend.MATH):
    math_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
    print(f"The math implementation runs in {math_time:.3f} microseconds")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        flash_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The flash attention implementation runs in {flash_time:.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        efficient_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")


######################################################################
# Hardware dependence
# ~~~~~~~~~~~~~~~~~~~
#
# Depending on what machine you ran the above cell on and what hardware is
# available, your results might be different.
# - If you don’t have a GPU and are running on CPU then with FP32 the context manager
# will have no effect and all three runs should return similar timings.
# - Depending on what compute capability your graphics card supports
# flash attention or memory efficient might have failed.


######################################################################
# Causal Self Attention
# ~~~~~~~~~~~~~~~~~~~~~
#
# Below is an example implementation of a multi-headed causal self
# attention block inspired by
# `Andrej Karpathy NanoGPT <https://github.com/karpathy/nanoGPT>`__ repository.
#

class CausalSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y


num_heads = 8
heads_per_dim = 64
embed_dimension = num_heads * heads_per_dim
dtype = torch.float16
model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to("cuda").to(dtype).eval()
print(model)


#####################################################################
# ``NestedTensor`` and Dense tensor support
# -----------------------------------------
#
# SDPA supports both ``NestedTensor`` and Dense tensor inputs. ``NestedTensors`` handle the case where the input is a batch of variable length sequences
# without needing to pad each sequence to the maximum length in the batch. For more information about ``NestedTensors`` see
# `torch.nested <https://pytorch.org/docs/stable/nested.html>`__ and `NestedTensors Tutorial <https://pytorch.org/tutorials/prototype/nestedtensor.html>`__.
#

import random
def generate_rand_batch(
    batch_size,
    max_sequence_len,
    embed_dimension,
    pad_percentage=None,
    dtype=torch.float16,
    device="cuda",
):
    if not pad_percentage:
        return (
            torch.randn(
                batch_size,
                max_sequence_len,
                embed_dimension,
                dtype=dtype,
                device=device,
            ),
            None,
        )
    # Random sequence lengths
    seq_len_list = [
        int(max_sequence_len * (1 - random.gauss(pad_percentage, 0.01)))
        for _ in range(batch_size)
    ]
    # Make random entry in the batch have max sequence length
    seq_len_list[random.randint(0, batch_size - 1)] = max_sequence_len
    return (
        torch.nested.nested_tensor(
            [
                torch.randn(seq_len, embed_dimension,
                            dtype=dtype, device=device)
                for seq_len in seq_len_list
            ]
        ),
        seq_len_list,
    )

random_nt, _ = generate_rand_batch(32, 512, embed_dimension, pad_percentage=0.5, dtype=dtype, device=device)
random_dense, _ = generate_rand_batch(32, 512, embed_dimension, pad_percentage=None, dtype=dtype, device=device)

# Currently the fused implementations don't support ``NestedTensor`` for training
model.eval()

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        print(f"Random NT runs in {benchmark_torch_function_in_microseconds(model, random_nt):.3f} microseconds")
        print(f"Random Dense runs in {benchmark_torch_function_in_microseconds(model, random_dense):.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")


######################################################################
# Using SDPA with ``torch.compile``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# With the release of PyTorch 2.0, a new feature called
# ``torch.compile()`` has been introduced, which can provide
# significant performance improvements over eager mode.
# Scaled dot product attention is fully composable with ``torch.compile()``.
# To demonstrate this, let's compile the ``CausalSelfAttention`` module using
# ``torch.compile()`` and observe the resulting performance improvements.
#

batch_size = 32
max_sequence_len = 256
x = torch.rand(batch_size, max_sequence_len,
               embed_dimension, device=device, dtype=dtype)
print(
    f"The non compiled module runs in  {benchmark_torch_function_in_microseconds(model, x):.3f} microseconds")


compiled_model = torch.compile(model)
# Let's compile it
compiled_model(x)
print(
    f"The compiled module runs in  {benchmark_torch_function_in_microseconds(compiled_model, x):.3f} microseconds")


######################################################################
#
# The exact execution time is dependent on machine, however the results for mine:
# The non compiled module runs in  166.616 microseconds
# The compiled module runs in  166.726 microseconds
# That is not what we were expecting. Let's dig a little deeper.
# PyTorch comes with an amazing built-in profiler that you can use to
# inspect the performance characteristics of your code.
#

from torch.profiler import profile, record_function, ProfilerActivity
activities = [ProfilerActivity.CPU]
if device == 'cuda':
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=False) as prof:
    with record_function(" Non-Compilied Causal Attention"):
        for _ in range(25):
            model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


with profile(activities=activities, record_shapes=False) as prof:
    with record_function("Compiled Causal Attention"):
        for _ in range(25):
            compiled_model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# For even more insights, you can export the trace and use ``chrome://tracing`` to view the results
#
# .. code-block:: python
#
#    prof.export_chrome_trace("compiled_causal_attention_trace.json").




######################################################################
# The previous code snippet generates a report of the top 10 PyTorch functions
# that consumed the most GPU execution time, for both the compiled and non-compiled module.
# The analysis reveals that the majority of time spent on the GPU is concentrated
# on the same set of functions for both modules.
# The reason for this here is that ``torch.compile`` is very good at removing the
# framework overhead associated with PyTorch. If your model is launching
# large, efficient CUDA kernels, which in this case ``CausalSelfAttention``
# is, then the overhead of PyTorch can be hidden.
#
# In reality, your module does not normally consist of a singular
# ``CausalSelfAttention`` block. When experimenting with `Andrej Karpathy NanoGPT <https://github.com/karpathy/nanoGPT>`__ repository, compiling
# the module took the time per train step from: ``6090.49ms`` to
# ``3273.17ms``! This was done on commit: ``ae3a8d5`` of NanoGPT training on
# the Shakespeare dataset.
#

######################################################################
# Using SDPA with attn_bias subclasses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# As of PyTorch 2.3, we have added a new submodule that contains tensor subclasses.
# Designed to be used with ``torch.nn.functional.scaled_dot_product_attention``.
# The module is named ``torch.nn.attention.bias`` and contains the following two
# utilities for generating causal attention variants:
#
# - ``torch.nn.attention.bias.causal_upper_left``
# - ``torch.nn.attention.bias.causal_lower_right``
#
# .. note::
#    The current argument ``is_causal`` in ``torch.nn.functional.scaled_dot_product_attention``
#    is the same as using ``torch.nn.attention.bias.causal_upper_left``.
#

from torch.nn.attention.bias import causal_lower_right, causal_upper_left

batch_size = 32
sequence_length_q = 2
sequence_length_kv = 10
num_heads = 16
embed_dimension = 32

dtype = torch.float16

query = torch.rand(batch_size, num_heads, sequence_length_q, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, sequence_length_kv, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, sequence_length_kv, embed_dimension, device=device, dtype=dtype)

upper_left_bias = causal_upper_left(sequence_length_q, sequence_length_kv)
lower_right_bias = causal_lower_right(sequence_length_q, sequence_length_kv)

print(type(upper_left_bias))
print(type(lower_right_bias))

assert type(upper_left_bias) == type(lower_right_bias)
assert issubclass(type(upper_left_bias), torch.Tensor)

# As you can see from the previous output, are the same type ``torch.nn.attention.bias.CausalBias``
# and subclass ``torch.Tensor``

# Lets see what these tensors look like
print(upper_left_bias)
print(lower_right_bias)

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
out_upper_left = F.scaled_dot_product_attention(query, key, value, upper_left_bias)
out_lower_right = F.scaled_dot_product_attention(query, key, value, lower_right_bias)
out_is_causal = F.scaled_dot_product_attention(query, key, value, is_causal=True)

assert torch.allclose(out_upper_left, out_is_causal)
assert not torch.allclose(out_upper_left, out_lower_right)

# These attention biases should also be compatible with torch.compile
compiled_sdpa = torch.compile(F.scaled_dot_product_attention, fullgraph=True)
out_upper_left = compiled_sdpa(query, key, value, upper_left_bias)

######################################################################
# Conclusion
# ~~~~~~~~~~~
#
# In this tutorial, we have demonstrated the basic usage of
# ``torch.nn.functional.scaled_dot_product_attention``. We have shown how
# the ``sdpa_kernel`` context manager can be used to assert a certain
# implementation is used on GPU. As well, we built a simple
# ``CausalSelfAttention`` module that works with ``NestedTensor`` and is torch
# compilable. In the process we have shown how to the profiling tools can
# be used to explore the performance characteristics of a user defined
# module.
#
