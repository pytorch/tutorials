Note

Go to the end
to download the full example code.

# Accelerating PyTorch Transformers by replacing `nn.Transformer` with Nested Tensors and `torch.compile()`

**Author:** [Mikayla Gawarecki](https://github.com/mikaylagawarecki)

 What you will learn

- Learn about the low-level building blocks PyTorch provides to build custom transformer layers (
nested tensors, `scaled_dot_product_attention`, `torch.compile()`, and `FlexAttention`)
- Discover how the above improve memory usage and performance using MultiHeadAttention as an example
- Explore advanced customizations using the aforementioned building blocks

 Prerequisites

- PyTorch v.2.6.0 or later

Over the past few years, the PyTorch team has developed various lower level
features that, when composed, can create a variety of transformer variants. These
include:

- Nested Tensors with the `torch.jagged` layout (AKA NJTs)
- `scaled_dot_product_attention`
- `torch.compile()`
- `FlexAttention`

This tutorial will give a brief overview of the above technologies and
demonstrate how they can be composed to yield flexible and performant transformer layers with improved user experience.

One may observe that the `torch.nn` module currently provides various `Transformer`-related layers.
In particular, it includes `TransformerEncoderLayer`, `TransformerEncoder`, `TransformerDecoderLayer`,
`TransformerDecoder`, `Transformer` and `MultiheadAttention`. This family
of layers was initially implemented following the [Attention is All
You Need](https://arxiv.org/abs/1706.03762) paper. The components discussed in
this tutorial provide improved user experience, flexibility and performance over
the existing `nn` layers.

# Is this tutorial for me?

If you are wondering about what building blocks the `torch` library provides
for writing your own transformer layers and best practices, you are in the
right place. Please keep reading!

If you are looking for an out-of-the-box implementation of a popular transformer
architecture, note that there are many open-source libraries that provide them,
including:

- [HuggingFace transformers](https://github.com/huggingface/transformers)
- [xformers](https://github.com/facebookresearch/xformers)
- [torchtune](https://github.com/pytorch/torchtune)

If you are only interested in performant attention score modifications, please
check out the [FlexAttention blog](https://pytorch.org/blog/flexattention/) that
contains a [gym of masks](https://github.com/meta-pytorch/attention-gym).

# Introducing the Building Blocks

First, we will briefly introduce the four technologies mentioned in the introduction

- [torch.nested](https://pytorch.org/tutorials/unstable/nestedtensor.html)

Nested tensors generalize the shape of regular dense tensors, allowing for
representation of ragged-sized data with the same tensor UX. In the context of
transformers, we can think of nested tensors as a tool for representing variable
sequence lengths. They eliminate the need for the bug-prone practices of explicit
padding and masking (think `key_padding_mask` in `nn.MultiHeadAttention`).

- [scaled_dot_product_attention](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)

`scaled_dot_product_attention` is a primitive for
\(\text{softmax}(\frac{QK^T}{\sqrt{E}} + B)V\) that dispatches into either fused
implementations of the operator or a fallback implementation. It works out of
the box in eager mode (i.e. the default mode of using PyTorch where operations
are executed on the fly as they are encountered) and also integrates seamlessly
with `torch.compile()`. As of 2.6, it will also offer grouped query attention
natively.

- [torch.compile()](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

`torch.compile()` is a compiler introduced in version 2.0 that is able to
capture a graph of PyTorch code and perform various optimizations on it, such as
fusing together sequences of ops. Nested tensors with the `torch.jagged` layout
and `scaled_dot_product_attention` work seamlessly with compile. In the
context of transformers, the value add of using compile with nested tensor
and SDPA is that compile can remove framework overhead ones sees in eager mode
and fuse sequences of ops in transformers together, such as projection and
activation.

- [FlexAttention](https://pytorch.org/blog/flexattention/)

`FlexAttention` is a primitive that allows users to modify attention scores
prior to the softmax operation. It generalizes the additive `B` term above
for `scaled_dot_product_attention`, allowing for arbitrary calculation. It
requires compile to achieve good performance.

# The above building blocks are "All You Need" (as of October 2024)

The main premise in this section is that most transformer variations are
GPT-style, consisting of layers like Embedding, Positional Encoding, Attention
Blocks and Feed Forward networks. If we were to try to classify the differences
in this space, we might land on something like:

1. Layer type (activation functions such as `SwiGLU` and others, normalization functions
such as `RMSNorm` and others, positional encodings, such as Sinusoidal, Rotary.)
2. Layer ordering, such as where to apply norms and positional encoding.
3. Modifications to attention score, such as `ALiBi`, Relative Positional Bias and so on.

In a pre-compiler environment, you might write a custom transformer and notice
that it functions correctly but is slow. To address this, you might develop a
custom fused kernel for the specific series of operations. In a compiler environment,
you can simply perform the initial step and then compile and benefit from improved performance.

## MultiheadAttention

Remember that MultiheadAttention takes in a query, key, and value, and consists
of an input projection, a `scaled_dot_product_attention` operator and an
output projection. The main takeaway we want to demonstrate here is the
improvement yielded when we replaced padded/masked inputs with nested tensors.
The improvements are threefold:

- **User Experience**
Remember that `nn.MultiheadAttention` requires `query`, `key`, and
`value` to be dense `torch.Tensors`. It also provides a
`key_padding_mask` that is used to mask out padding tokens in the `key`
that arise due to different sequence lengths within a batch. Since there is
no `query_padding_mask` in `nn.MHA`, users have to take care to mask/slice
the outputs appropriately to account for query sequence lengths. `NestedTensor`
cleanly removes the need for this sort of error-prone padding masks.
- **Memory**
Instead of materializing a dense `[B, S, D]` tensor with a `[B, S]`
padding mask (where `B` is batch size, `S` is max sequence length in the
batch and `D` is embedding size), nested tensors allow you to cleanly
represent the batch of varying sequence lengths. As a result, the input and
intermediate activations will use less memory.
- **Performance**
Since padding is not materialized and unnecessary computation on padding is
skipped, performance and memory usage improve.

We'll demonstrate the above by building upon the `MultiheadAttention` layer in the
[Nested Tensor tutorial](https://pytorch.org/tutorials/unstable/nestedtensor.html)
and comparing it to the `nn.MultiheadAttention` layer.

### Utilities

In this section, we include a utility to generate semi-realistic data using
`Zipf` distribution for sentence lengths. This is used to generate the nested
query, key, and value tensors. We also include a benchmark utility.

```
# Generate a batch of semi-realistic data using Zipf distribution for sentence lengths
# in the form of nested tensors with the jagged layout.
```

We will now demonstrate the performance improvements of using nested tensors
in the `MultiheadAttention` layer + compile for self attention. We compare this against
the traditional `nn.MultiheadAttention` + compile with padding and masking.

```
# ``nn.MultiheadAttention`` uses a non conventional initialization for layers, so do this for exact parity :(

# warmup compile

# benchmark

# For the vanilla ``nn.MultiheadAttention``, we need to construct the ``key_padding_mask``
# Further, ``nn.MultiheadAttention`` forces one to materialize the ``attn_mask`` even if using ``is_causal``

# warmup compile

# benchmark
```

For reference, here are some sample outputs on A100:

```
padded_time=0.03454, padded_peak_memory=4.14 GB
nested_time=0.00612, nested_peak_memory=0.76 GB
Max difference between vanilla and nested result 0.0
Nested speedup: 5.65
Nested peak memory reduction 3.39 GB
```

We can also see the same for backward pass

Sample outputs on A100:

```
padded_bw_time=2.09337, padded_bw_peak_mem=5.10 GB
nested_bw_time=0.01452, nested_bw_peak_mem=3.24 GB
Nested backward speedup: 144.13
Nested backward peak memory reduction 1.86 GB
Difference in out_proj.weight.grad 0.000244140625
Difference in packed_proj.weight.grad 0.001556396484375
Difference in out_proj.bias.grad 0.0
Difference in packed_proj.bias.grad 0.001953125
```

## GPT-style layer

A basic GPT-style transformer layer consists of a causal self-attention layer
followed by a feed-forward network (FFN) with skip connections. Implementing
this is fairly straightforward using the `MultiheadAttention` layer above and
gives equivalent results to an `nn.TransformerEncoderLayer` with
`is_causal=True`.

We demonstrate examples of implementing the rest of the `nn` layers
[here](https://github.com/mikaylagawarecki/transformer_tutorial_accompaniment)
but omit that from this tutorial for brevity.

## Going one step further

So far, we have demonstrated how to implement a performant `MultiheadAttention`
layer that follows the traditional `nn.MultiheadAttention`. Going back to our
classification of modifications to the transformer architecture, remember that we
classified the modifications into layer type, layer ordering, and modifications
to the attention score. We trust that changing layer type and layer ordering
(such as swapping `LayerNorm` for `RMSNorm`) is fairly straightforward.

In this section, we will discuss various functionalities using the
aforementioned building blocks, including the following:

- Cross Attention
- Fully masked rows no longer cause NaNs
- Packed Projection

## Cross Attention

Cross attention is a form of attention where the query and key/value tensors
are from different sequences.

One example of this is in `nn.TransformerDecoderLayer` where the query comes
from the decoder and the key/value come from the encoder.

The above MultiheadAttention layer nicely generalizes to this case with nested
tensors for both query and key/value.

As above, we can compare this against the vanilla compiled `nn.MultiheadAttention`.

```
# warmup compile
```

Sample outputs on A100:

```
Max difference between vanilla and nested result 0.0
Nested speedup: 4.01
Nested peak memory reduction 1.40 GB
```

## Fully masked rows no longer cause NaNs

There has been a long standing issue with `nn.MultiheadAttention` and
`scaled_dot_product_attention` where if a row was fully masked out, the output
of the attention layer would be NaN. See [issue](https://github.com/pytorch/pytorch/issues/41508).
This is because the softmax over an empty set is undefined.

Thanks to [this PR](https://github.com/pytorch/pytorch/pull/133882)
this is no longer the case. Instead, the output corresponding to fully masked rows
in `scaled_dot_product_attention` will be 0. For cases where `nn.MHA` does
not employ the "fast-path", this will also apply.

Using a custom MHA layer with NJTs is strongly recommended over the
existing "fast-path" in `nn.MultiheadAttention` as NJT's ability to model raggedness
appropriately makes it possible to properly express empty sequences.

## Packed Projection

Packed projection is a technique that makes use of the fact that when the input
for projection (matrix multiplications) are the same (self-attention), we can pack the projection
weights and biases into single tensors. It is especially useful when the individual
projections are memory bound rather than compute bound. There are
two examples that we will demonstrate here:

- Input projection for MultiheadAttention
- SwiGLU activation in feed-forward network of Transformer Layer

### Input projection for MultiheadAttention

When doing self-attention, the `query`, `key`, and `value`
are the same tensor. Each of these tensors is projected with a
`Linear(E_q, E_total)` layer. Instead, we can pack this into one layer,
which is what we do in the MultiheadAttention layer above.

Let us compare the performance of the packed projection against the usual method:

```
# warmup

# benchmark

# On my A100 prints 1.05x speedup
```

### SwiGLU feed forward network of Transformer Layer

Swish-Gated Linear Unit (SwiGLU) is a non-linear activation function that is increasingly popular in the feed-forward
network of the transformer layer (e.g. Llama). A feed-forward network with SwiGLU activation is defined as:

An alternative way of implementing this that uses packed projection is

We can compare the performance of the two implementations as follows
Depending on your hardware, you might see different results. On an A100 I see
1.12x speedup for D=128.

```
# warmup

# benchmark

# On my A100 prints 1.08x speedup
```

## Extended examples

We intend to update this tutorial to demonstrate more examples of how to use
the various performant building blocks such as KV-Caching, Grouped Query Attention
etc. Further, there are several good examples of using various performant building blocks to
implement various transformer architectures. Some examples include

- [gpt-fast](https://github.com/meta-pytorch/gpt-fast)
- [segment-anything-fast](https://github.com/meta-pytorch/segment-anything-fast)
- [lucidrains implementation of NaViT with nested tensors](https://github.com/lucidrains/vit-pytorch/blob/73199ab486e0fad9eced2e3350a11681db08b61b/vit_pytorch/na_vit_nested_tensor.py)
- [torchtune's implementation of VisionTransformer](https://github.com/pytorch/torchtune/blob/a8a64ec6a99a6ea2be4fdaf0cd5797b03a2567cf/torchtune/modules/vision_transformer.py#L16)

## Conclusion

In this tutorial, we have introduced the low level building blocks PyTorch
provides for writing transformer layers and demonstrated examples how to compose
them. It is our hope that this tutorial has educated the reader on the ease with
which flexible and performant transformer layers can be implemented by users of PyTorch.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: transformer_building_blocks.ipynb`](../_downloads/e201125c39959609ca168c306995205c/transformer_building_blocks.ipynb)

[`Download Python source code: transformer_building_blocks.py`](../_downloads/57114670a041b4c96ed6eb9fc17a6b3f/transformer_building_blocks.py)

[`Download zipped: transformer_building_blocks.zip`](../_downloads/9e1bf792ffacdc6aa490d8ac2246cba7/transformer_building_blocks.zip)