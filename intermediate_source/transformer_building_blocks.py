"""
Dismantling the ``nn.Transformer`` modules for gains and profits 
=================================================================
**Author:** `Mikayla Gawarecki <https://github.com/mikaylagawarecki>`_

.. note::
    This tutorial should be run with the latest nightly, or, when available, 2.6.

The ``torch.nn`` module currently provides various ``Transformer``-related layers.
In particular ``TransformerEncoderLayer``, ``TransformerEncoder``, ``TransformerDecoderLayer``,
``TransformerDecoder``, ``Transformer`` and ``MultiheadAttention``. This family
of layers was initially implemented following the `Attention is All
You Need <https://arxiv.org/abs/1706.03762>`_ paper. Since then, various improvements
were made to try to make these layers more flexible.

While historically these layers intended to provide out-of-the-box, performant
solutions, we make the observations that

1.  People want to add slight customizations to their transformer layers
2.  Writing these layers and customizations is not hard


Supporting all transformer variants via a small number of out of the box layers would
yield too many keyword arguments. This tutorial will describe how to build your
own performant transformer layers following our recommended best practices.
The technologies used will be the following

1.   Nested Tensors with the ``torch.jagged`` layout (AKA NJTs)
2.   ``scaled_dot_product_attention``
3.   ``torch.compile()``
4.   ``FlexAttention``

Is this tutorial for me?
========================

If you are looking for an out-of-the-box implementation of a popular transformer
architecture, note that there are many open-source libraries that provide them,
with some examples being:

* `HuggingFace transformers <https://github.com/huggingface/transformers>`_
* `xformers <https://github.com/facebookresearch/xformers>`_
* `torchtune <https://github.com/pytorch/torchtune>`_

Please head there instead!

If you are only interested in performant attention score modifications, please
head to the `FlexAttention blog <https://pytorch.org/blog/flexattention/>`_ that
contains a `gym of masks <https://github.com/pytorch-labs/attention-gym>`_.
If you are wondering about what building blocks the ``torch`` library provides
for writing your own transformer layers and best practices, you are in the
right place, please keep reading!


"""

################################################################################
# Introducing the Building Blocks
# ===============================
# First, we will briefly introduce the 4 technologies mentioned in the introduction

# * `torch.nested <https://pytorch.org/tutorials/prototype/nestedtensor.html>`_

# Nested tensors generalize the shape of regular dense tensors, allowing for
# representation of ragged-sized data with the same tensor UX. In the context of
# transformers, we can think of nested tensors as a tool for representing variable
# sequence lengths. They eliminate the need for the bug-prone practices of explicit
# padding and masking (think ``key_padding_mask`` in ``nn.MultiHeadAttention``).

# * `scaled_dot_product_attention <https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>`_

# ``scaled_dot_product_attention`` is a primitive for
# :math:`\text{softmax}(\frac{QK^T}{\sqrt{E}} + B)V` that dispatches into either fused
# implementations of the operator or a fallback implementation. It works out of
# the box in eager mode (i.e. the default mode of using PyTorch where operations
# are executed on the fly as they are encountered) and also integrates seamlessly
# with ``torch.compile()``. As of 2.6, it will also offer grouped query attention
# natively.

# * `torch.compile() <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_

# ``torch.compile()`` is a compiler introduced in version 2.0 that is able to
# capture a graph of PyTorch code and perform various optimizations on it, such as
# fusing together sequences of ops. Nested tensors with the ``torch.jagged`` layout
# and ``scaled_dot_product_attention`` work seamlessly with compile. In the
# context of transformers, the value add of using compile with nested tensor
# and SDPA is that compile can remove framework overhead ones sees in eager mode
# and fuse sequences of ops in transformers together (e.g. projection and
# activation).

# * `FlexAttention <https://pytorch.org/blog/flexattention/>`_

# ``FlexAttention`` is a primitive that allows users to modify attention scores
# prior to the softmax operation. It generalizes the additive ``B`` term above
# for ``scaled_dot_product_attention``, allowing for arbitrary calculation. It
# requires compile to achieve good performance.

# The above building blocks are "All You Need" (as of October 2024)
# ==================================================================

# The main premise in this section is that most transformer variations are
# GPT-style, consisting of layers like Embedding, Positional Encoding, Attention
# Blocks and Feed Forward networks. If we were to try to classify the differences
# in this space, we might land on something like:

# 1.   Layer type (activation functions e.g. ``SwiGLU``, normalization functions
#      e.g. ``RMSNorm`` etc., positional encodings e.g. Sinusoidal, Rotary etc.)
# 2.   Layer ordering (where to apply norms, where to apply positional encoding etc.)
# 3.   Modifications to attention score (``ALiBi``, Relative Positional Bias etc.)


# In a pre-compiler world, one might write their custom transformer and observe
# that it works but is slow. Then, one might write a custom fused kernel for
# the specific series of ops. In a compiler world, one can do the former, compile
# and profit.


###############################################################################
# MultiheadAttention
# ------------------
# Recall that MultiheadAttention takes in a query, key and value and consists
# of an input projection, a ``scaled_dot_product_attention`` operator and an
# output projection. The main takeaway we want to demonstrate here is the
# improvement yielded when we replaced padded/masked inputs with nested tensors.
# The improvements are threefold:
#
# * User Experience
#   Recall that ``nn.MultiheadAttention`` requires ``query``, ``key`` and
#   ``value`` to be dense ``torch.Tensors``. It also provides a
#   ``key_padding_mask`` that is used to mask out padding tokens in the ``key``
#   that arise due to different sequence lengths within a batch. Since there is
#   no ``query_padding_mask`` in ``nn.MHA``, users have to take care to mask/slice
#   the outputs appropriately to account for query sequence lengths. Nested tensor
#   cleanly removes the need for this sort of error-prone padding masks. 
#
# * Memory
#   Instead of materializing a dense ``[B, S, D]`` tensor with a ``[B, S]``
#   padding mask (where ``B`` is batch size, ``S`` is max sequence length in the
#   batch and ``D`` is embedding size), nested tensors allow you to cleanly
#   represent the batch of varying sequence lengths. As a result, the input and
#   intermediate activations will use less memory.
#
# * Performance
#   Since padding is not materialized and unnecessary computation on padding is
#   skipped, performance and memory usage improve.
#
# We'll demonstrate the above by building off the ``MultiheadAttention`` layer in the
# `Nested Tensor tutorial <https://pytorch.org/tutorials/prototype/nestedtensor.html>`_
# and comparing it to the ``nn.MultiheadAttention`` layer.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """
    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
          self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
          self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
          self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
          self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask=None,
                is_causal=False) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = F.linear(query, q_weight, q_bias), F.linear(key, k_weight, k_bias), F.linear(value, v_weight, v_bias)

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


###############################################################################
# Utilities
# =========
# In this section, we include a utility to generate semi-realistic data using
# Zipf distribution for sentence lengths. This is used to generate the nested
# query, key and value tensors. We also include a benchmark utility.


import numpy as np

def zipf_sentence_lengths(alpha: float, batch_size: int) -> torch.Tensor:
    # generate fake corpus by unigram Zipf distribution
    # from wikitext-2 corpus, we get rank "." = 3, "!" = 386, "?" = 858
    sentence_lengths = np.empty(batch_size, dtype=int)
    for ibatch in range(batch_size):
        sentence_lengths[ibatch] = 1
        word = np.random.zipf(alpha)
        while word != 3 and word != 386 and word != 858:
            sentence_lengths[ibatch] += 1
            word = np.random.zipf(alpha)
    return torch.tensor(sentence_lengths)

# Generate a batch of semi-realistic data using Zipf distribution for sentence lengths
# in the form of nested tensors with the jagged layout.
def gen_batch(N, E_q, E_k, E_v, device, dtype=torch.float32, query_seq_len_1=False):
    # generate semi-realistic data using Zipf distribution for sentence lengths
    sentence_lengths = zipf_sentence_lengths(alpha=1.2, batch_size=N)

    # Note: the torch.jagged layout is a nested tensor layout that supports a single ragged
    # dimension and works with torch.compile. The batch items each have shape (B, S*, D)
    # where B = batch size, S* = ragged sequence length, and D = embedding dimension.
    if query_seq_len_1:
        query = torch.nested.nested_tensor([
            torch.randn(1, E_q, dtype=dtype, device=device)
            for l in sentence_lengths
        ], layout=torch.jagged)
    else:
        query = torch.nested.nested_tensor([
            torch.randn(l.item(), E_q, dtype=dtype, device=device)
            for l in sentence_lengths
        ], layout=torch.jagged)

    key = torch.nested.nested_tensor([
        torch.randn(s.item(), E_k, dtype=dtype, device=device)
        for s in sentence_lengths
    ], layout=torch.jagged)

    value = torch.nested.nested_tensor([
        torch.randn(s.item(), E_v, dtype=dtype, device=device)
        for s in sentence_lengths
    ], layout=torch.jagged)

    return query, key, value, sentence_lengths

import timeit
import math

def benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = timeit.default_timer()
    output = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return output, (end - begin), torch.cuda.max_memory_allocated()

##############################################################################
# We will now demonstrate the performance improvements of using nested tensors
# in the ``MultiheadAttention`` layer + compile for self attention. We compare this against
# the traditional ``nn.MultiheadAttention`` + compile with padding and masking.

N, E_q, E_k, E_v, E_total = 512, 512, 512, 512, 512
E_out = E_q
d_model = E_q
nheads = 8
dropout = 0.0
bias = True
device='cuda'
torch.manual_seed(6)
query, key, value, sentence_lengths = gen_batch(N, E_q, E_k, E_v, device)
S = sentence_lengths.max().item()
print(f"Total sequence length in nested query {sentence_lengths.sum().item()}, max sequence length {S}")
padded_query, padded_key, padded_value = (
    t.to_padded_tensor(0.0) for t in (query, key, value)
)

torch.manual_seed(6)
mha_layer = MultiHeadAttention(E_q, E_k, E_v, E_total, nheads, dropout=dropout, bias=bias, device='cuda')
torch.manual_seed(6)
vanilla_mha_layer = nn.MultiheadAttention(E_q, nheads, dropout=dropout, batch_first=True, bias=bias, device='cuda')

# ``nn.MultiheadAttention`` uses a non conventional initialization for layers, so do this for exact parity :(
mha_layer.out_proj.weight = nn.Parameter(vanilla_mha_layer.out_proj.weight.clone().detach())
mha_layer.packed_proj.weight = nn.Parameter(vanilla_mha_layer.in_proj_weight.clone().detach())
mha_layer.out_proj.bias = nn.Parameter(vanilla_mha_layer.out_proj.bias.clone().detach())
mha_layer.packed_proj.bias = nn.Parameter(vanilla_mha_layer.in_proj_bias.clone().detach())

new_mha_layer = torch.compile(mha_layer)
# warmup compile
nested_result_warmup = new_mha_layer(query, query, query, is_causal=True)

# benchmark
nested_result, nested_time, nested_peak_memory = benchmark(new_mha_layer, query, query, query, is_causal=True)
padded_nested_result = nested_result.to_padded_tensor(0.0)

# For the vanilla ``nn.MultiheadAttention``, we need to construct the ``key_padding_mask``
# Further, ``nn.MultiheadAttention`` forces one to materialize the ``attn_mask`` even if using ``is_causal``
src_key_padding_mask = torch.where(padded_query == 0.0, -math.inf, 0)[:, :, 0]
attn_mask = torch.empty((N, S, S), device=device).fill_(float('-inf'))
for i, s in enumerate(sentence_lengths):
    attn_mask[i, :s, :s] = nn.Transformer.generate_square_subsequent_mask(s)
attn_mask = attn_mask.unsqueeze(1).expand(N, nheads, S, S).reshape(N*nheads, S, S)

vanilla_mha_layer = torch.compile(vanilla_mha_layer)
# warmup compile
warmup_vanilla_result = vanilla_mha_layer(padded_query,
                                          padded_query,
                                          padded_query,
                                          attn_mask=attn_mask,
                                          key_padding_mask=src_key_padding_mask,
                                          need_weights=False,
                                          is_causal=True)

# benchmark
(padded_result, _), padded_time, padded_peak_memory = benchmark(vanilla_mha_layer,
                                                                padded_query,
                                                                padded_query,
                                                                padded_query,
                                                                key_padding_mask=src_key_padding_mask,
                                                                need_weights=False,
                                                                attn_mask=attn_mask,
                                                                is_causal=True)

print(f"{padded_time=:.5f}, padded_peak_memory={padded_peak_memory/1e9:.2f} GB")
print(f"{nested_time=:.5f}, nested_peak_memory={nested_peak_memory/1e9:.2f} GB")
print("Difference between vanilla and nested result", (padded_result - padded_nested_result).abs().max().item())
print(f"Nested speedup: {(padded_time/nested_time):.2f}")
print(f"Nested peak memory reduction {((padded_peak_memory - nested_peak_memory)/1e9):.2f} GB")

######################################################################################
# For reference some sample outputs on A100:
# 
# ..code::
#   padded_time=0.03454, padded_peak_memory=4.14 GB
#   nested_time=0.00612, nested_peak_memory=0.76 GB
#   Difference between vanilla and nested result 0.0
#   Nested speedup: 5.65
#   Nested peak memory reduction 3.39 GB
#
# We can also see the same for backward pass

for i, entry_length in enumerate(sentence_lengths):
    # padding-specific step: remove output projection bias from padded entries for fair comparison
    padded_result[i, entry_length:, :] = 0.0

_, padded_bw_time, padded_bw_peak_mem = benchmark(lambda : padded_result.sum().backward())
_, nested_bw_time, nested_bw_peak_mem = benchmark(lambda : padded_nested_result.sum().backward())

print(f"{padded_bw_time=:.5f}, padded_bw_peak_mem={padded_bw_peak_mem/1e9:.2f} GB")
print(f"{nested_bw_time=:.5f}, nested_bw_peak_mem={nested_bw_peak_mem/1e9:.2f} GB")
print(f"Nested backward speedup: {(padded_bw_time/nested_bw_time):.2f}")
print(f"Nested backward peak memory reduction {((padded_bw_peak_mem - nested_bw_peak_mem)/1e9):.2f} GB")

print("Difference in out_proj.weight.grad", (mha_layer.out_proj.weight.grad - vanilla_mha_layer.out_proj.weight.grad).abs().max().item())
print("Difference in packed_proj.weight.grad", (mha_layer.packed_proj.weight.grad - vanilla_mha_layer.in_proj_weight.grad).abs().max().item())
print("Difference in out_proj.bias.grad", (mha_layer.out_proj.bias.grad - vanilla_mha_layer.out_proj.bias.grad).abs().max().item())
print("Difference in packed_proj.bias.grad", (mha_layer.packed_proj.bias.grad - vanilla_mha_layer.in_proj_bias.grad).abs().max().item())

##################################################################################
# Sample outputs on A100:
#
# ..code::
#   ``padded_bw_time``=2.09337, ``padded_bw_peak_mem``=5.10 GB
#   ``nested_bw_time``=0.01452, ``nested_bw_peak_mem``=3.24 GB
#   Nested backward speedup: 144.13
#   Nested backward peak memory reduction 1.86 GB
#   Difference in ``out_proj.weight.grad`` 0.000244140625
#   Difference in ``packed_proj.weight.grad`` 0.001556396484375
#   Difference in ``out_proj.bias.grad`` 0.0
#   Difference in ``packed_proj.bias.grad`` 0.001953125
#

##################################################################################
# GPT-style layer
# ---------------
# A basic GPT-style transformer layer consists of a causal self-attention layer
# followed by a feed-forward network (FFN) with skip connections. Implementing
# this is fairly straightforward using the ``MultiheadAttention`` layer above and
# gives equivalent results to an ``nn.TransformerEncoderLayer`` with
# ``is_causal=True``.
#
# We  demonstrate examples of implementing the rest of the ``nn`` layers
# `here <https://github.com/mikaylagawarecki/transformer_tutorial_accompaniment>`_
# but omit that from this tutorial for brevity.


###############################################################################
# Going one step further
# ----------------------
# So far, we have demonstrated how to implement a performant ``MultiheadAttention``
# layer that follows the traditional ``nn.MultiheadAttention``. Going back to our
# classification of modifications to the transformer architecture, recall that we
# classified the modifications into layer type, layer ordering, and modifications
# to the attention score. We trust that changing layer type and layer ordering
# (e.g. swapping ``LayerNorm`` for ``RMSNorm``) is fairly straightforward.
# 
# In this section, we will discuss various functionalities using the
# aforementioned building blocks. In particular,
# 
# * Cross Attention
# * Fully masked rows no longer cause NaNs
# * Modifying attention score: ALiBi with FlexAttention and NJT
# * Packed Projection

###############################################################################
# Cross Attention
# ---------------
# Cross attention is a form of attention where the query and key/value tensors
# are from different sequences.
#
# One example of this is in ``nn.TransformerDecoderLayer`` where the query comes
# from the decoder and the key/value come from the encoder.
#
# The above MultiheadAttention layer nicely generalizes to this case with nested
# tensors for both query and key/value.

query, _, _, q_len = gen_batch(N, E_q, E_k, E_v, device)
_, key, value, kv_len = gen_batch(N, E_q, E_k, E_v, device)

print(f"Total sequence length in nested query {q_len.sum().item()}, max sequence length {q_len.max().item()}")
print(f"Total sequence length in nested key/value {kv_len.sum().item()}, max sequence length {kv_len.max().item()}")
out = new_mha_layer(query, key, value, is_causal=False)


################################################################################
# Fully masked rows no longer cause NaNs
# --------------------------------------
# 
# There has been a long standing issue with ``nn.MultiheadAttention`` and
# ``scaled_dot_product_attention`` where if a row was fully masked out, the output
# of the attention layer would be NaN. See `issue <https://github.com/pytorch/pytorch/issues/41508>`_.
# This is because the softmax over an empty set is undefined.
# 
# Thanks to `this PR <https://github.com/pytorch/pytorch/pull/133882>`_
# this is no longer the case. Instead, fully masked rows in ``scaled_dot_product_attention``.
# For cases where ``nn.MHA`` does not employ the "fast-path", this will also apply.
#
# Using a custom MHA layer with NJTs is strongly recommended over the
# existing "fast-path" in ``nn.MultiheadAttention`` as NJT's ability to model raggedness
# appropriately makes it possible to properly express empty sequences.


################################################################################
# FlexAttention + NJT
# ---------------------------------------------------------------------
# NJT also composes with the ``FlexAttention`` module. This is a generalization
# of the ``MultiheadAttention`` layer that allows for arbitrary modifications
# to the attention score. The example below takes the ``alibi_mod``
# that implements `ALiBi <https://arxiv.org/abs/2108.12409>`_ from
# `attention gym <https://github.com/pytorch-labs/attention-gym>`_ and uses it
# with nested input tensors.

from torch.nn.attention.flex_attention import flex_attention

def generate_alibi_bias(H: int):
    """Returns an alibi bias score_mod given the number of heads H
    Args:
        H: number of heads
    Returns:
        alibi_bias: alibi bias score_mod
    """
    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (q_idx - kv_idx) * scale
        return score + bias
    return alibi_mod

query, key, value, _ = gen_batch(N, E_q, E_k, E_v, device)
n_heads, D = 8, E_q // 8
alibi_score_mod = generate_alibi_bias(n_heads)
query = (
    query.unflatten(-1, [n_heads, D]).transpose(1, 2).detach().requires_grad_()
)
key = key.unflatten(-1, [n_heads, D]).transpose(1, 2).detach().requires_grad_()
value = (
    value.unflatten(-1, [n_heads, D]).transpose(1, 2).detach().requires_grad_()
)
out_flex2 = flex_attention(query, key, value, score_mod=alibi_score_mod)

###############################################################################
# Packed Projection
# -----------------
# 
# Packed projection is a technique that makes use of the fact that when the input
# for projection (matrix multiplications) are the same (self-attention), we can pack the projection
# weights and biases into single tensors. It is especially useful when the individual
# projections are memory bound rather than compute bound. There are
# two examples that we will demonstrate here:
# 
# * Input projection for MultiheadAttention
# * SwiGLU activation in feed-forward network of Transformer Layer
# 
# Input projection for MultiheadAttention
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Recall that when doing self-attention, the ``query``, ``key`` and ``value``
# are the same tensor. Each of these tensors is projected with a 
# ``Linear(E_q, E_total)`` layer. Instead, we can pack this into one layer,
# which is what we do in the MultiheadAttention layer above.
# 
# Let us compare the performance of the packed projection against the usual method:

class InputProjection(nn.Module):
    def __init__(self, E_q, E_total, bias=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)

    def forward(self, query):  
        return self.q_proj(query), self.k_proj(query), self.v_proj(query)

class PackedInputProjection(nn.Module):
    def __init__(self, E_q, E_total, bias=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)

    def forward(self, query):  
        return torch.chunk(self.packed_proj(query), 3, dim=-1)

B, D, dtype = 256, 4096, torch.bfloat16

torch.set_float32_matmul_precision('high')
in_proj = torch.compile(InputProjection(D, D, device='cuda', dtype=torch.bfloat16))
packed_in_proj = torch.compile(PackedInputProjection(D, D, device='cuda', dtype=torch.bfloat16))

q, _, _, sequence_lengths = gen_batch(B, D, D, D, device='cuda', dtype=torch.bfloat16)

# warmup
in_proj(q)
packed_in_proj(q)

# benchmark
(q_out, k_out, v_out), time, _ = benchmark(in_proj, q)
(q_out, k_out, v_out), time_packed, _ = benchmark(packed_in_proj, q)
print(f"InputProjection: {time:5f} s, PackedInputProjection: {time_packed:5f} s, speedup: {time/time_packed:.2f}x")

##################################################
# SwiGLU feed forward network of Transformer Layer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SwiGLU is a non-linear activation function that is increasingly popular in the feed-forward
# network of the transformer layer (e.g. Llama). A feed-forward network with SwiGLU activation is defined as

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

########################################################################
# An alternative way of implementing this that uses packed projection is 

class PackedSwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)
    
    def forward(self, x):
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return self.w2(F.silu(x1) * x3)

################################################################################
# We can compare the performance of the two implementations as follows
# Depending on your hardware, you might see different results. On an A100 I see
# 1.12x speedup for D=128.
D = 128

swigluffn = torch.compile(SwiGLUFFN(D, D * 4, 256, device='cuda', dtype=torch.bfloat16))
packed_swigluffn = torch.compile(PackedSwiGLUFFN(D, D * 4, 256, device='cuda', dtype=torch.bfloat16))

q, _, _, sentence_lengths = gen_batch(D, D, D, D, device="cuda", dtype=torch.bfloat16)

# warmup
swigluffn(q)
packed_swigluffn(q)

# benchmark
_, time, _ = benchmark(swigluffn, q)
_, time_packed, _ = benchmark(packed_swigluffn, q)
print(f"SwiGLUFFN: {time} s, PackedSwiGLUFFN: {time_packed} s, speedup: {time/time_packed:.2f}x")

################################################################################
# Extended examples
# -----------------
# 
# We intend to update this tutorial to demonstrate more examples of how to use
# the various performant building blocks such as KV-Caching, Grouped Query Attention
# etc. Further, there are several good examples of using various performant building blocks to
# implement various transformer architectures. Some examples include
#
# * `gpt-fast <https://github.com/pytorch-labs/gpt-fast>`_
# * `segment-anything-fast <https://github.com/pytorch-labs/segment-anything-fast>`_
# * `lucidrains implementation of NaViT with nested tensors <https://github.com/lucidrains/vit-pytorch/blob/73199ab486e0fad9eced2e3350a11681db08b61b/vit_pytorch/na_vit_nested_tensor.py>`_
# * `torchtune's implementation of VisionTransformer <https://github.com/pytorch/torchtune/blob/a8a64ec6a99a6ea2be4fdaf0cd5797b03a2567cf/torchtune/modules/vision_transformer.py#L16>`_
