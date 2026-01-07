"""
Using Variable Length Attention in PyTorch
==========================================

.. meta::
   :description: Learn how to use PyTorch's varlen_attn API for efficient variable length attention without padding. Complete tutorial with code examples for training Transformers with packed sequences.
   :keywords: pytorch, attention, variable length, transformer, varlen_attn, packed sequences, memory optimization, torch.compile

**Author:** `Angel Li <https://github.com/liangel-02>`_


In this tutorial, we will introduce a variable length attention API.
This API is called ``varlen_attn`` and is a custom op in PyTorch,
meaning it is also compilable using ``torch.compile``.

"""

######################################################################
#    | **Note:**
#    | This tutorial currently requires you to use the PyTorch nightly
#      build. This op currently only works with NVIDIA CUDA on A100 machines
#      or newer. Supported dtypes include BF16 and FP16.
#
# What you will learn
# ~~~~~~~~~~~~~~~~~~~
#
# -  Variable length attention and how it differs from
#    Scaled Dot Product Attention (SDPA)
# -  Explore an example of how to use ``varlen_attn`` in a simple
#    Transformer attention layer
#
# Prerequisites
# ~~~~~~~~~~~~~
#
# -  PyTorch v2.10.0.dev or later
# -  NVIDIA A100 GPU or newer
# -  A basic understanding of attention and our current offerings. Please
#    reference these tutorials for more details on `FlexAttention <https://pytorch.org/blog/flexattention/>`__ and
#    `SDPA <https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>`__.
#


######################################################################
# Overview of Variable Length Attention
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In normal SDPA, sequences are expected to be a fixed length. In
# practice, this means that input tensors are often **padded** to the same
# length in a batch. However, this wastes both memory and compute through
# storing this padding and performing unnecessary computations.
# Variable length attention handles sequences of varying length by
# **packing** the tensors in a batch together and essentially collapsing
# the batch dimension.


######################################################################
# However, we still need to maintain the boundaries between documents. To
# do so, we compute cumulative sequence positions for query and key/value
# that mark the end of documents. In the diagram below, doc 1 is 7 tokens
# long, doc 2 is 10 tokens long, etc. so ``cu_seq_lens = [0, 7, 17, ...]``.
#
# .. figure:: ../_static/img/varlen_diagram.png
#    :alt: Padding vs Packing Diagram
#
#    Padding vs Packing Diagram


######################################################################
# Note that ``NestedTensor`` is another way to enable
# variable length with packed tensors (see tutorial
# `here <https://docs.pytorch.org/tutorials/unstable/nestedtensor.html>`__).


######################################################################
# Definition
# ----------
#
# Below is the definition of ``varlen_attn`` which returns the output
# tensor from the attention computation.
#
# .. code:: python
#
#    def varlen_attn(
#        query: torch.Tensor,
#        key: torch.Tensor,
#        value: torch.Tensor,
#        cu_seq_q: torch.Tensor,
#        cu_seq_k: torch.Tensor,
#        max_q: int,
#        max_k: int,
#        is_causal: bool = False,
#        return_aux: AuxRequest | None = None,
#    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
#
# ``query``, ``key``, and ``value`` correspond to the ``q``, ``k``, and
# ``v`` of the packed input. ``cu_seq_q`` and ``cu_seq_k`` are the
# cumulative indices for query and key/value, respectively. These mark the
# logical boundaries that separate the documents in our input. ``max_q``
# and ``max_k`` are the maximum sequence lengths of query and key,
# respectively. ``is_causal`` applies causal masking if set to True and
# ``return_aux`` specifies which auxiliary outputs to return (ie ``lse``).

######################################################################
# **Note on causal masking**
# When ``is_causal`` is set to True, causal masking is applied which means
# that tokens can only attend to previous tokens. For bidirectional
# attention, set this flag to False.
#
# In torchtitan (PyTorch's pretraining framework), we set
# ``is_causal = True`` uniformly to prevent the model from cheating and
# artificially driving the loss down too quickly.


######################################################################
# Example
# ~~~~~~~
#
# Let’s walk through a simple example of how we would use ``varlen_attn``
# in the context of training a Transformer model.
#


######################################################################
# Creating Required Metadata for ``varlen_attn`` from Input Batches
# -----------------------------------------------------------------
#
# Given an input batch, how would we construct the metadata that
# ``varlen_attn`` expects? More specifically, how do we calculate the
# cumulative sequence indices?
#
# The helper function ``create_varlen_metadata`` returns the required
# ``cu_seqlens`` and ``max_seqlen`` given ``input_batch`` and the end of
# sequence token ID that marks the end of documents.
#

import torch


def create_varlen_metadata(input_batch: torch.Tensor, eos_id: int):
    batch_size, seq_len = input_batch.shape
    device = input_batch.device
    cu_seqlens_list, all_seq_lengths = [], []
    offset = 0

    for b in range(batch_size):
        tokens = input_batch[b]
        eos_positions = (tokens == eos_id).nonzero(as_tuple=True)[0].to(torch.int32)

        # we use the position of the eos tokens to mark the end of documents
        sample_cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                eos_positions + 1,
                torch.tensor([seq_len], dtype=torch.int32, device=device),
            ]
        )
        sample_cu_seqlens = torch.unique_consecutive(sample_cu_seqlens)

        seq_lengths = torch.diff(sample_cu_seqlens)
        all_seq_lengths.append(seq_lengths)

        cu_seqlens_adjusted = sample_cu_seqlens[:-1] + offset
        cu_seqlens_list.append(cu_seqlens_adjusted)

        offset += seq_len

    packed_cu_seqlens = torch.cat(
        cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)]
    )

    max_seqlen = 0
    if len(all_seq_lengths) > 0:
        all_seq_lengths = torch.cat(all_seq_lengths)
        max_seqlen = all_seq_lengths.max().item()

    return packed_cu_seqlens, max_seqlen


######################################################################
# Implementing the Attention Block with ``varlen_attn``
# -----------------------------------------------------
#
# Let's explore how we would use ``varlen_attn`` in an Attention module.
# We define an attention module as usual, but in the ``forward`` method,
# we call the new ``varlen_attn`` custom op.
#
# This function expects the ``cu_seq`` indices and ``max_len`` that we
# computed earlier using ``create_varlen_metadata`` to mark the boundaries
# of the different documents.
#
# Before we call ``varlen_attn``, we also pack our input so that it has
# the shape ``(total tokens, dim)``. Recall that variable length attention
# allows us to collapse the ``batch_size`` dimension so that we can lay
# out our input samples contiguously.
#

import torch
import torch.nn as nn
from torch.nn.attention.varlen import varlen_attn


class SimpleVarlenAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self, x: torch.Tensor, cu_seq: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_packed = x.view(batch_size * seq_len, -1)  # pack x into (total_tokens, dim)

        qkv = self.qkv_proj(x_packed)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        attn_out = varlen_attn(
            query=q,
            key=k,
            value=v,
            cu_seq_q=cu_seq,
            cu_seq_k=cu_seq,
            max_q=max_len,
            max_k=max_len,
            is_causal=True,
        )
        attn_out = attn_out.view(-1, self.embed_dim)
        attn_out = self.out_proj(attn_out)
        return attn_out.view(batch_size, seq_len, self.embed_dim)


######################################################################
# We can also use ``torch.compile`` with ``varlen_attn`` and define:
#
# .. code:: python
#
#    compiled_varlen_attn: ClassVar[Callable] = torch.compile(
#        varlen_attn, mode="max-autotune-no-cudagraphs"
#    )
#
# We can call ``compiled_varlen_attn`` instead of ``varlen_attn`` in the
# Attention forward, and everything else stays the same.


######################################################################
# Creating a Transformer
# ----------------------
#
# Now, we can use this ``SimpleVarlenAttention`` module in a simple
# Transformer.
#


class SimpleVarlenTransformer(nn.Module):
    """
    simple 1 layer transformer with varlen attention
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.attention = SimpleVarlenAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, tokens: torch.Tensor, cu_seq: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        x = self.tok_embeddings(tokens)
        x = x + self.attention(x, cu_seq, max_len)
        x = self.norm(x)
        return x


######################################################################
# Running a Training Step
# -----------------------
#
# Now we’re ready to put all the pieces together! Let’s run a training
# step with our ``SimpleVarlenTransformer``. We define our model, compute
# ``cu_seq`` and ``max_len`` using ``create_varlen_metadata``, and run a
# forward and backward pass.
#


def main():
    torch.manual_seed(42)

    batch_size = 3
    seq_len = 64
    vocab_size = 1000
    embed_dim = 128
    num_heads = 4
    eos_id = 2
    num_docs = 3
    device = "cuda"
    dtype = torch.bfloat16

    model = SimpleVarlenTransformer(vocab_size, embed_dim, num_heads).to(
        device=device, dtype=dtype
    )

    # create input_batch tokens
    input_batch = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for b in range(batch_size):
        # getting random positions to cut the input into multiple documents
        doc_positions = torch.randint(10, seq_len - 1, (num_docs - 1,))
        for pos in doc_positions:
            input_batch[b, pos] = eos_id  # insert eos token to simulate end of sample
        input_batch[b, -1] = eos_id

    cu_seq, max_len = create_varlen_metadata(input_batch, eos_id)
    print(
        f"cu_seq: {cu_seq}, max_len: {max_len}"
    )  # cu_seq: tensor([0, 32, 47, 64, 92, 103, 128, 168, 177, 192]), max_len: 40

    # fwd pass
    output = model(input_batch, cu_seq, max_len)
    print(f"output shape: {output.shape}")  # (3, 64, 128)

    # bwd pass
    loss = output.mean()
    loss.backward()


if __name__ == "__main__":
    main()


######################################################################
# Conclusion
# ~~~~~~~~~~
#
# In this tutorial, we have covered how to use the ``varlen_attn`` API in PyTorch to efficiently
# handle sequences of varying lengths without padding. We explored how to create the
# necessary metadata including the cumulative sequence indices, implemented a simple
# Transformer attention layer with variable length attention, and ran a complete
# training step.

######################################################################
# This approach eliminates wasted computation on padding tokens
# and enables more efficient training and inference for models processing
# documents of different lengths.
#
# .. seealso::
#
#    - `Implementing High-Performance Transformers with Scaled Dot Product Attention <https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>`_
#    - `torch.nn.functional.scaled_dot_product_attention <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>`_
