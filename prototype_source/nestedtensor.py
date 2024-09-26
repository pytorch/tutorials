"""

Getting Started with Nested Tensors
===============================================================

Nested tensors generalize the shape of regular dense tensors, allowing for representation
of ragged-sized data.

* for a regular tensor, each dimension is regular and has a size

* for a nested tensor, not all dimensions have regular sizes; some of them are ragged

Nested tensors are a natural solution for representing sequential data within various domains:

* in NLP, sentences can have variable lengths, so a batch of sentences forms a nested tensor

* in CV, images can have variable shapes, so a batch of images forms a nested tensor

In this tutorial, we will demonstrate basic usage of nested tensors and motivate their usefulness
for operating on sequential data of varying lengths with a real-world example. In particular,
they are invaluable for building transformers that can efficiently operate on ragged sequential
inputs. Below, we present an implementation of multi-head attention using nested tensors that,
combined usage of ``torch.compile``, out-performs operating naively on tensors with padding.

Nested tensors are currently a prototype feature and are subject to change.
"""

import numpy as np
import timeit
import torch
import torch.nn.functional as F

from torch import nn

torch.manual_seed(1)
np.random.seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################################
# Nested tensor initialization
# ----------------------------
#
# From the Python frontend, a nested tensor can be created from a list of tensors.
# We denote nt[i] as the ith tensor component of a nestedtensor.
nt = torch.nested.nested_tensor([torch.arange(12).reshape(
    2, 6), torch.arange(18).reshape(3, 6)], dtype=torch.float, device=device)
print(f"{nt=}")

######################################################################
# By padding every underlying tensor to the same shape,
# a nestedtensor can be converted to a regular tensor.
padded_out_tensor = torch.nested.to_padded_tensor(nt, padding=0.0)
print(f"{padded_out_tensor=}")

######################################################################
# All tensors posses an attribute for determining if they are nested;
print(f"nt is nested: {nt.is_nested}")
print(f"padded_out_tensor is nested: {padded_out_tensor.is_nested}")

######################################################################
# It is common to construct nestedtensors from batches of irregularly shaped tensors.
# i.e. dimension 0 is assumed to be the batch dimension.
# Indexing dimension 0 gives back the first underlying tensor component.
print("First underlying tensor component:", nt[0], sep='\n')
print("last column of 2nd underlying tensor component:", nt[1, :, -1], sep='\n')

# When indexing a nestedtensor's 0th dimension, the result is a regular tensor.
print(f"First underlying tensor component is nested: {nt[0].is_nested}")

######################################################################
# An important note is that slicing in dimension 0 has not been supported yet.
# Which means it not currently possible to construct a view that combines the underlying
# tensor components.

######################################################################
# Nested Tensor Operations
# ------------------------
#
# As each operation must be explicitly implemented for nestedtensors,
# operation coverage for nestedtensors is currently narrower than that of regular tensors.
# For now, only basic operations such as index, dropout, softmax, transpose, reshape, linear, bmm are covered.
# However, coverage is being expanded.
# If you need certain operations, please file an `issue <https://github.com/pytorch/pytorch>`__
# to help us prioritize coverage.
#
# **reshape**
#
# The reshape op is for changing the shape of a tensor.
# Its full semantics for regular tensors can be found
# `here <https://pytorch.org/docs/stable/generated/torch.reshape.html>`__.
# For regular tensors, when specifying the new shape,
# a single dimension may be -1, in which case it is inferred
# from the remaining dimensions and the number of elements.
#
# The semantics for nestedtensors are similar, except that -1 no longer infers.
# Instead, it inherits the old size (here 2 for ``nt[0]`` and 3 for ``nt[1]``).
# -1 is the only legal size to specify for a jagged dimension.
nt_reshaped = nt.reshape(2, -1, 2, 3)
print(f"{nt_reshaped=}")

######################################################################
# **transpose**
#
# The transpose op is for swapping two dimensions of a tensor.
# Its full semantics can be found
# `here <https://pytorch.org/docs/stable/generated/torch.transpose.html>`__.
# Note that for nestedtensors dimension 0 is special;
# it is assumed to be the batch dimension,
# so transposes involving nestedtensor dimension 0 are not supported.
nt_transposed = nt_reshaped.transpose(1, 2)
print(f"{nt_transposed=}")

######################################################################
# **others**
#
# Other operations have the same semantics as for regular tensors.
# Applying the operation on a nestedtensor is equivalent to
# applying the operation to the underlying tensor components,
# with the result being a nestedtensor as well.
nt_mm = torch.nested.nested_tensor([torch.randn((2, 3, 4)), torch.randn((2, 3, 5))], device=device)
nt3 = torch.matmul(nt_transposed, nt_mm)
print(f"Result of Matmul:\n {nt3}")

nt4 = F.dropout(nt3, 0.1)
print(f"Result of Dropout:\n {nt4}")

nt5 = F.softmax(nt4, -1)
print(f"Result of Softmax:\n {nt5}")

######################################################################
# Why Nested Tensor
# -----------------
#

######################################################################
# When data is sequential, it is often the case that each sample has a different length.
# For example, in a batch of sentences, each sentence has a different number of words.
# A common technique for handling varying sequences is to manually pad each data tensor
# to the same shape in order to form a batch.
# For example, we have 2 sentences with different lengths and a vocabulary
# In order to represent his as single tensor we pad with 0 to the max length in the batch.
sentences = [["goodbye", "padding"],
             ["embrace", "nested", "tensor"]]
vocabulary = {"goodbye": 1.0, "padding": 2.0,
              "embrace": 3.0, "nested": 4.0, "tensor": 5.0}
padded_sentences = torch.tensor([[1.0, 2.0, 0.0],
                                 [3.0, 4.0, 5.0]])
nested_sentences = torch.nested.nested_tensor([torch.tensor([1.0, 2.0]),
                                               torch.tensor([3.0, 4.0, 5.0])])
print(f"{padded_sentences=}")
print(f"{nested_sentences=}")

######################################################################
# This technique of padding a batch of data to its max length is not optimal.
# The padded data is not needed for computation and wastes memory by allocating
# larger tensors than necessary.
# Further, not all operations have the same semnatics when applied to padded data.
# For matrix multiplications in order to ignore the padded entries, one needs to pad
# with 0 while for softmax one has to pad with -inf to ignore specific entries.
# The primary objective of nested tensor is to facilitate operations on ragged
# data using the standard PyTorch tensor UX, thereby eliminating the need
# for inefficient and complex padding and masking.
padded_sentences_for_softmax = torch.tensor([[1.0, 2.0, float("-inf")],
                                             [3.0, 4.0, 5.0]])
print(F.softmax(padded_sentences_for_softmax, -1))
print(F.softmax(nested_sentences, -1))

######################################################################
# Let us take a look at a practical example: the multi-head attention component
# utilized in `Transformers <https://arxiv.org/pdf/1706.03762.pdf>`__.
# We can implement this in such a way that it can operate on either padded
# or nested tensors.
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
        dropout_p (float, optional): Dropout probability. Default: 0.0
    """
    def __init__(self, E_q: int, E_k: int, E_v: int, E_total: int,
                 nheads: int, dropout_p: float = 0.0):
        super().__init__()
        self.nheads = nheads
        self.dropout_p = dropout_p
        self.query_proj = nn.Linear(E_q, E_total)
        self.key_proj = nn.Linear(E_k, E_total)
        self.value_proj = nn.Linear(E_v, E_total)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_t, E_q)
            key (torch.Tensor): key of shape (N, L_s, E_k)
            value (torch.Tensor): value of shape (N, L_s, E_v)

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        # TODO: demonstrate packed projection
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

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
            query, key, value, dropout_p=dropout_p, is_causal=True)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output

######################################################################
# set hyperparameters following `the Transformer paper <https://arxiv.org/pdf/1706.03762.pdf>`__
N = 512
E_q, E_k, E_v, E_total = 512, 512, 512, 512
E_out = E_q
nheads = 8

######################################################################
# except for dropout probability: set to 0 for correctness check
dropout_p = 0.0

######################################################################
# Let us generate some realistic fake data from Zipf's law.
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

######################################################################
# Create nested tensor batch inputs
def gen_batch(N, E_q, E_k, E_v, device):
    # generate semi-realistic data using Zipf distribution for sentence lengths
    sentence_lengths = zipf_sentence_lengths(alpha=1.2, batch_size=N)

    # Note: the torch.jagged layout is a nested tensor layout that supports a single ragged
    # dimension and works with torch.compile. The batch items each have shape (B, S*, D)
    # where B = batch size, S* = ragged sequence length, and D = embedding dimension.
    query = torch.nested.nested_tensor([
        torch.randn(l.item(), E_q, device=device)
        for l in sentence_lengths
    ], layout=torch.jagged)

    key = torch.nested.nested_tensor([
        torch.randn(s.item(), E_k, device=device)
        for s in sentence_lengths
    ], layout=torch.jagged)

    value = torch.nested.nested_tensor([
        torch.randn(s.item(), E_v, device=device)
        for s in sentence_lengths
    ], layout=torch.jagged)

    return query, key, value, sentence_lengths

query, key, value, sentence_lengths = gen_batch(N, E_q, E_k, E_v, device)

######################################################################
# Generate padded forms of query, key, value for comparison
def jagged_to_padded(jt, padding_val):
    # TODO: do jagged -> padded directly when this is supported
    return torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(list(jt.unbind())),
        padding_val)

padded_query, padded_key, padded_value = (
    jagged_to_padded(t, 0.0) for t in (query, key, value)
)

######################################################################
# Construct the model
mha = MultiHeadAttention(E_q, E_k, E_v, E_total, nheads, dropout_p).to(device=device)

######################################################################
# Check correctness and performance
def benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    begin = timeit.default_timer()
    output = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    return output, (end - begin)

output_nested, time_nested = benchmark(mha, query, key, value)
output_padded, time_padded = benchmark(mha, padded_query, padded_key, padded_value)

# padding-specific step: remove output projection bias from padded entries for fair comparison
for i, entry_length in enumerate(sentence_lengths):
    output_padded[i, entry_length:] = 0.0

print("=== without torch.compile ===")
print("nested and padded calculations differ by", (jagged_to_padded(output_nested, 0.0) - output_padded).abs().max().item())
print("nested tensor multi-head attention takes", time_nested, "seconds")
print("padded tensor multi-head attention takes", time_padded, "seconds")

# warm up compile first...
compiled_mha = torch.compile(mha)
compiled_mha(query, key, value)
# ...now benchmark
compiled_output_nested, compiled_time_nested = benchmark(
    compiled_mha, query, key, value)

# warm up compile first...
compiled_mha(padded_query, padded_key, padded_value)
# ...now benchmark
compiled_output_padded, compiled_time_padded = benchmark(
    compiled_mha, padded_query, padded_key, padded_value)

# padding-specific step: remove output projection bias from padded entries for fair comparison
for i, entry_length in enumerate(sentence_lengths):
    compiled_output_padded[i, entry_length:] = 0.0

print("=== with torch.compile ===")
print("nested and padded calculations differ by", (jagged_to_padded(compiled_output_nested, 0.0) - compiled_output_padded).abs().max().item())
print("nested tensor multi-head attention takes", compiled_time_nested, "seconds")
print("padded tensor multi-head attention takes", compiled_time_padded, "seconds")

######################################################################
# Note that without ``torch.compile``, the overhead of the python subclass nested tensor
# can make it slower than the equivalent computation on padded tensors. However, once
# ``torch.compile`` is enabled, operating on nested tensors gives a multiple x speedup.
# Avoiding wasted computation on padding becomes only more valuable as the percentage
# of padding in the batch increases.
print(f"Nested speedup: {compiled_time_padded / compiled_time_nested:.3f}")

######################################################################
# Conclusion
# ----------
# In this tutorial, we have learned how to perform basic operations with nested tensors and
# how implement multi-head attention for transformers in a way that avoids computation on padding.
# For more information, check out the docs for the
# `torch.nested <https://pytorch.org/docs/stable/nested.html>`__ namespace.
