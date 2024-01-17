"""

NestedTensors
===============================================================

NestedTensors are similar to regular tensors, except for their shape:

* for a regular tensor, each dimension has a size

* for a nestedtensor, not all dimensions have regular sizes; some of them are jagged

Nestedtensors are a natural solution for representing sequential data within various domains:

* in NLP, sentences can have variable lengths, so a batch of sentences forms a nestedtensor

* in CV, images can have variable shapes, so a batch of images forms a nestedtensor

In this tutorial, we will demonstrate basic usage of nestedtensors and motivate their usefulness
for operating on sequential data of varying lengths with a real-world example.

NestedTensor are currently a prototype feature and are subject to change.
"""

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################################
# NestedTensor Initialization
# ----------------------------
#
# From the Python frontend, a nestedtensor can be created from a list of tensors.
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
# This techinque of padding a batch of data to its max length is not optimal.
# The padded data is not needed for computation and wastes memory by allocating
# larger tensors than necessary.
# Further, not all operations have the same semnatics when applied to padded data.
# For matrix multiplications in order to ignore the padded entries, one needs to pad
# with 0 while for softmax one has to pad with -inf to ignore specific entries.
padded_sentences_for_softmax = torch.tensor([[1.0, 2.0, float("-inf")],
                                             [3.0, 4.0, 5.0]])
print(F.softmax(padded_sentences_for_softmax, -1))
print(F.softmax(nested_sentences, -1))

######################################################################
# Let us take a look at a practical example: the multi-head attention component
# utilized in `Transformers <https://arxiv.org/pdf/1706.03762.pdf>`__.
# The nestedtensor version is straightforward.
import math

def mha_nested(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, nheads: int,
               W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor, W_out: torch.Tensor,
               b_q: torch.Tensor = None, b_k: torch.Tensor = None, b_v: torch.Tensor = None, b_out: torch.Tensor = None,
               dropout_p: float = 0.0) -> torch.Tensor:
    """Compute multi-head attention with nested tensors.
    Args:
        query (torch.Tensor): query of shape (N, L_t, E_q)
        key (torch.Tensor): key of shape (N, L_s, E_k)
        value (torch.Tensor): value of shape (N, L_s, E_v)
        nheads (int): number of heads in multi-head attention
        W_q (torch.Tensor): Weight for query input projection of shape (E_total, E_q)
        W_k (torch.Tensor): Weight for key input projection of shape (E_total, E_k)
        W_v (torch.Tensor): Weight for value input projection of shape (E_total, E_v)
        W_out (torch.Tensor): Weight for output projection of shape (E_out, E_total)
        b_q (torch.Tensor, optional): Bias for query input projection of shape E_total. Default: None. Defaults to None.
        b_k (torch.Tensor, optional): Bias for key input projection of shape E_total. Default: None. Defaults to None.
        b_v (torch.Tensor, optional): Bias for value input projection of shape E_total. Default: None. Defaults to None.
        b_out (torch.Tensor, optional): Bias for output projection of shape E_out. Default: None. Defaults to None.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

        Where:
            N is the batch size
            L_t is the target sequence length (jagged)
            L_s is the source sequence length (jagged)
            E_q is the embedding size for query
            E_k is the embedding size for key
            E_v is the embedding size for value
            E_total is the embedding size for all heads combined
            E_out is the output embedding size
    Returns:
        torch.Tensor:  Output of shape (N, L_t, E_out)
    """

    N = query.size(0)
    E_total = W_q.size(0)
    assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
    E_head = E_total // nheads

    # apply input projection
    # (N, L_t, E_q) -> (N, L_t, E_total)
    query = F.linear(query, W_q, b_q)
    # (N, L_s, E_k) -> (N, L_s, E_total)
    key = F.linear(key, W_k, b_k)
    # (N, L_s, E_v) -> (N, L_s, E_total)
    value = F.linear(value, W_v, b_v)

    # reshape query, key, value to separate by head
    # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
    query = query.reshape(N, -1, nheads, E_head).transpose(1, 2)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
    key = key.reshape(N, -1, nheads, E_head).transpose(1, 2)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
    value = value.reshape(N, -1, nheads, E_head).transpose(1, 2)

    # query matmul key^T
    # (N, nheads, L_t, E_head) x (N, nheads, L_s, E_head)^T -> (N, nheads, L_t, L_s)
    keyT = key.transpose(-1, -2)
    attn_weights = torch.matmul(query, keyT)

    # scale down
    attn_weights = attn_weights * (1.0 / math.sqrt(E_head))

    # softmax
    attn_weights = F.softmax(attn_weights, dim=-1)

    # dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # attention_weights matmul value
    # (N, nheads, L_t, L_s) x (N, nheads, L_s, E_head) -> (N, nheads, L_t, E_head)
    attn_output = torch.matmul(attn_weights, value)

    # merge heads
    # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
    attn_output = attn_output.transpose(1, 2).reshape(N, -1, E_total)

    # apply output projection
    # (N, L_t, E_total) -> (N, L_t, E_out)
    attn_output = F.linear(attn_output, W_out, b_out)

    return attn_output

######################################################################
# The 0-padded tensor version additionally requires masks
# for more complicated treatments at padded entries.
def mha_padded(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, nheads: int,
               attn_mask_q: torch.Tensor, attn_mask_kv: torch.Tensor,
               W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor, W_out: torch.Tensor,
               b_q: torch.Tensor = None, b_k: torch.Tensor = None, b_v: torch.Tensor = None, b_out: torch.Tensor = None,
               dropout_p: float = 0.0) -> torch.Tensor:
    """Compute multi-head attention for padded out dense tensors.

    Args:
        query (torch.Tensor): query of shape (N, L_t, E_q)
        key (torch.Tensor): key of shape (N, L_s, E_k)
        value (torch.Tensor): value of shape (N, L_s, E_v)
        nheads (int): number of heads in multi-head attention
        attn_mask_q (torch.Tensor): boolean mask indicating locations that should not take part in attention for query, shape (N, L_t)
        attn_mask_kv (torch.Tensor): boolean mask indicating locations that should not take part in attention for key and value, shape (N, L_s)
        W_q (torch.Tensor): Weight for query input projection of shape (E_total, E_q)
        W_k (torch.Tensor): Weight for key input projection of shape (E_total, E_k)
        W_v (torch.Tensor): Weight for value input projection of shape (E_total, E_v)
        W_out (torch.Tensor): Weight for output projection of shape (E_out, E_total)
        b_q (torch.Tensor, optional): Bias for query input projection of shape E_total.. Defaults to None.
        b_k (torch.Tensor, optional): Bias for key input projection of shape E_total.. Defaults to None.
        b_v (torch.Tensor, optional): Bias for value input projection of shape E_total.. Defaults to None.
        b_out (torch.Tensor, optional): Bias for output projection of shape E_out. Defaults to None.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

        Where:
            N is the batch size
            L_t is the target sequence length (padded)
            L_s is the source sequence length (padded)
            E_q is the embedding size for query
            E_k is the embedding size for key
            E_v is the embedding size for value
            E_total is the embedding size for all heads combined
            E_out is the output embedding size
    Returns:
        torch.Tensor: Output of shape (N, L_t, E_out)
    """
    N = query.size(0)
    L_t = query.size(1)
    L_s = key.size(1)
    E_total = W_q.size(0)
    assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
    assert L_t == L_s, "This implementation assumes equal query and key sequence lengths"
    E_head = E_total // nheads

    # apply input projection
    # (N, L_t, E_q) -> (N, L_t, E_total)
    query = F.linear(query, W_q, b_q)
    # (N, L_s, E_k) -> (N, L_s, E_total)
    key = F.linear(key, W_k, b_k)
    # (N, L_s, E_v) -> (N, L_s, E_total)
    value = F.linear(value, W_v, b_v)

    # reshape query, key, value to separate by head
    # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head) -> (N * nheads, L_t, E_head)
    query = query.reshape(N, -1, nheads, E_head).transpose(1, 2).reshape(N * nheads, -1, E_head)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head) -> (N * nheads, L_s, E_head)
    key = key.reshape(N, -1, nheads, E_head).transpose(1, 2).reshape(N * nheads, -1, E_head)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head) -> (N * nheads, L_s, E_head)
    value = value.reshape(N, -1, nheads, E_head).transpose(1, 2).reshape(N * nheads, -1, E_head)

    # query bmm key^T
    # (N * nheads, L_t, E_head) x (N * nheads, L_s, E_head)^T -> (N * nheads, L_t, L_s)
    keyT = key.transpose(-1, -2)
    attn_weights = torch.bmm(query, keyT)

    # scale down
    attn_weights = attn_weights * (1.0 / math.sqrt(E_head))

    # Have to manipulate masks in order to apply them to the attention weights
    key_padding_mask = attn_mask_q.view(N, 1, 1, L_t).expand(-1, nheads, -1, -1).reshape(N*nheads, 1, L_t).to(device=device)
    attn_mask = torch.zeros(key_padding_mask.shape, device=device, dtype=torch.float32)
    attn_mask = attn_mask.masked_fill_(key_padding_mask, float("-inf"))

    # Zero out the attention weights where the mask is True by adding -inf prior to softmax
    attn_weights.add_(attn_mask)

    # softmax
    attn_weights = F.softmax(attn_weights, dim=-1).nan_to_num_(0.0)

    # dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # attention_weights bmm value
    # (N * nheads, L_t, L_s) x (N * nheads, L_s, E_head) -> (N * nheads, L_t, E_head)
    attn_output = attn_weights.bmm(value)

    # merge heads
    # (N * nheads, L_t, E_head) -> (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
    attn_output = attn_output.reshape(N, nheads, -1, E_head).transpose(1, 2).reshape(N, -1, E_total)

    # apply output projection
    # (N, L_t, E_total) -> (N, L_t, E_out)
    attn_output = F.linear(attn_output, W_out, b_out)

    # padding-specific step: remove output projection bias from padded entries
    attn_output[attn_mask_q, :] = 0.0

    return attn_output

######################################################################
# set hyperparameters following `the Transformer paper <https://arxiv.org/pdf/1706.03762.pdf>`__
N = 512
E_q, E_k, E_v, E_total, E_out = 512, 512, 512, 512, 512
nheads = 8

######################################################################
# except for dropout probability: set to 0 for correctness check
dropout_p = 0.0

######################################################################
# Let us generate some realistic fake data from Zipf's law.
import numpy as np

def zipf_sentence_lengths(alpha: float, batch_size: int) -> np.ndarray:
    # generate fake corpus by unigram Zipf distribution
    # from wikitext-2 corpus, we get rank "." = 3, "!" = 386, "?" = 858
    sentence_lengths = np.empty(batch_size, dtype=int)
    for ibatch in range(batch_size):
        sentence_lengths[ibatch] = 1
        word = np.random.zipf(alpha)
        while word != 3 and word != 386 and word != 858:
            sentence_lengths[ibatch] += 1
            word = np.random.zipf(alpha)
    return sentence_lengths

alpha = 1.2

sentence_lengths = zipf_sentence_lengths(alpha, N)
L_t = np.max(sentence_lengths)
L_s = L_t

######################################################################
# create inputs

# create parameters
W_q, b_q = torch.randn((E_total, E_q), device=device), torch.randn(E_total, device=device)
W_k, b_k = torch.randn((E_total, E_k), device=device), torch.randn(E_total, device=device)
W_v, b_v = torch.randn((E_total, E_v), device=device), torch.randn(E_total, device=device)
W_out, b_out = torch.randn((E_out, E_total), device=device), torch.randn(E_out, device=device)

# create nested input
queries = []
keys = []
values = []
for i in range(N):
    l = sentence_lengths[i]
    s = l
    queries.append(torch.randn((l, E_q), device=device))
    keys   .append(torch.randn((s, E_k), device=device))
    values .append(torch.randn((s, E_v), device=device))
query = torch.nested.nested_tensor(queries)
key = torch.nested.nested_tensor(keys)
value = torch.nested.nested_tensor(values)

# pad input
padded_query = torch.nested.to_padded_tensor(query, 0.0, (N, L_t, E_q))
padded_key   = torch.nested.to_padded_tensor(key, 0.0, (N, L_s, E_k))
padded_value = torch.nested.to_padded_tensor(value, 0.0, (N, L_s, E_v))

# create attention masks
attn_mask_q = torch.zeros((N, L_t), dtype=torch.bool)
attn_mask_kv = torch.zeros((N, L_s), dtype=torch.bool)

#  We need to mask out the padding entries in the attention weights.
for i, entry_length in enumerate(sentence_lengths):
    attn_mask_q[i, entry_length:] = True
    attn_mask_kv[i, entry_length:] = True

######################################################################
# check correctness and performance

import timeit

t0 = timeit.default_timer()
out_nested = mha_nested(
    query, key, value, nheads,
    W_q, W_k, W_v, W_out,
    b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
    dropout_p=dropout_p)

t1 = timeit.default_timer()
out_padded = mha_padded(
    padded_query, padded_key, padded_value, nheads,
    attn_mask_q, attn_mask_kv,
    W_q, W_k, W_v, W_out,
    b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
    dropout_p=dropout_p)
t2 = timeit.default_timer()

print("nested and padded calculations differ by", (torch.nested.to_padded_tensor(out_nested, 0.0, (N, L_t, E_out)) - out_padded).abs().max().item())
print("nestedtensor multi-head attention takes", t1 - t0, "seconds")
print("padded tensor multi-head attention takes", t2 - t1, "seconds")

######################################################################
# Although the nestedtensor version avoids wasted computation on padding, it is not faster
# then the equivalent padded tensor version. This is because the nestedtensor version
# has implemented a few of the kernels, like softmax, in a non optimal way.
#
# There are plans to implement performance critical operations using the new Pytorch 2.0 stack
# For now, some performant kernels are provided for specific use cases, e.g.
# self-attention evaluation by multi-head attention formula.

# embeddings are assumed to be the same
E = E_total
mha_lib = torch.nn.MultiheadAttention(E, nheads, batch_first=True, device=device)
mha_lib.eval()

######################################################################
# extract parameters for correctness check
mha_lib.in_proj_weight.requires_grad_(False)
mha_lib.in_proj_bias.requires_grad_(False)
mha_lib.out_proj.weight.requires_grad_(False)
mha_lib.out_proj.bias.requires_grad_(False)
W_q, b_q = mha_lib.in_proj_weight[: E, :], mha_lib.in_proj_bias[: E]
W_k, b_k = mha_lib.in_proj_weight[E : 2 * E, :], mha_lib.in_proj_bias[E : 2 * E]
W_v, b_v = mha_lib.in_proj_weight[2 * E :, :], mha_lib.in_proj_bias[2 * E :]
W_out, b_out = mha_lib.out_proj.weight, mha_lib.out_proj.bias

######################################################################
# If we set need_weights to False this will enable the fast path in the library.
# Under the hood this will call _scaled_dot_product_attention. If your tensors
# are on CUDA, than a fused, efficient attention kernel will be used. For
# more detailed performance characteristics look at the benchmark in
# pytorch/benchmarks/transformer/sdp.py

with torch.inference_mode():
    t0 = timeit.default_timer()
    out_lib, out_lib_weights = mha_lib(query, query, query, need_weights=False)

    t1 = timeit.default_timer()
    padded_out = mha_padded(
        padded_query, padded_query, padded_query, nheads,
        attn_mask_q, attn_mask_q,
        W_q, W_k, W_v, W_out,
        b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
        dropout_p=dropout_p)
    t2 = timeit.default_timer()

nested_time = t1 - t0
padded_time = t2 - t1
print("Nested and padded calculations differ by", (torch.nested.to_padded_tensor(out_lib, 0.0) - padded_out).abs().max().item())
print("Nested library multi-head attention takes", nested_time, "seconds")
print("Padded tensor multi-head attention takes", padded_time, "seconds")
print(f"Nested Speedup: {padded_time / nested_time:.3f}")
