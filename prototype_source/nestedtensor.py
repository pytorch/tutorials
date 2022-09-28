"""
Nested Tensors
===============================================================

Nested tensor is very similar to regular tensor, except for the shape:

* for a regular tensor, each dimension has a size

* for a nested tensor, not all dimensions have regular sizes; some of them are jagged

Nested tensors are a natural solution for representing sequential data within various domains:

* in NLP, sentences can have variable lengths, so a batch of sentences forms a nested tensor

* in CV, images can have variable shapes, so a batch of images forms a nested tensor

In this tutorial, we will demonstrate basic usage of nested tensors and motivate their usefulness
for operating on sequential data of varying lengths with a real-world example.

The nested tensor operations used here have not been released yet.
You will have to install the latest nightly to run this tutorial.
"""

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################################
# Nested Tensor Initialization
# ----------------
#

######################################################################
# From the Python frontend, a nested tensor can be created from a list of tensors.
nt = torch.nested_tensor([torch.randn((2, 6)), torch.randn((3, 6))], device=device)
print(nt)

######################################################################
# By padding every underlying tensor to the same shape,
# a nested tensor can be converted to a regular tensor.
pt = torch.nested.to_padded_tensor(nt, padding=0.0)
print(pt)

######################################################################
# For practical reasons, conceptually we implement nested tensor
# as a batch of tensors with different shapes,
# i.e. dimension 0 is assumed to be the batch dimension.
# Indexing dimension 0 gives back the underlying tensor.
print("0th underlying tensor:", nt[0], sep='\n')
print("last column of 1st underlying tensor:", nt[1, :, -1], sep='\n')

######################################################################
# Slicing in dimension 0 has not been supported yet.

######################################################################
# Nested Tensor Operations
# ----------------
#

######################################################################
# As each operation must be explicitly implemented for nested tensors,
# operation coverage for nested tensors is currently narrower than that of regular tensors.
# For now, only basic operations such as index, dropout, softmax, transpose, reshape, linear, bmm are covered.
# However, coverage is being expanded rapidly.
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
# The semantics for nested tensors are similar, except that -1 no longer infers.
# Instead, it inherits the old size (here 2 for ``nt[0]`` and 3 for ``nt[1]``).
# -1 is the only legal size to specify for a jagged dimension.
nt1 = nt.reshape(2, -1, 2, 3)
print(nt1)

######################################################################
# **transpose**
#
# The transpose op is for swapping two dimensions of a tensor.
# Its full semantics can be found
# `here <https://pytorch.org/docs/stable/generated/torch.transpose.html>`__.
# Note that nested tensor dimension 0 is special;
# it is assumed to be the batch dimension,
# so transposes involving nested tensor dimension 0 are forbidden.
nt2 = nt1.transpose(1, 2)
print(nt2)

######################################################################
# **others**
#
# Other operations have the same semantics as for regular tensors.
# Applying the operation on a nested tensor is equivalent to
# applying the operation to the underlying tensor components,
# with the result being a nested tensor as well.
nt_mm = torch.nested_tensor([torch.randn((2, 3, 4)), torch.randn((2, 3, 5))], device=device)
nt3 = torch.matmul(nt2, nt_mm)
print("matmul:", nt3, sep='\n')

nt4 = F.dropout(nt3, 0.1)
print("dropout:", nt4, sep='\n')

nt5 = F.softmax(nt4, -1)
print("softmax:", nt5, sep='\n')

######################################################################
# Why Nested Tensor
# ----------------
#

######################################################################
# In the age before nested tensor, one has to manually pad each data tensor
# to the same shape to form a batch as a regular tensor.
# For example, we have 2 sentences and a vocabulary, then pad with 0.
sentences = [["goodbye", "padding"],
             ["embrace", "nested", "tensor"]]
vocabulary = {"goodbye" : 1.0, "padding" : 2.0,
              "embrace" : 3.0, "nested" : 4.0, "tensor" : 5.0}
padded_sentences = torch.tensor([[1.0, 2.0, 0.0],
                                 [3.0, 4.0, 5.0]])
nested_sentences = torch.nested_tensor([torch.tensor([1.0, 2.0]),
                                        torch.tensor([3.0, 4.0, 5.0])])
print(padded_sentences)
print(nested_sentences)

######################################################################
# Clearly, padding introduces inefficiency.
# Further, padding with zeros does not correctly treat entries as padding for every operation,
# e.g. in softmax one has to pad with -inf rather than 0 to ignore specific entries.
padded_sentences_for_softmax = torch.tensor([[1.0, 2.0, float("-inf")],
                                             [3.0, 4.0, 5.0]])
print(F.softmax(padded_sentences_for_softmax, -1))
print(F.softmax(nested_sentences, -1))

######################################################################
# Let us take a look at a practical example: the multi-head attention component
# utilized in `Transformers <https://arxiv.org/pdf/1706.03762.pdf>`__.
# The nested tensor version is straightforward.
import math

"""
Args:
    query: query of shape (N, L_t, E_q)
    key: key of shape (N, L_s, E_k)
    value: value of shape (N, L_s, E_v)
    nheads: number of heads in multi-head attention
    W_q: Weight for query input projection of shape (E_total, E_q)
    W_k: Weight for key input projection of shape (E_total, E_k)
    W_v: Weight for value input projection of shape (E_total, E_v)
    W_out: Weight for output projection of shape (E_out, E_total)
    b_q (optional): Bias for query input projection of shape E_total. Default: None
    b_k (optional): Bias for key input projection of shape E_total. Default: None
    b_v (optional): Bias for value input projection of shape E_total. Default: None
    b_out (optional): Bias for output projection of shape E_out. Default: None
    dropout_p: dropout probability. Default: 0.0
    where:
        N is the batch size
        L_t is the target sequence length (jagged)
        L_s is the source sequence length (jagged)
        E_q is the embedding size for query
        E_k is the embedding size for key
        E_v is the embedding size for value
        E_total is the embedding size for all heads combined
        E_out is the output embedding size
Returns:
    attn_output: Output of shape (N, L_t, E_out)
"""
def mha_nested(query, key, value, nheads,
W_q, W_k, W_v, W_out,
b_q=None, b_k=None, b_v=None, b_out=None,
dropout_p=0.0):
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
    query = query.reshape(-1, -1, nheads, E_head).transpose(1, 2)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
    key = key.reshape(-1, -1, nheads, E_head).transpose(1, 2)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
    value = value.reshape(-1, -1, nheads, E_head).transpose(1, 2)

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
"""
Args:
    query: query of shape (N, L_t, E_q)
    key: key of shape (N, L_s, E_k)
    value: value of shape (N, L_s, E_v)
    nheads: number of heads in multi-head attention
    attn_mask_q: boolean mask indicating locations that should not take part in attention for query, shape (N, L_t)
    attn_mask_kv: boolean mask indicating locations that should not take part in attention for key and value, shape (N, L_s)
    W_q: Weight for query input projection of shape (E_total, E_q)
    W_k: Weight for key input projection of shape (E_total, E_k)
    W_v: Weight for value input projection of shape (E_total, E_v)
    W_out: Weight for output projection of shape (E_out, E_total)
    b_q (optional): Bias for query input projection of shape E_total. Default: None
    b_k (optional): Bias for key input projection of shape E_total. Default: None
    b_v (optional): Bias for value input projection of shape E_total. Default: None
    b_out (optional): Bias for output projection of shape E_out. Default: None
    dropout_p: dropout probability. Default: 0.0
    where:
        N is the batch size
        L_t is the target sequence length (padded)
        L_s is the source sequence length (padded)
        E_q is the embedding size for query
        E_k is the embedding size for key
        E_v is the embedding size for value
        E_total is the embedding size for all heads combined
        E_out is the output embedding size
Returns:
    attn_output: Output of shape (N, L_t, E_out)
"""
def mha_padded(query, key, value, nheads,
attn_mask_q, attn_mask_kv,
W_q, W_k, W_v, W_out,
b_q=None, b_k=None, b_v=None, b_out=None,
dropout_p=0.0):
    N = query.size(0)
    L_t = query.size(1)
    L_s = key.size(1)
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

    # padding-specific step: remove bias from padded entries
    # in the specific multihead-attention formula it is not necessary to remove these bias
    # because the -inf padding later on in softmax step can take care of it
    # but to be general here we demonstrate the bias removal
    for i in range(N):
        for j in range(L_t):
            if attn_mask_q[i, j]:
                query[i, j, :] = 0.0
        for j in range(L_s):
            if attn_mask_kv[i, j]:
                key[i, j, :] = 0.0
                value[i, j, :] = 0.0

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
    # padding-specific step: add -inf mask for padding in softmax
    attn_mask = query.new_zeros((N, nheads, L_t, L_s))
    for i in range(N):
        for j in range(L_t):
            for k in range(L_s):
                if attn_mask_q[i, j] or attn_mask_kv[i, k]:
                    attn_mask[i, :, j, k] = float("-inf")
    attn_mask = attn_mask.reshape((N * nheads, L_t, L_s))
    attn_weights = torch.baddbmm(attn_mask, query, keyT)
    # if no padding, it could have been as simple as
    #     attn_weights = torch.bmm(query, keyT)

    # scale down
    attn_weights = attn_weights * (1.0 / math.sqrt(E_head))

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
    for i in range(N):
        for j in range(L_t):
            if attn_mask_q[i, j]:
                attn_output[i, j, :] = 0.0

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
keys    = []
values  = []
for i in range(N):
    l = sentence_lengths[i]
    s = l
    queries.append(torch.randn((l, E_q), device=device))
    keys   .append(torch.randn((s, E_k), device=device))
    values .append(torch.randn((s, E_v), device=device))
query = torch.nested_tensor(queries)
key   = torch.nested_tensor(keys   )
value = torch.nested_tensor(values )

# pad input
padded_query = torch.nested.to_padded_tensor(query, 0.0, (N, L_t, E_q))
padded_key   = torch.nested.to_padded_tensor(key, 0.0, (N, L_s, E_k))
padded_value = torch.nested.to_padded_tensor(value, 0.0, (N, L_s, E_v))

# create attention masks
attn_mask_q = torch.zeros((N, L_t), dtype=torch.bool)
attn_mask_kv = torch.zeros((N, L_s), dtype=torch.bool)
for i in range(N):
    for j in range(L_t):
        if padded_query[i, j, :].abs().max().item() == 0.0:
            attn_mask_q[i, j] = True
    for j in range(L_s):
        if padded_key[i, j, :].abs().max().item() == 0.0:
            attn_mask_kv[i, j] = True

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
print("nested tensor multi-head attention takes", t1 - t0, "seconds")
print("padded tensor multi-head attention takes", t2 - t1, "seconds")

######################################################################
# The nested tensor version avoids wasted computation on padding,
# so in sequential CPU execution it is faster than padded tensor version as expected.
# Optimization for multi-threaded environment is underway.
#
# For now, performant kernels are provided for specific use cases, e.g.
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
# check correctness and performance

t0 = timeit.default_timer()
out_lib, out_lib_weights = mha_lib(query, query, query)

t1 = timeit.default_timer()
out_nested = mha_nested(
    query, query, query, nheads,
    W_q, W_k, W_v, W_out,
    b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
    dropout_p=dropout_p)

t2 = timeit.default_timer()
padded_out = mha_padded(
    padded_query, padded_query, padded_query, nheads,
    attn_mask_q, attn_mask_q,
    W_q, W_k, W_v, W_out,
    b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
    dropout_p=dropout_p)
t3 = timeit.default_timer()

print("nested general and library calculations differ by", (torch.nested.to_padded_tensor(out_nested, 0.0) - torch.nested.to_padded_tensor(out_lib, 0.0)).abs().max().item())
print("nested library multi-head attention takes", t1 - t0, "seconds")
print("nested general multi-head attention takes", t2 - t1, "seconds")
print("padded tensor multi-head attention takes", t3 - t2, "seconds")
