# -*- coding: utf-8 -*-
"""
(prototype) Introduction to Named Tensors in PyTorch
*******************************************************
**Author**: `Richard Zou <https://github.com/zou3519>`_

Named Tensors aim to make tensors easier to use by allowing users to associate
explicit names with tensor dimensions. In most cases, operations that take
dimension parameters will accept dimension names, avoiding the need to track
dimensions by position. In addition, named tensors use names to automatically
check that APIs are being used correctly at runtime, providing extra safety.
Names can also be used to rearrange dimensions, for example, to support
"broadcasting by name" rather than "broadcasting by position".

This tutorial is intended as a guide to the functionality that will
be included with the 1.3 launch. By the end of it, you will be able to:

- Create Tensors with named dimensions, as well as remove or rename those
  dimensions
- Understand the basics of how operations propagate dimension names
- See how naming dimensions enables clearer code in two key areas:
    - Broadcasting operations
    - Flattening and unflattening dimensions

Finally, we'll put this into practice by writing a multi-head attention module
using named tensors.

Named tensors in PyTorch are inspired by and done in collaboration with
`Sasha Rush <https://tech.cornell.edu/people/alexander-rush/>`_.
Sasha proposed the original idea and proof of concept in his
`January 2019 blog post <http://nlp.seas.harvard.edu/NamedTensor>`_.

Basics: named dimensions
========================

PyTorch now allows Tensors to have named dimensions; factory functions
take a new `names` argument that associates a name with each dimension.
This works with most factory functions, such as

- `tensor`
- `empty`
- `ones`
- `zeros`
- `randn`
- `rand`

Here we construct a tensor with names:
"""

import torch
imgs = torch.randn(1, 2, 2, 3, names=('N', 'C', 'H', 'W'))
print(imgs.names)

######################################################################
# Unlike in
# `the original named tensors blogpost <http://nlp.seas.harvard.edu/NamedTensor>`_,
# named dimensions are ordered: ``tensor.names[i]`` is the name of the ``i`` th
# dimension of ``tensor``.
#
# There are two ways to rename a ``Tensor``'s dimensions:

# Method #1: set the .names attribute (this changes name in-place)
imgs.names = ['batch', 'channel', 'width', 'height']
print(imgs.names)

# Method #2: specify new names (this changes names out-of-place)
imgs = imgs.rename(channel='C', width='W', height='H')
print(imgs.names)

######################################################################
# The preferred way to remove names is to call ``tensor.rename(None)``:

imgs = imgs.rename(None)
print(imgs.names)

######################################################################
# Unnamed tensors (tensors with no named dimensions) still work as
# normal and do not have names in their ``repr``.

unnamed = torch.randn(2, 1, 3)
print(unnamed)
print(unnamed.names)

######################################################################
# Named tensors do not require that all dimensions be named.

imgs = torch.randn(3, 1, 1, 2, names=('N', None, None, None))
print(imgs.names)

######################################################################
# Because named tensors can coexist with unnamed tensors, we need a nice way to
# write named tensor-aware code that works with both named and unnamed tensors.
# Use ``tensor.refine_names(*names)`` to refine dimensions and lift unnamed
# dims to named dims. Refining a dimension is defined as a "rename" with the
# following constraints:
#
# - A ``None`` dim can be refined to have any name
# - A named dim can only be refined to have the same name.

imgs = torch.randn(3, 1, 1, 2)
named_imgs = imgs.refine_names('N', 'C', 'H', 'W')
print(named_imgs.names)

# Refine the last two dims to 'H' and 'W'. In Python 2, use the string '...'
# instead of ...
named_imgs = imgs.refine_names(..., 'H', 'W')
print(named_imgs.names)


def catch_error(fn):
    try:
        fn()
        assert False
    except RuntimeError as err:
        err = str(err)
        if len(err) > 180:
            err = err[:180] + "..."
        print(err)


named_imgs = imgs.refine_names('N', 'C', 'H', 'W')

# Tried to refine an existing name to a different name
catch_error(lambda: named_imgs.refine_names('N', 'C', 'H', 'width'))

######################################################################
# Most simple operations propagate names. The ultimate goal for named tensors
# is for all operations to propagate names in a reasonable, intuitive manner.
# Support for many common operations has been added at the time of the 1.3
# release; here, for example, is ``.abs()``:

print(named_imgs.abs().names)

######################################################################
# Accessors and Reduction
# -----------------------
#
# One can use dimension names to refer to dimensions instead of the positional
# dimension. These operations also propagate names. Indexing (basic and
# advanced) has not been implemented yet but is on the roadmap. Using the
# ``named_imgs`` tensor from above, we can do:

output = named_imgs.sum('C')  # Perform a sum over the channel dimension
print(output.names)

img0 = named_imgs.select('N', 0)  # get one image
print(img0.names)

######################################################################
# Name inference
# --------------
#
# Names are propagated on operations in a two step process called
# **name inference**:
#
# 1. **Check names**: an operator may perform automatic checks at runtime that
#    check that certain dimension names must match.
# 2. **Propagate names**: name inference propagates output names to output
#    tensors.
#
# Let's go through the very small example of adding 2 one-dim tensors with no
# broadcasting.

x = torch.randn(3, names=('X',))
y = torch.randn(3)
z = torch.randn(3, names=('Z',))

######################################################################
# **Check names**: first, we will check whether the names of these two tensors
# *match*. Two names match if and only if they are equal (string equality) or
# at least one is ``None`` (``None`` is essentially a special wildcard name).
# The only one of these three that will error, therefore, is ``x + z``:

catch_error(lambda: x + z)

######################################################################
# **Propagate names**: *unify* the two names by returning the most refined name
# of the two. With ``x + y``,  ``X`` is more refined than ``None``.

print((x + y).names)

######################################################################
# Most name inference rules are straightforward but some of them can have
# unexpected semantics. Let's go through a couple you're likely to encounter:
# broadcasting and matrix multiply.
#
# Broadcasting
# ^^^^^^^^^^^^
#
# Named tensors do not change broadcasting behavior; they still broadcast by
# position. However, when checking two dimensions for if they can be
# broadcasted, PyTorch also checks that the names of those dimensions match.
#
# This results in named tensors preventing unintended alignment during
# operations that broadcast. In the below example, we apply a
# ``per_batch_scale`` to ``imgs``.

imgs = torch.randn(2, 2, 2, 2, names=('N', 'C', 'H', 'W'))
per_batch_scale = torch.rand(2, names=('N',))
catch_error(lambda: imgs * per_batch_scale)

######################################################################
# Without ``names``, the ``per_batch_scale`` tensor is aligned with the last
# dimension of ``imgs``, which is not what we intended. We really wanted to
# perform the operation by aligning ``per_batch_scale`` with the batch
# dimension of ``imgs``.
# See the new "explicit broadcasting by names" functionality for how to
# align tensors by name, covered below.
#
# Matrix multiply
# ^^^^^^^^^^^^^^^
#
# ``torch.mm(A, B)`` performs a dot product between the second dim of ``A``
# and the first dim of ``B``, returning a tensor with the first dim of ``A``
# and the second dim of ``B``. (other matmul functions, such as
# ``torch.matmul``, ``torch.mv``, and ``torch.dot``, behave similarly).

markov_states = torch.randn(128, 5, names=('batch', 'D'))
transition_matrix = torch.randn(5, 5, names=('in', 'out'))

# Apply one transition
new_state = markov_states @ transition_matrix
print(new_state.names)

######################################################################
# As you can see, matrix multiply does not check if the contracted dimensions
# have the same name.
#
# Next, we'll cover two new behaviors that named tensors enable: explicit
# broadcasting by names and flattening and unflattening dimensions by names
#
# New behavior: Explicit broadcasting by names
# --------------------------------------------
#
# One of the main complaints about working with multiple dimensions is the need
# to ``unsqueeze`` "dummy" dimensions so that operations can occur.
# For example, in our per-batch-scale example before, with unnamed tensors
# we'd do the following:

imgs = torch.randn(2, 2, 2, 2)  # N, C, H, W
per_batch_scale = torch.rand(2)  # N

correct_result = imgs * per_batch_scale.view(2, 1, 1, 1)  # N, C, H, W
incorrect_result = imgs * per_batch_scale.expand_as(imgs)
assert not torch.allclose(correct_result, incorrect_result)

######################################################################
# We can make these operations safer (and easily agnostic to the number of
# dimensions) by using names. We provide a new ``tensor.align_as(other)``
# operation that permutes the dimensions of tensor to match the order specified
# in ``other.names``, adding one-sized dimensions where appropriate
# (``tensor.align_to(*names)`` works as well):

imgs = imgs.refine_names('N', 'C', 'H', 'W')
per_batch_scale = per_batch_scale.refine_names('N')

named_result = imgs * per_batch_scale.align_as(imgs)
# note: named tensors do not yet work with allclose
assert torch.allclose(named_result.rename(None), correct_result)

######################################################################
# New behavior: Flattening and unflattening dimensions by names
# -------------------------------------------------------------
#
# One common operation is flattening and unflattening dimensions. Right now,
# users perform this using either ``view``, ``reshape``, or ``flatten``; use
# cases include flattening batch dimensions to send tensors into operators that
# must take inputs with a certain number of dimensions (i.e., conv2d takes 4D
# input).
#
# To make these operation more semantically meaningful than view or reshape, we
# introduce a new ``tensor.unflatten(dim, namedshape)`` method and update
# ``flatten`` to work with names: ``tensor.flatten(dims, new_dim)``.
#
# ``flatten`` can only flatten adjacent dimensions but also works on
# non-contiguous dims. One must pass into ``unflatten`` a **named shape**,
# which is a list of ``(dim, size)`` tuples, to specify how to unflatten the
# dim. It is possible to save the sizes during a ``flatten`` for ``unflatten``
# but we do not yet do that.

imgs = imgs.flatten(['C', 'H', 'W'], 'features')
print(imgs.names)

imgs = imgs.unflatten('features', (('C', 2), ('H', 2), ('W', 2)))
print(imgs.names)

######################################################################
# Autograd support
# ----------------
#
# Autograd currently ignores names on all tensors and just treats them like
# regular tensors. Gradient computation is correct but we lose the safety that
# names give us. It is on the roadmap to introduce handling of names to
# autograd.

x = torch.randn(3, names=('D',))
weight = torch.randn(3, names=('D',), requires_grad=True)
loss = (x - weight).abs()
grad_loss = torch.randn(3)
loss.backward(grad_loss)

correct_grad = weight.grad.clone()
print(correct_grad)  # Unnamed for now. Will be named in the future

weight.grad.zero_()
grad_loss = grad_loss.refine_names('C')
loss = (x - weight).abs()
# Ideally we'd check that the names of loss and grad_loss match, but we don't
# yet
loss.backward(grad_loss)

print(weight.grad)  # still unnamed
assert torch.allclose(weight.grad, correct_grad)

######################################################################
# Other supported (and unsupported) features
# ------------------------------------------
#
# `See here <https://pytorch.org/docs/stable/named_tensor.html>`_ for a
# detailed breakdown of what is supported with the 1.3 release.
#
# In particular, we want to call out three important features that are not
# currently supported:
#
# - Saving or loading named tensors via ``torch.save`` or ``torch.load``
# - Multi-processing via ``torch.multiprocessing``
# - JIT support; for example, the following will error

imgs_named = torch.randn(1, 2, 2, 3, names=('N', 'C', 'H', 'W'))


@torch.jit.script
def fn(x):
    return x


catch_error(lambda: fn(imgs_named))

######################################################################
# As a workaround, please drop names via ``tensor = tensor.rename(None)``
# before using anything that does not yet support named tensors.
#
# Longer example: Multi-head attention
# --------------------------------------
#
# Now we'll go through a complete example of implementing a common
# PyTorch ``nn.Module``: multi-head attention. We assume the reader is already
# familiar with multi-head attention; for a refresher, check out
# `this explanation <https://nlp.seas.harvard.edu/2018/04/03/attention.html>`_
# or
# `this explanation <http://jalammar.github.io/illustrated-transformer/>`_.
#
# We adapt the implementation of multi-head attention from
# `ParlAI <https://github.com/facebookresearch/ParlAI>`_; specifically
# `here <https://github.com/facebookresearch/ParlAI/blob/f7db35cba3f3faf6097b3e6b208442cd564783d9/parlai/agents/transformer/modules.py#L907>`_.
# Read through the code at that example; then, compare with the code below,
# noting that there are four places labeled (I), (II), (III), and (IV), where
# using named tensors enables more readable code; we will dive into each of
# these after the code block.

import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        # (I)
        query = query.refine_names(..., 'T', 'D')
        self_attn = key is None and value is None
        if self_attn:
            mask = mask.refine_names(..., 'T')
        else:
            mask = mask.refine_names(..., 'T', 'T_key')  # enc attn

        dim = query.size('D')
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        # (II)
        def prepare_head(tensor):
            tensor = tensor.refine_names(..., 'T', 'D')
            return (tensor.unflatten('D', [('H', n_heads), ('D_head', dim_per_head)])
                          .align_to(..., 'H', 'T', 'D_head'))

        assert value is None
        if self_attn:
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            key = key.refine_names(..., 'T', 'D')
            value = key
        dim = key.size('D')

        # Distinguish between query_len (T) and key_len (T_key) dims.
        k = prepare_head(self.k_lin(key)).rename(T='T_key')
        v = prepare_head(self.v_lin(value)).rename(T='T_key')
        q = prepare_head(self.q_lin(query))

        dot_prod = q.div_(scale).matmul(k.align_to(..., 'D_head', 'T_key'))
        dot_prod.refine_names(..., 'H', 'T', 'T_key')  # just a check

        # (III)
        attn_mask = (mask == 0).align_as(dot_prod)
        dot_prod.masked_fill_(attn_mask, -float(1e20))

        attn_weights = self.attn_dropout(F.softmax(dot_prod / scale,
                                                   dim='T_key'))

        # (IV)
        attentioned = (
            attn_weights.matmul(v).refine_names(..., 'H', 'T', 'D_head')
            .align_to(..., 'T', 'H', 'D_head')
            .flatten(['H', 'D_head'], 'D')
        )

        return self.out_lin(attentioned).refine_names(..., 'T', 'D')

######################################################################
# **(I) Refining the input tensor dims**

def forward(self, query, key=None, value=None, mask=None):
    # (I)
    query = query.refine_names(..., 'T', 'D')

######################################################################
# The ``query = query.refine_names(..., 'T', 'D')`` serves as enforcable documentation
# and lifts input dimensions to being named. It checks that the last two dimensions
# can be refined to ``['T', 'D']``, preventing potentially silent or confusing size
# mismatch errors later down the line.
#
# **(II)  Manipulating dimensions in prepare_head**

# (II)
def prepare_head(tensor):
    tensor = tensor.refine_names(..., 'T', 'D')
    return (tensor.unflatten('D', [('H', n_heads), ('D_head', dim_per_head)])
                  .align_to(..., 'H', 'T', 'D_head'))

######################################################################
# The first thing to note is how the code clearly states the input and
# output dimensions: the input tensor must end with the ``T`` and ``D`` dims
# and the output tensor ends in ``H``, ``T``, and ``D_head`` dims.
#
# The second thing to note is how clearly the code describes what is going on.
# prepare_head takes the key, query, and value and splits the embedding dim into
# multiple heads, finally rearranging the dim order to be ``[..., 'H', 'T', 'D_head']``.
# ParlAI implements ``prepare_head`` as the following, using ``view`` and ``transpose``
# operations:

def prepare_head(tensor):
    # input is [batch_size, seq_len, n_heads * dim_per_head]
    # output is [batch_size * n_heads, seq_len, dim_per_head]
    batch_size, seq_len, _ = tensor.size()
    tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
    tensor = (
        tensor.transpose(1, 2)
        .contiguous()
        .view(batch_size * n_heads, seq_len, dim_per_head)
    )
    return tensor

######################################################################
# Our named tensor variant uses ops that, though more verbose, have more
# semantic meaning than ``view`` and ``transpose`` and includes enforcable
# documentation in the form of names.
#
# **(III) Explicit broadcasting by names**

def ignore():
    # (III)
    attn_mask = (mask == 0).align_as(dot_prod)
    dot_prod.masked_fill_(attn_mask, -float(1e20))

######################################################################
# ``mask`` usually has dims ``[N, T]`` (in the case of self attention) or
# ``[N, T, T_key]`` (in the case of encoder attention) while ``dot_prod``
# has dims ``[N, H, T, T_key]``. To make ``mask`` broadcast correctly with
# ``dot_prod``, we would usually `unsqueeze` dims ``1`` and ``-1`` in the case
# of self attention or ``unsqueeze`` dim ``1`` in the case of encoder
# attention. Using named tensors, we simply align ``attn_mask`` to ``dot_prod``
# using ``align_as`` and stop worrying about where to ``unsqueeze`` dims.
#
# **(IV) More dimension manipulation using align_to and flatten**

def ignore():
    # (IV)
    attentioned = (
        attn_weights.matmul(v).refine_names(..., 'H', 'T', 'D_head')
        .align_to(..., 'T', 'H', 'D_head')
        .flatten(['H', 'D_head'], 'D')
    )

######################################################################
# Here, as in (II), ``align_to`` and ``flatten`` are more semantically
# meaningful than ``view`` and ``transpose`` (despite being more verbose).
#
# Running the example
# -------------------

n, t, d, h = 7, 5, 2 * 3, 3
query = torch.randn(n, t, d, names=('N', 'T', 'D'))
mask = torch.ones(n, t, names=('N', 'T'))
attn = MultiHeadAttention(h, d)
output = attn(query, mask=mask)
# works as expected!
print(output.names)

######################################################################
# The above works as expected. Furthermore, note that in the code we
# did not mention the name of the batch dimension at all. In fact,
# our ``MultiHeadAttention`` module is agnostic to the existence of batch
# dimensions.

query = torch.randn(t, d, names=('T', 'D'))
mask = torch.ones(t, names=('T',))
output = attn(query, mask=mask)
print(output.names)

######################################################################
# Conclusion
# ----------
#
# Thank you for reading! Named tensors are still very much in development;
# if you have feedback and/or suggestions for improvement, please let us
# know by creating `an issue <https://github.com/pytorch/pytorch/issues>`_.
