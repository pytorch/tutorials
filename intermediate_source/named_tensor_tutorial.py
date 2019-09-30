# -*- coding: utf-8 -*-
"""
Introduction to Named Tensors in PyTorch
****************************************
**Author**: `Richard Zou <https://github.com/zou3519>`_

`Sasha Rush <https://tech.cornell.edu/people/alexander-rush/>`_ proposed the idea of
`named tensors <http://nlp.seas.harvard.edu/NamedTensor>`_ in a January 2019 blog post as a
way to enable more readable code when writing with the manipulations of multidimensional
arrays necessary for coding up Transformer and Convolutional architectures. With PyTorch 1.3,
we begin supporting the concept of named tensors by allowing a ``Tensor`` to have **named
dimensions**; this tutorial is intended as a guide to the functionality that will
be included with the 1.3 launch. By the end of it, you will be able to:

- Initiate a ``Tensor`` with named dimensions, as well as removing or renmaing those dimensions
- Understand the basics of how dimension names are propagated through operations
- See how naming dimensions enables clearer code in two key areas:
    - Broadcasting operations
    - Flattening and unflattening dimensions

Finally, we'll put this into practice by coding the operations of multi-headed attention
using named tensors, and see that the code is significantly more readable than it would
be with regular, "unnamed" tensors!
"""

######################################################################
# Basics: named dimensions
# ------------------------
#
# Tensors now take a new ``names`` argument that represents a name for each dimension.
# Here we construct a tensor with names:
#

import torch
imgs = torch.randn(1, 2, 2, 3 , names=('N', 'C', 'H', 'W'))

######################################################################
# This works with most factory functions, such as:
#
# - ``tensor``
# -  ``empty``
# -  ``ones``
# -  ``zeros``
#
# There are two ways rename a ``Tensor``'s names:
#

print(imgs.names)

# Method #1: set .names attribute
imgs.names = ['batch', 'channel', 'width', 'height']
print(imgs.names)

# Method #2: specify new names:
imgs.rename(channel='C', width='W', height='H')
print(imgs.names)

######################################################################
# The preferred way to remove names is to call ``tensor.rename(None)``:

imgs.rename(None)

######################################################################
# Unnamed tensors (tensors with no named dimensions) still work as normal and do
# not have names in their repr.

unnamed = torch.randn(2, 1, 3)
print(unnamed)
print(unnamed.names)

######################################################################
# Named tensors do not require that all dimensions be named.

imgs = torch.randn(3, 1, 1, 2, names=('N', None, None, None))
print(imgs.names)

######################################################################
# Because named tensors coexist with unnamed tensors, we need a nice way to write named-tensor-aware
# code that works with both named and unnamed tensors. Use ``tensor.refine_names(*names)`` to refine
# dimensions and lift unnamed dims to named dims. Refining a dimension is defined as a "rename" with
# the following constraints:
#
# - A ``None`` dim can be refined to have any name
# - A named dim can only be refined to have the same name.

print(imgs.names)
print(imgs.refine_names('N', 'C', 'H', 'W').names)

# Coerces the last two dims to 'H' and 'W'. In Python 2, use the string '...' instead of ...
print(imgs.refine_names(..., 'H', 'W').names)

def catch_error(fn):
    try:
        fn()
    except RuntimeError as err:
        print(err)

# Tried to refine an existing name to a different name
print(catch_error(lambda: imgs.refine_names('batch', 'channel', 'height', 'width')))

######################################################################
# Most simple operations propagate names. The ultimate goal for named tensors is
# for all operations to propagate names in a reasonable, intuitive manner. Many
# common operations have been implemented at the time of the 1.3 release; here,
# for example, is `.abs()`:

named_imgs = imgs.refine_names('N', 'C', 'H', 'W')
print(named_imgs.abs().names)

######################################################################
# Accessors and Reduction
# -----------------------
#
# One can use dimension names to refer to dimensions instead of the positional
# dimension. These operations also propagate names. Indexing (basic and
# advanced) has not been implemented yet but is on the roadmap.

output = named_imgs.sum(['C'])  # Perform a sum over the channel dimension
print(output.names)

img0 = named_imgs.select('N', 0)  # get one image
print(img0.names)

######################################################################
# Name inference
# --------------
#
# Names are propagated on operations in a process called **name inference**. Name
# inference works in a two step process:
#
# - **Check names**: an operator may check that certain dimensions must match.
# - **Propagate names**: name inference computes and propagates output names to
#   output tensors.
#
# Let's go through the very small example of adding 2 one-dim tensors with no
# broadcasting.

x = torch.randn(3, names=('X',))
y = torch.randn(3)
z = torch.randn(3, names=('Z',))

# **Check names**: first, we will check whether the names of these two tensors
# match. Two names match if and only if they are equal (string equality) or at
# least one is ``None`` (``None``s are essentially a special wildcard name).
# The only one of these three that will error, therefore, is ``x+z``:

catch_error(lambda: x + z)

# **Propagate names**: unify the two names by returning the most refined name of
# the two. With ``x + y``,  ``X`` is more specific than ``None``.

print((x + y).names)

######################################################################
# Most name inference rules are straightforward but some of them (the dot
# product ones) can have unexpected semantics. Let's go through a few more of
# them.
#
# Broadcasting
# ------------
#
# Named tensors do not change broadcasting behavior; they still broadcast by
# position. However, when checking two dimensions for if they can be
# broadcasted, the names of those dimensions must match. Two names match if and
# only if they are equal (string equality), or if one is None.
#
# We do not support **automatic broadcasting** by names because the output
# ordering is ambiguous and does not work well with unnamed dimensions. However,
# we support **explicit broadcasting** by names. The two examples below help
# clarify this.

# Automatic broadcasting: expected to fail
imgs = torch.randn(6, 6, 6, 6, names=('N', 'C', 'H', 'W'))
per_batch_scale = torch.rand(6, names=('N',))
catch_error(lambda: imgs * per_batch_scale)

# Explicit broadcasting: the names check out and the more refined names are propagated.
imgs = torch.randn(6, 6, 6, 6, names=('N', 'C', 'H', 'W'))
per_batch_scale_4d = torch.rand(6, 1, 1, 1, names=('N', None, None, None))
print((imgs * per_batch_scale_4d).names)

######################################################################
# Matrix multiply
# ---------------
#
# Of course, many of you may be wondering about the very special operation of
# matrix multiplication. ``torch.mm(A, B)`` contracts away the second dimension
# of ``A`` with the first dimension of ``B``, returning a tensor with the first
# dim of ``A`` and the second dim of ``B``. (the other matmul functions,
# ``torch.matmul``, ``torch.mv``, ``torch.dot``, behave similarly):

markov_states = torch.randn(128, 5, names=('batch', 'D'))
transition_matrix = torch.randn(5, 5, names=('in', 'out'))

# Apply one transition
new_state = markov_states @ transition_matrix
print(new_state.names)

######################################################################
# New behavior: Explicit broadcasting by names
# --------------------------------------------
#
# One of the main complaints about working with multiple dimensions is the need
# to ``unsqueeze`` "dummy" dimensions so that operations can occur. For example, in
# our per-batch-scale example before, with unnamed tensors we'd do the
# following:

imgs = torch.randn(2, 2, 2, 2)  # N, C, H, W
per_batch_scale = torch.rand(2)  # N

correct_result = imgs * per_batch_scale.view(2, 1, 1, 1)  # N, C, H, W
incorrect_result = imgs * per_batch_scale.expand_as(imgs)
assert not torch.allclose(correct_result, incorrect_result)

######################################################################
# We can make these operations safer (and easily agnostic to the number of
# dimensions) by using names. We provide a new ``tensor.align_as(other)`` operation
# that permutes the dimensions of tensor to match the order specified in
# ``other.names``, adding one-sized dimensions where appropriate
# (``tensor.align_to(*names)`` works as well):

imgs = imgs.refine_names('N', 'C', 'H', 'W')
per_batch_scale = per_batch_scale.refine_names('N')

named_result = imgs * per_batch_scale.align_as(imgs)
assert torch.allclose(named_result.rename(None), correct_result)

######################################################################
# New behavior: Flattening and unflattening dimensions by names
# -------------------------------------------------------------
#
# One common operation is flattening and unflattening dimensions. Right now,
# users perform this using either ``view``, ``reshape``, or ``flatten``; use
# cases include flattening batch dimensions to send tensors into operators that
# must take inputs with a certain number of dimensions (i.e., conv2d takes 4D input).
#
# To make these operation more semantically meaningful than view or reshape, we
# introduce a new ``tensor.unflatten(dim, namedshape)`` method and update
# ``flatten`` to work with names: ``tensor.flatten(dims, new_dim)``.
#
# ``flatten`` can only flatten adjacent dimensions but also works on
# non-contiguous dims. One must pass into ``unflatten`` a **named shape**, which
# is a list of ``(dim, size)`` tuples, to specify how to unflatten the dim. It
# is possible to save the sizes during a ``flatten`` for ``unflatten`` but we
# do not yet do that.

imgs = imgs.flatten(['C', 'H', 'W'], 'features')
print(imgs.names)

imgs = imgs.unflatten('features', (('C', 2), ('H', 2), ('W', 2)))
print(imgs.names)

######################################################################
# Autograd support
# ----------------
#
# Autograd currently supports named tensors in a limited manner: autograd
# ignores names on all tensors. Gradient computation is still correct but we
# lose the safety that names give us. It is on the roadmap to introduce handling
# of names to autograd.

x = torch.randn(3, names=('D',))
weight = torch.randn(3, names=('D',), requires_grad=True)
loss = (x - weight).abs()
grad_loss = torch.randn(3)
loss.backward(grad_loss)

print(weight.grad)  # Unnamed for now. Will be named in the future

######################################################################
# Other supported features
# ------------------------
#
# See here (link to be included) for a detailed breakdown of what is
# supported with the 1.3 release, what is on the roadmap to be supported soon,
# and what will be supported in the future but not soon.
#
# In particular, three important features that we do not have plans to support
# soon are:
#
# - Retaining names when serializing or loading a serialized ``Tensor`` via
#   ``torch.save``
# - Multi-processing via ``torch.multiprocessing``
# - JIT support; for example, the following will error

@torch.jit.script
def fn(x):
    return x

catch_error(lambda: fn(named_tensor))

######################################################################
# Longer example: Multi-headed attention
# --------------------------------------
#
# Now we'll go through a complete example of implementing a common advanced
# PyTorch ``nn.Module``: multi-headed attention. We assume the reader is already
# familiar with multi-headed attention; for a refresher, check out
# `this explanation <http://jalammar.github.io/illustrated-transformer/>`_.
#
# We adapt the implementation of multi-headed attention from
# `ParlAI <https://github.com/facebookresearch/ParlAI>`_; specifically
# `here <https://github.com/facebookresearch/ParlAI/blob/f7db35cba3f3faf6097b3e6b208442cd564783d9/parlai/agents/transformer/modules.py#L907>`_.
# Read through the code at that example; then, compare with the code below,
# noting that there are four places labeled (I), (II), (III), and (IV), where
# using named tensors enables more readable code.

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
        query = query.refine_names('N', 'T', 'D')
        if mask.dim() is 2:
            mask = mask.refine_names('N', 'T')  # selfattn
        else:
            mask = mask.refine_names('N', 'T', 'T_key')  # enc attn

        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # (II)
            tensor = tensor.refine_names('N', 'T', 'D')
            return (tensor.unflatten('D', [('H', n_heads), ('D_head', dim_per_head)])
                          .align_to('N', 'H', 'T', 'D_head').contiguous())

        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            key = key.refine_names('N', 'T', 'D')
            value = key
        key_len = key.size('T')
        dim = key.size('D')

        # Distinguish between query_len (T) and key_len (T_key) dims.
        k = prepare_head(self.k_lin(key)).renamed(T='T_key')
        v = prepare_head(self.v_lin(value)).renamed(T='T_key')
        q = prepare_head(self.q_lin(query))

        dot_prod = q.matmul(k.transpose('D_head', 'T_key'))
        dot_prod.refine_names('N', 'H', 'T', 'T_key')  # just a check.

        # (III)
        # Named tensors doesn't support == yet; the following is a workaround.
        attn_mask = (mask.renamed(None) == 0).refine_names(*mask.names)
        attn_mask = attn_mask.align_as(dot_prod)
        dot_prod.masked_fill_(attn_mask, -float(1e20))

        attn_weights = self.attn_dropout(F.softmax(dot_prod / scale, dim='T_key'))

        # (IV)
        attentioned = (
            attn_weights.matmul(v).refine_names('N', 'H', 'T', 'D_head')
            .align_to('N', 'T', 'H', 'D_head')
            .flatten(['H', 'D_head'], 'D')
        )

        return self.out_lin(attentioned).refine_names('N', 'T', 'D')

######################################################################
# Let's dive into each of these areas in turn:
#
# **(I) Refining the input tensor dims**

def forward(self, query, key=None, value=None, mask=None):
    # (I)
    query = query.refine_names('N', 'T', 'D')

######################################################################
# The ``query = query.refine_names('N', 'T', 'D')`` serves as error checking and
# asserts that the the dimensions can be refined to ['N', 'T', 'D']. This prevents
# potentially silent or confusing size mismatch errors later down the line.
#
# **(II)  Manipulating dimensions in ``prepare_head``**

def prepare_head(tensor):
    # (II)
    tensor = tensor.refine_names('N', 'T', 'D')
    return (tensor.unflatten('D', [('H', n_heads), ('D_head', dim_per_head)])
                  .align_to('N', 'H', 'T', 'D_head').contiguous())

######################################################################
# Next, multihead attention takes the key, query, and value and splits their
# feature dimensions into multiple heads and rearranges the dim order to be
# ``['N', 'H', 'T', 'D_head']``. We can achieve something similar using view
# and transpose operations like the following:

def prepare_head(tensor):
    batch_size, seq_len, _ = tensor.size()  # N, T, D
    tensor = tensor.view(batch_size, seq_len, n_heads, dim_per_head)  # N, T, H, D
    return tensor.transpose(1, 2).contiguous()  # N, H, T, D

######################################################################
# but our named tensor variant provides ops that, although are more verbose, have
# more semantic meaning than ``view`` and "enforcable" documentation in the form
# of names.
#
# **(III) Explicit broadcasting by names**

def ignore():
    # (III)
    # Named tensors doesn't support == yet; the following is a workaround.
    attn_mask = (mask.renamed(None) == 0).refine_names(*mask.names)
    attn_mask = attn_mask.align_as(dot_prod)
    dot_prod.masked_fill_(attn_mask, -float(1e20))

######################################################################
# ``mask`` usually has dims ``[N, T]`` or ``[N, T, T_key]``, while ``dot_prod``
# has dims ``[N, H, T, T_key]``. To make ``mask`` broadcast correctly with
# ``dot_prod``, we would usually ``unsqueeze`` dim 1 (and also the last dim
# in the former). Using named tensors, we can simply align the two tensors a
# nd stop worrying about where to ``unsqueeze`` dims.
#
# **(IV) More dimension manipulation using ``align_to`` and ``flatten``**

def ignore():
    # (IV)
    attentioned = (
        attn_weights.matmul(v).refine_names('N', 'H', 'T', 'D_head')
        .align_to('N', 'T', 'H', 'D_head')
        .flatten(['H', 'D_head'], 'D')
    )

######################################################################
# (IV): Like (II), using ``align_to`` and ``flatten`` are more semantically
# meaningful than view.
#
# Running the example
# -------------------

n, t, d, h = 7, 5, 2 * 3, 3
query = torch.randn(n, t, d, names=('N', 'T', 'D'))
mask = torch.ones(n, t, names=('N', 'T'))
attn = MultiHeadAttention(h, d)
output = attn(query, mask=mask)
print(output.names)

######################################################################
# Conclusion
# ----------
#
# Thank you for reading! Named tensors are still very much in development;
# if you have feedback and/or suggestions for improvement, please let us
# know by creating `an issue <https://github.com/pytorch/pytorch/issues>`_.
