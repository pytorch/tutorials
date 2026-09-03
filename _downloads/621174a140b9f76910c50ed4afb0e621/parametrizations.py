# -*- coding: utf-8 -*-
"""
Parametrizations Tutorial
=========================
**Author**: `Mario Lezcano <https://github.com/lezcano>`_

Regularizing deep-learning models is a surprisingly challenging task.
Classical techniques such as penalty methods often fall short when applied
on deep models due to the complexity of the function being optimized.
This is particularly problematic when working with ill-conditioned models.
Examples of these are RNNs trained on long sequences and GANs. A number
of techniques have been proposed in recent years to regularize these
models and improve their convergence. On recurrent models, it has been
proposed to control the singular values of the recurrent kernel for the
RNN to be well-conditioned. This can be achieved, for example, by making
the recurrent kernel `orthogonal <https://en.wikipedia.org/wiki/Orthogonal_matrix>`_.
Another way to regularize recurrent models is via
"`weight normalization <https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html>`_".
This approach proposes to decouple the learning of the parameters from the
learning of their norms.  To do so, the parameter is divided by its
`Frobenius norm <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`_
and a separate parameter encoding its norm is learned.
A similar regularization was proposed for GANs under the name of
"`spectral normalization <https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html>`_". This method
controls the Lipschitz constant of the network by dividing its parameters by
their `spectral norm <https://en.wikipedia.org/wiki/Matrix_norm#Special_cases>`_,
rather than their Frobenius norm.

All these methods have a common pattern: they all transform a parameter
in an appropriate way before using it. In the first case, they make it orthogonal by
using a function that maps matrices to orthogonal matrices. In the case of weight
and spectral normalization, they divide the original parameter by its norm.

More generally, all these examples use a function to put extra structure on the parameters.
In other words, they use a function to constrain the parameters.

In this tutorial, you will learn how to implement and use this pattern to put
constraints on your model. Doing so is as easy as writing your own ``nn.Module``.

Requirements: ``torch>=1.9.0``

Implementing parametrizations by hand
-------------------------------------

Assume that we want to have a square linear layer with symmetric weights, that is,
with weights ``X`` such that ``X = Xᵀ``. One way to do so is
to copy the upper-triangular part of the matrix into its lower-triangular part
"""

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)

X = torch.rand(3, 3)
A = symmetric(X)
assert torch.allclose(A, A.T)  # A is symmetric
print(A)                       # Quick visual check

###############################################################################
# We can then use this idea to implement a linear layer with symmetric weights
class LinearSymmetric(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(n_features, n_features))

    def forward(self, x):
        A = symmetric(self.weight)
        return x @ A

###############################################################################
# The layer can be then used as a regular linear layer
layer = LinearSymmetric(3)
out = layer(torch.rand(8, 3))

###############################################################################
# This implementation, although correct and self-contained, presents a number of problems:
#
# 1) It reimplements the layer. We had to implement the linear layer as ``x @ A``. This is
#    not very problematic for a linear layer, but imagine having to reimplement a CNN or a
#    Transformer...
# 2) It does not separate the layer and the parametrization.  If the parametrization were
#    more difficult, we would have to rewrite its code for each layer that we want to use it
#    in.
# 3) It recomputes the parametrization every time we use the layer. If we use the layer
#    several times during the forward pass, (imagine the recurrent kernel of an RNN), it
#    would compute the same ``A`` every time that the layer is called.
#
# Introduction to parametrizations
# --------------------------------
#
# Parametrizations can solve all these problems as well as others.
#
# Let's start by reimplementing the code above using ``torch.nn.utils.parametrize``.
# The only thing that we have to do is to write the parametrization as a regular ``nn.Module``
class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

###############################################################################
# This is all we need to do. Once we have this, we can transform any regular layer into a
# symmetric layer by doing
layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Symmetric())

###############################################################################
# Now, the matrix of the linear layer is symmetric
A = layer.weight
assert torch.allclose(A, A.T)  # A is symmetric
print(A)                       # Quick visual check

###############################################################################
# We can do the same thing with any other layer. For example, we can create a CNN with
# `skew-symmetric <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_ kernels.
# We use a similar parametrization, copying the upper-triangular part with signs
# reversed into the lower-triangular part
class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)


cnn = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3)
parametrize.register_parametrization(cnn, "weight", Skew())
# Print a few kernels
print(cnn.weight[0, 1])
print(cnn.weight[2, 2])

###############################################################################
# Inspecting a parametrized module
# --------------------------------
#
# When a module is parametrized, we find that the module has changed in three ways:
#
# 1) ``model.weight`` is now a property
#
# 2) It has a new ``module.parametrizations`` attribute
#
# 3) The unparametrized weight has been moved to ``module.parametrizations.weight.original``
#
# |
# After parametrizing ``weight``, ``layer.weight`` is turned into a
# `Python property <https://docs.python.org/3/library/functions.html#property>`_.
# This property computes ``parametrization(weight)`` every time we request ``layer.weight``
# just as we did in our implementation of ``LinearSymmetric`` above.
#
# Registered parametrizations are stored under a ``parametrizations`` attribute within the module.
layer = nn.Linear(3, 3)
print(f"Unparametrized:\n{layer}")
parametrize.register_parametrization(layer, "weight", Symmetric())
print(f"\nParametrized:\n{layer}")

###############################################################################
# This ``parametrizations`` attribute is an ``nn.ModuleDict``, and it can be accessed as such
print(layer.parametrizations)
print(layer.parametrizations.weight)

###############################################################################
# Each element of this ``nn.ModuleDict`` is a ``ParametrizationList``, which behaves like an
# ``nn.Sequential``. This list will allow us to concatenate parametrizations on one weight.
# Since this is a list, we can access the parametrizations indexing it. Here's
# where our ``Symmetric`` parametrization sits
print(layer.parametrizations.weight[0])

###############################################################################
# The other thing that we notice is that, if we print the parameters, we see that the
# parameter ``weight`` has been moved
print(dict(layer.named_parameters()))

###############################################################################
# It now sits under ``layer.parametrizations.weight.original``
print(layer.parametrizations.weight.original)

###############################################################################
# Besides these three small differences, the parametrization is doing exactly the same
# as our manual implementation
symmetric = Symmetric()
weight_orig = layer.parametrizations.weight.original
print(torch.dist(layer.weight, symmetric(weight_orig)))

###############################################################################
# Parametrizations are first-class citizens
# -----------------------------------------
#
# Since ``layer.parametrizations`` is an ``nn.ModuleList``, it means that the parametrizations
# are properly registered as submodules of the original module. As such, the same rules
# for registering parameters in a module apply to register a parametrization.
# For example, if a parametrization has parameters, these will be moved from CPU
# to CUDA when calling ``model = model.cuda()``.
#
# Caching the value of a parametrization
# --------------------------------------
#
# Parametrizations come with an inbuilt caching system via the context manager
# ``parametrize.cached()``
class NoisyParametrization(nn.Module):
    def forward(self, X):
        print("Computing the Parametrization")
        return X

layer = nn.Linear(4, 4)
parametrize.register_parametrization(layer, "weight", NoisyParametrization())
print("Here, layer.weight is recomputed every time we call it")
foo = layer.weight + layer.weight.T
bar = layer.weight.sum()
with parametrize.cached():
    print("Here, it is computed just the first time layer.weight is called")
    foo = layer.weight + layer.weight.T
    bar = layer.weight.sum()

###############################################################################
# Concatenating parametrizations
# ------------------------------
#
# Concatenating two parametrizations is as easy as registering them on the same tensor.
# We may use this to create more complex parametrizations from simpler ones. For example, the
# `Cayley map <https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map>`_
# maps the skew-symmetric matrices to the orthogonal matrices of positive determinant. We can
# concatenate ``Skew`` and a parametrization that implements the Cayley map to get a layer with
# orthogonal weights
class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X):
        # (I + X)(I - X)^{-1}
        return torch.linalg.solve(self.Id - X, self.Id + X)

layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Skew())
parametrize.register_parametrization(layer, "weight", CayleyMap(3))
X = layer.weight
print(torch.dist(X.T @ X, torch.eye(3)))  # X is orthogonal

###############################################################################
# This may also be used to prune a parametrized module, or to reuse parametrizations. For example,
# the matrix exponential maps the symmetric matrices to the Symmetric Positive Definite (SPD) matrices
# But the matrix exponential also maps the skew-symmetric matrices to the orthogonal matrices.
# Using these two facts, we may reuse the parametrizations before to our advantage
class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)

layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", MatrixExponential())
X = layer_orthogonal.weight
print(torch.dist(X.T @ X, torch.eye(3)))         # X is orthogonal

layer_spd = nn.Linear(3, 3)
parametrize.register_parametrization(layer_spd, "weight", Symmetric())
parametrize.register_parametrization(layer_spd, "weight", MatrixExponential())
X = layer_spd.weight
print(torch.dist(X, X.T))                        # X is symmetric
print((torch.linalg.eigvalsh(X) > 0.).all())  # X is positive definite

###############################################################################
# Initializing parametrizations
# -----------------------------
#
# Parametrizations come with a mechanism to initialize them. If we implement a method
# ``right_inverse`` with signature
#
# .. code-block:: python
#
#     def right_inverse(self, X: Tensor) -> Tensor
#
# it will be used when assigning to the parametrized tensor.
#
# Let's upgrade our implementation of the ``Skew`` class to support this
class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)

    def right_inverse(self, A):
        # We assume that A is skew-symmetric
        # We take the upper-triangular elements, as these are those used in the forward
        return A.triu(1)

###############################################################################
# We may now initialize a layer that is parametrized with ``Skew``
layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Skew())
X = torch.rand(3, 3)
X = X - X.T                             # X is now skew-symmetric
layer.weight = X                        # Initialize layer.weight to be X
print(torch.dist(layer.weight, X))      # layer.weight == X

###############################################################################
# This ``right_inverse`` works as expected when we concatenate parametrizations.
# To see this, let's upgrade the Cayley parametrization to also support being initialized
class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X):
        # Assume X skew-symmetric
        # (I + X)(I - X)^{-1}
        return torch.linalg.solve(self.Id - X, self.Id + X)

    def right_inverse(self, A):
        # Assume A orthogonal
        # See https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
        # (A - I)(A + I)^{-1}
        return torch.linalg.solve(A + self.Id, self.Id - A)

layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", CayleyMap(3))
# Sample an orthogonal matrix with positive determinant
X = torch.empty(3, 3)
nn.init.orthogonal_(X)
if X.det() < 0.:
    X[0].neg_()
layer_orthogonal.weight = X
print(torch.dist(layer_orthogonal.weight, X))  # layer_orthogonal.weight == X

###############################################################################
# This initialization step can be written more succinctly as
layer_orthogonal.weight = nn.init.orthogonal_(layer_orthogonal.weight)

###############################################################################
# The name of this method comes from the fact that we would often expect
# that ``forward(right_inverse(X)) == X``. This is a direct way of rewriting that
# the forward after the initialization with value ``X`` should return the value ``X``.
# This constraint is not strongly enforced in practice. In fact, at times, it might be of
# interest to relax this relation. For example, consider the following implementation
# of a randomized pruning method:
class PruningParametrization(nn.Module):
    def __init__(self, X, p_drop=0.2):
        super().__init__()
        # sample zeros with probability p_drop
        mask = torch.full_like(X, 1.0 - p_drop)
        self.mask = torch.bernoulli(mask)

    def forward(self, X):
        return X * self.mask

    def right_inverse(self, A):
        return A

###############################################################################
# In this case, it is not true that for every matrix A ``forward(right_inverse(A)) == A``.
# This is only true when the matrix ``A`` has zeros in the same positions as the mask.
# Even then, if we assign a tensor to a pruned parameter, it will comes as no surprise
# that tensor will be, in fact, pruned
layer = nn.Linear(3, 4)
X = torch.rand_like(layer.weight)
print(f"Initialization matrix:\n{X}")
parametrize.register_parametrization(layer, "weight", PruningParametrization(layer.weight))
layer.weight = X
print(f"\nInitialized weight:\n{layer.weight}")

###############################################################################
# Removing parametrizations
# -------------------------
#
# We may remove all the parametrizations from a parameter or a buffer in a module
# by using ``parametrize.remove_parametrizations()``
layer = nn.Linear(3, 3)
print("Before:")
print(layer)
print(layer.weight)
parametrize.register_parametrization(layer, "weight", Skew())
print("\nParametrized:")
print(layer)
print(layer.weight)
parametrize.remove_parametrizations(layer, "weight")
print("\nAfter. Weight has skew-symmetric values but it is unconstrained:")
print(layer)
print(layer.weight)

###############################################################################
# When removing a parametrization, we may choose to leave the original parameter (i.e. that in
# ``layer.parametriations.weight.original``) rather than its parametrized version by setting
# the flag ``leave_parametrized=False``
layer = nn.Linear(3, 3)
print("Before:")
print(layer)
print(layer.weight)
parametrize.register_parametrization(layer, "weight", Skew())
print("\nParametrized:")
print(layer)
print(layer.weight)
parametrize.remove_parametrizations(layer, "weight", leave_parametrized=False)
print("\nAfter. Same as Before:")
print(layer)
print(layer.weight)
