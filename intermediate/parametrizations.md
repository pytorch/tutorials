Note

Go to the end
to download the full example code.

# Parametrizations Tutorial

**Author**: [Mario Lezcano](https://github.com/lezcano)

Regularizing deep-learning models is a surprisingly challenging task.
Classical techniques such as penalty methods often fall short when applied
on deep models due to the complexity of the function being optimized.
This is particularly problematic when working with ill-conditioned models.
Examples of these are RNNs trained on long sequences and GANs. A number
of techniques have been proposed in recent years to regularize these
models and improve their convergence. On recurrent models, it has been
proposed to control the singular values of the recurrent kernel for the
RNN to be well-conditioned. This can be achieved, for example, by making
the recurrent kernel [orthogonal](https://en.wikipedia.org/wiki/Orthogonal_matrix).
Another way to regularize recurrent models is via
"[weight normalization](https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html)".
This approach proposes to decouple the learning of the parameters from the
learning of their norms. To do so, the parameter is divided by its
[Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm)
and a separate parameter encoding its norm is learned.
A similar regularization was proposed for GANs under the name of
"[spectral normalization](https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html)". This method
controls the Lipschitz constant of the network by dividing its parameters by
their [spectral norm](https://en.wikipedia.org/wiki/Matrix_norm#Special_cases),
rather than their Frobenius norm.

All these methods have a common pattern: they all transform a parameter
in an appropriate way before using it. In the first case, they make it orthogonal by
using a function that maps matrices to orthogonal matrices. In the case of weight
and spectral normalization, they divide the original parameter by its norm.

More generally, all these examples use a function to put extra structure on the parameters.
In other words, they use a function to constrain the parameters.

In this tutorial, you will learn how to implement and use this pattern to put
constraints on your model. Doing so is as easy as writing your own `nn.Module`.

Requirements: `torch>=1.9.0`

## Implementing parametrizations by hand

Assume that we want to have a square linear layer with symmetric weights, that is,
with weights `X` such that `X = Xᵀ`. One way to do so is
to copy the upper-triangular part of the matrix into its lower-triangular part

We can then use this idea to implement a linear layer with symmetric weights

The layer can be then used as a regular linear layer

This implementation, although correct and self-contained, presents a number of problems:

1. It reimplements the layer. We had to implement the linear layer as `x @ A`. This is
not very problematic for a linear layer, but imagine having to reimplement a CNN or a
Transformer...
2. It does not separate the layer and the parametrization. If the parametrization were
more difficult, we would have to rewrite its code for each layer that we want to use it
in.
3. It recomputes the parametrization every time we use the layer. If we use the layer
several times during the forward pass, (imagine the recurrent kernel of an RNN), it
would compute the same `A` every time that the layer is called.

## Introduction to parametrizations

Parametrizations can solve all these problems as well as others.

Let's start by reimplementing the code above using `torch.nn.utils.parametrize`.
The only thing that we have to do is to write the parametrization as a regular `nn.Module`

This is all we need to do. Once we have this, we can transform any regular layer into a
symmetric layer by doing

Now, the matrix of the linear layer is symmetric

We can do the same thing with any other layer. For example, we can create a CNN with
[skew-symmetric](https://en.wikipedia.org/wiki/Skew-symmetric_matrix) kernels.
We use a similar parametrization, copying the upper-triangular part with signs
reversed into the lower-triangular part

```
# Print a few kernels
```

## Inspecting a parametrized module

When a module is parametrized, we find that the module has changed in three ways:

1. `model.weight` is now a property
2. It has a new `module.parametrizations` attribute
3. The unparametrized weight has been moved to `module.parametrizations.weight.original`

After parametrizing `weight`, `layer.weight` is turned into a
[Python property](https://docs.python.org/3/library/functions.html#property).
This property computes `parametrization(weight)` every time we request `layer.weight`
just as we did in our implementation of `LinearSymmetric` above.

Registered parametrizations are stored under a `parametrizations` attribute within the module.

This `parametrizations` attribute is an `nn.ModuleDict`, and it can be accessed as such

Each element of this `nn.ModuleDict` is a `ParametrizationList`, which behaves like an
`nn.Sequential`. This list will allow us to concatenate parametrizations on one weight.
Since this is a list, we can access the parametrizations indexing it. Here's
where our `Symmetric` parametrization sits

The other thing that we notice is that, if we print the parameters, we see that the
parameter `weight` has been moved

It now sits under `layer.parametrizations.weight.original`

Besides these three small differences, the parametrization is doing exactly the same
as our manual implementation

## Parametrizations are first-class citizens

Since `layer.parametrizations` is an `nn.ModuleList`, it means that the parametrizations
are properly registered as submodules of the original module. As such, the same rules
for registering parameters in a module apply to register a parametrization.
For example, if a parametrization has parameters, these will be moved from CPU
to CUDA when calling `model = model.cuda()`.

## Caching the value of a parametrization

Parametrizations come with an inbuilt caching system via the context manager
`parametrize.cached()`

## Concatenating parametrizations

Concatenating two parametrizations is as easy as registering them on the same tensor.
We may use this to create more complex parametrizations from simpler ones. For example, the
[Cayley map](https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map)
maps the skew-symmetric matrices to the orthogonal matrices of positive determinant. We can
concatenate `Skew` and a parametrization that implements the Cayley map to get a layer with
orthogonal weights

This may also be used to prune a parametrized module, or to reuse parametrizations. For example,
the matrix exponential maps the symmetric matrices to the Symmetric Positive Definite (SPD) matrices
But the matrix exponential also maps the skew-symmetric matrices to the orthogonal matrices.
Using these two facts, we may reuse the parametrizations before to our advantage

## Initializing parametrizations

Parametrizations come with a mechanism to initialize them. If we implement a method
`right_inverse` with signature

```
def right_inverse(self, X: Tensor) -> Tensor
```

it will be used when assigning to the parametrized tensor.

Let's upgrade our implementation of the `Skew` class to support this

We may now initialize a layer that is parametrized with `Skew`

This `right_inverse` works as expected when we concatenate parametrizations.
To see this, let's upgrade the Cayley parametrization to also support being initialized

```
# Sample an orthogonal matrix with positive determinant
```

This initialization step can be written more succinctly as

The name of this method comes from the fact that we would often expect
that `forward(right_inverse(X)) == X`. This is a direct way of rewriting that
the forward after the initialization with value `X` should return the value `X`.
This constraint is not strongly enforced in practice. In fact, at times, it might be of
interest to relax this relation. For example, consider the following implementation
of a randomized pruning method:

In this case, it is not true that for every matrix A `forward(right_inverse(A)) == A`.
This is only true when the matrix `A` has zeros in the same positions as the mask.
Even then, if we assign a tensor to a pruned parameter, it will comes as no surprise
that tensor will be, in fact, pruned

## Removing parametrizations

We may remove all the parametrizations from a parameter or a buffer in a module
by using `parametrize.remove_parametrizations()`

When removing a parametrization, we may choose to leave the original parameter (i.e. that in
`layer.parametriations.weight.original`) rather than its parametrized version by setting
the flag `leave_parametrized=False`

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: parametrizations.ipynb`](../_downloads/c9153ca254003481aecc7a760a7b046f/parametrizations.ipynb)

[`Download Python source code: parametrizations.py`](../_downloads/621174a140b9f76910c50ed4afb0e621/parametrizations.py)

[`Download zipped: parametrizations.zip`](../_downloads/6cae7ea1310692224a35738b4d91aa85/parametrizations.zip)