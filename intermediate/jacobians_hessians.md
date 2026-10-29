Note

Go to the end
to download the full example code.

# Jacobians, Hessians, hvp, vhp, and more: composing function transforms

Computing jacobians or hessians are useful in a number of non-traditional
deep learning models. It is difficult (or annoying) to compute these quantities
efficiently using PyTorch's regular autodiff APIs
(`Tensor.backward()`, `torch.autograd.grad`). PyTorch's
[JAX-inspired](https://github.com/google/jax)
[function transforms API](https://pytorch.org/docs/master/func.html)
provides ways of computing various higher-order autodiff quantities
efficiently.

Note

This tutorial requires PyTorch 2.0.0 or later.

## Computing the Jacobian

Let's start with a function that we'd like to compute the jacobian of.
This is a simple linear function with non-linear activation.

Let's add some dummy data: a weight, a bias, and a feature vector x.

Let's think of `predict` as a function that maps the input `x` from \(R^D \to R^D\).
PyTorch Autograd computes vector-Jacobian products. In order to compute the full
Jacobian of this \(R^D \to R^D\) function, we would have to compute it row-by-row
by using a different unit vector each time.

Instead of computing the jacobian row-by-row, we can use PyTorch's
`torch.vmap` function transform to get rid of the for-loop and vectorize the
computation. We can't directly apply `vmap` to `torch.autograd.grad`;
instead, PyTorch provides a `torch.func.vjp` transform that composes with
`torch.vmap`:

```
# let's confirm both methods compute the same result
```

In a later tutorial a composition of reverse-mode AD and `vmap` will give us
per-sample-gradients.
In this tutorial, composing reverse-mode AD and `vmap` gives us Jacobian
computation!
Various compositions of `vmap` and autodiff transforms can give us different
interesting quantities.

PyTorch provides `torch.func.jacrev` as a convenience function that performs
the `vmap-vjp` composition to compute jacobians. `jacrev` accepts an `argnums`
argument that says which argument we would like to compute Jacobians with
respect to.

```
# Confirm by running the following:
```

Let's compare the performance of the two ways to compute the jacobian.
The function transform version is much faster (and becomes even faster the
more outputs there are).

In general, we expect that vectorization via `vmap` can help eliminate overhead
and give better utilization of your hardware.

`vmap` does this magic by pushing the outer loop down into the function's
primitive operations in order to obtain better performance.

Let's make a quick function to evaluate performance and deal with
microseconds and milliseconds measurements:

And then run the performance comparison:

Let's do a relative performance comparison of the above with our `get_perf` function:

Furthermore, it's pretty easy to flip the problem around and say we want to
compute Jacobians of the parameters to our model (weight, bias) instead of the input

```
# note the change in input via ``argnums`` parameters of 0,1 to map to weight and bias
```

## Reverse-mode Jacobian (`jacrev`) vs forward-mode Jacobian (`jacfwd`)

We offer two APIs to compute jacobians: `jacrev` and `jacfwd`:

- `jacrev` uses reverse-mode AD. As you saw above it is a composition of our
`vjp` and `vmap` transforms.
- `jacfwd` uses forward-mode AD. It is implemented as a composition of our
`jvp` and `vmap` transforms.

`jacfwd` and `jacrev` can be substituted for each other but they have different
performance characteristics.

As a general rule of thumb, if you're computing the jacobian of an \(R^N \to R^M\)
function, and there are many more outputs than inputs (for example, \(M > N\)) then
`jacfwd` is preferred, otherwise use `jacrev`. There are exceptions to this rule,
but a non-rigorous argument for this follows:

In reverse-mode AD, we are computing the jacobian row-by-row, while in
forward-mode AD (which computes Jacobian-vector products), we are computing
it column-by-column. The Jacobian matrix has M rows and N columns, so if it
is taller or wider one way we may prefer the method that deals with fewer
rows or columns.

First, let's benchmark with more inputs than outputs:

```
# remember the general rule about taller vs wider... here we have a taller matrix:
```

and then do a relative benchmark:

and now the reverse - more outputs (M) than inputs (N):

and a relative performance comparison:

## Hessian computation with functorch.hessian

We offer a convenience API to compute hessians: `torch.func.hessiani`.
Hessians are the jacobian of the jacobian (or the partial derivative of
the partial derivative, aka second order).

This suggests that one can just compose functorch jacobian transforms to
compute the Hessian.
Indeed, under the hood, `hessian(f)` is simply `jacfwd(jacrev(f))`.

Note: to boost performance: depending on your model, you may also want to
use `jacfwd(jacfwd(f))` or `jacrev(jacrev(f))` instead to compute hessians
leveraging the rule of thumb above regarding wider vs taller matrices.

```
# lets reduce the size in order not to overwhelm Colab. Hessians require
# significant memory:
```

Let's verify we have the same result regardless of using hessian API or
using `jacfwd(jacfwd())`.

## Batch Jacobian and Batch Hessian

In the above examples we've been operating with a single feature vector.
In some cases you might want to take the Jacobian of a batch of outputs
with respect to a batch of inputs. That is, given a batch of inputs of
shape `(B, N)` and a function that goes from \(R^N \to R^M\), we would like
a Jacobian of shape `(B, M, N)`.

The easiest way to do this is to use `vmap`:

If you have a function that goes from (B, N) -> (B, M) instead and are
certain that each input produces an independent output, then it's also
sometimes possible to do this without using `vmap` by summing the outputs
and then computing the Jacobian of that function:

If you instead have a function that goes from \(R^N \to R^M\) but inputs that
are batched, you compose `vmap` with `jacrev` to compute batched jacobians:

Finally, batch hessians can be computed similarly. It's easiest to think
about them by using `vmap` to batch over hessian computation, but in some
cases the sum trick also works.

## Computing Hessian-vector products

The naive way to compute a Hessian-vector product (hvp) is to materialize
the full Hessian and perform a dot-product with a vector. We can do better:
it turns out we don't need to materialize the full Hessian to do this. We'll
go through two (of many) different strategies to compute Hessian-vector products:
- composing reverse-mode AD with reverse-mode AD
- composing reverse-mode AD with forward-mode AD

Composing reverse-mode AD with forward-mode AD (as opposed to reverse-mode
with reverse-mode) is generally the more memory efficient way to compute a
hvp because forward-mode AD doesn't need to construct an Autograd graph and
save intermediates for backward:

Here's some sample usage.

If PyTorch forward-AD does not have coverage for your operations, then we can
instead compose reverse-mode AD with reverse-mode AD:

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: jacobians_hessians.ipynb`](../_downloads/748f25c58a5ac0f57235c618e51c869b/jacobians_hessians.ipynb)

[`Download Python source code: jacobians_hessians.py`](../_downloads/089b69a49b6eb4080d35c4b983b939a5/jacobians_hessians.py)

[`Download zipped: jacobians_hessians.zip`](../_downloads/25c8abd555a1cea24f31c027d3a1a502/jacobians_hessians.zip)