Note

Go to the end
to download the full example code.

# Forward-mode Automatic Differentiation (Beta)

This tutorial demonstrates how to use forward-mode AD to compute
directional derivatives (or equivalently, Jacobian-vector products).

The tutorial below uses some APIs only available in versions >= 1.11
(or nightly builds).

Also note that forward-mode AD is currently in beta. The API is
subject to change and operator coverage is still incomplete.

## Basic Usage

Unlike reverse-mode AD, forward-mode AD computes gradients eagerly
alongside the forward pass. We can use forward-mode AD to compute a
directional derivative by performing the forward pass as before,
except we first associate our input with another tensor representing
the direction of the directional derivative (or equivalently, the `v`
in a Jacobian-vector product). When an input, which we call "primal", is
associated with a "direction" tensor, which we call "tangent", the
resultant new tensor object is called a "dual tensor" for its connection
to dual numbers[0].

As the forward pass is performed, if any input tensors are dual tensors,
extra computation is performed to propagate this "sensitivity" of the
function.

```
# All forward AD computation must be performed in the context of
# a ``dual_level`` context. All dual tensors created in such a context
# will have their tangents destroyed upon exit. This is to ensure that
# if the output or intermediate results of this computation are reused
# in a future forward AD computation, their tangents (which are associated
# with this computation) won't be confused with tangents from the later
# computation.
```

## Usage with Modules

To use `nn.Module` with forward AD, replace the parameters of your
model with dual tensors before performing the forward pass. At the
time of writing, it is not possible to create dual tensor
`nn.Parameter`s. As a workaround, one must register the dual tensor
as a non-parameter attribute of the module.

## Using the functional Module API (beta)

Another way to use `nn.Module` with forward AD is to utilize
the functional Module API (also known as the stateless Module API).

```
# We need a fresh module because the functional call requires the
# the model to have parameters registered.

# Check our results
```

## Custom autograd Function

Custom Functions also support forward-mode AD. To create custom Function
supporting forward-mode AD, register the `jvp()` static method. It is
possible, but not mandatory for custom Functions to support both forward
and backward AD. See the
[documentation](https://pytorch.org/docs/master/notes/extending.html#forward-mode-ad)
for more information.

```
# It is important to use ``autograd.gradcheck`` to verify that your
# custom autograd Function computes the gradients correctly. By default,
# ``gradcheck`` only checks the backward-mode (reverse-mode) AD gradients. Specify
# ``check_forward_ad=True`` to also check forward grads. If you did not
# implement the backward formula for your function, you can also tell ``gradcheck``
# to skip the tests that require backward-mode AD by specifying
# ``check_backward_ad=False``, ``check_undefined_grad=False``, and
# ``check_batched_grad=False``.
```

## Functional API (beta)

We also offer a higher-level functional API in functorch
for computing Jacobian-vector products that you may find simpler to use
depending on your use case.

The benefit of the functional API is that there isn't a need to understand
or use the lower-level dual tensor API and that you can compose it with
other [functorch transforms (like vmap)](https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html);
the downside is that it offers you less control.

Note that the remainder of this tutorial will require functorch
([pytorch/functorch](https://github.com/pytorch/functorch)) to run. Please find installation
instructions at the specified link.

```
# Here is a basic example to compute the JVP of the above function.
# The ``jvp(func, primals, tangents)`` returns ``func(*primals)`` as well as the
# computed Jacobian-vector product (JVP). Each primal must be associated with a tangent of the same shape.

# ``functorch.jvp`` requires every primal to be associated with a tangent.
# If we only want to associate certain inputs to `fn` with tangents,
# then we'll need to create a new function that captures inputs without tangents:
```

## Using the functional API with Modules

To use `nn.Module` with `functorch.jvp` to compute Jacobian-vector products
with respect to the model parameters, we need to reformulate the
`nn.Module` as a function that accepts both the model parameters and inputs
to the module.

```
# Given a ``torch.nn.Module``, ``ft.make_functional_with_buffers`` extracts the state
# (``params`` and buffers) and returns a functional version of the model that
# can be invoked like a function.
# That is, the returned ``func`` can be invoked like
# ``func(params, buffers, input)``.
# ``ft.make_functional_with_buffers`` is analogous to the ``nn.Modules`` stateless API
# that you saw previously and we're working on consolidating the two.

# Because ``jvp`` requires every input to be associated with a tangent, we need to
# create a new function that, when given the parameters, produces the output
```

[0] [https://en.wikipedia.org/wiki/Dual_number](https://en.wikipedia.org/wiki/Dual_number)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: forward_ad_usage.ipynb`](../_downloads/31e117c487018c27130cd7b1fd3e3cad/forward_ad_usage.ipynb)

[`Download Python source code: forward_ad_usage.py`](../_downloads/3a285734c191abde60d7db0362f294b1/forward_ad_usage.py)

[`Download zipped: forward_ad_usage.zip`](../_downloads/80a4e63d9d30c1740e06a75d0e4139f7/forward_ad_usage.zip)