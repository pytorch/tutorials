Note

Go to the end
to download the full example code.

# Understanding requires_grad, retain_grad, Leaf, and Non-leaf Tensors

**Author:** [Justin Silver](https://github.com/j-silv)

This tutorial explains the subtleties of `requires_grad`,
`retain_grad`, leaf, and non-leaf tensors using a simple example.

Before starting, make sure you understand [tensors and how to manipulate
them](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).
A basic knowledge of [how autograd
works](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
would also be useful.

## Setup

First, make sure [PyTorch is
installed](https://pytorch.org/get-started/locally/) and then import
the necessary libraries.

Next, we instantiate a simple network to focus on the gradients. This
will be an affine layer, followed by a ReLU activation, and ending with
a MSE loss between prediction and label tensors.

\[\mathbf{y}_{\text{pred}} = \text{ReLU}(\mathbf{x} \mathbf{W} + \mathbf{b})\]

\[L = \text{MSE}(\mathbf{y}_{\text{pred}}, \mathbf{y})\]

Note that the `requires_grad=True` is necessary for the parameters
(`W` and `b`) so that PyTorch tracks operations involving those
tensors. We'll discuss more about this in a future
section.

```
# tensor setup

# forward pass
```

## Leaf vs. non-leaf tensors

After running the forward pass, PyTorch autograd has built up a [dynamic
computational
graph](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#computational-graph)
which is shown below. This is a [Directed Acyclic Graph
(DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph) which
keeps a record of input tensors (leaf nodes), all subsequent operations
on those tensors, and the intermediate/output tensors (non-leaf nodes).
The graph is used to compute gradients for each tensor starting from the
graph roots (outputs) to the leaves (inputs) using the [chain
rule](https://en.wikipedia.org/wiki/Chain_rule) from calculus:

\[\mathbf{y} = \mathbf{f}_k\bigl(\mathbf{f}_{k-1}(\dots \mathbf{f}_1(\mathbf{x}) \dots)\bigr)\]

\[\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
\frac{\partial \mathbf{f}_k}{\partial \mathbf{f}_{k-1}} \cdot
\frac{\partial \mathbf{f}_{k-1}}{\partial \mathbf{f}_{k-2}} \cdot
\cdots \cdot
\frac{\partial \mathbf{f}_1}{\partial \mathbf{x}}\]

```
graph TD

 x["x<br/>is_leaf=True<br/>requires_grad=False<br/>retains_grad=False<br/>grad=None"]
 W["W<br/>is_leaf=True<br/>requires_grad=True<br/>retains_grad=False<br/>grad=None"]
 b["b<br/>is_leaf=True<br/>requires_grad=True<br/>retains_grad=False<br/>grad=None"]
 matmul["x @ W"]
 z["z = x @ W + b<br/>is_leaf=False<br/>requires_grad=True<br/>retains_grad=False<br/>grad=None"]
 relu["y_pred = relu(z)<br/>is_leaf=False<br/>requires_grad=True<br/>retains_grad=False<br/>grad=None"]
 y["y<br/>is_leaf=True<br/>requires_grad=False<br/>retains_grad=False<br/>grad=None"]
 loss["loss = mse(y_pred, y)<br/>is_leaf=False<br/>requires_grad=True<br/>retains_grad=False<br/>grad=None"]

 x --> matmul
 W --> matmul
 matmul --> z
 b --> z
 z --> relu
 relu --> loss
 y --> loss
```

PyTorch considers a node to be a *leaf* if it is not the result of a
tensor operation with at least one input having `requires_grad=True`
(e.g. `x`, `W`, `b`, and `y`), and everything else to be
*non-leaf* (e.g. `z`, `y_pred`, and `loss`). You can verify this
programmatically by probing the `is_leaf` attribute of the tensors:

```
# prints True because new tensors are leafs by convention

# prints False because tensor is the result of an operation with at
# least one input having requires_grad=True
```

The distinction between leaf and non-leaf determines whether the
tensor's gradient will be stored in the `grad` property after the
backward pass, and thus be usable for [gradient
descent](https://en.wikipedia.org/wiki/Gradient_descent). We'll cover
this some more in the following section.

Let's now investigate how PyTorch calculates and stores gradients for
the tensors in its computational graph.

## `requires_grad`

To build the computational graph which can be used for gradient
calculation, we need to pass in the `requires_grad=True` parameter to
a tensor constructor. By default, the value is `False`, and thus
PyTorch does not track gradients on any created tensors. To verify this,
try not setting `requires_grad`, re-run the forward pass, and then run
backpropagation. You will see:

```
>>> loss.backward()
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

This error means that autograd can't backpropagate to any leaf tensors
because `loss` is not tracking gradients. If you need to change the
property, you can call `requires_grad_()` on the tensor (notice the _
suffix).

We can sanity check which nodes require gradient calculation, just like
we did above with the `is_leaf` attribute:

It's useful to remember that a non-leaf tensor has
`requires_grad=True` by definition, since backpropagation would fail
otherwise. If the tensor is a leaf, then it will only have
`requires_grad=True` if it was specifically set by the user. Another
way to phrase this is that if at least one of the inputs to a tensor
requires the gradient, then it will require the gradient as well.

There are two exceptions to this rule:

1. Any `nn.Module` that has `nn.Parameter` will have
`requires_grad=True` for its parameters (see
[here](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#creating-models))
2. Locally disabling gradient computation with context managers (see
[here](https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation))

In summary, `requires_grad` tells autograd which tensors need to have
their gradients calculated for backpropagation to work. This is
different from which tensors have their `grad` field populated, which
is the topic of the next section.

## `retain_grad`

To actually perform optimization (e.g. SGD, Adam, etc.), we need to run
the backward pass so that we can extract the gradients.

Calling `backward()` populates the `grad` field of all leaf tensors
which had `requires_grad=True`. The `grad` is the gradient of the
loss with respect to the tensor we are probing. Before running
`backward()`, this attribute is set to `None`.

You might be wondering about the other tensors in our network. Let's
check the remaining leaf nodes:

```
# prints all None because requires_grad=False
```

The gradients for these tensors haven't been populated because we did
not explicitly tell PyTorch to calculate their gradient
(`requires_grad=False`).

Let's now look at an intermediate non-leaf node:

PyTorch returns `None` for the gradient and also warns us that a
non-leaf node's `grad` attribute is being accessed. Although autograd
has to calculate intermediate gradients for backpropagation to work, it
assumes you don't need to access the values afterwards. To change this
behavior, we can use the `retain_grad()` function on a tensor. This
tells the autograd engine to populate that tensor's `grad` after
calling `backward()`.

```
# we have to re-run the forward pass

# tell PyTorch to store the gradients after backward()

# have to zero out gradients otherwise they would accumulate

# backpropagation

# print gradients for all tensors that have requires_grad=True
```

We get the same result for `W.grad` as before. Also note that because
the loss is scalar, the gradient of the loss with respect to itself is
simply `1.0`.

If we look at the state of the computational graph now, we see that the
`retains_grad` attribute has changed for the intermediate tensors. By
convention, this attribute will print `False` for any leaf node, even
if it requires its gradient.

```
graph TD

 x["x<br/>is_leaf=True<br/>requires_grad=False<br/>retains_grad=False<br/>grad=None"]
 W["W<br/>is_leaf=True<br/>requires_grad=True<br/>retains_grad=False<br/>grad=torch.Tensor"]
 b["b<br/>is_leaf=True<br/>requires_grad=True<br/>retains_grad=False<br/>grad=torch.Tensor"]
 matmul["x @ W"]
 z["z = x @ W + b<br/>is_leaf=False<br/>requires_grad=True<br/>retains_grad=True<br/>grad=torch.Tensor"]
 relu["y_pred = relu(z)<br/>is_leaf=False<br/>requires_grad=True<br/>retains_grad=True<br/>grad=torch.Tensor"]
 y["y<br/>is_leaf=True<br/>requires_grad=True<br/>retains_grad=False<br/>grad=None"]
 loss["loss = mse(y_pred, y)<br/>is_leaf=False<br/>requires_grad=True<br/>retains_grad=True<br/>grad=torch.Tensor"]

 x --> matmul
 W --> matmul
 matmul --> z
 b --> z
 z --> relu
 relu --> loss
 y --> loss
```

If you call `retain_grad()` on a leaf tensor, it results in a no-op
since leaf tensors already retain their gradients by default (when
`requires_grad=True`).
If we call `retain_grad()` on a tensor that has `requires_grad=False`,
PyTorch actually throws an error, since it can't store the gradient if
it is never calculated.

```
>>> x.retain_grad()
RuntimeError: can't retain_grad on Tensor that has requires_grad=False
```

## Summary table

Using `retain_grad()` and `retains_grad` only make sense for
non-leaf nodes, since the `grad` attribute will already be populated
for leaf tensors that have `requires_grad=True`. By default, these
non-leaf nodes do not retain (store) their gradient after
backpropagation. We can change that by rerunning the forward pass,
telling PyTorch to store the gradients, and then performing
backpropagation.

The following table can be used as a reference which summarizes the
above discussions. The following scenarios are the only ones that are
valid for PyTorch tensors.

| `is_leaf` | `requires_grad` | `retains_grad` | `require_grad()` | `retain_grad()` |
| --- | --- | --- | --- | --- |
| `True` | `False` | `False` | sets `requires_grad` to `True` or `False` | throws error |
| `True` | `True` | `False` | sets `requires_grad` to `True` or `False` | no-op (already retains) |
| `False` | `True` | `False` | no-op | sets `retains_grad` to `True` |
| `False` | `True` | `True` | no-op | no-op (already retains) |

## Conclusion

In this tutorial, we covered when and how PyTorch computes gradients for
leaf and non-leaf tensors. By using `retain_grad`, we can access the
gradients of intermediate tensors within autograd's computational graph.

If you would like to learn more about how PyTorch's autograd system
works, please visit the references below. If you have
any feedback for this tutorial (improvements, typo fixes, etc.) then
please use the [PyTorch Forums](https://discuss.pytorch.org/) and/or
the [issue tracker](https://github.com/pytorch/tutorials/issues) to
reach out.

## References

- [A Gentle Introduction to
torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Automatic Differentiation with
torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial)
- [Autograd
mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)

%%%%%%RUNNABLE_CODE_REMOVED%%%%%%

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: understanding_leaf_vs_nonleaf_tutorial.ipynb`](../_downloads/701f907342d551c22082a3564b0b8027/understanding_leaf_vs_nonleaf_tutorial.ipynb)

[`Download Python source code: understanding_leaf_vs_nonleaf_tutorial.py`](../_downloads/7af2903c4a08c1c66c93c5186207497f/understanding_leaf_vs_nonleaf_tutorial.py)

[`Download zipped: understanding_leaf_vs_nonleaf_tutorial.zip`](../_downloads/ef20452dcc8de24156f5965e634110a0/understanding_leaf_vs_nonleaf_tutorial.zip)