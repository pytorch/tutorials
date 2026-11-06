Note

Go to the end
to download the full example code.

# Hooks for autograd saved tensors

PyTorch typically computes gradients using backpropagation. However,
certain operations require intermediary results to be saved in order to
perform backpropagation. This tutorial walks through how these tensors
are saved/retrieved and how you can define hooks to control the
packing/unpacking process.

This tutorial assumes you are familiar with how backpropagation works in
theory. If not, read [this](https://colab.research.google.com/drive/1aWNdmYt7RcHMbUk-Xz2Cv5-cGFSWPXe0#scrollTo=AHcEJ6nXUb7W) first.

## Saved tensors

Training a model usually consumes more memory than running it for
inference. Broadly speaking, one can say that it is because "PyTorch
needs to save the computation graph, which is needed to call
`backward`", hence the additional memory usage. One goal of this
tutorial is to finetune this understanding.

In fact, the graph in itself sometimes does not consume much more memory
as it never copies any tensors. However, the graph can keep *references*
to tensors that would otherwise have gone out of scope: those are
referred to as **saved tensors**.

### Why does training a model (typically) requires more memory than evaluating it?

We start with a simple example: \(y = a \cdot b\) , for which
we know the gradients of \(y\) with respect to \(a\) and
\(b\):

\[\frac{\partial y}{\partial a} = b

\]

\[\frac{\partial y}{\partial b} = a

\]

Using a torchviz, we can visualize the computation graph

> [![https://user-images.githubusercontent.com/8019486/130124513-72e016a3-c36f-42b9-88e2-53baf3e016c5.png](https://user-images.githubusercontent.com/8019486/130124513-72e016a3-c36f-42b9-88e2-53baf3e016c5.png)](https://user-images.githubusercontent.com/8019486/130124513-72e016a3-c36f-42b9-88e2-53baf3e016c5.png)

In this example, PyTorch saves intermediary values \(a\) and
\(b\) in order to compute the gradient during the backward.

> [![https://user-images.githubusercontent.com/8019486/130124538-3da50977-6f0b-46d0-8909-5456ade9b598.png](https://user-images.githubusercontent.com/8019486/130124538-3da50977-6f0b-46d0-8909-5456ade9b598.png)](https://user-images.githubusercontent.com/8019486/130124538-3da50977-6f0b-46d0-8909-5456ade9b598.png)

Those intermediary values (in orange above) can be accessed (for
debugging purposes) by looking for attributes of the `grad_fn` of
`y` which start with the prefix `_saved`:

As the computation graph grows in depth, it will store more *saved
tensors*. Meanwhile, those tensors would have gone out of scope if not
for the graph.

[![https://user-images.githubusercontent.com/8019486/130124570-f1074098-1bb3-459e-bf5a-03bf6f65b403.png](https://user-images.githubusercontent.com/8019486/130124570-f1074098-1bb3-459e-bf5a-03bf6f65b403.png)](https://user-images.githubusercontent.com/8019486/130124570-f1074098-1bb3-459e-bf5a-03bf6f65b403.png)

In the example above, executing without grad would only have kept `x`
and `y` in the scope, But the graph additionally stores `f(x)` and
`f(f(x))`. Hence, running a forward pass during training will be more
costly in memory usage than during evaluation (more precisely, when
autograd is not required).

### The concept of packing / unpacking

Going back to the first example: `y.grad_fn._saved_self` and
`y.grad_fn._saved_other` point to the original tensor object,
respectively `a` and `b`.

However, that may not always be the case.

Under the hood, PyTorch has **packed** and **unpacked** the tensor
`y` to prevent reference cycles.

As a rule of thumb, you should *not* rely on the fact that accessing
the tensor saved for backward will yield the same tensor object as the
original tensor. They will however share the same *storage*.

## Saved tensors hooks

PyTorch provides an API to control how saved tensors should be packed /
unpacked.

The `pack_hook` function will be called every time an operation saves
a tensor for backward.
The output of `pack_hook` is then stored in the computation graph
instead of the original tensor.
The `unpack_hook` uses that return value to compute a new tensor,
which is the one actually used during the backward pass.
In general, you want `unpack_hook(pack_hook(t))` to be equal to
`t`.

One thing to note is that the output of `pack_hook` can be *any Python
object*, as long as `unpack_hook` can derive a tensor with the correct
value from it.

### Some unconventional examples

First, some silly examples to illustrate what is possible but you
probably don't ever want to do it.

#### Returning an `int`

Returning the index of a Python list
Relatively harmless but with debatable usefulness

#### Returning a tuple

Returning some tensor and a function how to unpack it
Quite unlikely to be useful in its current form

#### Returning a `str`

Returning the `__repr__ of` the tensor
Probably never do this

Although those examples will not be useful in practice, they
illustrate that the output of `pack_hook` can really be any Python
object as long as it contains enough information to retrieve the
content of the original tensor.
In the next sections, we focus on more useful applications.

### Saving tensors to CPU

Very often, the tensors involved in the computation graph live on GPU.
Keeping a reference to those tensors in the graph is what causes most
models to run out of GPU memory during training while they would have
done fine during evaluation.

Hooks provide a very simple way to implement that.

In fact, PyTorch provides an API to conveniently use those hooks (as
well as the ability to use pinned memory).

In practice, on a A100 GPU, for a ResNet-152 with batch size 256, this
corresponds to a GPU memory usage reduction from 48GB to 5GB, at the
cost of a 6x slowdown.

Of course, you can modulate the tradeoff by only saving to CPU certain
parts of the network.

For instance, you could define a special `nn.Module` that wraps any
module and saves its tensors to CPU.

### Saving tensors to disk

Similarly, you may want to save those tensors to disk. Again, this is
achievable with those hooks.

A naive version would look like this.

```
# Naive version - HINT: Don't do this
```

The reason the above code is bad is that we are leaking files on the
disk and they are never cleared. Fixing this is not as trivial as it
seems.

```
# Incorrect version - HINT: Don't do this
```

The reason the above code doesn't work is that `unpack_hook` can be
called multiple times. If we delete the file during unpacking the first
time, it will not be available when the saved tensor is accessed a
second time, which will raise an error.

To fix this, we can write a version of those hooks that takes advantage
of the fact that PyTorch automatically releases (deletes) the saved data
when it is no longer needed.

When we call `backward`, the output of `pack_hook` will be deleted,
which causes the file to be removed, so we're no longer leaking the
files.

This can then be used in your model, in the following way:

```
# Only save on disk tensors that have size >= 1000
```

In this last example, we also demonstrate how to filter which tensors
should be saved (here, those whose number of elements is greater than
1000) and how to combine this feature with `nn.DataParallel`.

If you've made it this far, congratulations! You now know how to use
saved tensor hooks and how they can be useful in a few scenarios to
tradeoff memory for compute.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: autograd_saved_tensors_hooks_tutorial.ipynb`](../_downloads/f688d6adc733eed7380d4c726b8e9643/autograd_saved_tensors_hooks_tutorial.ipynb)

[`Download Python source code: autograd_saved_tensors_hooks_tutorial.py`](../_downloads/bc9c404ccbb9fd600e9d44a56fe16bee/autograd_saved_tensors_hooks_tutorial.py)

[`Download zipped: autograd_saved_tensors_hooks_tutorial.zip`](../_downloads/f3101d1878b25f5793fdd76aa5083c9d/autograd_saved_tensors_hooks_tutorial.zip)