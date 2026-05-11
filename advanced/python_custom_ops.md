Note

Go to the end
to download the full example code.

# Custom Python Operators

 What you will learn

- How to integrate custom operators written in Python with PyTorch
- How to test custom operators using `torch.library.opcheck`

 Prerequisites

- PyTorch 2.4 or later

PyTorch offers a large library of operators that work on Tensors (e.g.
`torch.add`, `torch.sum`, etc). However, you might wish to use a new customized
operator with PyTorch, perhaps written by a third-party library. This tutorial
shows how to wrap Python functions so that they behave like PyTorch native
operators. Reasons why you may wish to create a custom operator in PyTorch include:

- Treating an arbitrary Python function as an opaque callable with respect
to `torch.compile` (that is, prevent `torch.compile` from tracing
into the function).
- Adding training support to an arbitrary Python function

Use [`torch.library.custom_op()`](https://docs.pytorch.org/docs/stable/library.html#torch.library.custom_op) to create Python custom operators.
Use the C++ `TORCH_LIBRARY` APIs to create C++ custom operators (these
work in Python-less environments).
See the [Custom Operators Landing Page](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
for more details.

Please note that if your operation can be expressed as a composition of
existing PyTorch operators, then there is usually no need to use the custom operator
API - everything (for example `torch.compile`, training support) should
just work.

## Example: Wrapping PIL's crop into a custom operator

Let's say that we are using PIL's `crop` operation.

`crop` is not handled effectively out-of-the-box by
`torch.compile`: `torch.compile` induces a
["graph break"](https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks)
on functions it is unable to handle and graph breaks are bad for performance.
The following code demonstrates this by raising an error
(`torch.compile` with `fullgraph=True` raises an error if a
graph break occurs).

```
# The following raises an error. Uncomment the line to see it.
# cropped_img = f(img)
```

In order to black-box `crop` for use with `torch.compile`, we need to
do two things:

1. wrap the function into a PyTorch custom operator.
2. add a "`FakeTensor` kernel" (aka "meta kernel") to the operator.
Given some `FakeTensors` inputs (dummy Tensors that don't have storage),
this function should return dummy Tensors of your choice with the correct
Tensor metadata (shape/strides/`dtype`/device).

```
# Use torch.library.custom_op to define a new custom operator.
# If your operator mutates any input Tensors, their names must be specified
# in the ``mutates_args`` argument.

# Use register_fake to add a ``FakeTensor`` kernel for the operator
```

After this, `crop` now works without graph breaks:

## Adding training support for crop

Use `torch.library.register_autograd` to add training support for an operator.
Prefer this over directly using `torch.autograd.Function`; some compositions of
`autograd.Function` with PyTorch operator registration APIs can lead to (and
has led to) silent incorrectness when composed with `torch.compile`.

If you don't need training support, there is no need to use
`torch.library.register_autograd`.
If you end up training with a `custom_op` that doesn't have an autograd
registration, we'll raise an error message.

The gradient formula for `crop` is essentially `PIL.paste` (we'll leave the
derivation as an exercise to the reader). Let's first wrap `paste` into a
custom operator:

And now let's use `register_autograd` to specify the gradient formula for `crop`:

Note that the backward must be a composition of PyTorch-understood operators,
which is why we wrapped paste into a custom operator instead of directly using
PIL's paste.

This is the correct gradient, with 1s (white) in the cropped region and 0s
(black) in the unused region.

## Testing Python Custom operators

Use `torch.library.opcheck` to test that the custom operator was registered
correctly. This does not test that the gradients are mathematically correct;
please write separate tests for that (either manual ones or `torch.autograd.gradcheck`).

To use `opcheck`, pass it a set of example inputs to test against. If your
operator supports training, then the examples should include Tensors that
require grad. If your operator supports multiple devices, then the examples
should include Tensors from each device.

## Mutable Python Custom operators

You can also wrap a Python function that mutates its inputs into a custom
operator.
Functions that mutate inputs are common because that is how many low-level
kernels are written; for example, a kernel that computes `sin` may take in
the input and an output tensor and write `input.sin()` to the output tensor.

We'll use `numpy.sin` to demonstrate an example of a mutable Python
custom operator.

Because the operator doesn't return anything, there is no need to register
a `FakeTensor` kernel (meta kernel) to get it to work with `torch.compile`.

And here's an `opcheck` run telling us that we did indeed register the operator correctly.
`opcheck` would error out if we forgot to add the output to `mutates_args`, for example.

## Conclusion

In this tutorial, we learned how to use `torch.library.custom_op` to
create a custom operator in Python that works with PyTorch subsystems
such as `torch.compile` and autograd.

This tutorial provides a basic introduction to custom operators.
For more detailed information, see:

- [the torch.library documentation](https://pytorch.org/docs/stable/library.html)
- [the Custom Operators Manual](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#the-custom-operators-manual)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: python_custom_ops.ipynb`](../_downloads/9878ff22933dc5322c65087cfef530a2/python_custom_ops.ipynb)

[`Download Python source code: python_custom_ops.py`](../_downloads/ce0cb1cce555cead1bcaba8a6d337c6f/python_custom_ops.py)

[`Download zipped: python_custom_ops.zip`](../_downloads/f7f21519a06aff88cc7a5a2be58e9038/python_custom_ops.zip)