Note

Go to the end
to download the full example code.

# Functional Python Custom Operators

Use this path when the operator mutates no Tensor inputs and returns fresh
Tensor outputs.

If the operator must work with `torch.compile` or `torch.export`,
register a fake kernel.
The fake kernel describes output metadata without running the real kernel.

Before writing the operator, read the required schema and mutation/aliasing
contract rules in [Required: schema and mutation/aliasing contract](python_custom_ops.html#python-custom-ops-schema-contract).

Checklist:

- use `mutates_args=()`;
- return tensors that do not alias any input;
- register a fake kernel for `torch.compile` and `torch.export`;
- validate the operator with `torch.library.opcheck`.

## Example: wrapping NumPy sin into a custom operator

Let's say that we are using NumPy's `sin` operation. This is an ordinary
Python function from PyTorch's point of view: it converts the Tensor to a
NumPy array, calls NumPy, and returns a fresh Tensor.

```
# This small example focuses on the custom-operator mechanics. More complex
# Python or third-party library calls may not be handled effectively
# out-of-the-box by ``torch.compile``: ``torch.compile`` may induce a
# `"graph break" <https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks>`_
# on functions it is unable to handle, and graph breaks are bad for performance.
# A custom operator gives PyTorch an explicit boundary for such code.
#
# To make ``numpy_sin_impl`` available as a custom operator that works with
# ``torch.compile`` and ``torch.export``, we need to do two things:
#
# 1. wrap the function into a PyTorch custom operator.
# 2. add a "``FakeTensor`` kernel" (aka "meta kernel") to the operator.
# Given some ``FakeTensors`` inputs (dummy Tensors that don't have storage),
# this function should return dummy Tensors of your choice with the correct
# Tensor metadata (shape/strides/``dtype``/device).
```

Use `register_fake` to add a `FakeTensor` kernel for the operator.
`numpy_sin` returns one Tensor with the same shape, strides, dtype, device,
and storage offset as `torch.empty_like(x)`, so the fake kernel can return
`empty_like(x)`. In general, the fake kernel must match all output metadata,
including storage offset when relevant.

After this, `numpy_sin` can be used under `torch.compile`:

A PIL image transform, Python binding to a C++ extension, or another
third-party library call follows the same pattern. If it returns tensors,
write the fake kernel to match the real output metadata exactly: shape,
strides, dtype, device, layout, and storage offset when relevant.

## Example: fake kernels must match strides

The fake kernel must match the real output strides, not only the shape. This
operator returns a fresh Tensor with the same shape as `x` but different
strides.

## Testing Python custom operators

Use `torch.library.opcheck` to test that the custom operator was registered
correctly. This does not test numerical correctness; write separate tests for
that.

To use `opcheck`, pass it a set of example inputs to test against. If your
operator supports training, then the examples should include Tensors that
require grad. If your operator supports multiple devices, then the examples
should include Tensors from each device.

To add autograd, `torch.vmap`, or other subsystem support, continue to
[Adding Training and Other Registrations to Python Custom Operators](python_custom_ops_registrations.html#python-custom-ops-registrations).

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: python_custom_ops_functional.ipynb`](../_downloads/250aeb3d0973d836d22091de6347054e/python_custom_ops_functional.ipynb)

[`Download Python source code: python_custom_ops_functional.py`](../_downloads/c3fe0deb0f9f1792bf1495fc53252c56/python_custom_ops_functional.py)

[`Download zipped: python_custom_ops_functional.zip`](../_downloads/d81446822c6c5fa819f186c8dfafbe39/python_custom_ops_functional.zip)