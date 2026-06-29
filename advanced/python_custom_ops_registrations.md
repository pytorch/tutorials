Note

Go to the end
to download the full example code.

# Adding Training and Other Registrations to Python Custom Operators

Start here after a base operator passes `torch.library.opcheck`:

- [Functional Python Custom Operators](python_custom_ops_functional.html#python-custom-ops-functional)
- [Mutable Python Custom Operators](python_custom_ops_mutable.html#python-custom-ops-mutable)

Registrations do not change the base contract. After adding one, rerun
`torch.library.opcheck` on representative inputs for that subsystem.

## Adding training support for NumPy sin

Use `torch.library.register_autograd` to add training support for an
operator. Prefer this over directly using `torch.autograd.Function`; some
compositions of `autograd.Function` with PyTorch operator registration APIs
can lead to (and has led to) silent incorrectness when composed with
`torch.compile`.

If you don't need training support, there is no need to use
`torch.library.register_autograd`. If you end up training with a
`custom_op` that doesn't have an autograd registration, we'll raise an error
message.

This page uses the same `numpy.sin` operation as the functional and mutable
pages so the only new concept is the autograd registration.

The fake kernel must describe the same output metadata as the real kernel,
including shape, strides, dtype, device, layout, and storage offset when
relevant. Here the real kernel returns `torch.empty_like(x)`, so the fake
kernel does the same.

The gradient formula for `sin(x)` is `cos(x)`. The backward formula must
be written in terms of PyTorch-understood operations or other custom
operators. Do not directly use non-traceable Python or NumPy code from the
backward formula.

Register the backward formula and the context setup function:

## Testing autograd registration

`opcheck` verifies that autograd was registered in a supported way, but it
does not prove that the gradient formula is mathematically correct. Use
separate numerical tests for that, either manual ones or
`torch.autograd.gradcheck`.

## Other registrations

Add these only when users need them.

- **Multiple device kernels:** pass `device_types="cpu"` or
`device_types="cuda"` if the implementation only works on one device.
Register device-specific kernels when devices need different code.
- **``torch.vmap``:** register a vmap rule with `torch.library.register_vmap`
when batching over the operator should do something different from a Python
loop over the batch dimension.
- **Tensor subclasses or modes:** use `torch.library.register_torch_dispatch`
when a Tensor subclass or `TorchDispatchMode` needs special behavior.
- **Autocast:** for C++/CUDA operators that should participate in autocast,
add an autocast registration as described in the C++ custom operator guide.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: python_custom_ops_registrations.ipynb`](../_downloads/747f02f817c7eb24314dc1fca2e23ee3/python_custom_ops_registrations.ipynb)

[`Download Python source code: python_custom_ops_registrations.py`](../_downloads/4ed0648c0afab0300f4355b221c59931/python_custom_ops_registrations.py)

[`Download zipped: python_custom_ops_registrations.zip`](../_downloads/ad2b486d0f6811fbee8824d2cd102a42/python_custom_ops_registrations.zip)