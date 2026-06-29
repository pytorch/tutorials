# Custom Python Operators

 What you will learn

- When to create a Python custom operator
- How to choose between functional and mutable operator contracts
- Why the schema and mutation/aliasing contract are required
- Where fake kernels, autograd, and other registrations fit

PyTorch offers a large library of operators that work on Tensors, such as
`torch.add` and `torch.sum`. However, you might wish to use a new custom
operator with PyTorch, perhaps written by a third-party library. This guide
shows how to wrap Python functions so that they behave like PyTorch native
operators.

Reasons why you may wish to create a custom operator in PyTorch include:

- treating an arbitrary Python function as an opaque callable with respect to
`torch.compile` and/or `torch.export`
- adding training support to an arbitrary Python function.

Please note that if your operation can be expressed as a composition of
existing PyTorch operators, then there is usually no need to use the custom
operator API. `torch.compile`, training support, and other PyTorch subsystems
should usually work.

Every custom operator needs:

- a stable schema and mutation/aliasing contract;
- validation with `torch.library.opcheck`;
- a fake kernel if it returns tensors and must work with `torch.compile` or
`torch.export`.

Choose one path:

- [Functional custom operators](python_custom_ops_functional.html#python-custom-ops-functional): the
operator returns fresh tensors and mutates no inputs.
- [Mutable custom operators](python_custom_ops_mutable.html#python-custom-ops-mutable): the operator
mutates an input or writes into an output buffer. Starting in PyTorch 2.13,
this includes PyTorch-style in-place and `out=` custom operators.
- [Optional registrations](python_custom_ops_registrations.html#python-custom-ops-registrations): add
autograd, `torch.vmap`, Tensor subclass behavior, or other subsystem
support after the base operator passes `opcheck`.

Choose your path

- **Any custom operator:** read
Schema and mutation/aliasing contract
and Validation. You need a stable
schema, representative examples, and `opcheck`.
- **Code that returns new tensors and does not mutate inputs:** read
[Functional custom operators](python_custom_ops_functional.html#python-custom-ops-functional). You need
`custom_op(..., mutates_args=())`, a fake kernel for `torch.compile`,
and `opcheck`.
- **A kernel that writes into existing memory:** read
[Mutable custom operators](python_custom_ops_mutable.html#python-custom-ops-mutable). You need
accurate `mutates_args` and one clear mutation pattern.
- **In-place, ``out=``, or maybe-out behavior:** read
[Mutable custom operators](python_custom_ops_mutable.html#python-custom-ops-mutable) and
Schema contract. Starting in
PyTorch 2.13, tagged in-place and `out=` custom operators are available;
split maybe-out behavior into separate operators.
- **Training support, ``vmap``, or Tensor subclass behavior:** read
[Adding registrations](python_custom_ops_registrations.html#python-custom-ops-registrations). Start with a
validated base operator, then add the registration for that subsystem.

For Python-less environments or AOTInductor, define the operator and backend
kernels in C++ instead. See the
[C++ custom operator tutorial](cpp_custom_ops.html#cpp-custom-ops-tutorial).

## Before you start

A kernel is the implementation. An operator is the PyTorch-facing contract:
name, inputs, outputs, mutation behavior, and subsystem registrations.

A custom operator gives PyTorch an explicit boundary. Use it when tracing into
the implementation is impossible or undesirable.

## Required: schema and mutation/aliasing contract

Decide the schema and mutation/aliasing contract before writing registrations.
PyTorch uses the schema and registrations to reason about mutation/aliasing;
it does not infer the contract from the Python function body.

Two Tensors alias when they share the same underlying storage. For example,
`y = x.view(-1)` creates a
[view](https://docs.pytorch.org/docs/main/tensor_view.html) `y` that aliases
`x`, so writing to `y` can change `x`.

- The schema must be stable: the mutation and aliasing behavior must be correct
and consistent. This means that an operator must not return an output that
sometimes aliases its input. Also, the operator may not mutate an input that
is not marked as being mutated.
- A functional custom operator must return fresh tensors. Do not return an
input tensor, a view of an input, or two outputs that alias each other.
- A mutable custom operator must list every mutated argument in `mutates_args`.
- A fake kernel must return tensors with the same metadata as the real kernel:
shape, dtype, device, layout, strides, and storage offset when relevant.
`empty_like(x)` is only correct when the real output has the same metadata
as `x`. The functional custom operator page shows an executable example of
this metadata mismatch.
- Fake kernels may inspect metadata, but must not read tensor data.
- Avoid "maybe-out" operators. An operator that sometimes allocates a new
tensor and sometimes writes into an output buffer has different aliasing
contracts for different calls.

Split maybe-out behavior into two operators: one functional operator that
allocates and one mutable operator that writes into an output buffer.

## Required: validate with opcheck

`torch.library.opcheck` validates the registration contract: schema, fake
kernel, autograd registration, and behavior under compilation APIs.

Run `opcheck` on representative inputs:

- each supported device;
- important dtypes;
- edge shapes such as empty tensors;
- important memory formats or non-contiguous strides;
- inputs with `requires_grad=True` if the operator supports training.

`opcheck` is not a numerical correctness test. Use
`torch.testing.assert_close` or ordinary unit tests for forward correctness,
and `torch.autograd.gradcheck` for gradient formulas.

## Next steps

Read one base-contract page first, then add registrations only if needed:

- [Functional custom operators](python_custom_ops_functional.html#python-custom-ops-functional)
- [Mutable custom operators](python_custom_ops_mutable.html#python-custom-ops-mutable)
- [Adding training and other registrations](python_custom_ops_registrations.html#python-custom-ops-registrations)