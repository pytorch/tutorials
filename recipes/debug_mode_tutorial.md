Note

Go to the end
to download the full example code.

# DebugMode: Recording Dispatched Operations and Numerical Debugging

**Authors:** Pian Pawakapan, Shangdi Yu

 What you will learn

- How to capture dispatched ops for eager and `torch.compile` runs
- How to use tensor hashes and stack traces in DebugMode to pinpoint numerical divergence

 Prerequisites

- PyTorch 2.10 or later

## Overview

`DebugMode` (`torch.utils._debug_mode.DebugMode`) is a
`TorchDispatchMode` that intercepts PyTorch runtime calls and emits a
hierarchical log of operations. It is particularly useful when you need to
understand *what* actually runs, both in eager mode and under `torch.compile`
or when you need to pinpoint numerical divergence between two runs.

Key capabilities:

- **Runtime logging** - Records dispatched operations and TorchInductor compiled
Triton kernels.
- **Tensor hashing** - Attaches deterministic hashes to inputs/outputs to enable
diffing runs to locate numerical divergences.
- **Dispatch hooks** - Allows registration of custom hooks to annotate calls

Note

This recipe describes a prototype feature. Prototype features are typically
at an early stage for feedback and testing and are subject to change.

## Quick start

The snippet below captures a small eager workload and prints the debug string:

## Getting more metadata

For most investigations, you'll want to enable stack traces, tensor IDs, and tensor hashing.
These features provide metadata to correlate operations back to model code.

`DebugMode.log_tensor_hashes` decorates the log with hashes for every call.
The `hash_tensor` hash function uses `torch.hash_tensor`, which returns 0 for tensors whose
elements are all the same. The `norm` hash function uses `norm` with `p=1`.
With both these functions, especially `norm`, tensor closeness in numerics is related to hash closeness,
so it's rather interpretable. The default `hash_fn` is `norm`.

Each line follows `op(args) -> outputs`. When `record_ids` is enabled,
tensors are suffixed with `$<id>` and DTensors are labeled `dt`.

## Log Triton kernels

Though Triton kernels are not dispatched, DebugMode has custom logic that logs their inputs and outputs.

Inductor-generated Triton kernels show up with a `[triton]` prefix.
Pre/post hash annotations report buffer hashes around each kernel call, which
is helpful when isolating incorrect kernels.

## Numerical debugging with tensor hashes

If you have numerical divergence between modes, you can use DebugMode to find where the
numerical divergence originates.
In the example below, you can see that all tensor hashes are the same for eager mode and compiled mode.
If any hash is different, then that's where the numerical divergence is coming from.

Now let's look at an example where the tensor hashes are different.
I intentionally wrote a wrong decomposition that decomposes cosine to sin.
This will cause numerical divergence.

In the eager log, we have `aten::cos`, but in the compiled log, we have `aten::sin`.
Moreover, the output hash is different between eager and compiled mode.
Diffing the two logs would show that the first numerical divergence shows up in the `aten::cos` call.

## Custom dispatch hooks

Hooks allow you to annotate each call with custom metadata such as GPU memory usage. `log_hook` returns a mapping
that is rendered inline with the debug string.

## Module boundaries

`record_nn_module=True` inserts `[nn.Mod]` markers that show which
module executed each set of operations. As of PyTorch 2.10 it only works in eager mode,
but support for compiled modes is under development.

## Conclusion

In this tutorial, we saw how DebugMode gives you a lightweight, runtime-only
view of what PyTorch actually executed, whether you are running eager code or
compiled graphs. By layering tensor hashing, Triton logging, and custom
dispatch hooks you can quickly track down numerical differences. This is
especially helpful in debugging bit-wise equivalence between runs.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: debug_mode_tutorial.ipynb`](../_downloads/e375168fa13eded0a0693eff61b3e027/debug_mode_tutorial.ipynb)

[`Download Python source code: debug_mode_tutorial.py`](../_downloads/7f7fa85318a98807950265add8370195/debug_mode_tutorial.py)

[`Download zipped: debug_mode_tutorial.zip`](../_downloads/e37ca59ecb52d3916829ab3a6d9881a7/debug_mode_tutorial.zip)