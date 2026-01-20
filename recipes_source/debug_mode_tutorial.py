# -*- coding: utf-8 -*-

"""
DebugMode: Recording Dispatched Operations and Numerical Debugging
=================================================================

**Authors:** Pian Pawakapan, Shangdi Yu

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to capture dispatched ops for eager and ``torch.compile`` runs
       * How to use tensor hashes and stack traces in DebugMode to pinpoint numerical divergence

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch 2.10 or later

"""

######################################################################
# Overview
# --------
#
# ``DebugMode`` (:class:`torch.utils._debug_mode.DebugMode`) is a
# ``TorchDispatchMode`` that intercepts PyTorch runtime calls and emits a
# hierarchical log of operations. It is particularly useful when you need to
# understand *what* actually runs, both in eager mode and under ``torch.compile``
# or when you need to pinpoint numerical divergence between two runs.
#
# Key capabilities:
#
# * **Runtime logging** – Records dispatched operations and TorchInductor compiled
#   Triton kernels.
# * **Tensor hashing** – Attaches deterministic hashes to inputs/outputs to enable
#   diffing runs to locate numerical divergences.
# * **Dispatch hooks** – Allows registration of custom hooks to annotate calls
#
# .. note::
#
#    This recipe describes a prototype feature. Prototype features are typically
#    at an early stage for feedback and testing and are subject to change.
#

######################################################################
# Quick start
# -----------
#
# The snippet below captures a small eager workload and prints the debug string:

from torch._inductor.decomposition import decomps_to_exclude
import torch
from torch.utils._debug_mode import DebugMode

def run_once():
    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    return torch.mm(torch.relu(x), y)

with DebugMode() as debug_mode:
    out = run_once()

print("DebugMode output:")
print(debug_mode.debug_string())


######################################################################
# Getting more metadata
# -----------
#
# For most investigations, you'll want to enable stack traces, tensor IDs, and tensor hashing.
# These features provide metadata to correlate operations back to model code.
#
# ``DebugMode.log_tensor_hashes`` decorates the log with hashes for every call.
# The ``hash_tensor`` hash function uses ``torch.hash_tensor``, which returns 0 for tensors whose
# elements are all the same. The ``norm`` hash function uses ``norm`` with ``p=1``.
# With both these functions, especially ``norm``, tensor closeness in numerics is related to hash closeness,
# so it's rather interpretable. The default ``hash_fn`` is ``norm``.

with (
    DebugMode(
        # record_stack_trace is only supported for eager in pytorch 2.10
        record_stack_trace=True,
        record_ids=True,
    ) as debug_mode,
    DebugMode.log_tensor_hashes(
        hash_fn=["norm"], # this is the default
        hash_inputs=True,
    ),
):
    result = run_once()

print("DebugMode output with more metadata:")
print(
    debug_mode.debug_string(show_stack_trace=True)
)

######################################################################
# Each line follows ``op(args) -> outputs``. When ``record_ids`` is enabled,
# tensors are suffixed with ``$<id>`` and DTensors are labeled ``dt``.


######################################################################
# Log Triton kernels
# ------------------
#
# Though Triton kernels are not dispatched, DebugMode has custom logic that logs their inputs and outputs.
#
# Inductor-generated Triton kernels show up with a ``[triton]`` prefix.
# Pre/post hash annotations report buffer hashes around each kernel call, which
# is helpful when isolating incorrect kernels.
def f(x):
    return torch.mm(torch.relu(x), x.T)

x = torch.randn(3, 3, device="cuda")

with (
    DebugMode(record_output=True) as debug_mode,
    DebugMode.log_tensor_hashes(
        hash_inputs=True,
    )
):
    a = torch.compile(f)(x)

print("Triton in DebugMode logs:")
print(debug_mode.debug_string())

######################################################################
# Numerical debugging with tensor hashes
# --------------------------------------
#
# If you have numerical divergence between modes, you can use DebugMode to find where the
# numerical divergence originates.
# In the example below, you can see that all tensor hashes are the same for eager mode and compiled mode.
# If any hash is different, then that's where the numerical divergence is coming from.

def run_model(model, data, *, compile_with=None):
    if compile_with is not None:
        model = torch.compile(model, backend=compile_with)
    with DebugMode(record_output=True) as dm, DebugMode.log_tensor_hashes(
        hash_inputs=True,
    ):
        dm_out = model(*data)
    return dm, dm_out

class Toy(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x).mm(x.T)

inputs = (torch.randn(4, 4),)
dm_eager, _ = run_model(Toy(), inputs)
dm_compiled, _ = run_model(Toy(), inputs, compile_with="aot_eager")

print("Eager mode:")
print(dm_eager.debug_string())
print("Compiled aot_eager mode:")
print(dm_compiled.debug_string())

###############################################################################################
# Now let's look at an example where the tensor hashes are different.
# I intentionally wrote a wrong decomposition that decomposes cosine to sin.
# This will cause numerical divergence.


from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.debugging import get_nop_func

def wrong_decomp(x):
    return torch.sin(x)

decomp_table = {}
decomp_table[torch.ops.aten.cos.default] = wrong_decomp

backend = aot_autograd(
    fw_compiler=get_nop_func(),
    bw_compiler=get_nop_func(),
    decompositions=decomp_table
)

def f(x):
    y = x.relu()
    z = torch.cos(x)
    return y + z

x = torch.randn(3, 3)
with DebugMode(record_output=True) as dm_eager, DebugMode.log_tensor_hashes(
    hash_inputs=True,
):
    f(x)

with DebugMode(record_output=True) as dm_compiled, DebugMode.log_tensor_hashes(
    hash_inputs=True,
):
    torch.compile(f, backend=backend)(x)

print("Eager:")
print(dm_eager.debug_string(show_stack_trace=True))
print()
print("Compiled with wrong decomposition:")
print(dm_compiled.debug_string())

###############################################################################################
# In the eager log, we have ``aten::cos``, but in the compiled log, we have ``aten::sin``.
# Moreover, the output hash is different between eager and compiled mode.
# Diffing the two logs would show that the first numerical divergence shows up in the ``aten::cos`` call.




######################################################################
# Custom dispatch hooks
# ---------------------
#
# Hooks allow you to annotate each call with custom metadata such as GPU memory usage. ``log_hook`` returns a mapping
# that is rendered inline with the debug string.

MB = 1024 * 1024.0

def memory_hook(func, types, args, kwargs, result):
    mem = torch.cuda.memory_allocated() / MB if torch.cuda.is_available() else 0.0
    peak = torch.cuda.max_memory_allocated() / MB if torch.cuda.is_available() else 0.0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    return {"mem": f"{mem:.3f} MB", "peak": f"{peak:.3f} MB"}

with (
    DebugMode() as dm,
    DebugMode.dispatch_hooks(log_hook=memory_hook),
):
    run_once()

print("DebugMode output with memory usage:")
print(dm.debug_string())

######################################################################
# Module boundaries
# ----------------------------------
#
# ``record_nn_module=True`` inserts ``[nn.Mod]`` markers that show which
# module executed each set of operations. As of PyTorch 2.10 it only works in eager mode,
# but support for compiled modes is under development.

class Foo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(4, 4)
            self.l2 = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.l2(self.l1(x))

class Bar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.abc = Foo()
        self.xyz = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.xyz(self.abc(x))

mod = Bar()
inp = torch.randn(4, 4)
with DebugMode(record_nn_module=True, record_output=False) as debug_mode:
    _ = mod(inp)

print("DebugMode output with stack traces and module boundaries:")
print(debug_mode.debug_string(show_stack_trace=True))

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we saw how DebugMode gives you a lightweight, runtime-only
# view of what PyTorch actually executed, whether you are running eager code or
# compiled graphs. By layering tensor hashing, Triton logging, and custom
# dispatch hooks you can quickly track down numerical differences. This is
# especially helpful in debugging bit-wise equivalence between runs.
