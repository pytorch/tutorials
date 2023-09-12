# -*- coding: utf-8 -*-

"""
torch.export Tutorial
================
**Author:** William Wen
"""

######################################################################
# ``torch.export`` is the PyTorch 2.0 way to export PyTorch models intended
# to be run on high performance environments.
#
# ``torch.export`` is built using the components of ``torch.compile``,
# so it may be helpful to familiarize yourself with ``torch.compile``.
# For an introduction to ``torch.compile``, see the ` ``torch.compile`` tutorial <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__.
#
# This tutorial focuses on using ``torch.export`` to extract
# `ExportedProgram`s (i.e. single-graph representations) from PyTorch programs.
#
# **Contents**
#
# - Exporting a PyTorch model using ``torch.export``
# - Comparison to ``torch.compile``
# - Control Flow Ops
# - Constraints
# - Custom Ops
# - ExportDB
# - Conclusion

######################################################################
# Exporting a PyTorch model using ``torch.export``
# ------------------------------------------------
#
# ``torch.export`` takes in a callable (including ``torch.nn.Module``s),
# a tuple of positional arguments, and optionally (not shown in the example below),
# a dictionary of keyword arguments.

import torch
from torch.export import export

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x, y):
        return torch.nn.functional.relu(self.lin(x + y), inplace=True)

mod = MyModule()
exported_mod = export(mod, (torch.randn(8, 100), torch.randn(8, 100)))
print(type(exported_mod))

######################################################################
# ``torch.export`` returns an ``ExportedProgram``, which is not a ``torch.nn.Module``,
# but can still be ran as a function:

print(exported_mod(torch.randn(8, 100), torch.randn(8, 100)))

######################################################################
# ``ExportedProgram`` has some attributes that are of interest.
# The ``graph`` attribute is an FX graph traced from the function we exported,
# that is, the computation graph of all PyTorch operations.
# The FX graph has some important properties:
# - The operations are "ATen-level" operations.
# - The graph is "functionalized", meaning that no operations are mutatations.
#
# The ``graph_module`` attribute is the ``GraphModule`` that wraps the ``graph`` attribute
# so that it can be ran as a ``torch.nn.Module``.
# We can use ``graph_module``'s ``print_readable` to print a Python code representation
# of ``graph``:

print(exported_mod)
exported_mod.graph_module.print_readable()

######################################################################
# The printed code shows that FX graph only contains ATen-level ops (i.e. ``torch.ops.aten``)
# and that mutations were removed (e.g. the mutating op ``torch.nn.functional.relu(..., inplace=True)``
# is represented in the printed code by ``torch.ops.aten.relu.default``, which does not mutate).

######################################################################
# Other attributes of interest in ``ExportedProgram`` include:
# - ``graph_signature`` -- the inputs, outputs, parameters, buffers, etc. of the exported graph.
# - ``range_constraints`` and ``equality_constraints`` -- Constraints, covered later

print(exported_mod.graph_signature)

######################################################################
# Comparison to ``torch.compile``
# -------------------------------
# Although ``torch.export`` is built on top of the ``torch.compile``
# components, the key limitation of ``torch.export`` is that it does not
# support graph breaks. This is because handling graph breaks involves interpreting
# the unsupported operation with default Python evaluation, which is incompatible
# with the export use case.
#
# A graph break is necessary in cases such as:
# - data-dependent control flow

def bad1(x):
    if x.sum() > 0:
        return torch.sin(x)
    return torch.cos(x)

import traceback as tb
try:
    export(bad1, (torch.randn(3, 3),))
except Exception:
    tb.print_exc()

######################################################################
# - accessing tensor data with ``.data``

def bad2(x):
    x.data[0, 0] = 3
    return x

try:
    export(bad2, (torch.randn(3, 3),))
except Exception:
    tb.print_exc()


######################################################################
# - calling unsupported functions (e.g. many builtins)

def bad3(x):
    x = x + 1
    return x + id(x)

try:
    export(bad3, (torch.randn(3, 3),))
except Exception:
    tb.print_exc()

######################################################################
# - unsupported Python language features (e.g. throwing exceptions, match statements)

def bad4(x):
    try:
        x = x + 1
        raise RuntimeError("bad")
    except:
        x = x + 2
    return x

try:
    export(bad4, (torch.randn(3, 3),))
except Exception:
    tb.print_exc()

######################################################################
# Control Flow Ops
# ----------------
# ``torch.export`` actually does support data-dependent control flow.
# But these need to be expressed using control flow ops. For example,
# we can fix the control flow example above using the ``cond`` op, like so:

from functorch.experimental import control_flow

def bad1_fixed(x):
    def true_fn(x):
        return torch.sin(x)
    def false_fn(x):
        return torch.cos(x)
    return control_flow.cond(x.sum() > 0, true_fn, false_fn, [x])

exported_bad1_fixed = export(bad1_fixed, (torch.randn(3, 3),))
print(exported_bad1_fixed(torch.ones(3, 3)))
print(exported_bad1_fixed(-torch.ones(3, 3)))

######################################################################
# There are some limitations one should be aware of:
# - The predicate (i.e. ``x.sum() > 0``) must result in a boolean or a single-element tensor.
# - The operands (i.e. ``[x]``) must be tensors.
# - The branch function (i.e. ``true_fn`` and ``false_fn``) signature must match with the
# operands and they must both return a single tensor with the same metadata (e.g. dtype, shape, etc.)
# - Branch functions cannot mutate inputs or globals
# - Branch functions cannot access closure variables, except for ``self`` if the function is
# defined in the scope of a method.

######################################################################
# We can also use ``map``, which applies a function across the first dimension
# of the first tensor argument.

from functorch.experimental.control_flow import map

def map_example(xs):
    def map_fn(x, const):
        def true_fn(x):
            return x + const
        def false_fn(x):
            return x - const
        return control_flow.cond(x.sum() > 0, true_fn, false_fn, [x])
    return control_flow.map(map_fn, xs, torch.tensor([2.0]))

exported_map_example= export(map_example, (torch.randn(4, 3),))
inp = torch.cat((torch.ones(2, 3), -torch.ones(2, 3)))
print(exported_map_example(inp))

######################################################################
# Constraints
# -----------
# Ops can have different specializations for different tensor shapes, so
# ``ExportedProgram``s uses constraints on tensor shapes in order to ensure
# correctness with other inputs.
# If we try to run the first ``ExportedProgram`` example with a tensor
# with a different shape, we get an error:

try:
    exported_mod(torch.randn(10, 100), torch.randn(10, 100))
except Exception:
    tb.print_exc()

######################################################################
# By default, ``torch.export`` requires all tensors to have the same shape
# as the example inputs, but we can modify the ``torch.export`` call to
# relax some of these constraints. We use ``torch.export.dynamic_dim`` to
# express shape constraints manually.
#
# We can use ``dynamic_dim`` to remove a dimension's constraints, or to
# manually provide an upper or lower bound. In the example below, our input
# ``inp1`` has an unconstrained first dimension, but the size of the second
# dimension must be in the interval (1, 18].

from torch.export import dynamic_dim

inp1 = torch.randn(10, 10)

def constraints_example1(x):
    x = x[:, 2:]
    return torch.relu(x)

constraints1 = [
    dynamic_dim(inp1, 0),
    3 < dynamic_dim(inp1, 1),
    dynamic_dim(inp1, 1) <= 18,
]

exported_constraints_example1 = export(constraints_example1, (inp1,), constraints=constraints1)

print(exported_constraints_example1(torch.randn(5, 5)))

try:
    exported_constraints_example1(torch.randn(8, 1))
except Exception:
    tb.print_exc()

try:
    exported_constraints_example1(torch.randn(8, 20))
except Exception:
    tb.print_exc()

######################################################################
# We can also use ``dynamic_dim`` to enforce expected equalities between
# dimensions, for example, in matrix multiplication:

inp2 = torch.randn(4, 8)
inp3 = torch.randn(8, 2)

def constraints_example2(x, y):
    return x @ y

constraints2 = [
    dynamic_dim(inp2, 0),
    dynamic_dim(inp2, 1) == dynamic_dim(inp3, 0),
    dynamic_dim(inp3, 1),
]

exported_constraints_example2 = export(constraints_example2, (inp2, inp3), constraints=constraints2)

print(exported_constraints_example2(torch.randn(2, 16), torch.randn(16, 4)))

try:
    exported_constraints_example2(torch.randn(4, 8), torch.randn(4, 2))
except Exception:
    tb.print_exc()

######################################################################
# We can actually use ``torch.export`` to guide us as to which constraints
# are necessary. We can do this by relaxing all constraints and letting ``torch.export``
# error out.

inp4 = torch.randn(8, 16)
inp5 = torch.randn(16, 32)

def constraints_example3(x, y):
    if x.shape[0] <= 16:
        return x @ y[:, :16]
    return y

constraints3 = (
    [dynamic_dim(inp4, i) for i in range(inp4.dim())] +
    [dynamic_dim(inp5, i) for i in range(inp4.dim())]
)

try:
    export(constraints_example3, (inp4, inp5), constraints=constraints3)
except Exception:
    tb.print_exc()

def specify_constraints(x, y):
    return [
        # x:
        dynamic_dim(x, 0),
        dynamic_dim(x, 1),
        dynamic_dim(x, 0) <= 16,

        # y:
        16 < dynamic_dim(y, 1),
        dynamic_dim(y, 0) == dynamic_dim(x, 1),
    ]

constraints3_fixed = specify_constraints(inp4, inp5)
exported_constraints_example3 = export(constraints_example3, (inp4, inp5), constraints=constraints3_fixed)
print(exported_constraints_example3(torch.randn(4, 32), torch.randn(32, 64)))

######################################################################
# Note that in the example above, because we constrained the value of ``x.shape[0]`` in
# ``constraints_example3``, the exported program is sound even though there is a
# raw ``if`` statement.
#
# If you want to see why ``torch.export`` generated these constraints, you can
# re-run the script with the environment variable ``TORCH_LOGS=dynamic,dynamo``,
# or use ``torch._logging.set_logs``.

import logging
torch._logging.set_logs(dynamic=logging.INFO, dynamo=logging.INFO)
exported_constraints_example3 = export(constraints_example3, (inp4, inp5), constraints=constraints3_fixed)
torch._logging.set_logs(dynamic=logging.WARNING, dynamo=logging.WARNING)

######################################################################
# We can view an ``ExportedProgram``'s constraints using the ``range_constraints`` and
# ``equality_constraints`` attributes. The logging above reveals what the symbols ``s0, s1, ...``
# represent.

print(exported_constraints_example3.range_constraints)
print(exported_constraints_example3.equality_constraints)

######################################################################
# We can also constrain on individual values in the source code itself using
# ``constrain_as_value`` and ``constrain_as_size``. ``constrain_as_value`` specifies
# that a given integer value is expected to fall within the provided minimum/maximum bounds (inclusive).
# If a bound is not provided, then it is assumed to be unbounded. ``constrain_as_size``
# is similar to ``constrain_as_value``, except that it should be used on integer values that
# will be used to specify tensor shapes -- in particular, the value must not be 0 or 1 because

from torch.export import constrain_as_size, constrain_as_value

def constraints_example4(x, y):
    b = y.item()
    constrain_as_value(b, 3, 5)
    if b >= 3:
       return x.cos()
    return x.sin()

exported_constraints_example4 = export(constraints_example4, (torch.randn(3, 3), torch.tensor([4])))
print(exported_constraints_example4(torch.randn(3, 3), torch.tensor([5])))
try:
    exported_constraints_example4(torch.randn(3, 3), torch.randn([2]))
except Exception:
    tb.print_exc()

def constraints_example5(x, y):
    b = y.item()
    constrain_as_size(b)
    z = torch.ones(b, 4)
    return x.sum() + z.sum()

exported_constraints_example5 = export(constraints_example5, (torch.randn(2, 2), torch.tensor([4])))
print(exported_constraints_example5(torch.randn(2, 2), torch.tensor([5])))
try:
    exported_constraints_example5(torch.randn(2, 2), torch.randn([1]))
except Exception:
    tb.print_exc()

######################################################################
# Custom Ops
# ----------
# ``torch.export`` can export PyTorch programs with custom operators.
#
# NOTE: the API for registering custom ops is still under active development
# and may change without notice.
#
# Currently, the steps to register a custom op for use by ``torch.export`` are:
# - Define the custom op using ``torch.library`` as with any other custom op

from torch.library import Library, impl

m = Library("my_custom_library", "DEF")

m.define("custom_op(Tensor input) -> Tensor")

@impl(m, "custom_op", "CompositeExplicitAutograd")
def custom_op(x):
    print("custom_op called!")
    return torch.relu(x)

# - Define a ``Meta`` implementation of the custom op that returns an empty
# tensor with the same shape as the expected output

@impl(m, "custom_op", "Meta")
def custom_op_meta(x):
    return torch.empty_like(x)

# - Call the custom op from the code you want to export using ``torch.ops``

def custom_op_example(x):
    x = torch.sin(x)
    x = torch.ops.my_custom_library.custom_op(x)
    x = torch.cos(x)
    return x

# - Export the code as before

exported_custom_op_example = export(custom_op_example, (torch.randn(3, 3),))
exported_custom_op_example.graph_module.print_readable()
print(exported_custom_op_example(torch.randn(3, 3)))

# Note in the above outputs that the custom op is included in the exported graph.
# And when we call the exported graph as a function, the original custom op is called,
# as evidenced by the ``print`` call.

######################################################################
# ExportDB
# --------
# `ExportDB <https://pytorch.org/docs/main/generated/exportdb/index.html>`__ is a resource for ``torch.export``
# developers. It documents types of code supported by ``torch.export`` and provides example code to
# assist developers in modifying existing code to be compatible with ``torch.export``.
# ExportDB is not exhaustive and is intended to only cover the most common confusing use cases.
#
# We can look through ExportDB in order to understand how to modify the below code to make it
# compatible with ``torch.export``.

# TODO actually come up with an example

######################################################################
# Conclusion
# ----------
# We introduced ``torch.export``, the new PyTorch 2.0 way to export single computation
# graphs from PyTorch programs. In particular, we demonstrate several code modifications
# and considerations (control flow ops, constraints, etc.) that need to be made in order to export a graph.
