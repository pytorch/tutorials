# -*- coding: utf-8 -*-

"""
torch.export Tutorial
===================================================
**Author:** William Wen, Zhengxu Chen, Angela Yi, Pian Pawakapan
"""

######################################################################
#
# .. warning::
#
#     ``torch.export`` and its related features are in prototype status and are subject to backwards compatibility
#     breaking changes. This tutorial provides a snapshot of ``torch.export`` usage as of PyTorch 2.5.
#
# :func:`torch.export` is the PyTorch 2.X way to export PyTorch models into
# standardized model representations, intended
# to be run on different (i.e. Python-less) environments. The official
# documentation can be found `here <https://pytorch.org/docs/main/export.html>`__.
#
# In this tutorial, you will learn how to use :func:`torch.export` to extract
# ``ExportedProgram``'s (i.e. single-graph representations) from PyTorch programs.
# We also detail some considerations/modifications that you may need
# to make in order to make your model compatible with ``torch.export``.
#
# **Contents**
#
# .. contents::
#     :local:

######################################################################
# Basic Usage
# -----------
#
# ``torch.export`` extracts single-graph representations from PyTorch programs
# by tracing the target function, given example inputs.
# ``torch.export.export()`` is the main entry point for ``torch.export``.
#
# In this tutorial, ``torch.export`` and ``torch.export.export()`` are practically synonymous,
# though ``torch.export`` generally refers to the PyTorch 2.X export process, and ``torch.export.export()``
# generally refers to the actual function call.
#
# The signature of ``torch.export.export()`` is:
#
# .. code-block:: python
#
#     export(
#         mod: torch.nn.Module,
#         args: Tuple[Any, ...],
#         kwargs: Optional[Dict[str, Any]] = None,
#         *,
#         dynamic_shapes: Optional[Dict[str, Dict[int, Dim]]] = None
#     ) -> ExportedProgram
#
# ``torch.export.export()`` traces the tensor computation graph from calling ``mod(*args, **kwargs)``
# and wraps it in an ``ExportedProgram``, which can be serialized or executed later with
# different inputs. To execute the ``ExportedProgram`` we can call ``.module()``
# on it to return a ``torch.nn.Module`` which is callable, just like the
# original program.
# We will detail the ``dynamic_shapes`` argument later in the tutorial.

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
print(exported_mod.module()(torch.randn(8, 100), torch.randn(8, 100)))


######################################################################
# Let's review some attributes of ``ExportedProgram`` that are of interest.
#
# The ``graph`` attribute is an `FX graph <https://pytorch.org/docs/stable/fx.html#torch.fx.Graph>`__
# traced from the function we exported, that is, the computation graph of all PyTorch operations.
# The FX graph is in "ATen IR" meaning that it contains only "ATen-level" operations.
#
# The ``graph_signature`` attribute gives a more detailed description of the
# input and output nodes in the exported graph, describing which ones are
# parameters, buffers, user inputs, or user outputs.
#
# The ``range_constraints`` attributes will be covered later.

print(exported_mod)

######################################################################
# See the ``torch.export`` `documentation <https://pytorch.org/docs/main/export.html#torch.export.export>`__
# for more details.

######################################################################
# Graph Breaks
# ------------
#
# Although ``torch.export`` shares components with ``torch.compile``,
# the key limitation of ``torch.export``, especially when compared to
# ``torch.compile``, is that it does not support graph breaks. This is because
# handling graph breaks involves interpreting the unsupported operation with
# default Python evaluation, which is incompatible with the export use case.
# Therefore, in order to make your model code compatible with ``torch.export``,
# you will need to modify your code to remove graph breaks.
#
# A graph break is necessary in cases such as:
#
# - data-dependent control flow

class Bad1(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return torch.sin(x)
        return torch.cos(x)

import traceback as tb
try:
    export(Bad1(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()

######################################################################
# - accessing tensor data with ``.data``

class Bad2(torch.nn.Module):
    def forward(self, x):
        x.data[0, 0] = 3
        return x

try:
    export(Bad2(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()

######################################################################
# - calling unsupported functions (such as many built-in functions)

class Bad3(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        return x + id(x)

try:
    export(Bad3(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()


######################################################################
# Non-Strict Export
# -----------------
#
# To trace the program, ``torch.export`` uses TorchDynamo by default, a byte
# code analysis engine, to symbolically analyze the Python code and build a
# graph based on the results. This analysis allows ``torch.export`` to provide
# stronger guarantees about safety, but not all Python code is supported,
# causing these graph breaks.
#
# To address this issue, in PyTorch 2.3, we introduced a new mode of
# exporting called non-strict mode, where we trace through the program using the
# Python interpreter executing it exactly as it would in eager mode, allowing us
# to skip over unsupported Python features. This is done through adding a
# ``strict=False`` flag.
#
# Looking at some of the previous examples which resulted in graph breaks:

######################################################################
# - Calling unsupported functions (such as many built-in functions) traces
# through, but in this case, ``id(x)`` gets specialized as a constant integer in
# the graph. This is because ``id(x)`` is not a tensor operation, so the
# operation is not recorded in the graph.

class Bad3(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        return x + id(x)

bad3_nonstrict = export(Bad3(), (torch.randn(3, 3),), strict=False)
print(bad3_nonstrict)
print(bad3_nonstrict.module()(torch.ones(3, 3)))


######################################################################
# However, there are still some features that require rewrites to the original
# module:

######################################################################
# Control Flow Ops
# ----------------
#
# ``torch.export`` actually does support data-dependent control flow.
# But these need to be expressed using control flow ops. For example,
# we can fix the control flow example above using the ``cond`` op, like so:

class Bad1Fixed(torch.nn.Module):
    def forward(self, x):
        def true_fn(x):
            return torch.sin(x)
        def false_fn(x):
            return torch.cos(x)
        return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

exported_bad1_fixed = export(Bad1Fixed(), (torch.randn(3, 3),))
print(exported_bad1_fixed)
print(exported_bad1_fixed.module()(torch.ones(3, 3)))
print(exported_bad1_fixed.module()(-torch.ones(3, 3)))

######################################################################
# There are limitations to ``cond`` that one should be aware of:
#
# - The predicate (i.e. ``x.sum() > 0``) must result in a boolean or a single-element tensor.
# - The operands (i.e. ``[x]``) must be tensors.
# - The branch function (i.e. ``true_fn`` and ``false_fn``) signature must match with the
#   operands and they must both return a single tensor with the same metadata (for example, ``dtype``, ``shape``, etc.).
# - Branch functions cannot mutate input or global variables.
# - Branch functions cannot access closure variables, except for ``self`` if the function is
#   defined in the scope of a method.
#
# For more details about ``cond``, check out the `cond documentation <https://pytorch.org/docs/main/cond.html>`__.

######################################################################
# We can also use ``map``, which applies a function across the first dimension
# of the first tensor argument.

from torch._higher_order_ops.map import map as torch_map

class MapModule(torch.nn.Module):
    def forward(self, xs, y, z):
        def body(x, y, z):
            return x + y + z

        return torch_map(body, xs, y, z)

inps = (torch.ones(6, 4), torch.tensor(5), torch.tensor(4))
exported_map_example = export(MapModule(), inps)
print(exported_map_example)
print(exported_map_example.module()(*inps))

######################################################################
# Other control flow ops include ``while_loop``, ``associative_scan``, and
# ``scan``. For more documentation on each operator, please refer to
# `this page <https://github.com/pytorch/pytorch/tree/main/torch/_higher_order_ops>`__.

######################################################################
# Constraints/Dynamic Shapes
# --------------------------
#
# This section covers dynamic behavior and representation of exported programs. Dynamic behavior is
# subjective to the particular model being exported, so for the most part of this tutorial, we'll focus
# on this particular toy model (with the resulting tensor shapes annotated):

class DynamicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(5, 3)

    def forward(
        self,
        w: torch.Tensor,  # [6, 5]
        x: torch.Tensor,  # [4]
        y: torch.Tensor,  # [8, 4]
        z: torch.Tensor,  # [32]
    ):
        x0 = x + y  # [8, 4]
        x1 = self.l(w)  # [6, 3]
        x2 = x0.flatten()  # [32]
        x3 = x2 + z  # [32]
        return x1, x3

######################################################################
# By default, ``torch.export`` produces a static program. One consequence of this is that at runtime,
# the program won't work on inputs with different shapes, even if they're valid in eager mode.

w = torch.randn(6, 5)
x = torch.randn(4)
y = torch.randn(8, 4)
z = torch.randn(32)
model = DynamicModel()
ep = export(model, (w, x, y, z))
model(w, x, torch.randn(3, 4), torch.randn(12))
try:
    ep.module()(w, x, torch.randn(3, 4), torch.randn(12))
except Exception:
    tb.print_exc()

######################################################################
# Basic concepts: symbols and guards
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To enable dynamism, ``export()`` provides a ``dynamic_shapes`` argument. The easiest way to work with
# dynamic shapes is using ``Dim.AUTO`` and looking at the program that's returned. Dynamic behavior is specified
# at a input dimension-level; for each input we can specify a tuple of values:

from torch.export.dynamic_shapes import Dim

dynamic_shapes = {
    "w": (Dim.AUTO, Dim.AUTO),
    "x": (Dim.AUTO,),
    "y": (Dim.AUTO, Dim.AUTO),
    "z": (Dim.AUTO,),
}
ep = export(model, (w, x, y, z), dynamic_shapes=dynamic_shapes)

######################################################################
# Before we look at the program that's produced, let's understand what specifying ``dynamic_shapes`` entails,
# and how that interacts with export. For every input dimension where a ``Dim`` object is specified, a symbol is
# `allocated <https://pytorch.org/docs/main/export.programming_model.html#basics-of-symbolic-shapes>`_,
# taking on a range of ``[2, inf]`` (why not ``[0, inf]`` or ``[1, inf]``? we'll explain later in the
# 0/1 specialization section).
#
# Export then runs model tracing, looking at each operation that's performed by the model. Each individual operation can emit
# what's called "guards"; basically boolean condition that are required to be true for the program to be valid.
# When guards involve symbols allocated for input dimensions, the program contains restrictions on what input shapes are valid;
# i.e. the program's dynamic behavior. The symbolic shapes subsystem is the part responsible for taking in all the emitted guards
# and producing a final program representation that adheres to all of these guards. Before we see this "final representation" in
# an ``ExportedProgram``, let's look at the guards emitted by the toy model we're tracing.
#
# Here, each forward input tensor is annotated with the symbol allocated at the start of tracing:

class DynamicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(5, 3)

    def forward(
        self,
        w: torch.Tensor,  # [s0, s1]
        x: torch.Tensor,  # [s2]
        y: torch.Tensor,  # [s3, s4]
        z: torch.Tensor,  # [s5]
    ):
        x0 = x + y  # guard: s2 == s4
        x1 = self.l(w)  # guard: s1 == 5
        x2 = x0.flatten()  # no guard added here
        x3 = x2 + z  # guard: s3 * s4 == s5
        return x1, x3

######################################################################
# Let's understand each of the operations and the emitted guards:
#
# - ``x0 = x + y``: This is an element-wise add with broadcasting, since ``x`` is a 1-d tensor and ``y`` a 2-d tensor. ``x`` is broadcasted along the last dimension of ``y``, emitting the guard ``s2 == s4``.
# - ``x1 = self.l(w)``: Calling ``nn.Linear()`` performs a matrix multiplication with model parameters. In export, parameters, buffers, and constants are considered program state, which is considered static, and so this is a matmul between a dynamic input (``w: [s0, s1]``), and a statically-shaped tensor. This emits the guard ``s1 == 5``.
# - ``x2 = x0.flatten()``: This call actually doesn't emit any guards! (at least none relevant to input shapes)
# - ``x3 = x2 + z``: ``x2`` has shape ``[s3*s4]`` after flattening, and this element-wise add emits ``s3 * s4 == s5``.
#
# Writing all of these guards down and summarizing is almost like a mathematical proof, which is what the symbolic shapes
# subsystem tries to do! In summary, we can conclude that the program must have the following input shapes to be valid:
#
# - ``w: [s0, 5]``
# - ``x: [s2]``
# - ``y: [s3, s2]``
# - ``z: [s2*s3]``
#
# And when we do finally print out the exported program to see our result, those shapes are what we see annotated on the
# corresponding inputs:

print(ep)

######################################################################
# Another feature to notice is the range_constraints field above, which contains a valid range for each symbol. This isn't
# so interesting currently, since this export call doesn't emit any guards related to symbol bounds and each base symbol has
# a generic bound, but this will come up later.
#
# So far, because we've been exporting this toy model, this experience has not been representative of how hard
# it typically is to debug dynamic shapes guards & issues. In most cases it isn't obvious what guards are being emitted,
# and which operations and parts of user code are responsible. For this toy model we pinpoint the exact lines, and the guards
# are rather intuitive.
#
# In more complicated cases, a helpful first step is always to enable verbose logging. This can be done either with the environment
# variable ``TORCH_LOGS="+dynamic"``, or interactively with ``torch._logging.set_logs(dynamic=10)``:

torch._logging.set_logs(dynamic=10)
ep = export(model, (w, x, y, z), dynamic_shapes=dynamic_shapes)

######################################################################
# This spits out quite a handful, even with this simple toy model. The log lines here have been cut short at front and end
# to ignore unnecessary info, but looking through the logs we can see the lines relevant to what we described above;
# e.g. the allocation of symbols:

"""
create_symbol s0 = 6 for L['w'].size()[0] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>)
create_symbol s1 = 5 for L['w'].size()[1] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>)
runtime_assert True == True [statically known]
create_symbol s2 = 4 for L['x'].size()[0] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>)
create_symbol s3 = 8 for L['y'].size()[0] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>)
create_symbol s4 = 4 for L['y'].size()[1] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>)
create_symbol s5 = 32 for L['z'].size()[0] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>)
"""

######################################################################
# The lines with `create_symbol` show when a new symbol has been allocated, and the logs also identify the tensor variable names
# and dimensions they've been allocated for. In other lines we can also see the guards emitted:

"""
runtime_assert Eq(s2, s4) [guard added] x0 = x + y  # output shape: [8, 4]  # dynamic_shapes_tutorial.py:16 in forward (_subclasses/fake_impls.py:845 in infer_size), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s2, s4)"
runtime_assert Eq(s1, 5) [guard added] x1 = self.l(w)  # [6, 3]  # dynamic_shapes_tutorial.py:17 in forward (_meta_registrations.py:2127 in meta_mm), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s1, 5)"
runtime_assert Eq(s2*s3, s5) [guard added] x3 = x2 + z  # [32]  # dynamic_shapes_tutorial.py:19 in forward (_subclasses/fake_impls.py:845 in infer_size), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s2*s3, s5)"
"""

######################################################################
# Next to the ``[guard added]`` messages, we also see the responsible user lines of code - luckily here the model is simple enough.
# In many real-world cases it's not so straightforward: high-level torch operations can have complicated fake-kernel implementations
# or operator decompositions that complicate where and what guards are emitted. In such cases the best way to dig deeper and investigate
# is to follow the logs' suggestion, and re-run with environment variable ``TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="..."``, to further
# attribute the guard of interest.
#
# ``Dim.AUTO`` is just one of the available options for interacting with ``dynamic_shapes``; as of writing this 2 other options are available:
# ``Dim.DYNAMIC``, and ``Dim.STATIC``. ``Dim.STATIC`` simply marks a dimension static, while ``Dim.DYNAMIC`` is similar to ``Dim.AUTO`` in all
# ways except one: it raises an error when specializing to a constant; this is designed to maintain dynamism. See for example what happens when a
# static guard is emitted on a dynamically-marked dimension:

dynamic_shapes["w"] = (Dim.AUTO, Dim.DYNAMIC)
try:
    export(model, (w, x, y, z), dynamic_shapes=dynamic_shapes)
except Exception:
    tb.print_exc()

######################################################################
# Static guards also aren't always inherent to the model; they can also come from user specifications. In fact, a common pitfall leading to shape
# specializations is when the user specifies conflicting markers for equivalent dimensions; one dynamic and another static. The same error type is
# raised when this is the case for ``x.shape[0]`` and ``y.shape[1]``:

dynamic_shapes["w"] = (Dim.AUTO, Dim.AUTO)
dynamic_shapes["x"] = (Dim.STATIC,)
dynamic_shapes["y"] = (Dim.AUTO, Dim.DYNAMIC)
try:
    export(model, (w, x, y, z), dynamic_shapes=dynamic_shapes)
except Exception:
    tb.print_exc()

######################################################################
# Here you might ask why export "specializes", i.e. why we resolve this static/dynamic conflict by going with the static route. The answer is because
# of the symbolic shapes system described above, of symbols and guards. When ``x.shape[0]`` is marked static, we don't allocate a symbol, and compile
# treating this shape as a concrete integer 4. A symbol is allocated for ``y.shape[1]``, and so we finally emit the guard ``s3 == 4``, leading to
# specialization.
#
# One feature of export is that during tracing, statements like asserts, ``torch._check()``, and ``if/else`` conditions will also emit guards.
# See what happens when we augment the existing model with such statements:

class DynamicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(5, 3)

    def forward(self, w, x, y, z):
        assert w.shape[0] <= 512
        torch._check(x.shape[0] >= 4)
        if w.shape[0] == x.shape[0] + 2:
            x0 = x + y
            x1 = self.l(w)
            x2 = x0.flatten()
            x3 = x2 + z
            return x1, x3
        else:
            return w

dynamic_shapes = {
    "w": (Dim.AUTO, Dim.AUTO),
    "x": (Dim.AUTO,),
    "y": (Dim.AUTO, Dim.AUTO),
    "z": (Dim.AUTO,),
}
try:
    ep = export(DynamicModel(), (w, x, y, z), dynamic_shapes=dynamic_shapes)
except Exception:
    tb.print_exc()

######################################################################
# Each of these statements emits an additional guard, and the exported program shows the changes; ``s0`` is eliminated in favor of ``s2 + 2``,
# and ``s2`` now contains lower and upper bounds, reflected in ``range_constraints``.
#
# For the if/else condition, you might ask why the True branch was taken, and why it wasn't the ``w.shape[0] != x.shape[0] + 2`` guard that
# got emitted from tracing. The answer is that export is guided by the sample inputs provided by tracing, and specializes on the branches taken.
# If different sample input shapes were provided that fail the ``if`` condition, export would trace and emit guards corresponding to the ``else`` branch.
# Additionally, you might ask why we traced only the ``if`` branch, and if it's possible to maintain control-flow in your program and keep both branches
# alive. For that, refer to rewriting your model code following the ``Control Flow Ops`` section above.

######################################################################
# 0/1 specialization
# ^^^^^^^^^^^^^^^^^^
#
# Since we're talking about guards and specializations, it's a good time to talk about the 0/1 specialization issue we brought up earlier.
# The bottom line is that export will specialize on sample input dimensions with value 0 or 1, because these shapes have trace-time properties that
# don't generalize to other shapes. For example, size 1 tensors can broadcast while other sizes fail; and size 0 ... . This just means that you should
# specify 0/1 sample inputs when you'd like your program to hardcode them, and non-0/1 sample inputs when dynamic behavior is desirable. See what happens
# at runtime when we export this linear layer:

ep = export(
    torch.nn.Linear(4, 3),
    (torch.randn(1, 4),),
    dynamic_shapes={
        "input": (Dim.AUTO, Dim.STATIC),
    },
)
try:
    ep.module()(torch.randn(2, 4))
except Exception:
    tb.print_exc()

######################################################################
# Named Dims
# ^^^^^^^^^^
#
# So far we've only been talking about 3 ways to specify dynamic shapes: ``Dim.AUTO``, ``Dim.DYNAMIC``, and ``Dim.STATIC``. The attraction of these is the
# low-friction user experience; all the guards emitted during model tracing are adhered to, and dynamic behavior like min/max ranges, relations, and static/dynamic
# dimensions are automatically figured out underneath export. The dynamic shapes subsystem essentially acts as a "discovery" process, summarizing these guards
# and presenting what export believes is the overall dynamic behavior of the program. The drawback of this design appears once the user has stronger expectations or
# beliefs about the dynamic behavior of these models - maybe there is a strong desire on dynamism and specializations on particular dimensions are to be avoided at
# all costs, or maybe we just want to catch changes in dynamic behavior with changes to the original model code, or possibly underlying decompositions or meta-kernels.
# These changes won't be detected and the ``export()`` call will most likely succeed, unless tests are in place that check the resulting ``ExportedProgram`` representation.
#
# For such cases, our stance is to recommend the "traditional" way of specifying dynamic shapes, which longer-term users of export might be familiar with: named ``Dims``:

dx = Dim("dx", min=4, max=256)
dh = Dim("dh", max=512)
dynamic_shapes = {
    "x": (dx, None),
    "y": (2 * dx, dh),
}

######################################################################
# This style of dynamic shapes allows the user to specify what symbols are allocated for input dimensions, min/max bounds on those symbols, and places restrictions on the
# dynamic behavior of the ``ExportedProgram`` produced; ``ConstraintViolation`` errors will be raised if model tracing emits guards that conflict with the relations or static/dynamic
# specifications given. For example, in the above specification, the following is asserted:
#
# - ``x.shape[0]`` is to have range ``[4, 256]``, and related to ``y.shape[0]`` by ``y.shape[0] == 2 * x.shape[0]``.
# - ``x.shape[1]`` is static.
# - ``y.shape[1]`` has range ``[2, 512]``, and is unrelated to any other dimension.
#
# In this design, we allow relations between dimensions to be specified with univariate linear expressions: ``A * dim + B`` can be specified for any dimension. This allows users
# to specify more complex constraints like integer divisibility for dynamic dimensions:

dx = Dim("dx", min=4, max=512)
dynamic_shapes = {
    "x": (4 * dx, None)  # x.shape[0] has range [16, 2048], and is divisible by 4.
}

######################################################################
# Constraint violations, suggested fixes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# One common issue with this specification style (before ``Dim.AUTO`` was introduced), is that the specification would often be mismatched with what was produced by model tracing.
# That would lead to ``ConstraintViolation`` errors and export suggested fixes - see for example with this model & specification, where the model inherently requires equality between
# dimensions 0 of ``x`` and ``y``, and requires dimension 1 to be static.

class Foo(torch.nn.Module):
    def forward(self, x, y):
        w = x + y
        return w + torch.ones(4)

dx, dy, d1 = torch.export.dims("dx", "dy", "d1")
try:
    ep = export(
        Foo(),
        (torch.randn(6, 4), torch.randn(6, 4)),
        dynamic_shapes={
            "x": (dx, d1),
            "y": (dy, d1),
        },
    )
except Exception:
    tb.print_exc()

######################################################################
# The expectation with suggested fixes is that the user can interactively copy-paste the changes into their dynamic shapes specification, and successfully export afterwards.
#
# Lastly, there's couple nice-to-knows about the options for specification:
#
# - ``None`` is a good option for static behavior:
#   - ``dynamic_shapes=None`` (default) exports with the entire model being static.
#   - specifying ``None`` at an input-level exports with all tensor dimensions static, and is also required for non-tensor inputs.
#   - specifying ``None`` at a dimension-level specializes that dimension, though this is deprecated in favor of ``Dim.STATIC``.
# - specifying per-dimension integer values also produces static behavior, and will additionally check that the provided sample input matches the specification.
#
# These options are combined in the inputs & dynamic shapes spec below:

inputs = (
    torch.randn(4, 4),
    torch.randn(3, 3),
    16,
    False,
)
dynamic_shapes = {
    "tensor_0": (Dim.AUTO, None),
    "tensor_1": None,
    "int_val": None,
    "bool_val": None,
}

######################################################################
# Data-dependent errors
# ---------------------
#
# While trying to export models, you have may have encountered errors like "Could not guard on data-dependent expression", or Could not extract specialized integer from data-dependent expression".
# These errors exist because ``torch.export()`` compiles programs using FakeTensors, which symbolically represent their real tensor counterparts. While these have equivalent symbolic properties
# (e.g. sizes, strides, dtypes), they diverge in that FakeTensors do not contain any data values. While this avoids unnecessary memory usage and expensive computation, it does mean that export may be
# unable to out-of-the-box compile parts of user code where compilation relies on data values. In short, if the compiler requires a concrete, data-dependent value in order to proceed, it will error out,
# complaining that the value is not available.
#
# Data-dependent values appear in many places, and common sources are calls like ``item()``, ``tolist()``, or ``torch.unbind()`` that extract scalar values from tensors.
# How are these values represented in the exported program? In the `Constraints/Dynamic Shapes <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html#constraints-dynamic-shapes>`_
# section, we talked about allocating symbols to represent dynamic input dimensions.
# The same happens here: we allocate symbols for every data-dependent value that appears in the program. The important distinction is that these are "unbacked" symbols,
# in contrast to the "backed" symbols allocated for input dimensions. The `"backed/unbacked" <https://pytorch.org/docs/main/export.programming_model.html#basics-of-symbolic-shapes>`_
# nomenclature refers to the presence/absence of a "hint" for the symbol: a concrete value backing the symbol, that can inform the compiler on how to proceed.
#
# In the input shape symbol case (backed symbols), these hints are simply the sample input shapes provided, which explains why control-flow branching is determined by the sample input properties.
# For data-dependent values, the symbols are taken from FakeTensor "data" during tracing, and so the compiler doesn't know the actual values (hints) that these symbols would take on.
#
# Let's see how these show up in exported programs:

class Foo(torch.nn.Module):
    def forward(self, x, y):
        a = x.item()
        b = y.tolist()
        return b + [a]

inps = (
    torch.tensor(1),
    torch.tensor([2, 3]),
)
ep = export(Foo(), inps)
print(ep)

######################################################################
# The result is that 3 unbacked symbols (notice they're prefixed with "u", instead of the usual "s" for input shape/backed symbols) are allocated and returned:
# 1 for the ``item()`` call, and 1 for each of the elements of ``y`` with the ``tolist()`` call.
# Note from the range constraints field that these take on ranges of ``[-int_oo, int_oo]``, not the default ``[0, int_oo]`` range allocated to input shape symbols,
# since we have no information on what these values are - they don't represent sizes, so don't necessarily have positive values.

######################################################################
# Guards, torch._check()
# ^^^^^^^^^^^^^^^^^^^^^^
#
# But the case above is easy to export, because the concrete values of these symbols aren't used in any compiler decision-making; all that's relevant is that the return values are unbacked symbols.
# The data-dependent errors highlighted in this section are cases like the following, where `data-dependent guards <https://pytorch.org/docs/main/export.programming_model.html#control-flow-static-vs-dynamic>`_ are encountered:

class Foo(torch.nn.Module):
    def forward(self, x, y):
        a = x.item()
        if a // 2 >= 5:
            return y + 2
        else:
            return y * 5

######################################################################
# Here we actually need the "hint", or the concrete value of ``a`` for the compiler to decide whether to trace ``return y + 2`` or ``return y * 5`` as the output.
# Because we trace with FakeTensors, we don't know what ``a // 2 >= 5`` actually evaluates to, and export errors out with "Could not guard on data-dependent expression ``u0 // 2 >= 5 (unhinted)``".
#
# So how do we export this toy model? Unlike ``torch.compile()``, export requires full graph compilation, and we can't just graph break on this. Here are some basic options:
#
# 1. Manual specialization: we could intervene by selecting the branch to trace, either by removing the control-flow code to contain only the specialized branch, or using ``torch.compiler.is_compiling()`` to guard what's traced at compile-time.
# 2. ``torch.cond()``: we could rewrite the control-flow code to use ``torch.cond()`` so we don't specialize on a branch.
#
# While these options are valid, they have their pitfalls. Option 1 sometimes requires drastic, invasive rewrites of the model code to specialize, and ``torch.cond()`` is not a comprehensive system for handling data-dependent errors.
# As we will see, there are data-dependent errors that do not involve control-flow.
#
# The generally recommended approach is to start with ``torch._check()`` calls. While these give the impression of purely being assert statements, they are in fact a system of informing the compiler on properties of symbols.
# While a ``torch._check()`` call does act as an assertion at runtime, when traced at compile-time, the checked expression is sent to the symbolic shapes subsystem for reasoning, and any symbol properties that follow from the expression being true,
# are stored as symbol properties (provided it's smart enough to infer those properties). So even if unbacked symbols don't have hints, if we're able to communicate properties that are generally true for these symbols via
# ``torch._check()`` calls, we can potentially bypass data-dependent guards without rewriting the offending model code.
#
# For example in the model above, inserting ``torch._check(a >= 10)`` would tell the compiler that ``y + 2`` can always be returned, and ``torch._check(a == 4)`` tells it to return ``y * 5``.
# See what happens when we re-export this model.

class Foo(torch.nn.Module):
    def forward(self, x, y):
        a = x.item()
        torch._check(a >= 10)
        torch._check(a <= 60)
        if a // 2 >= 5:
            return y + 2
        else:
            return y * 5

inps = (
    torch.tensor(32),
    torch.randn(4),
)
ep = export(Foo(), inps)
print(ep)

######################################################################
# Export succeeds, and note from the range constraints field that ``u0`` takes on a range of ``[10, 60]``.
#
# So what information do ``torch._check()`` calls actually communicate? This varies as the symbolic shapes subsystem gets smarter, but at a fundamental level, these are generally true:
#
# 1. Equality with non-data-dependent expressions: ``torch._check()`` calls that communicate equalities like ``u0 == s0 + 4`` or ``u0 == 5``.
# 2. Range refinement: calls that provide lower or upper bounds for symbols, like the above.
# 3. Some basic reasoning around more complicated expressions: inserting ``torch._check(a < 4)`` will typically tell the compiler that ``a >= 4`` is false. Checks on complex expressions like ``torch._check(a ** 2 - 3 * a <= 10)`` will typically get you past identical guards.
#
# As mentioned previously, ``torch._check()`` calls have applicability outside of data-dependent control flow. For example, here's a model where ``torch._check()`` insertion
# prevails while manual specialization & ``torch.cond()`` do not:

class Foo(torch.nn.Module):
    def forward(self, x, y):
        a = x.item()
        return y[a]

inps = (
    torch.tensor(32),
    torch.randn(60),
)
try:
    export(Foo(), inps)
except Exception:
    tb.print_exc()

######################################################################
# Here is a scenario where ``torch._check()`` insertion is required simply to prevent an operation from failing. The export call will fail with
# "Could not guard on data-dependent expression ``-u0 > 60``", implying that the compiler doesn't know if this is a valid indexing operation -
# if the value of ``x`` is out-of-bounds for ``y`` or not. Here, manual specialization is too prohibitive, and ``torch.cond()`` has no place.
# Instead, informing the compiler of ``u0``'s range is sufficient:

class Foo(torch.nn.Module):
    def forward(self, x, y):
        a = x.item()
        torch._check(a >= 0)
        torch._check(a < y.shape[0])
        return y[a]

inps = (
    torch.tensor(32),
    torch.randn(60),
)
ep = export(Foo(), inps)
print(ep)

######################################################################
# Specialized values
# ^^^^^^^^^^^^^^^^^^
#
# Another category of data-dependent error happens when the program attempts to extract a concrete data-dependent integer/float value
# while tracing. This looks something like "Could not extract specialized integer from data-dependent expression", and is analogous to
# the previous class of errors - if these occur when attempting to evaluate concrete integer/float values, data-dependent guard errors arise
# with evaluating concrete boolean values.
#
# This error typically occurs when there is an explicit or implicit ``int()`` cast on a data-dependent expression. For example, this list comprehension
# has a `range()` call that implicitly does an ``int()`` cast on the size of the list:

class Foo(torch.nn.Module):
    def forward(self, x, y):
        a = x.item()
        b = torch.cat([y for y in range(a)], dim=0)
        return b + int(a)

inps = (
    torch.tensor(32),
    torch.randn(60),
)
try:
    export(Foo(), inps, strict=False)
except Exception:
    tb.print_exc()

######################################################################
# For these errors, some basic options you have are:
#
# 1. Avoid unnecessary ``int()`` cast calls, in this case the ``int(a)`` in the return statement.
# 2. Use ``torch._check()`` calls; unfortunately all you may be able to do in this case is specialize (with ``torch._check(a == 60)``).
# 3. Rewrite the offending code at a higher level. For example, the list comprehension is semantically a ``repeat()`` op, which doesn't involve an ``int()`` cast. The following rewrite avoids data-dependent errors:

class Foo(torch.nn.Module):
    def forward(self, x, y):
        a = x.item()
        b = y.unsqueeze(0).repeat(a, 1)
        return b + a

inps = (
    torch.tensor(32),
    torch.randn(60),
)
ep = export(Foo(), inps, strict=False)
print(ep)

######################################################################
# Data-dependent errors can be much more involved, and there are many more options in your toolkit to deal with them: ``torch._check_is_size()``, ``guard_size_oblivious()``, or real-tensor tracing, as starters.
# For more in-depth guides, please refer to the `Export Programming Model <https://pytorch.org/docs/main/export.programming_model.html>`_,
# or `Dealing with GuardOnDataDependentSymNode errors <https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs>`_.

######################################################################
# Custom Ops
# ----------
#
# ``torch.export`` can export PyTorch programs with custom operators. Please
# refer to `this page <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html>`__
# on how to author a custom operator in either C++ or Python.
#
# The following is an example of registering a custom operator in python to be
# used by ``torch.export``. The important thing to note is that the custom op
# must have a `FakeTensor kernel <https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.xvrg7clz290>`__.

@torch.library.custom_op("my_custom_library::custom_op", mutates_args={})
def custom_op(x: torch.Tensor) -> torch.Tensor:
    print("custom_op called!")
    return torch.relu(x)

@custom_op.register_fake
def custom_op_meta(x):
    # Returns an empty tensor with the same shape as the expected output
    return torch.empty_like(x)

######################################################################
# Here is an example of exporting a program with the custom op.

class CustomOpExample(torch.nn.Module):
    def forward(self, x):
        x = torch.sin(x)
        x = torch.ops.my_custom_library.custom_op(x)
        x = torch.cos(x)
        return x

exported_custom_op_example = export(CustomOpExample(), (torch.randn(3, 3),))
print(exported_custom_op_example)
print(exported_custom_op_example.module()(torch.randn(3, 3)))

######################################################################
# Note that in the ``ExportedProgram``, the custom operator is included in the graph.

######################################################################
# IR/Decompositions
# -----------------
#
# The graph produced by ``torch.export`` returns a graph containing only
# `ATen operators <https://pytorch.org/cppdocs/#aten>`__, which are the
# basic unit of computation in PyTorch. As there are over 3000 ATen operators,
# export provides a way to narrow down the operator set used in the graph based
# on certain characteristics, creating different IRs.
#
# By default, export produces the most generic IR which contains all ATen
# operators, including both functional and non-functional operators. A functional
# operator is one that does not contain any mutations or aliasing of the inputs.
# You can find a list of all ATen operators
# `here <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>`__
# and you can inspect if an operator is functional by checking
# ``op._schema.is_mutable``, for example:

print(torch.ops.aten.add.Tensor._schema.is_mutable)
print(torch.ops.aten.add_.Tensor._schema.is_mutable)

######################################################################
# This generic IR can be used to train in eager PyTorch Autograd. This IR can be
# more explicitly reached through the API ``torch.export.export_for_training``,
# which was introduced in PyTorch 2.5, but calling ``torch.export.export``
# should produce the same graph as of PyTorch 2.6.

class DecompExample(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export_for_training(DecompExample(), (torch.randn(1, 1, 3, 3),))
print(ep_for_training.graph)

######################################################################
# We can then lower this exported program to an operator set which only contains
# functional ATen operators through the API ``run_decompositions``, which
# decomposes the ATen operators into the ones specified in the decomposition
# table, and functionalizes the graph. By specifying an empty set, we're only
# performing functionalization, and does not do any additional decompositions.
# This results in an IR which contains ~2000 operators (instead of the 3000
# operators above), and is ideal for inference cases.

ep_for_inference = ep_for_training.run_decompositions(decomp_table={})
print(ep_for_inference.graph)

######################################################################
# As we can see, the previously mutable operator,
# ``torch.ops.aten.add_.default`` has now been replaced with
# ``torch.ops.aten.add.default``, a l operator.

######################################################################
# We can also further lower this exported program to an operator set which only
# contains the
# `Core ATen Operator Set <https://pytorch.org/docs/main/torch.compiler_ir.html#core-aten-ir>`__,
# which is a collection of only ~180 operators. This IR is optimal for backends
# who do not want to reimplement all ATen operators.

from torch.export import default_decompositions

core_aten_decomp_table = default_decompositions()
core_aten_ep = ep_for_training.run_decompositions(decomp_table=core_aten_decomp_table)
print(core_aten_ep.graph)

######################################################################
# We now see that ``torch.ops.aten.conv2d.default`` has been decomposed
# into ``torch.ops.aten.convolution.default``. This is because ``convolution``
# is a more "core" operator, as operations like ``conv1d`` and ``conv2d`` can be
# implemented using the same op.

######################################################################
# We can also specify our own decomposition behaviors:

my_decomp_table = torch.export.default_decompositions()

def my_awesome_custom_conv2d_function(x, weight, bias, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1):
    return 2 * torch.ops.aten.convolution(x, weight, bias, stride, padding, dilation, False, [0, 0], groups)

my_decomp_table[torch.ops.aten.conv2d.default] = my_awesome_custom_conv2d_function
my_ep = ep_for_training.run_decompositions(my_decomp_table)
print(my_ep.graph)

######################################################################
# Notice that instead of ``torch.ops.aten.conv2d.default`` being decomposed
# into ``torch.ops.aten.convolution.default``, it is now decomposed into
# ``torch.ops.aten.convolution.default`` and ``torch.ops.aten.mul.Tensor``,
# which matches our custom decomposition rule.

######################################################################
# ExportDB
# --------
#
# ``torch.export`` will only ever export a single computation graph from a PyTorch program. Because of this requirement,
# there will be Python or PyTorch features that are not compatible with ``torch.export``, which will require users to
# rewrite parts of their model code. We have seen examples of this earlier in the tutorial -- for example, rewriting
# if-statements using ``cond``.
#
# `ExportDB <https://pytorch.org/docs/main/generated/exportdb/index.html>`__ is the standard reference that documents
# supported and unsupported Python/PyTorch features for ``torch.export``. It is essentially a list a program samples, each
# of which represents the usage of one particular Python/PyTorch feature and its interaction with ``torch.export``.
# Examples are also tagged by category so that they can be more easily searched.
#
# For example, let's use ExportDB to get a better understanding of how the predicate works in the ``cond`` operator.
# We can look at the example called ``cond_predicate``, which has a ``torch.cond`` tag. The example code looks like:

def cond_predicate(x):
    """
    The conditional statement (aka predicate) passed to ``cond()`` must be one of the following:
    - ``torch.Tensor`` with a single element
    - boolean expression
    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """
    pred = x.dim() > 2 and x.shape[2] > 10
    return cond(pred, lambda x: x.cos(), lambda y: y.sin(), [x])

######################################################################
# More generally, ExportDB can be used as a reference when one of the following occurs:
#
# 1. Before attempting ``torch.export``, you know ahead of time that your model uses some tricky Python/PyTorch features
#    and you want to know if ``torch.export`` covers that feature.
# 2. When attempting ``torch.export``, there is a failure and it's unclear how to work around it.
#
# ExportDB is not exhaustive, but is intended to cover all use cases found in typical PyTorch code. Feel free to reach
# out if there is an important Python/PyTorch feature that should be added to ExportDB or supported by ``torch.export``.

######################################################################
# Running the Exported Program
# ----------------------------
#
# As ``torch.export`` is only a graph capturing mechanism, calling the artifact
# produced by ``torch.export`` eagerly will be equivalent to running the eager
# module. To optimize the execution of the Exported Program, we can pass this
# exported artifact to backends such as Inductor through ``torch.compile``,
# `AOTInductor <https://pytorch.org/docs/main/torch.compiler_aot_inductor.html>`__,
# or `TensorRT <https://pytorch.org/TensorRT/dynamo/dynamo_export.html>`__.

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        x = self.linear(x)
        return x

inp = torch.randn(2, 3, device="cuda")
m = M().to(device="cuda")
ep = torch.export.export(m, (inp,))

# Run it eagerly
res = ep.module()(inp)
print(res)

# Run it with torch.compile
res = torch.compile(ep.module(), backend="inductor")(inp)
print(res)

######################################################################
# .. code-block:: python
#
#    import torch._inductor
#
#    # Note: these APIs are subject to change
#    # Compile the exported program to a PT2 archive using ``AOTInductor``
#    with torch.no_grad():
#        pt2_path = torch._inductor.aoti_compile_and_package(ep)
#
#    # Load and run the .so file in Python.
#    # To load and run it in a C++ environment, see:
#    # https://pytorch.org/docs/main/torch.compiler_aot_inductor.html
#    aoti_compiled = torch._inductor.aoti_load_package(pt2_path)
#    res = aoti_compiled(inp)

######################################################################
# Conclusion
# ----------
#
# We introduced ``torch.export``, the new PyTorch 2.X way to export single computation
# graphs from PyTorch programs. In particular, we demonstrate several code modifications
# and considerations (control flow ops, constraints, etc.) that need to be made in order to export a graph.
