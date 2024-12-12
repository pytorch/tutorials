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
#     breaking changes. This tutorial provides a snapshot of ``torch.export`` usage as of PyTorch 2.3.
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
#         f: Callable,
#         args: Tuple[Any, ...],
#         kwargs: Optional[Dict[str, Any]] = None,
#         *,
#         dynamic_shapes: Optional[Dict[str, Dict[int, Dim]]] = None
#     ) -> ExportedProgram
#
# ``torch.export.export()`` traces the tensor computation graph from calling ``f(*args, **kwargs)``
# and wraps it in an ``ExportedProgram``, which can be serialized or executed later with
# different inputs. Note that while the output ``ExportedGraph`` is callable and can be
# called in the same way as the original input callable, it is not a ``torch.nn.Module``.
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
# The FX graph has some important properties:
#
# - The operations are "ATen-level" operations.
# - The graph is "functionalized", meaning that no operations are mutations.
#
# The ``graph_module`` attribute is the ``GraphModule`` that wraps the ``graph`` attribute
# so that it can be ran as a ``torch.nn.Module``.

print(exported_mod)
print(exported_mod.graph_module)

######################################################################
# The printed code shows that FX graph only contains ATen-level ops (such as ``torch.ops.aten``)
# and that mutations were removed. For example, the mutating op ``torch.nn.functional.relu(..., inplace=True)``
# is represented in the printed code by ``torch.ops.aten.relu.default``, which does not mutate.
# Future uses of input to the original mutating ``relu`` op are replaced by the additional new output
# of the replacement non-mutating ``relu`` op.
#
# Other attributes of interest in ``ExportedProgram`` include:
#
# - ``graph_signature`` -- the inputs, outputs, parameters, buffers, etc. of the exported graph.
# - ``range_constraints`` -- constraints, covered later

print(exported_mod.graph_signature)

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
# - unsupported Python language features (e.g. throwing exceptions, match statements)

class Bad4(torch.nn.Module):
    def forward(self, x):
        try:
            x = x + 1
            raise RuntimeError("bad")
        except:
            x = x + 2
        return x

try:
    export(Bad4(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()

######################################################################
# Non-Strict Export
# -----------------
#
# To trace the program, ``torch.export`` uses TorchDynamo, a byte code analysis
# engine, to symbolically analyze the Python code and build a graph based on the
# results. This analysis allows ``torch.export`` to provide stronger guarantees
# about safety, but not all Python code is supported, causing these graph
# breaks.
#
# To address this issue, in PyTorch 2.3, we introduced a new mode of
# exporting called non-strict mode, where we trace through the program using the
# Python interpreter executing it exactly as it would in eager mode, allowing us
# to skip over unsupported Python features. This is done through adding a
# ``strict=False`` flag.
#
# Looking at some of the previous examples which resulted in graph breaks:
#
# - Accessing tensor data with ``.data`` now works correctly

class Bad2(torch.nn.Module):
    def forward(self, x):
        x.data[0, 0] = 3
        return x

bad2_nonstrict = export(Bad2(), (torch.randn(3, 3),), strict=False)
print(bad2_nonstrict.module()(torch.ones(3, 3)))

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
# - Unsupported Python language features (such as throwing exceptions, match
# statements) now also get traced through.

class Bad4(torch.nn.Module):
    def forward(self, x):
        try:
            x = x + 1
            raise RuntimeError("bad")
        except:
            x = x + 2
        return x

bad4_nonstrict = export(Bad4(), (torch.randn(3, 3),), strict=False)
print(bad4_nonstrict.module()(torch.ones(3, 3)))


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

from functorch.experimental.control_flow import cond

class Bad1Fixed(torch.nn.Module):
    def forward(self, x):
        def true_fn(x):
            return torch.sin(x)
        def false_fn(x):
            return torch.cos(x)
        return cond(x.sum() > 0, true_fn, false_fn, [x])

exported_bad1_fixed = export(Bad1Fixed(), (torch.randn(3, 3),))
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
# ..
#     [NOTE] map is not documented at the moment
#     We can also use ``map``, which applies a function across the first dimension
#     of the first tensor argument.
#
#     from functorch.experimental.control_flow import map
#
#     def map_example(xs):
#         def map_fn(x, const):
#             def true_fn(x):
#                 return x + const
#             def false_fn(x):
#                 return x - const
#             return control_flow.cond(x.sum() > 0, true_fn, false_fn, [x])
#         return control_flow.map(map_fn, xs, torch.tensor([2.0]))
#
#     exported_map_example= export(map_example, (torch.randn(4, 3),))
#     inp = torch.cat((torch.ones(2, 3), -torch.ones(2, 3)))
#     print(exported_map_example(inp))

######################################################################
# Constraints/Dynamic Shapes
# --------------------------
#
# This section covers dynamic behavior and representation of exported programs. Dynamic behavior is
# subjective to the particular model being exported, so for the most part of this tutorial, we'll focus
# on this particular toy model (with the sample input shapes annotated):

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
        x0 = x + y  # output shape: [8, 4]
        x1 = self.l(w)  # [6, 3]
        x2 = x0.flatten()  # [32]
        x3 = x2 + z  # [32]
        return x1, x3

######################################################################
# By default, ``torch.export`` produces a static program. One clear consequence of this is that at runtime,
# the program won't work on inputs with different shapes, even if they're valid in eager mode.

w = torch.randn(6, 5)
x = torch.randn(4)
y = torch.randn(8, 4)
z = torch.randn(32)
model = DynamicModel()
ep = export(model, (w, x, y, z))
model(w, x, torch.randn(3, 4), torch.randn(12))
ep.module()(w, x, torch.randn(3, 4), torch.randn(12))

######################################################################
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
# allocated, taking on a range of ``[2, inf]`` (why not ``[0, inf]`` or [1, inf]``? we'll explain later in the
# 0/1 specialization section).
#
# Export then runs model tracing, looking at each operation that's performed by the model. Each individual operation can emit
# what's called "guards"; basically boolean condition that are required to be true for the program to be valid.
# When guards involve symbols allocated for input dimensions, the program contains restrictions on what input shapes are valid;
# i.e. the program's dynamic behavior. The symbolic shapes subsystem is the part responsible for taking in all the emitted guards
# and producing a final program representation that adheres to all of these guards. Before we see this "final representation" in
# an ExportedProgram, let's look at the guards emitted by the toy model we're tracing.
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
        x2 = x0.flatten()
        x3 = x2 + z  # guard: s3 * s4 == s5
        return x1, x3

######################################################################
# Let's understand each of the operations and the emitted guards:
#
# - ``x0 = x + y``: This is an element-wise add with broadcasting, since ``x`` is a 1-d tensor and ``y`` a 2-d tensor.
# ``x`` is broadcasted along the last dimension of ``y``, emitting the guard ``s2 == s4``.
# - ``x1 = self.l(w)``: Calling ``nn.Linear()`` performs a matrix multiplication with model parameters. In export,
# parameters, buffers, and constants are considered program state, which is considered static, and so this is
# a matmul between a dynamic input (``w: [s0, s1]``), and a statically-shaped tensor. This emits the guard ``s1 == 5``.
# - ``x2 = x0.flatten()``: This call actually doesn't emit any guards! (at least none relevant to input shapes)
# - ``x3 = x2 + z``: ``x2`` has shape ``[s3*s4]`` after flattening, and this element-wise add emits ``s3 * s4 == s5``.
#
# Writing all of these guards down and summarizing is almost like a mathematical proof, which is what the symbolic shapes
# subsystem tries to do! In summary, we can conclude that the program must have the following input shapes to be valid:
# 
# ``w: [s0, 5]``
# ``x: [s2]``
# ``y: [s3, s2]``
# ``z: [s2*s3]``
#
# And when we do finally print out the exported program to see our result, those shapes are what we see annotated on the
# corresponding inputs:

print(ep)

######################################################################
# Another feature to notice is the range_constraints field above, which contains a valid range for each symbol. This isn't
# so interesting currently, since this export call doesn't emit any guards related to symbol bounds and each base symbol has
# a generic bound, but this will come up later.
#
# So far, because we've been exporting this toy model, this experience has been misrepresentative of how hard
# it typically is to debug dynamic shapes guards & issues. In most cases it isn't obvious what guards are being emitted,
# and which operations and parts of user code are responsible. For this toy model we pinpoint the exact lines, and the guards
# are rather intuitive.
#
# In more complicated cases, a helpful first step is always to enable verbose logging. This can be done either with the environment
# variable ``TORCH_LOGS="+dynamic"``, or interactively with ``torch._logging.set_logs(dynamic=10)``:

torch._logging.set_logs(dynamic=10)
ep = export(model, (w, x, y, z), dynamic_shapes=dynamic_shapes)

######################################################################
# This spits out quite a handful, even with this simple toy model. But looking through the logs we can see the lines relevant
# to what we described above; e.g. the allocation of symbols:

"""
I1210 16:20:19.720000 3417744 torch/fx/experimental/symbolic_shapes.py:4404] [1/0] create_symbol s0 = 6 for L['w'].size()[0] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s0" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
I1210 16:20:19.722000 3417744 torch/fx/experimental/symbolic_shapes.py:4404] [1/0] create_symbol s1 = 5 for L['w'].size()[1] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s1" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
V1210 16:20:19.722000 3417744 torch/fx/experimental/symbolic_shapes.py:6535] [1/0] runtime_assert True == True [statically known]
I1210 16:20:19.727000 3417744 torch/fx/experimental/symbolic_shapes.py:4404] [1/0] create_symbol s2 = 4 for L['x'].size()[0] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s2" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
I1210 16:20:19.729000 3417744 torch/fx/experimental/symbolic_shapes.py:4404] [1/0] create_symbol s3 = 8 for L['y'].size()[0] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s3" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
I1210 16:20:19.731000 3417744 torch/fx/experimental/symbolic_shapes.py:4404] [1/0] create_symbol s4 = 4 for L['y'].size()[1] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s4" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
I1210 16:20:19.734000 3417744 torch/fx/experimental/symbolic_shapes.py:4404] [1/0] create_symbol s5 = 32 for L['z'].size()[0] [2, int_oo] (_dynamo/variables/builder.py:2841 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s5" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
"""

######################################################################
# Or the guards emitted:

"""
I1210 16:20:19.743000 3417744 torch/fx/experimental/symbolic_shapes.py:6234] [1/0] runtime_assert Eq(s2, s4) [guard added] x0 = x + y  # output shape: [8, 4]  # dynamic_shapes_tutorial.py:16 in forward (_subclasses/fake_impls.py:845 in infer_size), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s2, s4)"
I1210 16:20:19.754000 3417744 torch/fx/experimental/symbolic_shapes.py:6234] [1/0] runtime_assert Eq(s1, 5) [guard added] x1 = self.l(w)  # [6, 3]  # dynamic_shapes_tutorial.py:17 in forward (_meta_registrations.py:2127 in meta_mm), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s1, 5)"
I1210 16:20:19.775000 3417744 torch/fx/experimental/symbolic_shapes.py:6234] [1/0] runtime_assert Eq(s2*s3, s5) [guard added] x3 = x2 + z  # [32]  # dynamic_shapes_tutorial.py:19 in forward (_subclasses/fake_impls.py:845 in infer_size), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s2*s3, s5)"
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
# ways except one: it raises an error when specializing to a constant; designed to maintain dynamism. See for example what happens when a
# static guard is emitted on a dynamically-marked dimension:

dynamic_shapes["w"] = (Dim.AUTO, Dim.DYNAMIC)
export(model, (w, x, y, z), dynamic_shapes=dynamic_shapes)

######################################################################
# Static guards also aren't always inherent to the model; they can also come from user-specifications. In fact, a common pitfall leading to shape
# specializations is when the user specifies conflicting markers for equivalent dimensions; one dynamic and another static. The same error type is
# raised when this is the case for ``x.shape[0]`` and ``y.shape[1]``:

dynamic_shapes["w"] = (Dim.AUTO, Dim.AUTO)
dynamic_shapes["x"] = (Dim.STATIC,)
dynamic_shapes["y"] = (Dim.AUTO, Dim.DYNAMIC)
export(model, (w, x, y, z), dynamic_shapes=dynamic_shapes)

######################################################################
# Here you might ask why export "specializes"; why we resolve this static/dynamic conflict by going with the static route. The answer is because
# of the symbolic shapes system described above, of symbols and guards. When ``x.shape[0]`` is marked static, we don't allocate a symbol, and compile
# treating this shape as a concrete integer 4. A symbol is allocated for ``y.shape[1]``, and so we finally emit the guard ``s3 == 4``, leading to
# specialization.
#
# One feature of export is that during tracing, statements like asserts, ``torch._checks()``, and ``if/else`` conditions will also emit guards.
# See what happens when we augment the existing model with such statements:

class DynamicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(5, 3)

    def forward(self, w, x, y, z):
        assert w.shape[0] <= 512
        torch._check(x.shape[0] >= 16)
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
ep = export(DynamicModel(), (w, x, y, z), dynamic_shapes=dynamic_shapes)
print(ep)

######################################################################
# Each of these statements emits an additional guard, and the exported program shows the changes; ``s0`` is eliminated in favor of ``s2 + 2``,
# and ``s2`` now contains lower and upper bounds, reflected in ``range_constraints``.
#
# For the if/else condition, you might ask why the True branch was taken, and why it wasn't the ``w.shape[0] != x.shape[0] + 2`` guard that
# got emitted from tracing. The answer is that export is guided by the sample inputs provided by tracing, and specializes on the branches taken.
# If different sample input shapes were provided that fail the ``if`` condition, export would trace and emit guards corresponding to the ``else`` branch.
# Additionally, you might ask why we traced only the ``if`` branch, and if it's possible to maintain control-flow in your program and keep both branches
# alive. For that, refer to rewriting your model code following the ``Control Flow Ops`` section above.
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
ep.module()(torch.randn(2, 4))

######################################################################
# So far we've only been talking about 3 ways to specify dynamic shapes: ``Dim.AUTO``, ``Dim.DYNAMIC``, and ``Dim.STATIC``. The attraction of these is the
# low-friction user experience; all the guards emitted during model tracing are adhered to, and dynamic behavior like min/max ranges, relations, and static/dynamic
# dimensions are automatically figured out underneath export. The dynamic shapes subsystem essentially acts as a "discovery" process, summarizing these guards
# and presenting what export believes is the overall dynamic behavior of the program. The drawback of this design appears once the user has stronger expectations or
# beliefs about the dynamic behavior of these models - maybe there is a strong desire on dynamism and specializations on particular dimensions are to be avoided at
# all costs, or maybe we just want to catch changes in dynamic behavior with changes to the original model code, or possibly underlying decompositions or meta-kernels.
# These changes won't be detected and the ``export()`` call will most likely succeed, unless tests are in place that check the resulting ExportedProgram representation.
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
# dynamic behavior of the ExportedProgram produced; ConstraintViolation errors will be raised if model tracing emits guards that conflict with the relations or static/dynamic
# specifications given. For example, in the above specification, the following is asserted:
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
# One common issue with this specification style (before ``Dim.AUTO`` was introduced), is that the specification would often be mismatched with what was produced by model tracing.
# That would lead to ConstraintViolation errors and export suggested fixes - see for example with this model & specification, where the model inherently requires equality between
# dimensions 0 of ``x`` and ``y``, and requires dimension 1 to be static.

class Foo(torch.nn.Module):
    def forward(self, x, y):
        w = x + y
        return w + torch.ones(4)

dx, dy, d1 = torch.export.dims("dx", "dy", "d1")
ep = export(
    Foo(),
    (torch.randn(6, 4), torch.randn(6, 4)),
    dynamic_shapes={
        "x": (dx, d1),
        "y": (dy, d1),
    },
)

######################################################################
# The expectation with suggested fixes is that the user can interactively copy-paste the changes into their dynamic shapes specification, and successfully export afterwards.
#
# Lastly, there's couple nice-to-knows about the options for specification:
# - ``None`` is a good option for static behavior:
#   - ``dynamic_shapes=None`` (default) exports with the entire model being static.
#   - specifying ``None`` at an input-level exports with all tensor dimensions static, and alternatively is also required for non-tensor inputs.
#   - specfiying ``None`` at a dimension-level specializes that dimension, though this is deprecated in favor of ``Dim.STATIC``.
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
# Custom Ops
# ----------
#
# ``torch.export`` can export PyTorch programs with custom operators.
#
# Currently, the steps to register a custom op for use by ``torch.export`` are:
#
# - Define the custom op using ``torch.library`` (`reference <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html>`__)
#   as with any other custom op

@torch.library.custom_op("my_custom_library::custom_op", mutates_args={})
def custom_op(input: torch.Tensor) -> torch.Tensor:
    print("custom_op called!")
    return torch.relu(x)

######################################################################
# - Define a ``"Meta"`` implementation of the custom op that returns an empty
#   tensor with the same shape as the expected output

@custom_op.register_fake 
def custom_op_meta(x):
    return torch.empty_like(x)

######################################################################
# - Call the custom op from the code you want to export using ``torch.ops``

class CustomOpExample(torch.nn.Module):
    def forward(self, x):
        x = torch.sin(x)
        x = torch.ops.my_custom_library.custom_op(x)
        x = torch.cos(x)
        return x

######################################################################
# - Export the code as before

exported_custom_op_example = export(CustomOpExample(), (torch.randn(3, 3),))
exported_custom_op_example.graph_module.print_readable()
print(exported_custom_op_example.module()(torch.randn(3, 3)))

######################################################################
# Note in the above outputs that the custom op is included in the exported graph.
# And when we call the exported graph as a function, the original custom op is called,
# as evidenced by the ``print`` call.
#
# If you have a custom operator implemented in C++, please refer to
# `this document <https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz>`__
# to make it compatible with ``torch.export``.

######################################################################
# Decompositions
# --------------
#
# The graph produced by ``torch.export`` by default returns a graph containing
# only functional ATen operators. This functional ATen operator set (or "opset") contains around 2000
# operators, all of which are functional, that is, they do not
# mutate or alias inputs.  You can find a list of all ATen operators
# `here <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>`__
# and you can inspect if an operator is functional by checking
# ``op._schema.is_mutable``, for example:

print(torch.ops.aten.add.Tensor._schema.is_mutable)
print(torch.ops.aten.add_.Tensor._schema.is_mutable)

######################################################################
# By default, the environment in which you want to run the exported graph
# should support all ~2000 of these operators.
# However, you can use the following API on the exported program
# if your specific environment is only able to support a subset of
# the ~2000 operators.
#
# .. code-block:: python
#
#     def run_decompositions(
#         self: ExportedProgram,
#         decomposition_table: Optional[Dict[torch._ops.OperatorBase, Callable]]
#     ) -> ExportedProgram
#
# ``run_decompositions`` takes in a decomposition table, which is a mapping of
# operators to a function specifying how to reduce, or decompose, that operator
# into an equivalent sequence of other ATen operators.
#
# The default decomposition table for ``run_decompositions`` is the
# `Core ATen decomposition table <https://github.com/pytorch/pytorch/blob/b460c3089367f3fadd40aa2cb3808ee370aa61e1/torch/_decomp/__init__.py#L252>`__
# which will decompose the all ATen operators to the
# `Core ATen Operator Set <https://pytorch.org/docs/main/torch.compiler_ir.html#core-aten-ir>`__
# which consists of only ~180 operators.

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)

ep = export(M(), (torch.randn(2, 3),))
print(ep.graph)

core_ir_ep = ep.run_decompositions()
print(core_ir_ep.graph)

######################################################################
# Notice that after running ``run_decompositions`` the
# ``torch.ops.aten.t.default`` operator, which is not part of the Core ATen
# Opset, has been replaced with ``torch.ops.aten.permute.default`` which is part
# of the Core ATen Opset.
#
# Most ATen operators already have decompositions, which are located
# `here <https://github.com/pytorch/pytorch/blob/b460c3089367f3fadd40aa2cb3808ee370aa61e1/torch/_decomp/decompositions.py>`__.
# If you would like to use some of these existing decomposition functions,
# you can pass in a list of operators you would like to decompose to the
# `get_decompositions <https://github.com/pytorch/pytorch/blob/b460c3089367f3fadd40aa2cb3808ee370aa61e1/torch/_decomp/__init__.py#L191>`__
# function, which will return a decomposition table using existing
# decomposition implementations.

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)

ep = export(M(), (torch.randn(2, 3),))
print(ep.graph)

from torch._decomp import get_decompositions
decomp_table = get_decompositions([torch.ops.aten.t.default, torch.ops.aten.transpose.int])
core_ir_ep = ep.run_decompositions(decomp_table)
print(core_ir_ep.graph)

######################################################################
# If there is no existing decomposition function for an ATen operator that you would
# like to decompose, feel free to send a pull request into PyTorch
# implementing the decomposition!

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
#    import torch._export
#    import torch._inductor
#
#    # Note: these APIs are subject to change
#    # Compile the exported program to a .so using ``AOTInductor``
#    with torch.no_grad():
#    so_path = torch._inductor.aot_compile(ep.module(), [inp])
#
#    # Load and run the .so file in Python.
#    # To load and run it in a C++ environment, see:
#    # https://pytorch.org/docs/main/torch.compiler_aot_inductor.html
#    res = torch._export.aot_load(so_path, device="cuda")(inp)

######################################################################
# Conclusion
# ----------
#
# We introduced ``torch.export``, the new PyTorch 2.X way to export single computation
# graphs from PyTorch programs. In particular, we demonstrate several code modifications
# and considerations (control flow ops, constraints, etc.) that need to be made in order to export a graph.
