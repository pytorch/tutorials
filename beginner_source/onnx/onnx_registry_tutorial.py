"""
`Introduction to ONNX <intro_onnx.html>`_ ||
`Exporting a PyTorch model to ONNX <export_simple_model_to_onnx_tutorial.html>`_ ||
**Extending the ONNX Exporter Operator Support**

Extending the ONNX Exporter Operator Support
============================================

**Authors:** Ti-Tai Wang (titaiwang@microsoft.com), Justin Chu (justinchu@microsoft.com)
"""


###############################################################################
# Overview
# --------
#
# This tutorial describes how you can create ONNX implementation for unsupported Torch operators
# or replace existing implementation with your own.
#
# We will cover three scenarios that require extending the ONNX registry with custom operators:
#
# * Unsupported Torch operators
# * Custom operators with existing ONNX Runtime support
# * Custom operators without ONNX Runtime support
#
# Unsupported Torch operators
# --------------------------
#
# Although the ONNX exporter team does their best efforts to support all Torch operators, some of them
# might not be supported yet. In this section, we will demonstrate how you can add
# unsupported Torch operators to the ONNX Registry.
#
# .. note::
#       The steps to implement unsupported Torch operators are the same to replace the implementation of an existing
#       Torch operator with a custom implementation.
#       Because we don't actually have an unsupported Torch operator to use in this tutorial, we are going to leverage
#       this and replace the implementation of ``torch.ops.aten.add.Tensor`` with a custom implementation the same way we would
#       if the operator was not implemented by the ONNX exporter.
#
# When a model cannot be exported to ONNX due to an unsupported operator, the ONNX exporter will show an error message
# similar to:
#
# .. code-block:: python
#
#   No decompositions registered for [...]
#
# The error message indicates that the unsupported Torch operator is ``torch.ops.aten.add.Tensor``.
# The operator is of type ``<class 'torch._ops.OpOverload'>``, and this operator is what we will use as the
# target to register our custom implementation.
#
# To add support for an unsupported Torch operator or to replace the implementation for an existing one, we need:
#
# * The target Torch operator.
# * The implementation of the operator using `ONNX Script <https://github.com/microsoft/onnxscript>`__.
#   ONNX Script is a prerequisite for this tutorial. Please make sure you have read the
#   `ONNX Script tutorial <https://github.com/microsoft/onnxscript/blob/main/docs/tutorial/index.md>`_
#   before proceeding.

import torch
import onnxruntime
import onnxscript

# Opset 18 is the standard supported version as of PyTorch 2.6
from onnxscript import opset18 as op


# Create a model that uses the operator torch.ops.aten.add.Tensor
class Model(torch.nn.Module):
    def forward(self, input_x, input_y):
        return torch.ops.aten.add.Tensor(input_x, input_y)


# NOTE: The function signature (including param names) must match the signature of the unsupported Torch operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/Torch/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
def custom_aten_add(self, other, alpha: float = 1.0):
    if alpha != 1.0:
        alpha = op.CastLike(alpha, other)
        other = op.Mul(other, alpha)
    # To distinguish the custom implementation from the builtin one, we switch the order of the inputs
    return op.Add(other, self)


x = torch.tensor([1.0])
y = torch.tensor([2.0])

# Then we provide the custom implementation to the ONNX exporter as a ``custom_translation_table``.
onnx_program = torch.onnx.export(
    Model().eval(),
    (x, y),
    dynamo=True,
    custom_translation_table={
        torch.ops.aten.add.Tensor: custom_aten_add,
    },
)
# Optimize the ONNX graph to remove redundant nodes
onnx_program.optimize()

######################################################################
# Now let's inspect the model and verify the model is using the custom implementation.

print(onnx_program.model)

# We get
#
# .. code-block:: python
#     <
#         ir_version=10,
#         opset_imports={'pkg.onnxscript.torch_lib.common': 1, '': 18},
#         producer_name='pytorch',
#         producer_version='2.7.0.dev20250124+cu124',
#         domain=None,
#         model_version=None,
#     >
#     graph(
#         name=main_graph,
#         inputs=(
#             %"input_x"<FLOAT,[1]>,
#             %"input_y"<FLOAT,[1]>
#         ),
#         outputs=(
#             %"add"<FLOAT,[1]>
#         ),
#     ) {
#         0 |  # node_Add_0
#             %"add"<FLOAT,[1]> ⬅️ ::Add(%"input_y", %"input_x")
#         return %"add"<FLOAT,[1]>
#     }
#
# The translation is using our custom implementation: In node ``node_Add_0``, ``input_y`` now
# comes first, and ``input_x`` comes second.
#
# We can use ONNX Runtime to run the model and verify the results by calling
# the ONNXProgram directly on the input tensors.

result = onnx_program(x, y)[0]
torch.testing.assert_close(result, torch.tensor([3.0]))


######################################################################
# Custom operators with existing ONNX Runtime support
# ---------------------------------------------------
#
# In this case, the user creates a model with standard PyTorch operators, but the runtime
# (e.g. Microsoft's ONNX Runtime) can provide a custom implementation for that kernel, effectively replacing the
# existing implementation.
#
# In the following example, we use the ``com.microsoft.Gelu`` operator provided by ONNX Runtime,
# which is not the same ``Gelu`` from ONNX spec.

class GeluModel(torch.nn.Module):
    def forward(self, input_x):
        return torch.ops.aten.gelu(input_x)

# Create a namespace for the custom operator using ONNX Script
# com.microsoft is an official ONNX Runtime namespace
microsoft_op = onnxscript.values.Opset(domain="com.microsoft", version=1)

# NOTE: The function signature (including param names) must match the signature of the unsupported Torch operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/Torch/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
# The function must be scripted using the ``@onnxscript.script()`` decorator when
# using operators from custom domains. This may be improved in future versions.
from onnxscript import FLOAT

@onnxscript.script(microsoft_op)
def custom_aten_gelu(self: FLOAT, approximate: str = "none") -> FLOAT:
    return microsoft_op.Gelu(self)

onnx_program = torch.onnx.export(
    GeluModel().eval(),
    (x,),
    dynamo=True,
    custom_translation_table={
        torch.ops.aten.gelu.default: custom_aten_gelu,
    },
)

# Optimize the ONNX graph to remove redundant nodes
onnx_program.optimize()


######################################################################
# Let's inspect the model and verify the model uses op_type ``Gelu``
# from namespace ``com.microsoft``.
#

print(onnx_program.model)

# We get
#
# .. code-block:: python
#     <
#         ir_version=10,
#         opset_imports={'pkg.onnxscript.torch_lib.common': 1, 'com.microsoft': 1, '': 18},
#         producer_name='pytorch',
#         producer_version='2.7.0.dev20250124+cu124',
#         domain=None,
#         model_version=None,
#     >
#     graph(
#         name=main_graph,
#         inputs=(
#             %"input_x"<FLOAT,[1]>
#         ),
#         outputs=(
#             %"gelu"<FLOAT,[1]>
#         ),
#     ) {
#         0 |  # n0
#              %"gelu"<FLOAT,[1]> ⬅️ com.microsoft::Gelu(%"input_x")
#         return %"gelu"<FLOAT,[1]>
#     }


######################################################################
# Similar to the previous example, we can use ONNX Runtime to run the model and verify the results.

result = onnx_program(x)[0]
torch.testing.assert_close(result, torch.ops.aten.gelu(x))


######################################################################
# Custom operators without ONNX Runtime support
# ---------------------------------------------
#
# In this case, the operator is not supported by any ONNX runtime, but we
# would like to use it as custom operator in ONNX graph. Therefore, we need to implement
# the operator in three places:
#
# 1. PyTorch FX graph
# 2. ONNX Registry
# 3. ONNX Runtime
#
# In the following example, we would like to use a custom operator
# that takes one tensor input, and returns one output. The operator adds
# the input to itself, and returns the rounded result.
#
#
# Custom Ops Registration in PyTorch FX Graph (Beta)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Firstly, we need to implement the operator in PyTorch FX graph.
# This can be done by using ``torch._custom_op``.
#

# NOTE: This is a beta feature in PyTorch, and is subject to change.
from torch._custom_op import impl as custom_op


@custom_op.custom_op("mylibrary::addandround_op")
def addandround_op(tensor_x: torch.Tensor) -> torch.Tensor: ...


@addandround_op.impl_abstract()
def addandround_op_impl_abstract(tensor_x):
    return torch.empty_like(tensor_x)


@addandround_op.impl("cpu")
def addandround_op_impl(tensor_x):
    return torch.round(tensor_x + tensor_x)  # add x to itself, and round the result


torch._dynamo.allow_in_graph(addandround_op)


class CustomFoo(torch.nn.Module):
    def forward(self, tensor_x):
        return addandround_op(tensor_x)


input_addandround_x = torch.randn(3)
custom_addandround_model = CustomFoo()


######################################################################
#
# Custom Ops Registration in ONNX Registry
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For the step 2 and 3, we need to implement the operator in ONNX registry.
# In this example, we will implement the operator in ONNX registry
# with the namespace ``test.customop`` and operator name ``CustomOpOne``,
# and ``CustomOpTwo``. These two ops are registered and built in
# `cpu_ops.cc <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/testdata/custom_op_library/cpu/cpu_ops.cc>`__.
#


custom_opset = onnxscript.values.Opset(domain="test.customop", version=1)


# NOTE: The function signature must match the signature of the unsupported Torch operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/Torch/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
@onnxscript.script(custom_opset)
def custom_addandround(input_x):
    # The same as opset18.Add(x, x)
    add_x = custom_opset.CustomOpOne(input_x, input_x)
    # The same as opset18.Round(x, x)
    round_x = custom_opset.CustomOpTwo(add_x)
    # Cast to FLOAT to match the ONNX type
    return opset18.Cast(round_x, to=1)


onnx_registry = torch.onnx.OnnxRegistry()
onnx_registry.register_op(
    namespace="mylibrary",
    op_name="addandround_op",
    overload="default",
    function=custom_addandround,
)

export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)
onnx_program = torch.onnx.dynamo_export(
    custom_addandround_model, input_addandround_x, export_options=export_options
)
onnx_program.save("./custom_addandround_model.onnx")


######################################################################
# The ``onnx_program`` exposes the exported model as protobuf through ``onnx_program.model_proto``.
# The graph has one graph nodes for ``custom_addandround``, and inside ``custom_addandround``,
# there are two function nodes, one for each operator.
#

assert onnx_program.model_proto.graph.node[0].domain == "test.customop"
assert onnx_program.model_proto.graph.node[0].op_type == "custom_addandround"
assert onnx_program.model_proto.functions[0].node[0].domain == "test.customop"
assert onnx_program.model_proto.functions[0].node[0].op_type == "CustomOpOne"
assert onnx_program.model_proto.functions[0].node[1].domain == "test.customop"
assert onnx_program.model_proto.functions[0].node[1].op_type == "CustomOpTwo"


######################################################################
# This is how ``custom_addandround_model`` ONNX graph looks using Netron:
#
# .. image:: /_static/img/onnx/custom_addandround_model.png
#    :width: 70%
#    :align: center
#
# Inside the ``custom_addandround`` function, we can see the two custom operators we
# used in the function (``CustomOpOne``, and ``CustomOpTwo``), and they are from module
# ``test.customop``:
#
# .. image:: /_static/img/onnx/custom_addandround_function.png
#
# Custom Ops Registration in ONNX Runtime
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To link your custom op library to ONNX Runtime, you need to
# compile your C++ code into a shared library and link it to ONNX Runtime.
# Follow the instructions below:
#
# 1. Implement your custom op in C++ by following
#    `ONNX Runtime instructions <`https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/reference/operators/add-custom-op.md>`__.
# 2. Download ONNX Runtime source distribution from
#    `ONNX Runtime releases <https://github.com/microsoft/onnxruntime/releases>`__.
# 3. Compile and link your custom op library to ONNX Runtime, for example:
#
# .. code-block:: bash
#
#    $ gcc -shared -o libcustom_op_library.so custom_op_library.cc -L /path/to/downloaded/ort/lib/ -lonnxruntime -fPIC
#
# 4. Run the model with ONNX Runtime Python API and compare the results with PyTorch.
#
# .. code-block:: python
#
#    ort_session_options = onnxruntime.SessionOptions()
#
#    # NOTE: Link the custom op library to ONNX Runtime and replace the path
#    # with the path to your custom op library
#    ort_session_options.register_custom_ops_library(
#        "/path/to/libcustom_op_library.so"
#    )
#    ort_session = onnxruntime.InferenceSession(
#        "./custom_addandround_model.onnx", providers=['CPUExecutionProvider'], sess_options=ort_session_options)
#
#    def to_numpy(tensor):
#        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
#    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(input_addandround_x)
#    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
#    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
#
#    torch_outputs = custom_addandround_model(input_addandround_x)
#    torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)
#
#    assert len(torch_outputs) == len(onnxruntime_outputs)
#    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
#        torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))
#
# Conclusion
# ----------
#
# Congratulations! In this tutorial, we explored the :class:`ONNXRegistry` API and
# discovered how to create custom implementations for unsupported or existing Torch operators
# using ONNX Script.
# Finally, we leveraged ONNX Runtime to execute the model and compare the results with PyTorch,
# providing us with a comprehensive understanding of handling unsupported
# operators in the ONNX ecosystem.
#
# Further reading
# ---------------
#
# The list below refers to tutorials that ranges from basic examples to advanced scenarios,
# not necessarily in the order they are listed.
# Feel free to jump directly to specific topics of your interest or
# sit tight and have fun going through all of them to learn all there is about the ONNX exporter.
#
# .. include:: /beginner_source/onnx/onnx_toc.txt
#
# .. toctree::
#    :hidden:
#
