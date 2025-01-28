# -*- coding: utf-8 -*-

"""
`Introduction to ONNX <intro_onnx.html>`_ ||
`Exporting a PyTorch model to ONNX <export_simple_model_to_onnx_tutorial.html>`_ ||
**Extending the ONNX Registry**

Extending the ONNX Registry
===========================

**Authors:** Ti-Tai Wang (titaiwang@microsoft.com)
"""


###############################################################################
# Overview
# --------
#
# This tutorial is an introduction to ONNX registry, which empowers users to implement new ONNX operators
# or even replace existing operators with a new implementation.
#
# During the model export to ONNX, the PyTorch model is lowered to an intermediate
# representation composed of `ATen operators <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.
# While ATen operators are maintained by PyTorch core team, it is the responsibility of the ONNX exporter team
# to independently implement each of these operators to ONNX through `ONNX Script <https://onnxscript.ai/>`_.
# The users can also replace the behavior implemented by the ONNX exporter team with their own implementation
# to fix bugs or improve performance for a specific ONNX runtime.
#
# The ONNX Registry manages the mapping between PyTorch operators and the ONNX operators counterparts and provides
# APIs to extend the registry.
#
# In this tutorial, we will cover three scenarios that require extending the ONNX registry with custom operators:
#
# * Custom operators with existing ONNX Runtime support
# * Custom operators without ONNX Runtime support
#

import torch
import onnxruntime
import onnxscript
from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now


######################################################################
# Custom operators with existing ONNX Runtime support
# ---------------------------------------------------
#
# In this case, the user creates a model with standard PyTorch operators, but the ONNX runtime
# (e.g. Microsoft's ONNX Runtime) can provide a custom implementation for that kernel, effectively replacing the
# existing implementation in the ONNX Registry. Another use case is when the user wants to use a custom implementation
# of an existing ONNX operator to fix a bug or improve performance of a specific operator.
# To achieve this, we only need to register the new implementation with the existing ATen fully qualified name.
#
# In the following example, we use the ``com.microsoft.Gelu`` from ONNX Runtime,
# which is not the same ``Gelu`` from ONNX spec. Thus, we register the Gelu with
# the namespace ``com.microsoft`` and operator name ``Gelu``.
#
# Before we begin, let's check whether ``aten::gelu.default`` is really supported by the ONNX registry.

onnx_registry = torch.onnx.OnnxRegistry()
print(f"aten::gelu.default is supported by ONNX registry: \
    {onnx_registry.is_registered_op(namespace='aten', op_name='gelu', overload='default')}")


######################################################################
# In our example, ``aten::gelu.default`` operator is supported by the ONNX registry,
# so :meth:`onnx_registry.is_registered_op` returns ``True``.

class CustomGelu(torch.nn.Module):
    def forward(self, input_x):
        return torch.ops.aten.gelu(input_x)

# com.microsoft is an official ONNX Runtime namspace
custom_ort = onnxscript.values.Opset(domain="com.microsoft", version=1)

# NOTE: The function signature must match the signature of the unsupported ATen operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
@onnxscript.script(custom_ort)
def custom_aten_gelu(input_x, approximate: str = "none"):
    # We know com.microsoft::Gelu is supported by ONNX Runtime
    # It's only not supported by ONNX
    return custom_ort.Gelu(input_x)


onnx_registry = torch.onnx.OnnxRegistry()
onnx_registry.register_op(
    namespace="aten", op_name="gelu", overload="default", function=custom_aten_gelu)
export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)

aten_gelu_model = CustomGelu()
input_gelu_x = torch.randn(3, 3)

onnx_program = torch.onnx.dynamo_export(
    aten_gelu_model, input_gelu_x, export_options=export_options
    )


######################################################################
# Let's inspect the model and verify the model uses op_type ``Gelu``
# from namespace ``com.microsoft``.
#
# .. note::
#     :func:`custom_aten_gelu` does not exist in the graph because
#     functions with fewer than three operators are inlined automatically.
#

# graph node domain is the custom domain we registered
assert onnx_program.model_proto.graph.node[0].domain == "com.microsoft"
# graph node name is the function name
assert onnx_program.model_proto.graph.node[0].op_type == "Gelu"


######################################################################
# The following diagram shows ``custom_aten_gelu_model`` ONNX graph using Netron,
# we can see the ``Gelu`` node from module ``com.microsoft`` used in the function:
#
# .. image:: /_static/img/onnx/custom_aten_gelu_model.png
#
# That is all we need to do. As an additional step, we can use ONNX Runtime to run the model,
# and compare the results with PyTorch.
#

onnx_program.save("./custom_gelu_model.onnx")
ort_session = onnxruntime.InferenceSession(
    "./custom_gelu_model.onnx", providers=['CPUExecutionProvider']
    )

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_input = [input_gelu_x]
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

torch_outputs = aten_gelu_model(input_gelu_x)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

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
def addandround_op(tensor_x: torch.Tensor) -> torch.Tensor:
    ...

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

# NOTE: The function signature must match the signature of the unsupported ATen operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
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
    namespace="mylibrary", op_name="addandround_op", overload="default", function=custom_addandround
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
assert onnx_program.model_proto.graph.node[0].op_type == "CustomOpOne"
assert onnx_program.model_proto.graph.node[1].domain == "test.customop"
assert onnx_program.model_proto.graph.node[1].op_type == "CustomOpTwo"


######################################################################
# This is how ``custom_addandround_model`` ONNX graph looks using Netron. 
# We can see the two custom operators we used in the function (``CustomOpOne``, and ``CustomOpTwo``), 
# and they are from module ``test.customop``:
#
# .. image:: /_static/img/onnx/custom_addandround.png
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
# discovered how to create custom implementations for unsupported or existing ATen operators
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
