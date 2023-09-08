# %%
"""
Introduction to ONNX Registry
===========================

**Authors:** Ti-Tai Wang (titaiwang@microsoft.com)

This tutorial is an introduction to ONNX registry, which 
empowers us to create our own ONNX registry, granting us 
the capability to address unsupported operators in ONNX.

In this tutorial we will cover the following scenarios:

1. Unsupported ATen operators
2. Unsupported ATen operators with existing ONNX RUNTIME support
3. Unsupported PyTorch operators with no ONNX RUNTIME support

"""

# %%
import torch
print(torch.__version__)
torch.manual_seed(191009)  # set the seed for reproducibility

import onnxscript  # pip install onnxscript-preview
print(onnxscript.__version__)

# NOTE: opset18 is the only version of ONNX operators we are 
# using in torch.onnx.dynamo_export for now.
from onnxscript import opset18

import onnxruntime  # pip install onnxruntime
print(onnxruntime.__version__)

# For demonstration purpose, warnings are ignored in this tutorial
import warnings
warnings.filterwarnings('ignore')

# %%
######################################################################
# Unsupported ATen operators
# ---------------------------------
#
# ATen operators are the operators that are implemented in PyTorch, and
# ONNX exporter team has to manually implement the conversion from ATen
# operators to ONNX operators through onnxscript. Although ONNX exporter
# team has been doing their best to support as many ATen operators as
# possible, there are still some ATen operators that are not supported.
# In this section, we will demonstrate how to address unsupported ATen 
# operators.
#
# If the model cannot be exported to ONNX because aten::add.Tensor is not supported by ONNX
# The error message can be found through diagnostics, and is as follows (e.g. aten::add.Tensor):
#    ``RuntimeErrorWithDiagnostic: Unsupported FX nodes: {'call_function': ['aten.add.Tensor']}. ``
#
# To support unsupported ATen operators, we need two things:
# 1. The unsupported ATen operator namespace, operator name, and the
#    corresponding overload. (e.g. <namespace>::<op_name>.<overload> - aten::add.Tensor),
#    which can be found in the error message.
# 2. The implementation of the operator in onnxscript.

# %%
# NOTE: ``is_registered_op`` is a method in ONNX registry that checks
# whether the operator is supported by ONNX. If the operator is not
# supported, it will return False. Otherwise, it will return True.

onnx_registry = torch.onnx.OnnxRegistry()
# aten::add.default and aten::add.Tensor are supported by ONNX
print(f"aten::add.default is supported by ONNX registry: {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='default')}")
# aten::add.Tensor is the one invoked by torch.ops.aten.add
print(f"aten::add.Tensor is supported by ONNX registry: {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='Tensor')}")

# %%
# In this example, we will pretend aten::add.Tensor is not supported by ONNX
# registry, and we will show how to support it. ONNX registry supports operator
# registration overrided by user. In this case, we will override the registration
# of aten::add.Tensor with our own implementation.

# %%
class Model(torch.nn.Module):
    def forward(self, x, y):
        # specifically call out aten::add
        return torch.ops.aten.add(x, y)

input_add_x = torch.randn(3, 4)
input_add_y = torch.randn(3, 4)
aten_add_model = Model()

# %%
# Let's create a onnxscript function to support aten::add.Tensor.
# This can be named anything, and shows later on Netron graph.
custom_aten = onnxscript.values.Opset(domain="custom.aten", version=1)

# NOTE: The function signature must match the signature of the unsupported ATen operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
@onnxscript.script(custom_aten)
def custom_aten_add(x, y, alpha: float = 1.0):
    alpha = opset18.CastLike(alpha, y)
    y = opset18.Mul(y, alpha)
    return opset18.Add(x, y)



# %%
# Now we have both things we need to support unsupported ATen operators.
# Let's register the custom_aten_add function to ONNX registry, and 
# export the model to ONNX again.

onnx_registry.register_op(namespace="aten", op_name="add", overload="Tensor", function=custom_aten_add)
print(f"aten::add.Tensor is supported by ONNX registry: {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='Tensor')}")
export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)
export_output = torch.onnx.dynamo_export(aten_add_model, input_add_x, input_add_y, export_options=export_options)

# Make sure the model uses custom_aten_add instead of aten::add.Tensor
# The graph has one graph nodes for custom_aten_add, and inside 
# custom_aten_add, there are four function nodes, one for each
# operator, and one for constant attribute.
# graph node domain is the custom domain we registered
assert export_output.model_proto.graph.node[0].domain == "custom.aten"
assert len(export_output.model_proto.graph.node) == 1
# graph node name is the function name
assert export_output.model_proto.graph.node[0].op_type == "custom_aten_add"
# function node domain is empty because we use standard ONNX operators
assert export_output.model_proto.functions[0].node[3].domain == ""
# function node name is the standard ONNX operator name
assert export_output.model_proto.functions[0].node[3].op_type == "Add"

# %%
# TODO: Check the ONNX model with Netron
# ...

# %%
# Now we can use ONNX Runtime to run the model, and compare the results with PyTorch
export_output.save("./custom_add_model.onnx")
ort_session = onnxruntime.InferenceSession("./custom_add_model.onnx", providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_input = export_output.adapt_torch_inputs_to_onnx(input_add_x, input_add_y)
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

# The output can be a single tensor or a list of tensors, depending on the model.
# Let's execute the PyTorch model and use it as benchmark next
torch_outputs = aten_add_model(input_add_x, input_add_y)
torch_outputs = export_output.adapt_torch_outputs_to_onnx(torch_outputs)

# Now we can compare both results
assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

# %%
######################################################################
# Unsupported ATen operators with existing ONNX RUNTIME support
# -----------------------------------------------------------------
#
# 

# %%
class CustomGelu(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.gelu(x)

custom_ort = onnxscript.values.Opset(domain="com.microsoft", version=1)

@onnxscript.script(custom_ort)
def custom_aten_gelu(x, approximate: str = "none"):
    # We know com.microsoft::Gelu is supported by ONNX RUNTIME
    # It's only not supported by ONNX
    return custom_ort.Gelu(x)

# %%
onnx_registry = torch.onnx.OnnxRegistry()
onnx_registry.register_op(namespace="aten", op_name="gelu", overload="default", function=custom_aten_gelu)
export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)

aten_gelu_model = CustomGelu()
input_gelu_x = torch.randn(3, 3)

export_output = torch.onnx.dynamo_export(aten_gelu_model, input_gelu_x, export_options=export_options)

# Make sure the model uses custom_aten_add instead of aten::add.Tensor
# graph node domain is the custom domain we registered
assert export_output.model_proto.graph.node[0].domain == "com.microsoft"
# graph node name is the function name
assert export_output.model_proto.graph.node[0].op_type == "custom_aten_gelu"
# function node domain is the custom domain we registered
assert export_output.model_proto.functions[0].node[0].domain == "com.microsoft"
# function node name is the node name used in the function
assert export_output.model_proto.functions[0].node[0].op_type == "Gelu"


# %%
# Now we can use ONNX Runtime to run the model, and compare the results with PyTorch
export_output.save("./custom_gelu_model.onnx")
ort_session = onnxruntime.InferenceSession("./custom_gelu_model.onnx", providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_input = export_output.adapt_torch_inputs_to_onnx(input_gelu_x)
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

# The output can be a single tensor or a list of tensors, depending on the model.
# Let's execute the PyTorch model and use it as benchmark next
torch_outputs = aten_gelu_model(input_gelu_x)
torch_outputs = export_output.adapt_torch_outputs_to_onnx(torch_outputs)

# Now we can compare both results
assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

# %%
######################################################################
# Unsupported PyTorch operators with no ONNX RUNTIME support
# ----------------------------------------------------------
#
#


# %%
# NOTE: This is a beta feature in PyTorch, and is subject to change.

from torch._custom_op import impl as custom_op

@custom_op.custom_op("mylibrary::foo_op")
def foo_op(x: torch.Tensor) -> torch.Tensor:
    ...

@foo_op.impl_abstract()
def foo_op_impl_abstract(x):
    return torch.empty_like(x)

@foo_op.impl("cpu")
def foo_op_impl(x):
    return torch.round(x + x)

torch._dynamo.allow_in_graph(foo_op)

class CustomFoo(torch.nn.Module):
    def forward(self, x):
        return foo_op(x)

input_foo_x = torch.randn(3)
custom_foo_model = CustomFoo()

# %%
custom_opset = onnxscript.values.Opset(domain="test.customop", version=1)

# Exporter for torch.ops.foo.bar.default.
@onnxscript.script(custom_opset)
def custom_foo(x):
    # The same as opset18.Add(x, x)
    add_x = custom_opset.CustomOpOne(x, x)
    # The same as opset18.Round(x, x)
    round_x = custom_opset.CustomOpTwo(add_x)
    # Cast to FLOAT to match the ONNX type
    return opset18.Cast(round_x, to=1)

# %%
onnx_registry = torch.onnx.OnnxRegistry()
onnx_registry.register_op(namespace="mylibrary", op_name="foo_op", overload="default", function=custom_foo)

export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)
export_output = torch.onnx.dynamo_export(custom_foo_model, input_foo_x, export_options=export_options)

assert export_output.model_proto.graph.node[0].domain == "test.customop"
assert export_output.model_proto.graph.node[0].op_type == "custom_foo"
assert export_output.model_proto.functions[0].node[0].domain == "test.customop"
assert export_output.model_proto.functions[0].node[0].op_type == "CustomOpOne"
assert export_output.model_proto.functions[0].node[1].domain == "test.customop"
assert export_output.model_proto.functions[0].node[1].op_type == "CustomOpTwo"

# %%
# Now we can use ONNX Runtime to run the model, and compare the results with PyTorch
export_output.save("./custom_foo_model.onnx")
ort_session_options = onnxruntime.SessionOptions()
ort_session_options.register_custom_ops_library("/home/titaiwang/onnxruntime/build/Linux/RelWithDebInfo/libcustom_op_library.so")
ort_session = onnxruntime.InferenceSession("./custom_foo_model.onnx", providers=['CPUExecutionProvider'], sess_options=ort_session_options)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_input = export_output.adapt_torch_inputs_to_onnx(input_foo_x)
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

# The output can be a single tensor or a list of tensors, depending on the model.
# Let's execute the PyTorch model and use it as benchmark next
torch_outputs = custom_foo_model(input_foo_x)
torch_outputs = export_output.adapt_torch_outputs_to_onnx(torch_outputs)

# Now we can compare both results
assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))


