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
2. Unsupported PyTorch operators with existing ONNX RUNTIME support
3. Unsupported PyTorch operators with no ONNX RUNTIME support

"""

# %%
import torch
print(torch.__version__)
torch.manual_seed(191009)  # set the seed for reproducibility

import onnxscript  # pip install onnxscript-preview
print(onnxscript.__version__)

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
# To support unsupported ATen operators, we need two things:
# 1. The unsupported ATen operator namespace, operator name, and the
#    corresponding overload. (e.g. <namespace>::<op_name>.<overload> - aten::add.Tensor)
# 2. The implementation of the operator in onnxscript.
#

# %%
# NOTE: Setup the environment for an unsupported ATen operator
# To observe the error of unsupported ATen operators,
# we can start out by making a simple model that uses an ATen operator
# that is supported by ONNX, but deleted purposely in this tutorial. 
# We’ll use ``aten::add`` operator.

onnx_registry = torch.onnx.OnnxRegistry()
# aten::add.default and aten::add.Tensor is supported by ONNX
print(f"aten::add.default is supported by ONNX registry: {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='default')}")
print(f"aten::add.Tensor is supported by ONNX registry: {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='Tensor')}")

# %%
# NOTE: This is a hack to delete an existing operator from ONNX registry
from torch.onnx._internal.fx import registration
hack_to_delete_aten_add_default = registration.OpName.from_name_parts(namespace="aten", op_name="add")
hack_to_delete_aten_add_tensor = registration.OpName.from_name_parts(namespace="aten", op_name="add", overload="Tensor")
del onnx_registry._registry[hack_to_delete_aten_add_default]
del onnx_registry._registry[hack_to_delete_aten_add_tensor]
print(f"aten::add.default is supported by ONNX registry: {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='default')}")
print(f"aten::add.Tensor is supported by ONNX registry: {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='Tensor')}")

# %%
# Now aten::add is not supported by ONNX, Let's try to export 
# a model with aten::add operator to ONNX to see what happens.

class Model(torch.nn.Module):
    def forward(self, x, y):
        # specifically call out aten::add
        return torch.ops.aten.add(x, y)

x = torch.randn(3, 4)
y = torch.randn(3, 4)
model = Model()
export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)

try:
    export_output = torch.onnx.dynamo_export(model, x, y, export_options=export_options)
except torch.onnx.OnnxExporterError as e:
    print(f"Caught exception: {e}")

# %%
# The model cannot be exported to ONNX because aten::add.Tensor is not supported by ONNX
# The error message can be found through diagnostics, and is as follows:
#    ``RuntimeErrorWithDiagnostic: Unsupported FX nodes: {'call_function': ['aten.add.Tensor']}. ``

# The error message gives us the first thing we need to support unsupported ATen operators:
# the unsupported ATen operator namespace, operator name, and the corresponding overload.
# (e.g. <namespace>::<op_name>.<overload> - aten::add.Tensor)

# %%
# Let's create a onnxscript function to support aten::add.Tensor.
# This can be named anything, and shows later on Netron graph.
custom_aten = onnxscript.values.Opset(domain="custom.aten", version=1)

# NOTE: opset18 is the only version of ONNX operators we are 
# using in torch.onnx.dynamo_export for now.
@onnxscript.script(custom_aten)
def custom_aten_add(x, y):
    return onnxscript.opset18.Add(x, y)



# %%
# Now we have both things we need to support unsupported ATen operators.
# Let's register the custom_aten_add function to ONNX registry, and 
# export the model to ONNX again.

onnx_registry.register_op(namespace="aten", op_name="add", overload="Tensor", function=custom_aten_add)
print(f"aten::add.Tensor is supported by ONNX registry: {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='Tensor')}")
export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)
export_output = torch.onnx.dynamo_export(model, x, y, export_options=export_options)
print(export_output.model_proto)

# %%
# Check the ONNX model with Netron
# ...

# %%
######################################################################
# Unsupported PyTorch operators with existing ONNX RUNTIME support
# -----------------------------------------------------------------
#
# 


