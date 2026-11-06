Note

Go to the end
to download the full example code.

[Introduction to ONNX](intro_onnx.html) ||
[Exporting a PyTorch model to ONNX](export_simple_model_to_onnx_tutorial.html) ||
**Extending the ONNX exporter operator support** ||
[Export a model with control flow to ONNX](export_control_flow_model_to_onnx_tutorial.html)

# Extending the ONNX Exporter Operator Support

**Authors:** [Ti-Tai Wang](mailto:titaiwang%40microsoft.com), [Justin Chu](mailto:justinchu%40microsoft.com)

## Overview

This tutorial describes how you can create ONNX implementation for unsupported PyTorch operators
or replace existing implementation with your own.

We will cover three scenarios that require extending the ONNX exporter's operator support:

- Overriding the implementation of an existing PyTorch operator
- Using custom ONNX operators
- Supporting a custom PyTorch operator

What you will learn:

- How to override or add support for PyTorch operators in ONNX.
- How to integrate custom ONNX operators for specialized runtimes.
- How to implement and translate custom PyTorch operators to ONNX.

### Prerequisites

Before starting this tutorial, make sure you have completed the following prerequisites:

- `torch >= 2.6`
- The target PyTorch operator
- Completed the
[ONNX Script tutorial](https://github.com/microsoft/onnxscript/blob/main/docs/tutorial/index.md)
before proceeding
- The implementation of the operator using [ONNX Script](https://github.com/microsoft/onnxscript)

## Overriding the implementation of an existing PyTorch operator

Although the ONNX exporter team does their best efforts to support all PyTorch operators, some of them
might not be supported yet. In this section, we will demonstrate how you can add
unsupported PyTorch operators to the ONNX Registry.

Note

The steps to implement unsupported PyTorch operators are the same as those for replacing the implementation of an existing
PyTorch operator with a custom one.
Because we don't actually have an unsupported PyTorch operator to use in this tutorial, we are going to leverage
this and replace the implementation of `torch.ops.aten.add.Tensor` with a custom implementation the same way we would
if the operator was not implemented by the ONNX exporter.

When a model cannot be exported to ONNX due to an unsupported operator, the ONNX exporter will show an error message
similar to:

```
No decompositions registered for [...]
```

The error message indicates that the unsupported PyTorch operator is `torch.ops.aten.add.Tensor`.
The operator is of type `<class 'torch._ops.OpOverload'>`, and this operator is what we will use as the
target to register our custom implementation.

```
import torch
import onnxscript

# Opset 18 is the standard supported version as of PyTorch 2.6
from onnxscript import opset18 as op

# Create a model that uses the operator torch.ops.aten.add.Tensor
class Model(torch.nn.Module):
 def forward(self, input_x, input_y):
 return torch.ops.aten.add.Tensor(input_x, input_y)

# NOTE: The function signature (including parameter names) must match the signature of the unsupported PyTorch operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# All attributes must be annotated with type hints.
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
```

```
[torch.onnx] Obtain model graph for `Model()` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `Model()` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decompositions...
[torch.onnx] Run decompositions... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
[torch.onnx] Optimize the ONNX graph...
[torch.onnx] Optimize the ONNX graph... ✅
```

Now let's inspect the model and verify the model is using the custom implementation.

```
print(onnx_program.model)
```

```
<
 ir_version=10,
 opset_imports={'': 20},
 producer_name='pytorch',
 producer_version='2.15.0+cu132',
 domain=None,
 model_version=None,
>
graph(
 name=main_graph,
 inputs=(
 %"input_x"<FLOAT,[1]>,
 %"input_y"<FLOAT,[1]>
 ),
 outputs=(
 %"add"<FLOAT,[1]>
 ),
) {
 0 | # node_add
 %"add"<FLOAT,[1]> ⬅️ ::Add(%"input_y", %"input_x")
 return %"add"<FLOAT,[1]>
}
```

The translation is using our custom implementation: In node `node_Add_0`, `input_y` now
comes first, and `input_x` comes second.

We can use ONNX Runtime to run the model and verify the results by calling
the [`torch.onnx.ONNXProgram`](https://docs.pytorch.org/docs/stable/onnx_export.html#torch.onnx.ONNXProgram) directly on the input tensors.

```
result = onnx_program(x, y)[0]
torch.testing.assert_close(result, torch.tensor([3.0]))
```

## Using custom ONNX operators

In this case, we create a model with standard PyTorch operators, but the runtime
(such as Microsoft's ONNX Runtime) can provide a custom implementation for that kernel, effectively replacing the
existing implementation.

In the following example, we use the `com.microsoft.Gelu` operator provided by ONNX Runtime,
which is not the same `Gelu` from ONNX spec.

```
class GeluModel(torch.nn.Module):
 def forward(self, input_x):
 return torch.ops.aten.gelu(input_x)

# Create a namespace for the custom operator using ONNX Script
# ``com.microsoft`` is an official ONNX Runtime namespace
microsoft_op = onnxscript.values.Opset(domain="com.microsoft", version=1)

# NOTE: The function signature (including parameter names) must match the signature of the unsupported PyTorch operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
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
```

```
[torch.onnx] Obtain model graph for `GeluModel()` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `GeluModel()` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decompositions...
[torch.onnx] Run decompositions... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
[torch.onnx] Optimize the ONNX graph...
[torch.onnx] Optimize the ONNX graph... ✅
```

Let's inspect the model and verify the model uses op_type `Gelu`
from namespace `com.microsoft`.

```
print(onnx_program.model)
```

```
<
 ir_version=10,
 opset_imports={'com.microsoft': 1, '': 20},
 producer_name='pytorch',
 producer_version='2.15.0+cu132',
 domain=None,
 model_version=None,
>
graph(
 name=main_graph,
 inputs=(
 %"input_x"<FLOAT,[1]>
 ),
 outputs=(
 %"gelu"<FLOAT,[1]>
 ),
) {
 0 | # n0
 %"gelu"<FLOAT,[1]> ⬅️ com.microsoft::Gelu(%"input_x")
 return %"gelu"<FLOAT,[1]>
}
```

Similar to the previous example, we can use ONNX Runtime to run the model and verify the results.

```
result = onnx_program(x)[0]
torch.testing.assert_close(result, torch.ops.aten.gelu(x))
```

## Supporting a custom PyTorch operator

In this case, the operator is an operator that is user implemented and registered to PyTorch.

In the following example, we would like to use a custom operator
that takes one tensor input, and returns one output. The operator adds
the input to itself, and returns the rounded result.

Firstly, we assume the custom operator is implemented and registered with `torch.library.custom_op()`.
You can refer to [Creating new custom ops in Python](https://pytorch.org/docs/stable/library.html#torch.library.custom_op)
for a detailed guide on how to create custom operators.

```
# Define and use the operator in PyTorch
@torch.library.custom_op("mylibrary::add_and_round_op", mutates_args=())
def add_and_round_op(input: torch.Tensor) -> torch.Tensor:
 return torch.round(input + input)

@add_and_round_op.register_fake
def _add_and_round_op_fake(tensor_x):
 return torch.empty_like(tensor_x)

class AddAndRoundModel(torch.nn.Module):
 def forward(self, input):
 return add_and_round_op(input)

# Implement the custom operator in ONNX using ONNX Script
def onnx_add_and_round(input):
 return op.Round(op.Add(input, input))

onnx_program = torch.onnx.export(
 AddAndRoundModel().eval(),
 (x,),
 dynamo=True,
 custom_translation_table={
 torch.ops.mylibrary.add_and_round_op.default: onnx_add_and_round,
 },
)

# Optimize the ONNX graph to remove redundant nodes
onnx_program.optimize()
print(onnx_program)
```

```
[torch.onnx] Obtain model graph for `AddAndRoundModel()` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `AddAndRoundModel()` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decompositions...
[torch.onnx] Run decompositions... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
[torch.onnx] Optimize the ONNX graph...
[torch.onnx] Optimize the ONNX graph... ✅
ONNXProgram(
 model=
 <
 ir_version=10,
 opset_imports={'': 20},
 producer_name='pytorch',
 producer_version='2.15.0+cu132',
 domain=None,
 model_version=None,
 >
 graph(
 name=main_graph,
 inputs=(
 %"input"<FLOAT,[1]>
 ),
 outputs=(
 %"add_and_round_op"<FLOAT,[1]>
 ),
 ) {
 0 | # node_Add_0
 %"val_0"<FLOAT,[1]> ⬅️ ::Add(%"input", %"input")
 1 | # node_add_and_round_op
 %"add_and_round_op"<FLOAT,[1]> ⬅️ ::Round(%"val_0")
 return %"add_and_round_op"<FLOAT,[1]>
 }

 ,
 exported_program=
 ExportedProgram:
 class GraphModule(torch.nn.Module):
 def forward(self, input: "f32[1]"):
 input_1 = input

 # File: /__w/tutorials/tutorials/pytorch/tutorials/beginner_source/onnx/onnx_registry_tutorial.py:215 in forward, code: return add_and_round_op(input)
 add_and_round_op: "f32[1]" = torch.ops.mylibrary.add_and_round_op.default(input_1); input_1 = None
 return (add_and_round_op,)

 Graph signature:
 # inputs
 input: USER_INPUT

 # outputs
 add_and_round_op: USER_OUTPUT

 Range constraints: {}

)
```

The translation is using our custom implementation to translate the `torch.ops.mylibrary.add_and_round_op.default`
operator in the torch.export.ExportedProgram` to the ONNX operator `Add` and `Round`.

Finally we verify the results.

```
result = onnx_program(x)[0]
torch.testing.assert_close(result, add_and_round_op(x))
```

## Conclusion

Congratulations! In this tutorial, we explored the `custom_translation_table` option and
discovered how to create custom implementations for unsupported or existing PyTorch operators
using ONNX Script.

Finally, we leveraged ONNX Runtime to execute the model and compare the results with PyTorch,
providing us with a comprehensive understanding of handling unsupported
operators in the ONNX ecosystem.

## Further reading

The list below refers to tutorials that ranges from basic examples to advanced scenarios,
not necessarily in the order they are listed.
Feel free to jump directly to specific topics of your interest or
sit tight and have fun going through all of them to learn all there is about the ONNX exporter.

1. [Exporting a PyTorch model to ONNX](export_simple_model_to_onnx_tutorial.html)
2. [Extending the ONNX exporter operator support](onnx_registry_tutorial.html)
3. [Export a model with control flow to ONNX](export_control_flow_model_to_onnx_tutorial.html)

**Total running time of the script:** (0 minutes 2.511 seconds)

[`Download Jupyter notebook: onnx_registry_tutorial.ipynb`](../../_downloads/0bd6b9a8e47e1d64e4d20ef356a6095d/onnx_registry_tutorial.ipynb)

[`Download Python source code: onnx_registry_tutorial.py`](../../_downloads/94896e8c36969aff0b2abe5e3848a487/onnx_registry_tutorial.py)

[`Download zipped: onnx_registry_tutorial.zip`](../../_downloads/2e4c2828ba60bca3ba657692711bc16e/onnx_registry_tutorial.zip)