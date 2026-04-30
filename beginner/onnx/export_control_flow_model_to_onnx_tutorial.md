Note

Go to the end
to download the full example code.

[Introduction to ONNX](intro_onnx.html) ||
[Exporting a PyTorch model to ONNX](export_simple_model_to_onnx_tutorial.html) ||
[Extending the ONNX exporter operator support](onnx_registry_tutorial.html) ||
**`Export a model with control flow to ONNX**

# Export a model with control flow to ONNX

**Author**: [Xavier Dupré](https://github.com/xadupre)

## Overview

This tutorial demonstrates how to handle control flow logic while exporting
a PyTorch model to ONNX. It highlights the challenges of exporting
conditional statements directly and provides solutions to circumvent them.

Conditional logic cannot be exported into ONNX unless they refactored
to use `torch.cond()`. Let's start with a simple model
implementing a test.

What you will learn:

- How to refactor the model to use `torch.cond()` for exporting.
- How to export a model with control flow logic to ONNX.

### Prerequisites

- `torch >= 2.8`

## Define the Models

Two models are defined:

`ForwardWithControlFlowTest`: A model with a forward method containing an
if-else conditional.

`ModelWithControlFlowTest`: A model that incorporates `ForwardWithControlFlowTest`
as part of a simple MLP. The models are tested with
a random input tensor to confirm they execute as expected.

## Exporting the Model: First Attempt

Exporting this model using torch.export.export fails because the control
flow logic in the forward pass creates a graph break that the exporter cannot
handle. This behavior is expected, as conditional logic not written using
`torch.cond()` is unsupported.

A try-except block is used to capture the expected failure during the export
process. If the export unexpectedly succeeds, an `AssertionError` is raised.

## Suggested Patch: Refactoring with `torch.cond()`

To make the control flow exportable, the tutorial demonstrates replacing the
forward method in `ForwardWithControlFlowTest` with a refactored version that
uses torch.cond`().

Details of the Refactoring:

Two helper functions (identity2 and neg) represent the branches of the conditional logic:
* torch.cond`() is used to specify the condition and the two branches along with the input arguments.
* The updated forward method is then dynamically assigned to the `ForwardWithControlFlowTest` instance within the model. A list of submodules is printed to confirm the replacement.

Let's see what the FX graph looks like.

Let's export again.

## Conclusion

This tutorial demonstrates the challenges of exporting models with conditional
logic to ONNX and presents a practical solution using `torch.cond()`.
While the default exporters may fail or produce imperfect graphs, refactoring the
model's logic ensures compatibility and generates a faithful ONNX representation.

By understanding these techniques, we can overcome common pitfalls when
working with control flow in PyTorch models and ensure smooth integration with ONNX workflows.

## Further reading

The list below refers to tutorials that ranges from basic examples to advanced scenarios,
not necessarily in the order they are listed.
Feel free to jump directly to specific topics of your interest or
sit tight and have fun going through all of them to learn all there is about the ONNX exporter.

1. [Exporting a PyTorch model to ONNX](export_simple_model_to_onnx_tutorial.html)
2. [Extending the ONNX exporter operator support](onnx_registry_tutorial.html)
3. [Export a model with control flow to ONNX](export_control_flow_model_to_onnx_tutorial.html)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: export_control_flow_model_to_onnx_tutorial.ipynb`](../../_downloads/1c5e336a8b06aec51ee1f67538b41344/export_control_flow_model_to_onnx_tutorial.ipynb)

[`Download Python source code: export_control_flow_model_to_onnx_tutorial.py`](../../_downloads/7c1717301d5ef4e2f2b6c974660ec4a9/export_control_flow_model_to_onnx_tutorial.py)

[`Download zipped: export_control_flow_model_to_onnx_tutorial.zip`](../../_downloads/0080a38f22bd92aa2613545074840623/export_control_flow_model_to_onnx_tutorial.zip)