Note

Go to the end
to download the full example code.

**Introduction to ONNX** ||
[Exporting a PyTorch model to ONNX](export_simple_model_to_onnx_tutorial.html) ||
[Extending the ONNX exporter operator support](onnx_registry_tutorial.html) ||
[Export a model with control flow to ONNX](export_control_flow_model_to_onnx_tutorial.html)

# Introduction to ONNX

Authors:
[Ti-Tai Wang](https://github.com/titaiwangms), [Thiago Crepaldi](https://github.com/thiagocrepaldi).

[Open Neural Network eXchange (ONNX)](https://onnx.ai/) is an open standard
format for representing machine learning models. The `torch.onnx` module provides APIs to
capture the computation graph from a native PyTorch [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) model and convert
it into an [ONNX graph](https://github.com/onnx/onnx/blob/main/docs/IR.md).

The exported model can be consumed by any of the many
[runtimes that support ONNX](https://onnx.ai/supported-tools.html#deployModel),
including Microsoft's [ONNX Runtime](https://www.onnxruntime.ai).

When setting `dynamo=True`, the exporter will use [torch.export](https://pytorch.org/docs/stable/export.html) to capture an `ExportedProgram`,
before translating the graph into ONNX representations. This approach is the new and recommended way to export models to ONNX.
It works with PyTorch 2.0 features more robustly, has better support for newer ONNX operator sets, and consumes less resources
to make exporting larger models possible.

## Dependencies

PyTorch 2.5.0 or newer is required.

The ONNX exporter depends on extra Python packages:

> - [ONNX](https://onnx.ai) standard library
> - [ONNX Script](https://microsoft.github.io/onnxscript/) library that enables developers to author ONNX operators,
> functions and models using a subset of Python in an expressive, and yet simple fashion
> - [ONNX Runtime](https://onnxruntime.ai) accelerated machine learning library.

They can be installed through [pip](https://pypi.org/project/pip/):

```
pip install --upgrade onnx onnxscript onnxruntime
```

To validate the installation, run the following commands:

```
import torch
print(torch.__version__)

import onnxscript
print(onnxscript.__version__)

import onnxruntime
print(onnxruntime.__version__)
```

Each import must succeed without any errors and the library versions must be printed out.

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

[`Download Jupyter notebook: intro_onnx.ipynb`](../../_downloads/33f8140bedc02273a55c752fe79058e5/intro_onnx.ipynb)

[`Download Python source code: intro_onnx.py`](../../_downloads/ea6986634c1fca7a6c0eaddbfd7f799c/intro_onnx.py)

[`Download zipped: intro_onnx.zip`](../../_downloads/0462d96a7f13c32ea84053ad86b9793c/intro_onnx.zip)