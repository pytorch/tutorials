"""
**Introduction to ONNX** ||
`Exporting a PyTorch model to ONNX <export_simple_model_to_onnx_tutorial.html>`_ ||
`Extending the ONNX Registry <onnx_registry_tutorial.html>`_

Introduction to ONNX
====================

Authors:
`Thiago Crepaldi <https://github.com/thiagocrepaldi>`_,

`Open Neural Network eXchange (ONNX) <https://onnx.ai/>`_ is an open standard
format for representing machine learning models. The ``torch.onnx`` module provides APIs to
capture the computation graph from a native PyTorch :class:`torch.nn.Module` model and convert
it into an `ONNX graph <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_.

The exported model can be consumed by any of the many
`runtimes that support ONNX <https://onnx.ai/supported-tools.html#deployModel>`_,
including Microsoft's `ONNX Runtime <https://www.onnxruntime.ai>`_.

.. note::
    Currently, there are two flavors of ONNX exporter APIs,
    but this tutorial will focus on the ``torch.onnx.dynamo_export``.

The TorchDynamo engine is leveraged to hook into Python's frame evaluation API and dynamically rewrite its
bytecode into an `FX graph <https://pytorch.org/docs/stable/fx.html>`_.
The resulting FX Graph is polished before it is finally translated into an
`ONNX graph <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_.

The main advantage of this approach is that the `FX graph <https://pytorch.org/docs/stable/fx.html>`_ is captured using
bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques.

Dependencies
------------

PyTorch 2.1.0 or newer is required.

The ONNX exporter depends on extra Python packages:

  - `ONNX <https://onnx.ai>`_ standard library
  - `ONNX Script <https://onnxscript.ai>`_ library that enables developers to author ONNX operators,
    functions and models using a subset of Python in an expressive, and yet simple fashion.

They can be installed through `pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

  pip install --upgrade onnx onnxscript

To validate the installation, run the following commands:

.. code-block:: python

  import torch
  print(torch.__version__)

  import onnxscript
  print(onnxscript.__version__)

  from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now

  import onnxruntime
  print(onnxruntime.__version__)

Each `import` must succeed without any errors and the library versions must be printed out.

Further reading
---------------

The list below refers to tutorials that ranges from basic examples to advanced scenarios,
not necessarily in the order they are listed.
Feel free to jump directly to specific topics of your interest or
sit tight and have fun going through all of them to learn all there is about the ONNX exporter.

.. include:: /beginner_source/onnx/onnx_toc.txt

.. toctree::
   :hidden:

"""
