"""
**Introduction to ONNX** ||
`Exporting a PyTorch model to ONNX <export_simple_model_to_onnx_tutorial.html>`_ ||
`Extending the ONNX exporter operator support <onnx_registry_tutorial.html>`_ ||
`Export a model with control flow to ONNX <export_control_flow_model_to_onnx_tutorial.html>`_

Introduction to ONNX
====================

Authors:
`Ti-Tai Wang <https://github.com/titaiwangms>`_, `Thiago Crepaldi <https://github.com/thiagocrepaldi>`_.

`Open Neural Network eXchange (ONNX) <https://onnx.ai/>`_ is an open standard
format for representing machine learning models. The ``torch.onnx`` module provides APIs to
capture the computation graph from a native PyTorch :class:`torch.nn.Module` model and convert
it into an `ONNX graph <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_.

The exported model can be consumed by any of the many
`runtimes that support ONNX <https://onnx.ai/supported-tools.html#deployModel>`_,
including Microsoft's `ONNX Runtime <https://www.onnxruntime.ai>`_.

.. note::
    Currently, you can choose either through `TorchScript https://pytorch.org/docs/stable/jit.html`_ or
    `ExportedProgram https://pytorch.org/docs/stable/export.html`_ to export the model to ONNX by the
    boolean parameter dynamo in `torch.onnx.export <https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export>`_.
    In this tutorial, we will focus on the ``ExportedProgram`` approach.

When setting ``dynamo=True``, the exporter will use `torch.export <https://pytorch.org/docs/stable/export.html>`_ to capture an ``ExportedProgram``,
before translating the graph into ONNX representations. This approach is the new and recommended way to export models to ONNX.
It works with PyTorch 2.0 features more robustly, has better support for newer ONNX opsets, and consumes less resources
to make exporting larger models possible.

Dependencies
------------

PyTorch 2.5.0 or newer is required.

The ONNX exporter depends on extra Python packages:

  - `ONNX <https://onnx.ai>`_ standard library
  - `ONNX Script <https://onnxscript.ai>`_ library that enables developers to author ONNX operators,
    functions and models using a subset of Python in an expressive, and yet simple fashion
  - `ONNX Runtime <https://onnxruntime.ai>`_ accelerated machine learning library.

They can be installed through `pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

  pip install --upgrade onnx onnxscript onnxruntime

To validate the installation, run the following commands:

.. code-block:: python

  import torch
  print(torch.__version__)

  import onnxscript
  print(onnxscript.__version__)

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