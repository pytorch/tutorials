"""
**Introduction to ONNX**

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

.. Note::
    Currently there are two flavors of ONNX exporter APIs,
    but this tutorial will focus on the ``torch.onnx.dynamo_export``.

TorchDynamo engine is leveraged to hook into Python's frame evaluation API and dynamically rewrite its
bytecode into an  `FX graph <https://pytorch.org/docs/stable/fx.html>`_.
The resulting FX Graph is polished before it is finally translated into an
`ONNX graph <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_.

The main advantage of this approach is that the `FX graph <https://pytorch.org/docs/stable/fx.html>`_ is captured using
bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques.

Dependencies
------------

The ONNX exporter depends on extra Python packages:

  - `ONNX <https://onnx.ai>`_
  - `ONNX Script <https://onnxscript.ai>`_

They can be installed through `pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

  pip install --upgrade onnx onnxscript

.. include:: /beginner_source/basics/onnx_toc.txt

.. toctree::
   :hidden:

"""
