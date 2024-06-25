.. _custom-ops-landing-page:

PyTorch Custom Operators Landing Page
=====================================

PyTorch offers a large library of operators that work on Tensors (e.g. ``torch.add``,
``torch.sum``, etc). However, you may wish to bring a new custom operation to PyTorch
and get it to work with subsystems like ``torch.compile``, autograd, and ``torch.vmap``.
In order to do so, you must register the custom operation with PyTorch via the Python
`torch.library docs <https://pytorch.org/docs/stable/library.html>`_ or C++ ``TORCH_LIBRARY``
APIs.

TL;DR
-----

Authoring a custom operator from Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see :ref:`python-custom-ops-tutorial`.

You may wish to author a custom operator from Python (as opposed to C++) if:

- you have a Python function you want PyTorch to treat as an opaque callable, especially with
  respect to ``torch.compile`` and ``torch.export``.
- you have some Python bindings to C++/CUDA kernels and want those to compose with PyTorch
  subsystems (like ``torch.compile`` or ``torch.autograd``)

Integrating custom C++ and/or CUDA code with PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see :ref:`cpp-custom-ops-tutorial`.

You may wish to author a custom operator from C++ (as opposed to Python) if:

- you have custom C++ and/or CUDA code.
- you plan to use this code with ``AOTInductor`` to do Python-less inference.

The Custom Operators Manual
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For information not covered in the tutorials and this page, please see
`The Custom Operators Manual <https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU>`_
(we're working on moving the information to our docs site). We recommend that you
first read one of the tutorials above and then use the Custom Operators Manual as a reference;
it is not meant to be read head to toe.

When should I create a Custom Operator?
---------------------------------------
If your operation is expressible as a composition of built-in PyTorch operators
then please write it as a Python function and call it instead of creating a
custom operator. Use the operator registration APIs to create a custom operator if you
are calling into some library that PyTorch doesn't understand (e.g. custom C/C++ code,
a custom CUDA kernel, or Python bindings to C/C++/CUDA extensions).

Why should I create a Custom Operator?
--------------------------------------

It is possible to use a C/C++/CUDA kernel by grabbing a Tensor's data pointer
and passing it to a pybind'ed kernel. However, this approach doesn't compose with
PyTorch subsystems like autograd, torch.compile, vmap, and more. In order
for an operation to compose with PyTorch subsystems, it must be registered
via the operator registration APIs.
