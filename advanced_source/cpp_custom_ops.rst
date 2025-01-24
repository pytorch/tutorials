.. _cpp-custom-ops-tutorial:

Custom C++ and CUDA Operators
=============================

**Author:** `Richard Zou <https://github.com/zou3519>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to integrate custom operators written in C++/CUDA with PyTorch
       * How to test custom operators using ``torch.library.opcheck``

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch 2.4 or later
       * Basic understanding of C++ and CUDA programming

.. note::

  This tutorial will also work on AMD ROCm with no additional modifications.

PyTorch offers a large library of operators that work on Tensors (e.g. torch.add, torch.sum, etc).
However, you may wish to bring a new custom operator to PyTorch. This tutorial demonstrates the
blessed path to authoring a custom operator written in C++/CUDA.

For our tutorial, we’ll demonstrate how to author a fused multiply-add C++
and CUDA operator that composes with PyTorch subsystems. The semantics of
the operation are as follows:

.. code-block:: python

  def mymuladd(a: Tensor, b: Tensor, c: float):
      return a * b + c

You can find the end-to-end working example for this tutorial
`here <https://github.com/pytorch/extension-cpp>`_ .

Setting up the Build System
---------------------------

If you are developing custom C++/CUDA code, it must be compiled.
Note that if you’re interfacing with a Python library that already has bindings
to precompiled C++/CUDA code, you might consider writing a custom Python operator
instead (:ref:`python-custom-ops-tutorial`).

Use `torch.utils.cpp_extension <https://pytorch.org/docs/stable/cpp_extension.html>`_
to compile custom C++/CUDA code for use with PyTorch
C++ extensions may be built either "ahead of time" with setuptools, or "just in time"
via `load_inline <https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline>`_;
we’ll focus on the "ahead of time" flavor.

Using ``cpp_extension`` is as simple as writing the following ``setup.py``:

.. code-block:: python

  from setuptools import setup, Extension
  from torch.utils import cpp_extension

  setup(name="extension_cpp",
        ext_modules=[
            cpp_extension.CppExtension(
              "extension_cpp",
              ["muladd.cpp"],
              # define Py_LIMITED_API with min version 3.9 to expose only the stable
              # limited API subset from Python.h
              extra_compile_args={"cxx": ["-DPy_LIMITED_API=0x03090000"]}, 
              py_limited_api=True)],  # Build 1 wheel across multiple Python versions
        cmdclass={'build_ext': cpp_extension.BuildExtension},
        options={"bdist_wheel": {"py_limited_api": "cp39"}}  # 3.9 is minimum supported Python version
  )

If you need to compile CUDA code (for example, ``.cu`` files), then instead use
`torch.utils.cpp_extension.CUDAExtension <https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension>`_.
Please see `extension-cpp <https://github.com/pytorch/extension-cpp>`_ for an
example for how this is set up.

The above example represents what we refer to as a CPython agnostic wheel, meaning
we are building a single wheel that can be run across multiple CPython versions (similar
to pure Python packages). CPython agnosticism is desirable in minimizing the number of wheels your
custom library needs to support and release. To achieve this, there are three key lines to note.

The first is the specification of ``Py_LIMITED_API`` in ``extra_compile_args`` to the
minimum CPython version you would like to support:

.. code-block:: python
  extra_compile_args={"cxx": ["-DPy_LIMITED_API=0x03090000"]},

Defining the ``Py_LIMITED_API`` flag helps guarantee that the extension is in fact
only using the `CPython Stable Limited API <https://docs.python.org/3/c-api/stable.html>`_,
which is a requirement for the building a CPython agnostic wheel. If this requirement
is not met, it is possible to build a wheel that looks CPython agnostic but will crash,
or worse, be silently incorrect, in another CPython environment. Take care to avoid
using unstable CPython APIs, for example APIs from libtorch_python (in particular
pytorch/python bindings,) and to only use APIs from libtorch (ATen objects, operators
and the dispatcher). We strongly recommend defining the ``Py_LIMITED_API`` flag to
ensure the extension is compliant and safe as a CPython agnostic wheel.

The second and third lines inform setuptools that you intend to build a CPython agnostic
wheel and will influence the naming of the wheel accordingly. It is necessary to specify
``py_limited_api=True`` as an argument to CppExtension/CUDAExtension and also as an option
to the ``"bdist_wheel"`` command with the minimal supported CPython version (in this case,
3.9):

.. code-block:: python
  setup(name="extension_cpp",
        ext_modules=[
            cpp_extension.CppExtension(
              ...,
              py_limited_api=True)],  # Build 1 wheel across multiple Python versions
        ...,
        options={"bdist_wheel": {"py_limited_api": "cp39"}}  # 3.9 is minimum supported Python version
  )

This ``setup`` would build one wheel that could be installed across multiple CPython
versions ``>=3.9``.

If your extension uses CPython APIs outside the stable limited set, then you should build
a wheel per CPython version instead, like so:

.. code-block:: python

  from setuptools import setup, Extension
  from torch.utils import cpp_extension

  setup(name="extension_cpp",
        ext_modules=[
            cpp_extension.CppExtension(
              "extension_cpp",
              ["muladd.cpp"])],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
  )


Defining the custom op and adding backend implementations
---------------------------------------------------------
First, let's write a C++ function that computes ``mymuladd``:

.. code-block:: cpp

   at::Tensor mymuladd_cpu(at::Tensor a, const at::Tensor& b, double c) {
     TORCH_CHECK(a.sizes() == b.sizes());
     TORCH_CHECK(a.dtype() == at::kFloat);
     TORCH_CHECK(b.dtype() == at::kFloat);
     TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
     TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
     at::Tensor a_contig = a.contiguous();
     at::Tensor b_contig = b.contiguous();
     at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
     const float* a_ptr = a_contig.data_ptr<float>();
     const float* b_ptr = b_contig.data_ptr<float>();
     float* result_ptr = result.data_ptr<float>();
     for (int64_t i = 0; i < result.numel(); i++) {
       result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
     }
     return result;
   }

In order to use this from PyTorch’s Python frontend, we need to register it
as a PyTorch operator using the ``TORCH_LIBRARY`` API. This will automatically
bind the operator to Python.

Operator registration is a two step-process:

- **Defining the operator** - This step ensures that PyTorch is aware of the new operator.
- **Registering backend implementations** - In this step, implementations for various
  backends, such as CPU and CUDA, are associated with the operator.

Defining an operator
^^^^^^^^^^^^^^^^^^^^
To define an operator, follow these steps:

1. select a namespace for an operator. We recommend the namespace be the name of your top-level
   project; we’ll use "extension_cpp" in our tutorial.
2. provide a schema string that specifies the input/output types of the operator and if an
   input Tensors will be mutated. We support more types in addition to Tensor and float;
   please see `The Custom Operators Manual <https://pytorch.org/docs/main/notes/custom_operators.html>`_
   for more details.

   * If you are authoring an operator that can mutate its input Tensors, please see here
     (:ref:`mutable-ops`) for how to specify that.

.. code-block:: cpp

  TORCH_LIBRARY(extension_cpp, m) {
     // Note that "float" in the schema corresponds to the C++ double type
     // and the Python float type.
     m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
   }

This makes the operator available from Python via ``torch.ops.extension_cpp.mymuladd``.

Registering backend implementations for an operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use ``TORCH_LIBRARY_IMPL`` to register a backend implementation for the operator.

.. code-block:: cpp

   TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
     m.impl("mymuladd", &mymuladd_cpu);
   }

If you also have a CUDA implementation of ``myaddmul``, you can register it
in a separate ``TORCH_LIBRARY_IMPL`` block:

.. code-block:: cpp

  __global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) result[idx] = a[idx] * b[idx] + c;
  }

  at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    int numel = a_contig.numel();
    muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);
    return result;
  }

  TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
    m.impl("mymuladd", &mymuladd_cuda);
  }

Adding ``torch.compile`` support for an operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add ``torch.compile`` support for an operator, we must add a FakeTensor kernel (also
known as a "meta kernel" or "abstract impl"). FakeTensors are Tensors that have
metadata (such as shape, dtype, device) but no data: the FakeTensor kernel for an
operator specifies how to compute the metadata of output tensors given the metadata of input tensors.
The FakeTensor kernel should return dummy Tensors of your choice with
the correct Tensor metadata (shape/strides/``dtype``/device).

We recommend that this be done from Python via the ``torch.library.register_fake`` API,
though it is possible to do this from C++ as well (see
`The Custom Operators Manual <https://pytorch.org/docs/main/notes/custom_operators.html>`_
for more details).

.. code-block:: python

  # Important: the C++ custom operator definitions should be loaded first
  # before calling ``torch.library`` APIs that add registrations for the
  # C++ custom operator(s). The following import loads our
  # C++ custom operator definitions.
  # Note that if you are striving for Python agnosticism, you should use
  # the ``load_library(...)`` API call instead. See the next section for
  # more details.
  from . import _C

  @torch.library.register_fake("extension_cpp::mymuladd")
  def _(a, b, c):
      torch._check(a.shape == b.shape)
      torch._check(a.dtype == torch.float)
      torch._check(b.dtype == torch.float)
      torch._check(a.device == b.device)
      return torch.empty_like(a)

Setting up hybrid Python/C++ registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this tutorial, we defined a custom operator in C++, added CPU/CUDA
implementations in C++, and added ``FakeTensor`` kernels and backward formulas
in Python. The order in which these registrations are loaded (or imported)
matters (importing in the wrong order will lead to an error).

To use the custom operator with hybrid Python/C++ registrations, we must
first load the C++ library that holds the custom operator definition
and then call the ``torch.library`` registration APIs. This can happen in
three ways:


1. The first way to load the C++ library that holds the custom operator definition
   is to define a dummy Python module for _C. Then, in Python, when you import the
   module with ``import _C``, the ``.so`` files corresponding to the extension will
   be loaded and the ``TORCH_LIBRARY`` and ``TORCH_LIBRARY_IMPL`` static initializers
   will run. One can create a dummy Python module with ``PYBIND11_MODULE`` like below,
   but you will notice that this does not compile with ``Py_LIMITED_API``, because
   ``pybind11`` does not promise to only use the stable limited CPython API! With
   the below code, you sadly cannot build a CPython agnostic wheel for your extension!
   (Foreshadowing: I wonder what the second way is ;) ).

.. code-block:: cpp

  // in, say, not_agnostic/csrc/extension_BAD.cpp
  #include <pybind11/pybind11.h>

  PYBIND11_MODULE("_C", m) {}

.. code-block:: python

  # in, say, extension/__init__.py
  from . import _C

2. In this tutorial, because we value being able to build a single wheel across multiple
   CPython versions, we will replace the unstable ``PYBIND11`` call with stable API calls.
   The below code compiles with ``-DPy_LIMITED_API=0x03090000`` and successfully creates
   a dummy Python module for our ``_C`` extension so that it can be imported from Python.
   See `extension_cpp/__init__.py <https://github.com/pytorch/extension-cpp/blob/38ec45e/extension_cpp/__init__.py>`_
   and `extension_cpp/csrc/muladd.cpp  <https://github.com/pytorch/extension-cpp/blob/38ec45e/extension_cpp/csrc/muladd.cpp>`_
   for more details:

.. code-block:: cpp
  
  #include <Python.h>

  extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
      The import from Python will load the .so consisting of this file
      in this extension, so that the TORCH_LIBRARY static initializers
      below are run. */
    PyObject* PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",   /* name of module */
            NULL,   /* module documentation, may be NULL */
            -1,     /* size of per-interpreter state of the module,
                      or -1 if the module keeps state in global variables. */
            NULL,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
  }

.. code-block:: python

  # in, say, extension/__init__.py
  from . import _C

3. If you want to avoid ``Python.h`` entirely in your C++ custom operator, you may
   use ``torch.ops.load_library("/path/to/library.so")`` in Python to load the ``.so``
   file(s) compiled from the extension. Note that, with this method, there is no ``_C``
   Python module created for the extension so you cannot call ``import _C`` from Python.
   Instead of relying on the import statement to trigger the custom operators to be
   registered, ``torch.ops.load_library("/path/to/library.so")`` will do the trick.
   The challenge then is shifted towards understanding where the ``.so`` files are
   located so that you can load them, which is not always trivial:

.. code-block:: python

  import torch
  from pathlib import Path

  so_files = list(Path(__file__).parent.glob("_C*.so"))
  assert (
      len(so_files) == 1
  ), f"Expected one _C*.so file, found {len(so_files)}"
  torch.ops.load_library(so_files[0])

  from . import ops


Adding training (autograd) support for an operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use ``torch.library.register_autograd`` to add training support for an operator. Prefer
this over directly using Python ``torch.autograd.Function`` or C++ ``torch::autograd::Function``;
you must use those in a very specific way to avoid silent incorrectness (see
`The Custom Operators Manual <https://pytorch.org/docs/main/notes/custom_operators.html>`_
for more details).

.. code-block:: python

  def _backward(ctx, grad):
      a, b = ctx.saved_tensors
      grad_a, grad_b = None, None
      if ctx.needs_input_grad[0]:
          grad_a = grad * b
      if ctx.needs_input_grad[1]:
          grad_b = grad * a
      return grad_a, grad_b, None

  def _setup_context(ctx, inputs, output):
      a, b, c = inputs
      saved_a, saved_b = None, None
      if ctx.needs_input_grad[0]:
          saved_b = b
      if ctx.needs_input_grad[1]:
          saved_a = a
      ctx.save_for_backward(saved_a, saved_b)

  # This code adds training support for the operator. You must provide us
  # the backward formula for the operator and a `setup_context` function
  # to save values to be used in the backward.
  torch.library.register_autograd(
      "extension_cpp::mymuladd", _backward, setup_context=_setup_context)

Note that the backward must be a composition of PyTorch-understood operators.
If you wish to use another custom C++ or CUDA kernel in your backwards pass,
it must be wrapped into a custom operator.

If we had our own custom ``mymul`` kernel, we would need to wrap it into a
custom operator and then call that from the backward:

.. code-block:: cpp

  // New! a mymul_cpu kernel
  at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_CHECK(a.device().type() == at::DeviceType::CPU);
    TORCH_CHECK(b.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
      result_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
  }

  TORCH_LIBRARY(extension_cpp, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    // New! defining the mymul operator
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
  }


  TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
    m.impl("mymuladd", &mymuladd_cpu);
    // New! registering the cpu kernel for the mymul operator
    m.impl("mymul", &mymul_cpu);
  }

.. code-block:: python

  def _backward(ctx, grad):
      a, b = ctx.saved_tensors
      grad_a, grad_b = None, None
      if ctx.needs_input_grad[0]:
          grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
      if ctx.needs_input_grad[1]:
          grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
      return grad_a, grad_b, None


  def _setup_context(ctx, inputs, output):
      a, b, c = inputs
      saved_a, saved_b = None, None
      if ctx.needs_input_grad[0]:
          saved_b = b
      if ctx.needs_input_grad[1]:
          saved_a = a
      ctx.save_for_backward(saved_a, saved_b)


  # This code adds training support for the operator. You must provide us
  # the backward formula for the operator and a `setup_context` function
  # to save values to be used in the backward.
  torch.library.register_autograd(
      "extension_cpp::mymuladd", _backward, setup_context=_setup_context)

Testing an operator
-------------------
Use ``torch.library.opcheck`` to test that the custom op was registered correctly.
Note that this function does not test that the gradients are mathematically correct
-- plan to write separate tests for that, either manual ones or by using
``torch.autograd.gradcheck``.

.. code-block:: python

  def sample_inputs(device, *, requires_grad=False):
      def make_tensor(*size):
          return torch.randn(size, device=device, requires_grad=requires_grad)

      def make_nondiff_tensor(*size):
          return torch.randn(size, device=device, requires_grad=False)

      return [
          [make_tensor(3), make_tensor(3), 1],
          [make_tensor(20), make_tensor(20), 3.14],
          [make_tensor(20), make_nondiff_tensor(20), -123],
          [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
      ]

  def reference_muladd(a, b, c):
      return a * b + c

  samples = sample_inputs(device, requires_grad=True)
  samples.extend(sample_inputs(device, requires_grad=False))
  for args in samples:
      # Correctness test
      result = torch.ops.extension_cpp.mymuladd(*args)
      expected = reference_muladd(*args)
      torch.testing.assert_close(result, expected)

      # Use opcheck to check for incorrect usage of operator registration APIs
      torch.library.opcheck(torch.ops.extension_cpp.mymuladd.default, args)

.. _mutable-ops:

Creating mutable operators
--------------------------
You may wish to author a custom operator that mutates its inputs. Use ``Tensor(a!)``
to specify each mutable Tensor in the schema; otherwise, there will be undefined
behavior. If there are multiple mutated Tensors, use different names (for example, ``Tensor(a!)``,
``Tensor(b!)``, ``Tensor(c!)``) for each mutable Tensor.

Let's author a ``myadd_out(a, b, out)`` operator, which writes the contents of ``a+b`` into ``out``.

.. code-block:: cpp

  // An example of an operator that mutates one of its inputs.
  void myadd_out_cpu(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(b.sizes() == out.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_CHECK(out.dtype() == at::kFloat);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = out.data_ptr<float>();
    for (int64_t i = 0; i < out.numel(); i++) {
      result_ptr[i] = a_ptr[i] + b_ptr[i];
    }
  }

When defining the operator, we must specify that it mutates the out Tensor in the schema:

.. code-block:: cpp

  TORCH_LIBRARY(extension_cpp, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
    // New!
    m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
  }

  TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
    m.impl("mymuladd", &mymuladd_cpu);
    m.impl("mymul", &mymul_cpu);
    // New!
    m.impl("myadd_out", &myadd_out_cpu);
  }

.. note::

  Do not return any mutated Tensors as outputs of the operator as this will
  cause incompatibility with PyTorch subsystems like ``torch.compile``.

Conclusion
----------
In this tutorial, we went over the recommended approach to integrating Custom C++
and CUDA operators with PyTorch. The ``TORCH_LIBRARY/torch.library`` APIs are fairly
low-level. For more information about how to use the API, see
`The Custom Operators Manual <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#the-custom-operators-manual>`_.
