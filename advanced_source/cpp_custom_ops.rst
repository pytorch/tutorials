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

       * PyTorch 2.10 or later
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
in the `extension-cpp <https://github.com/pytorch/extension-cpp>`_ repository,
which contains two parallel implementations:

- `extension_cpp_stable/ <https://github.com/pytorch/extension-cpp/tree/main/extension_cpp_stable>`_:
  Uses APIs supported by the LibTorch Stable ABI (recommended for PyTorch 2.10+). The main body of this
  tutorial uses code snippets from this implementation.
- `extension_cpp/ <https://github.com/pytorch/extension-cpp/tree/main/extension_cpp>`_:
  Uses the standard ATen/LibTorch API. Use this if you need APIs not yet available in the
  stable ABI. Code snippets from this implementation are shown in the
  :ref:`reverting-to-non-stable-api` section.

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
              extra_compile_args={
                  "cxx": [
                      # define Py_LIMITED_API with min version 3.9 to expose only the stable
                      # limited API subset from Python.h
                      "-DPy_LIMITED_API=0x03090000",
                      # define TORCH_TARGET_VERSION with min version 2.10 to expose only the
                      # stable API subset from torch
                      "-DTORCH_TARGET_VERSION=0x020a000000000000",
                  ]
              },
              py_limited_api=True)],  # Build 1 wheel across multiple Python versions
        cmdclass={'build_ext': cpp_extension.BuildExtension},
        options={"bdist_wheel": {"py_limited_api": "cp39"}}  # 3.9 is minimum supported Python version
  )

If you need to compile CUDA code (for example, ``.cu`` files), then instead use
`torch.utils.cpp_extension.CUDAExtension <https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension>`_.
Please see `extension-cpp <https://github.com/pytorch/extension-cpp>`_ for an
example for how this is set up.

CPython Agnosticism
^^^^^^^^^^^^^^^^^^^

The above example represents what we refer to as a CPython agnostic wheel, meaning we are
building a single wheel that can be run across multiple CPython versions (similar to pure
Python packages). CPython agnosticism is desirable in minimizing the number of wheels your
custom library needs to support and release. The minimum version we'd like to support is
3.9, since it is the oldest supported version currently, so we use the corresponding hexcode
and specifier throughout the setup code. We suggest building the extension in the same
environment as the minimum CPython version you'd like to support to minimize unknown behavior,
so, here, we build the extension in a CPython 3.9 environment. When built, this single wheel
will be runnable in any CPython environment 3.9+. To achieve this, there are three key lines
to note.

The first is the specification of ``Py_LIMITED_API`` in ``extra_compile_args`` to the
minimum CPython version you would like to support:

.. code-block:: python

  extra_compile_args={"cxx": ["-DPy_LIMITED_API=0x03090000"]},

Defining the ``Py_LIMITED_API`` flag helps verify that the extension is in fact
only using the `CPython Stable Limited API <https://docs.python.org/3/c-api/stable.html>`_,
which is a requirement for the building a CPython agnostic wheel. If this requirement
is not met, it is possible to build a wheel that looks CPython agnostic but will crash,
or worse, be silently incorrect, in another CPython environment. Take care to avoid
using unstable CPython APIs, for example APIs from libtorch_python (in particular
pytorch/python bindings,) and to only use APIs from libtorch (ATen objects, operators
and the dispatcher). We strongly recommend defining the ``Py_LIMITED_API`` flag to
help ascertain the extension is compliant and safe as a CPython agnostic wheel. Note that
defining this flag is not a full guarantee that the built wheel is CPython agnostic, but
it is better than the wild wild west. There are several caveats mentioned in the
`Python docs <https://docs.python.org/3/c-api/stable.html#limited-api-caveats>`_,
and you should test and verify yourself that the wheel is truly agnostic for the relevant
CPython versions.

The second and third lines specifying ``py_limited_api`` inform setuptools that you intend
to build a CPython agnostic wheel and will influence the naming of the wheel accordingly:

.. code-block:: python

  setup(name="extension_cpp",
        ext_modules=[
            cpp_extension.CppExtension(
              ...,
              py_limited_api=True)],  # Build 1 wheel across multiple Python versions
        ...,
        options={"bdist_wheel": {"py_limited_api": "cp39"}}  # 3.9 is minimum supported Python version
  )

It is necessary to specify ``py_limited_api=True`` as an argument to CppExtension/
CUDAExtension and also as an option to the ``"bdist_wheel"`` command with the minimal
supported CPython version (in this case, 3.9). Consequently, the ``setup`` in our
tutorial would build one properly named wheel that could be installed across multiple
CPython versions ``>=3.9``.

If your extension uses CPython APIs outside the stable limited set, then you cannot
build a CPython agnostic wheel! You should build one wheel per CPython version instead,
like so:

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

LibTorch Stable ABI (PyTorch Agnosticism)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to CPython agnosticism, there is a second axis of wheel compatibility:
LibTorch agnosticism. While CPython agnosticism allows building a single wheel
that works across multiple Python versions (3.9, 3.10, 3.11, etc.), LibTorch agnosticism
allows building a single wheel that works across multiple PyTorch versions (2.10, 2.11, 2.12, etc.).
These two concepts are orthogonal and can be combined.

To achieve LibTorch agnosticism, you must use the LibTorch Stable ABI, which provides
a stable C API for interacting with PyTorch tensors and operators. For example, instead of
using ``at::Tensor``, you must use ``torch::stable::Tensor``. For comprehensive
documentation on the stable ABI, including migration guides, supported types, and
stack-based API conventions, see the
`LibTorch Stable ABI documentation <https://pytorch.org/docs/main/notes/libtorch_stable_abi.html>`_.

The setup.py above already includes ``TORCH_TARGET_VERSION=0x020a000000000000``, which indicates that
the extension targets the LibTorch Stable ABI with a minimum supported PyTorch version of 2.10. The version format is:
``[MAJ 1 byte][MIN 1 byte][PATCH 1 byte][ABI TAG 5 bytes]``, so 2.10.0 = ``0x020a000000000000``.

The sections below contain examples of code using the LibTorch Stable ABI.
If the stable API/ABI does not contain what you need, see the :ref:`reverting-to-non-stable-api` section
or the `extension_cpp/ subdirectory <https://github.com/pytorch/extension-cpp/tree/main/extension_cpp>`_
in the extension-cpp repository for the equivalent examples using the non-stable API.


Defining the custom op and adding backend implementations
---------------------------------------------------------
First, let's write a C++ function that computes ``mymuladd`` using the LibTorch Stable ABI:

.. code-block:: cpp

   #include <torch/csrc/stable/library.h>
   #include <torch/csrc/stable/ops.h>
   #include <torch/csrc/stable/tensor.h>
   #include <torch/headeronly/core/ScalarType.h>
   #include <torch/headeronly/macros/Macros.h>

   torch::stable::Tensor mymuladd_cpu(
       const torch::stable::Tensor& a,
       const torch::stable::Tensor& b,
       double c) {
     STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
     STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
     STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
     STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU);
     STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU);

     torch::stable::Tensor a_contig = torch::stable::contiguous(a);
     torch::stable::Tensor b_contig = torch::stable::contiguous(b);
     torch::stable::Tensor result = torch::stable::empty_like(a_contig);

     const float* a_ptr = a_contig.const_data_ptr<float>();
     const float* b_ptr = b_contig.const_data_ptr<float>();
     float* result_ptr = result.mutable_data_ptr<float>();

     for (int64_t i = 0; i < result.numel(); i++) {
       result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
     }
     return result;
   }

In order to use this from PyTorch’s Python frontend, we need to register it
as a PyTorch operator using the ``STABLE_TORCH_LIBRARY`` macro. This will automatically
bind the operator to Python.

Operator registration is a two step-process:

- **Defining the operator** - This step ensures that PyTorch is aware of the new operator.
- **Registering backend implementations** - In this step, implementations for various
  backends, such as CPU and CUDA, are associated with the operator.

Defining an operator
^^^^^^^^^^^^^^^^^^^^
To define an operator, follow these steps:

1. select a namespace for an operator. We recommend the namespace be the name of your top-level
   project; we'll use "extension_cpp" in our tutorial.
2. provide a schema string that specifies the input/output types of the operator and if an
   input Tensors will be mutated. We support more types in addition to Tensor and float;
   please see `The Custom Operators Manual <https://pytorch.org/docs/main/notes/custom_operators.html>`_
   for more details.

   * If you are authoring an operator that can mutate its input Tensors, please see here
     (:ref:`mutable-ops`) for how to specify that.

.. code-block:: cpp

  STABLE_TORCH_LIBRARY(extension_cpp, m) {
     // Note that "float" in the schema corresponds to the C++ double type
     // and the Python float type.
     m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
   }

This makes the operator available from Python via ``torch.ops.extension_cpp.mymuladd``.

Registering backend implementations for an operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use ``STABLE_TORCH_LIBRARY_IMPL`` to register a backend implementation for the operator.
Note that we wrap the function pointer with ``TORCH_BOX()`` - this is required for
stable ABI functions to handle argument boxing/unboxing correctly.

.. code-block:: cpp

   STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
     m.impl("mymuladd", TORCH_BOX(&mymuladd_cpu));
   }

If you also have a CUDA implementation of ``myaddmul``, you can register it
in a separate ``STABLE_TORCH_LIBRARY_IMPL`` block:

.. code-block:: cpp

  #include <torch/csrc/stable/library.h>
  #include <torch/csrc/stable/ops.h>
  #include <torch/csrc/stable/tensor.h>
  #include <torch/csrc/stable/c/shim.h>
  #include <cuda.h>
  #include <cuda_runtime.h>

  __global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) result[idx] = a[idx] * b[idx] + c;
  }

  torch::stable::Tensor mymuladd_cuda(
      const torch::stable::Tensor& a,
      const torch::stable::Tensor& b,
      double c) {
    STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CUDA);

    torch::stable::Tensor a_contig = torch::stable::contiguous(a);
    torch::stable::Tensor b_contig = torch::stable::contiguous(b);
    torch::stable::Tensor result = torch::stable::empty_like(a_contig);

    const float* a_ptr = a_contig.const_data_ptr<float>();
    const float* b_ptr = b_contig.const_data_ptr<float>();
    float* result_ptr = result.mutable_data_ptr<float>();

    int numel = a_contig.numel();

    // For now, we rely on the raw shim API to get the current CUDA stream.
    // This will be improved in a future release.
    // When using a raw shim API, we need to use TORCH_ERROR_CODE_CHECK to
    // check the error code and throw an appropriate runtime_error otherwise.
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(a.get_device_index(), &stream_ptr));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    muladd_kernel<<<(numel+255)/256, 256, 0, stream>>>(numel, a_ptr, b_ptr, c, result_ptr);
    return result;
  }

  STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
    m.impl("mymuladd", TORCH_BOX(&mymuladd_cuda));
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
this over directly using Python ``torch.autograd.Function`` (see
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
  torch::stable::Tensor mymul_cpu(
      const torch::stable::Tensor& a,
      const torch::stable::Tensor& b) {
    STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU);
    STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU);

    torch::stable::Tensor a_contig = torch::stable::contiguous(a);
    torch::stable::Tensor b_contig = torch::stable::contiguous(b);
    torch::stable::Tensor result = torch::stable::empty_like(a_contig);

    const float* a_ptr = a_contig.const_data_ptr<float>();
    const float* b_ptr = b_contig.const_data_ptr<float>();
    float* result_ptr = result.mutable_data_ptr<float>();

    for (int64_t i = 0; i < result.numel(); i++) {
      result_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
  }

  STABLE_TORCH_LIBRARY(extension_cpp, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    // New! defining the mymul operator
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
  }


  STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
    m.impl("mymuladd", TORCH_BOX(&mymuladd_cpu));
    // New! registering the cpu kernel for the mymul operator
    m.impl("mymul", TORCH_BOX(&mymul_cpu));
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
  void myadd_out_cpu(
      const torch::stable::Tensor& a,
      const torch::stable::Tensor& b,
      torch::stable::Tensor& out) {
    STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
    STD_TORCH_CHECK(b.sizes().equals(out.sizes()));
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Float);
    STD_TORCH_CHECK(out.is_contiguous());
    STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU);
    STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU);
    STD_TORCH_CHECK(out.device().type() == torch::headeronly::DeviceType::CPU);

    torch::stable::Tensor a_contig = torch::stable::contiguous(a);
    torch::stable::Tensor b_contig = torch::stable::contiguous(b);

    const float* a_ptr = a_contig.const_data_ptr<float>();
    const float* b_ptr = b_contig.const_data_ptr<float>();
    float* result_ptr = out.mutable_data_ptr<float>();

    for (int64_t i = 0; i < out.numel(); i++) {
      result_ptr[i] = a_ptr[i] + b_ptr[i];
    }
  }

When defining the operator, we must specify that it mutates the out Tensor in the schema:

.. code-block:: cpp

  STABLE_TORCH_LIBRARY(extension_cpp, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
    // New!
    m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
  }

  STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
    m.impl("mymuladd", TORCH_BOX(&mymuladd_cpu));
    m.impl("mymul", TORCH_BOX(&mymul_cpu));
    // New!
    m.impl("myadd_out", TORCH_BOX(&myadd_out_cpu));
  }

.. note::

  Do not return any mutated Tensors as outputs of the operator as this will
  cause incompatibility with PyTorch subsystems like ``torch.compile``.

.. _reverting-to-non-stable-api:

Reverting to the Non-Stable LibTorch API
----------------------------------------

The LibTorch Stable ABI/API is still under active development, and certain APIs may not
yet be available in ``torch/csrc/stable``, ``torch/headeronly``, or the C shims
(``torch/csrc/stable/c/shim.h``).

If you need an API that is not yet available in the stable ABI/API, you can revert to
the regular ATen API. Note that doing so means you will need to build separate wheels
for each PyTorch version you want to support.

We provide code snippets for ``mymuladd`` below to illustrate. The changes for the
CUDA variant, ``mymul`` and ``myadd_out`` are similar in nature and can be found in the
`extension_cpp/ <https://github.com/pytorch/extension-cpp/tree/main/extension_cpp>`_
subdirectory of the extension-cpp repository.

**Setup (setup.py)**

Remove ``-DTORCH_TARGET_VERSION`` from your ``extra_compile_args``:

.. code-block:: python

  extra_compile_args = {
      "cxx": [
          "-O3" if not debug_mode else "-O0",
          "-fdiagnostics-color=always",
          "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
          # Note: No -DTORCH_TARGET_VERSION flag
      ],
      "nvcc": [
          "-O3" if not debug_mode else "-O0",
      ],
  }

**C++ Implementation (muladd.cpp)**

Use ATen headers and types instead of the stable API:

.. code-block:: cpp

  // Use ATen/torch headers instead of torch/csrc/stable headers
  #include <ATen/Operators.h>
  #include <torch/all.h>
  #include <torch/library.h>

  namespace extension_cpp {

  // Use at::Tensor instead of torch::stable::Tensor
  at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
    // Use TORCH_CHECK instead of STD_TORCH_CHECK
    TORCH_CHECK(a.sizes() == b.sizes());
    // Use at::kFloat instead of torch::headeronly::ScalarType::Float
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    // Use at::DeviceType instead of torch::headeronly::DeviceType
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    // Use tensor.contiguous() instead of torch::stable::contiguous(tensor)
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    // Use torch::empty() instead of torch::stable::empty_like()
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    // Use data_ptr<T>() instead of const_data_ptr<T>()
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
      result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
    }
    return result;
  }

  // Use TORCH_LIBRARY instead of STABLE_TORCH_LIBRARY
  TORCH_LIBRARY(extension_cpp, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  }

  // Use TORCH_LIBRARY_IMPL instead of STABLE_TORCH_LIBRARY_IMPL
  TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
    // Pass function pointer directly instead of wrapping with TORCH_BOX()
    m.impl("mymuladd", &mymuladd_cpu);
  }

  }

Conclusion
----------
In this tutorial, we went over the recommended approach to integrating Custom C++
and CUDA operators with PyTorch. The ``STABLE_TORCH_LIBRARY/torch.library`` APIs are fairly
low-level. For more information about how to use the API, see
`The Custom Operators Manual <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#the-custom-operators-manual>`_.
