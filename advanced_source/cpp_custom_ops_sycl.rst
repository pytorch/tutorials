.. _cpp-custom-ops-tutorial-sycl:

Custom SYCL Operators
=====================

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to integrate custom operators written in SYCL with PyTorch

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch 2.8 or later for Linux
       * PyTorch 2.10 or later for Windows
       * Basic understanding of SYCL programming

.. note::

  ``SYCL`` serves as the backend programming language for Intel GPUs (device label ``xpu``). For configuration details, see:
  `Getting Started on Intel GPUs <https://docs.pytorch.org/docs/main/notes/get_start_xpu.html>`_. The Intel Compiler, which comes bundled with `Intel Deep Learning Essentials <https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html>`_, handles ``SYCL`` compilation. Ensure you install and activate the compiler environment prior to executing the code examples in this tutorial.

PyTorch offers a large library of operators that work on Tensors (e.g. torch.add, torch.sum, etc).
However, you may wish to bring a new custom operator to PyTorch. This tutorial demonstrates the
best path to authoring a custom operator written in SYCL. Tutorials for C++ and CUDA operators are available in the :ref:`cpp-custom-ops-tutorial`.

Follow the structure to create a custom SYCL operator:

.. code-block:: text

  sycl_example/
  ├── setup.py
  ├── sycl_extension
  │   ├── __init__.py
  │   ├── muladd.sycl
  │   └── ops.py
  └── test_sycl_extension.py

Setting up the Build System
---------------------------

If you need to compile **SYCL** code (for example, ``.sycl`` files), use `torch.utils.cpp_extension.SyclExtension <https://docs.pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.SyclExtension>`_.
The setup process is very similar to C++/CUDA, except the compilation arguments need to be adjusted for SYCL.

Using ``sycl_extension`` is as straightforward as writing the following ``setup.py``:

.. code-block:: python

    import os
    import torch
    import glob
    import platform  
    from setuptools import find_packages, setup
    from torch.utils.cpp_extension import SyclExtension, BuildExtension

    library_name = "sycl_extension"
    py_limited_api = True

    IS_WINDOWS = (platform.system() == 'Windows')

    if IS_WINDOWS:
        cxx_args = [
            "/O2",                        
            "/std:c++17",                 
            "/DPy_LIMITED_API=0x03090000",
        ]
        sycl_args = ["/O2", "/std:c++17"] 
    else:
        cxx_args = [
            "-O3",
            "-fdiagnostics-color=always", 
            "-DPy_LIMITED_API=0x03090000"
        ]
        sycl_args = ["-O3"]

    extra_compile_args = {
        "cxx": cxx_args,
        "sycl": sycl_args
    }

    assert(torch.xpu.is_available()), "XPU is not available, please check your environment"

    # Source files collection
    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name)
    sources = list(glob.glob(os.path.join(extensions_dir, "*.sycl")))

    # Construct extension
    ext_modules = [
        SyclExtension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            py_limited_api=py_limited_api,
        )
    ]

    setup(
        name=library_name,
        packages=find_packages(),
        ext_modules=ext_modules,
        install_requires=["torch"],
        description="Simple Example of PyTorch Sycl extensions",
        cmdclass={"build_ext": BuildExtension},
        options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
    )

Defining the custom op and adding backend implementations
---------------------------------------------------------
First, let's write a SYCL function that computes ``mymuladd``:

In order to use this from PyTorch’s Python frontend, we need to register it
as a PyTorch operator using the ``TORCH_LIBRARY`` API. This will automatically
bind the operator to Python.


If you also have a SYCL implementation of ``myaddmul``, you can also register it
in a separate ``TORCH_LIBRARY_IMPL`` block:

.. code-block:: cpp

    #include <c10/xpu/XPUStream.h>
    #include <sycl/sycl.hpp>
    #include <ATen/Operators.h>
    #include <torch/all.h>
    #include <torch/library.h> 


    #include <Python.h>

    namespace sycl_extension {

    // ==========================================================
    // 1. Kernel 
    // ==========================================================
    static void muladd_kernel(
        int numel, const float* a, const float* b, float c, float* result,
        const sycl::nd_item<1>& item) {
        int idx = item.get_global_id(0);
        if (idx < numel) {
            result[idx] = a[idx] * b[idx] + c;
        }
    }

    class MulAddKernelFunctor {
    public:
        MulAddKernelFunctor(int _numel, const float* _a, const float* _b, float _c, float* _result)
            : numel(_numel), a(_a), b(_b), c(_c), result(_result) {}
        void operator()(const sycl::nd_item<1>& item) const {
            muladd_kernel(numel, a, b, c, result, item);
        }

    private:
        int numel;
        const float* a;
        const float* b;
        float c;
        float* result;
    };

    // ==========================================================
    // 2. Wrapper 
    // ==========================================================
    at::Tensor mymuladd_xpu(const at::Tensor& a, const at::Tensor& b, double c) {
        TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
        TORCH_CHECK(a.dtype() == at::kFloat, "a must be a float tensor");
        TORCH_CHECK(b.dtype() == at::kFloat, "b must be a float tensor");
        TORCH_CHECK(a.device().is_xpu(), "a must be an XPU tensor");
        TORCH_CHECK(b.device().is_xpu(), "b must be an XPU tensor");

        at::Tensor a_contig = a.contiguous();
        at::Tensor b_contig = b.contiguous();
        at::Tensor result = at::empty_like(a_contig);

        const float* a_ptr = a_contig.data_ptr<float>();
        const float* b_ptr = b_contig.data_ptr<float>();
        float* res_ptr = result.data_ptr<float>();
        int numel = a_contig.numel();

        sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
        constexpr int threads = 256;
        int blocks = (numel + threads - 1) / threads;

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<MulAddKernelFunctor>(
                sycl::nd_range<1>(blocks * threads, threads),
                MulAddKernelFunctor(numel, a_ptr, b_ptr, static_cast<float>(c), res_ptr)
            );
        });

        return result;
    }

    // ==========================================================
    // 3. Registration 
    // ==========================================================
    TORCH_LIBRARY(sycl_extension, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(sycl_extension, XPU, m) {
        m.impl("mymuladd", &mymuladd_xpu);
    }

    } // namespace sycl_extension

    // ==========================================================
    // 4. Windows Linker
    // ==========================================================
    extern "C" {
        #ifdef _WIN32
        __declspec(dllexport)
        #endif
        PyObject* PyInit__C(void) {
            static struct PyModuleDef moduledef = {
                PyModuleDef_HEAD_INIT,
                "_C",                 
                "XPU Extension Shim", 
                -1,                   
                NULL                  
            };
            return PyModule_Create(&moduledef);
        }
    }


Create a Python Interface
-------------------------

Create a Python interface for our operator in the ``sycl_extension/ops.py`` file:

.. code-block:: python

  import torch
  from torch import Tensor
  __all__ = ["mymuladd"]

  def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
      """Performs a * b + c in an efficient fused kernel"""
      return torch.ops.sycl_extension.mymuladd.default(a, b, c)

Initialize Package
------------------

Create ``sycl_extension/__init__.py`` file to make the package importable:

.. code-block:: python

    import ctypes
    import platform
    from pathlib import Path

    import torch

    current_dir = Path(__file__).parent.parent
    build_dir = current_dir / "build"

    if platform.system() == 'Windows':
        file_pattern = "**/*.pyd"
    else:
        file_pattern = "**/*.so"

    lib_files = list(build_dir.glob(file_pattern))

    if not lib_files:
        current_package_dir = Path(__file__).parent
        lib_files = list(current_package_dir.glob(file_pattern))

    assert len(lib_files) > 0, f"Could not find any {file_pattern} file in {build_dir} or {current_dir}"
    lib_file = lib_files[0]


    with torch._ops.dl_open_guard():
        loaded_lib = ctypes.CDLL(str(lib_file))

    from . import ops

    __all__ = [
        "loaded_lib",
        "ops",
    ]

Testing SYCL extension operator
-------------------

Use simple test to verify that the operator works correctly.

.. code-block:: python

  import torch
  from torch.testing._internal.common_utils import TestCase
  import unittest
  import sycl_extension

  def reference_muladd(a, b, c):
      return a * b + c

  class TestMyMulAdd(TestCase):
      def sample_inputs(self, device, *, requires_grad=False):
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

      def _test_correctness(self, device):
          samples = self.sample_inputs(device)
          for args in samples:
              result = sycl_extension.ops.mymuladd(*args)
              expected = reference_muladd(*args)
              torch.testing.assert_close(result, expected)

      @unittest.skipIf(not torch.xpu.is_available(), "requires Intel GPU")
      def test_correctness_xpu(self):
          self._test_correctness("xpu")

  if __name__ == "__main__":
      unittest.main()

This test checks the correctness of the custom operator by comparing its output against a reference implementation.

Conclusion
----------

In this tutorial, we demonstrated how to implement and compile custom SYCL operators for PyTorch. We specifically showcased an inference operation ``muladd``. For adding backward support or enabling torch.compile compatibility, please refer to :ref:`cpp-custom-ops-tutorial`.
