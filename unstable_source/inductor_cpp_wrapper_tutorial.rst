TorchInductor C++ Wrapper Tutorial
==============================================================

**Author**: `Chunyuan Wu <https://github.com/chunyuan-w>`_, `Bin Bao <https://github.com/desertfire>`__, `Jiong Gong <https://github.com/jgong5>`__

Prerequisites:
----------------
-  `torch.compile and TorchInductor concepts in PyTorch <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__

Introduction
------------

In ``torch.compile``, the default backend **TorchInductor** emits Python wrapper
code that manages memory allocation and kernel invocation. This design provides
flexibility and ease of debugging, but the interpreted nature of Python
introduces runtime overhead in performance-sensitive environments.

To address this limitation, TorchInductor includes a specialized mode that
generates **C++ wrapper code** in place of the Python wrapper, enabling faster
execution with minimal Python involvement.


Enabling the C++ wrapper mode
----------------
To enable this C++ wrapper mode for TorchInductor, add the following config to your code:

.. code:: python

    import torch._inductor.config as config
    config.cpp_wrapper = True


Example code
------------

We will use the following model code as an example:

.. code:: python

    import torch
    import torch._inductor.config as config

    config.cpp_wrapper = True

    def fn(x, y):
        return (x + y).sum()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(128, 128, device=device)
    y = torch.randn(128, 128, device=device)

    opt_fn = torch.compile(fn)
    result = opt_fn(x, y)


**For CPU**

The main part of TorchInductor-generated code with the default Python wrapper will look like this:

.. code:: python

    class Runner:
        def __init__(self, partitions):
            self.partitions = partitions

        def call(self, args):
            arg0_1, arg1_1 = args
            args.clear()
            assert_size_stride(arg0_1, (128, 128), (128, 1))
            assert_size_stride(arg1_1, (128, 128), (128, 1))
            buf0 = empty_strided_cpu((), (), torch.float32)
            cpp_fused_add_sum_0(arg0_1, arg1_1, buf0)
            del arg0_1
            del arg1_1
            return (buf0, )

By turning on the C++ wrapper, the generated code for the ``call`` function becomes a C++ function
``inductor_entry_impl``:

.. code:: python

    cpp_wrapper_src = (
    r'''
    #include <torch/csrc/inductor/cpp_wrapper/cpu.h>
    extern "C"  void  cpp_fused_add_sum_0(const float* in_ptr0,
                        const float* in_ptr1,
                        float* out_ptr0);
    CACHE_TORCH_DTYPE(float32);
    CACHE_TORCH_DEVICE(cpu);

    void inductor_entry_impl(
        AtenTensorHandle*
            input_handles, // array of input AtenTensorHandle; handles
                            // are stolen; the array itself is borrowed
        AtenTensorHandle*
            output_handles  // array for writing output AtenTensorHandle; handles
                            // will be stolen by the caller; the array itself is
                            // borrowed)
    ) {
        py::gil_scoped_release_simple release;

        auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 2);
        auto arg0_1 = std::move(inputs[0]);
        auto arg1_1 = std::move(inputs[1]);
        static constexpr int64_t *int_array_0=nullptr;
        AtenTensorHandle buf0_handle;
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(0, int_array_0, int_array_0, cached_torch_dtype_float32, cached_torch_device_type_cpu,  0, &buf0_handle));
        RAIIAtenTensorHandle buf0(buf0_handle);
        cpp_fused_add_sum_0((const float*)(arg0_1.data_ptr()), (const float*)(arg1_1.data_ptr()), (float*)(buf0.data_ptr()));
        arg0_1.reset();
        arg1_1.reset();
        output_handles[0] = buf0.release();
    } // inductor_entry_impl
    ...
    '''
    )

    inductor_entry = CppWrapperCodeCache.load_pybinding(
        argtypes=["std::vector<AtenTensorHandle>"],
        main_code=cpp_wrapper_src,
        device_type="cpu",
        num_outputs=1,
        kernel_code=None,
    )

    call = _wrap_func(inductor_entry)

**For GPU**

Based on the same example code, the generated code for GPU will look like this:

.. code:: python

    def call(args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (1, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0) # no-op to ensure context
            buf0 = empty_strided((19, ), (1, ), device='cuda', dtype=torch.float32)
            # Source Nodes: [add, tensor], Original ATen: [aten.add, aten.lift_fresh]
            stream0 = get_cuda_stream(0)
            triton_poi_fused_add_lift_fresh_0.run(constant0, arg0_1, buf0, 19, grid=grid(19), stream=stream0)
            run_intermediate_hooks('add', buf0)
            del arg0_1
            return (buf0, )

With the C++ wrapper turned on, the below equivalent C++ code will be generated:

.. code:: python

    inductor_entry = CppWrapperCodeCache.load_pybinding(
        argtypes=["std::vector<AtenTensorHandle>"],
        main_code=cpp_wrapper_src,
        device_type="cuda",
        num_outputs=1,
        kernel_code=None,
    )

    def _wrap_func(f):
        def g(args):
            input_tensors = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg, device='cpu') for arg in args]
            input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)

            args.clear()
            del input_tensors

            output_handles = f(input_handles)
            output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
            return output_tensors

        return g

    call = _wrap_func(inductor_entry)


Conclusion
------------

This tutorial introduced the **C++ wrapper** feature in TorchInductor, designed
to improve model performance with minimal code modification. We described the
motivation for this feature, detailed the experimental API used to enable it,
and compared the generated outputs of the default Python wrapper and the new
C++ wrapper on both CPU and GPU backends to illustrate their distinctions.

.. For more information on torch.compile, see
..
.. .. _torch.compile tutorial: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
.. .. TORCH_LOGS tutorial: https://docs.pytorch.org/tutorials/recipes/torch_logs.html
