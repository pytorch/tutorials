Inductor C++ Wrapper Tutorial
==============================================================

**Author**: `Chunyuan Wu <https://github.com/chunyuan-w>`_, `Bin Bao <https://github.com/desertfire>`__, `Jiong Gong <https://github.com/jgong5>`__

Prerequisites:
----------------
-  `torch.compile and TorchInductor concepts in PyTorch <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__

Introduction
------------

Python as the primary interface of PyTorch is ease-of-use and efficient for development and debugging.
Inductor default wrapper generates Python code to invoke generated kernels and external kernels.
However, in deployment that requires high performance, Python as an interpreted language is slower compared
with compiled language.

We implemented Inductor cpp wrapper by leveraging the PyTorch C++ APIs
to generate pure cpp code to combine the generated and external kernels, which makes the
execution of each captured dynamo graph in pure cpp. This reduces the Python overhead within the graph.


API
------------
This feature is still in prototype stage. To turn it on, the below code change is needed:

.. code:: python

    import torch._inductor.config as config
    config.cpp_wrapper = True

This will speed up your models by reducing the python overhead of the inductor wrapper.


Example code
------------
Taken the below frontend code as an example:

.. code:: python
    
    import torch

    def fn(x):
        return torch.tensor(list(range(2, 40, 2)), device=x.device) + x

    x = torch.randn(1)
    opt_fn = torch.compile()(fn)
    y = opt_fn(x)


1. CPU

The main part of inductor generated code with the default python wrapper will be:

.. code:: python

    def call(args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (1, ), (1, ))
        buf0 = empty_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
        cpp_fused_add_lift_fresh_0(c_void_p(constant0.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
        del arg0_1
        return (buf0, )

By turning on cpp wrapper, the generated code for the ``call`` function becomes a cpp function
``inductor_entry_cpp`` of the CPP extension ``module``:

.. code:: python

    std::vector<at::Tensor> inductor_entry_cpp(const std::vector<at::Tensor>& args) {
        at::Tensor arg0_1 = args[0];
        at::Tensor constant0 = args[1];
        auto buf0 = at::empty_strided({19L, }, {1L, }, at::device(at::kCPU).dtype(at::kFloat));
        cpp_fused_add_lift_fresh_0((long*)(constant0.data_ptr()), (float*)(arg0_1.data_ptr()), (float*)(buf0.data_ptr()));
        arg0_1.reset();
        return {buf0};
    }

    module = CppWrapperCodeCache.load(cpp_wrapper_src, 'inductor_entry_cpp', 'c2buojsvlqbywxe3itb43hldieh4jqulk72iswa2awalwev7hjn2', False)

    def _wrap_func(f):
        def g(args):
            args_tensor = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]
            constants_tensor = [constant0]
            args_tensor.extend(constants_tensor)                    

            return f(args_tensor)
        return g
    call = _wrap_func(module.inductor_entry_cpp)

2. GPU

Based on the same example code, below demonstrated the generated code on GPU.
With the default python wrapper, the main generated code will be:

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

With cpp wrapper turned on, the below equivalent cpp code will be generated:

.. code:: python

    std::vector<at::Tensor> inductor_entry_cpp(const std::vector<at::Tensor>& args) {
        at::Tensor arg0_1 = args[0];
        at::Tensor constant0 = args[1];

        at::cuda::CUDAGuard device_guard(0);
        auto buf0 = at::empty_strided({19L, }, {1L, }, at::TensorOptions(c10::Device(at::kCUDA, 0)).dtype(at::kFloat));
        // Source Nodes: [add, tensor], Original ATen: [aten.add, aten.lift_fresh]
        if (triton_poi_fused_add_lift_fresh_0 == nullptr) {
            triton_poi_fused_add_lift_fresh_0 = loadKernel("/tmp/torchinductor_user/mm/cmm6xjgijjffxjku4akv55eyzibirvw6bti6uqmfnruujm5cvvmw.cubin", "triton_poi_fused_add_lift_fresh_0_0d1d2d3");
        }
        CUdeviceptr var_0 = reinterpret_cast<CUdeviceptr>(constant0.data_ptr());
        CUdeviceptr var_1 = reinterpret_cast<CUdeviceptr>(arg0_1.data_ptr());
        CUdeviceptr var_2 = reinterpret_cast<CUdeviceptr>(buf0.data_ptr());
        auto var_3 = 19;
        void* kernel_args_var_0[] = {&var_0, &var_1, &var_2, &var_3};
        cudaStream_t stream0 = at::cuda::getCurrentCUDAStream(0);
        launchKernel(triton_poi_fused_add_lift_fresh_0, 1, 1, 1, 1, 0, kernel_args_var_0, stream0);
        arg0_1.reset();
        return {buf0};
    }

    module = CppWrapperCodeCache.load(cpp_wrapper_src, 'inductor_entry_cpp', 'czbpeilh4qqmbyejdgsbpdfuk2ss5jigl2qjb7xs4gearrjvuwem', True)

    def _wrap_func(f):
        def g(args):
            args_tensor = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]
            constants_tensor = [constant0]
            args_tensor.extend(constants_tensor)

            return f(args_tensor)
        return g
    call = _wrap_func(module.inductor_entry_cpp)


Conclusion
------------
With this tutorial, we introduces a new cpp wrapper in TorchInductor to speed up your
models with two lines of code change.