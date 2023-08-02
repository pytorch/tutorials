Inductor cpp wrapper tutorial
==============================================================

**Author**: `Chunyuan Wu <https://github.com/chunyuan-w>`_, `Bin Bao <https://github.com/desertfire>`__, `Jiong Gong <https://github.com/jgong5>`__

Prerequisites:
------------
-  `torch.compile and TorchInductor concepts in PyTorch <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__

Introduction
------------

Inductor default wrapper generates python code to invoke generated kernels and external kernels.
Python as the primary interface of PyTorch is ease-of-use and efficient for development and debugging.
However, in deployment that requires high performance, Python as an interpreted language is slower compared
with compiled language. We implemented Inductor cpp wrapper by leveraging the PyTorch C++ APIs
to generate pure cpp code to combine the generated and external kernels, which makes the
execution of each captured dynamo graph in pure cpp.


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

Conclusion
------------
With this tutorial, we introduces a new cpp wrapper in TorchInductor to speed up your
models with two lines of code change.