PyTorch Vulkan Backend User Workflow
====================================

**Author**: `Ivan Kobzarev <https://github.com/IvanKobzarev>`_

Introduction
------------
PyTorch 1.7 supports the ability to run model inference on GPUs that support the Vulkan graphics and compute API. The primary target devices are mobile GPUs on Android devices. The Vulkan backend can also be used on Linux, Mac, and Windows desktop builds to use Vulkan devices like Intel integrated GPUs. This feature is in the prototype stage and is subject to change.

Building PyTorch with Vulkan backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vulkan backend is not included by default. The main switch to include Vulkan backend is cmake option ``USE_VULKAN``, that can be set by environment variable ``USE_VULKAN``.

To use PyTorch with Vulkan backend, we need to build it from source with additional settings. Checkout the PyTorch source code from GitHub master branch.

Optional usage of vulkan wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, Vulkan library will be loaded at runtime using the vulkan_wrapper library. If you specify the environment variable ``USE_VULKAN_WRAPPER=0`` libvulkan will be linked directly.

Desktop build
^^^^^^^^^^^^^

Vulkan SDK
^^^^^^^^^^
Download VulkanSDK from https://vulkan.lunarg.com/sdk/home and set environment variable ``VULKAN_SDK``

Unpack VulkanSDK to ``VULKAN_SDK_ROOT`` folder, install VulkanSDK following VulkanSDK instructions for your system.

For Mac:

::

    cd $VULKAN_SDK_ROOT
    source setup-env.sh
    sudo python install_vulkan.py


Building PyTorch:

For Linux:

::

    cd PYTORCH_ROOT
    USE_VULKAN=1 USE_VULKAN_SHADERC_RUNTIME=1 USE_VULKAN_WRAPPER=0 python setup.py install

For Mac:

::

    cd PYTORCH_ROOT
    USE_VULKAN=1 USE_VULKAN_SHADERC_RUNTIME=1 USE_VULKAN_WRAPPER=0 MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

After successful build, open another terminal and verify the version of installed PyTorch.

::

    import torch
    print(torch.__version__)

At the time of writing of this recipe, the version is 1.8.0a0+41237a4. You might be seeing different numbers depending on when you check out the code from master, but it should be greater than 1.7.0.


Android build
^^^^^^^^^^^^^

To build LibTorch for android with Vulkan backend for specified ``ANDROID_ABI``.

::

    cd PYTORCH_ROOT
    ANDROID_ABI=arm64-v8a USE_VULKAN=1 sh ./scripts/build_android.sh


To prepare pytorch_android aars that you can use directly in your app:

::

    cd $PYTORCH_ROOT
    USE_VULKAN=1 sh ./scripts/build_pytorch_android.sh


Model preparation
-----------------

Install torchvision, get the default pretrained float model.

::

    pip install torchvision

Python script to save pretrained mobilenet_v2 to a file:

::

    import torch
    import torchvision

    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.eval()
    script_model = torch.jit.script(model)
    torch.jit.save(script_model, "mobilenet2.pt")

PyTorch 1.7 Vulkan backend supports only float 32bit operators. The default model needs additional step that will optimize operators fusing

::

    from torch.utils.mobile_optimizer import optimize_for_mobile
    script_model_vulkan = optimize_for_mobile(script_model, backend='vulkan')
    torch.jit.save(script_model_vulkan, "mobilenet2-vulkan.pt")

The result model can be used only on Vulkan backend as it contains specific to the Vulkan backend operators.

By default, ``optimize_for_mobile`` with ``backend='vulkan'`` rewrites the graph so  that inputs are transferred to the Vulkan backend, and outputs are transferred to the CPU backend, therefore, the model can be run on CPU inputs and produce CPU outputs. To disable this, add the argument ``optimization_blocklist={MobileOptimizerType.VULKAN_AUTOMATIC_GPU_TRANSFER}`` to ``optimize_for_mobile``. (``MobileOptimizerType`` can be imported from ``torch.utils.mobile_optimizer``)

For more information, see the `torch.utils.mobile_optimizer` `API documentation <https://pytorch.org/docs/stable/mobile_optimizer.html>`_.

Using Vulkan backend in code
----------------------------

C++ API
-------

::

    at::is_vulkan_available()
    auto tensor = at::rand({1, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
    auto tensor_vulkan = t.vulkan();
    auto module = torch::jit::load("$PATH");
    auto tensor_output_vulkan = module.forward(inputs).toTensor();
    auto tensor_output = tensor_output.cpu();

``at::is_vulkan_available()`` function tries to initialize Vulkan backend and if Vulkan device is successfully found and context is created - it will return true, false otherwise.

``.vulkan()`` function called on Tensor will copy tensor to Vulkan device, and for operators called with this tensor as input - the operator will run on Vulkan device, and its output will be on the Vulkan device.

``.cpu()`` function called on Vulkan tensor will copy its data to CPU tensor (default)

Operators called with a tensor on a Vulkan device as an input will be executed on a Vulkan device. If an operator is not supported for the Vulkan backend the exception will be thrown.

List of supported operators:

::

    _adaptive_avg_pool2d
    _cat
    add.Scalar
    add.Tensor
    add_.Tensor
    addmm
    avg_pool2d
    clamp
    convolution
    empty.memory_format
    empty_strided
    hardtanh_
    max_pool2d
    mean.dim
    mm
    mul.Scalar
    relu_
    reshape
    select.int
    slice.Tensor
    transpose.int
    transpose_
    unsqueeze
    upsample_nearest2d
    view

Those operators allow to use torchvision models for image classification on Vulkan backend.


Python API
----------

``torch.is_vulkan_available()`` is exposed to Python API.

``tensor.to(device='vulkan')`` works as ``.vulkan()`` moving tensor to the Vulkan device.

``.vulkan()`` at the moment of writing of this tutorial is not exposed to Python API, but it is planned to be there.

Android Java API
----------------

For Android API to run model on Vulkan backend we have to specify this during model loading:

::

    import org.pytorch.Device;
    Module module = Module.load("$PATH", Device.VULKAN)
    FloatBuffer buffer = Tensor.allocateFloatBuffer(1 * 3 * 224 * 224);
    Tensor inputTensor = Tensor.fromBlob(buffer, new int[]{1, 3, 224, 224});
    Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();

In this case, all inputs will be transparently copied from CPU to the Vulkan device, and model will be run on Vulkan device, the output will be copied transparently to CPU.

The example of using Vulkan backend can be found in test application within the PyTorch repository:
https://github.com/pytorch/pytorch/blob/master/android/test_app/app/src/main/java/org/pytorch/testapp/MainActivity.java#L133

Building android test app with Vulkan
-------------------------------------

1. Build pytorch android with Vulkan backend for all android ABIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    cd $PYTORCH_ROOT
    USE_VULKAN=1 sh ./scripts/build_pytorch_android.sh

Or if you need only specific abi you can set it as an argument:

::

    cd $PYTORCH_ROOT
    USE_VULKAN=1 sh ./scripts/build_pytorch_android.sh $ANDROID_ABI

2. Add vulkan model to test application assets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add prepared model ``mobilenet2-vulkan.pt`` to test applocation assets:

::

  cp mobilenet2-vulkan.pt $PYTORCH_ROOT/android/test_app/app/src/main/assets/


3. Build and Install test applocation to connected android device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    cd $PYTORCH_ROOT
    gradle -p android test_app:installMbvulkanLocalBaseDebug

After successful installation, the application with the name 'MBQ' can be launched on the device.





Testing models without uploading to android device
--------------------------------------------------

Software implementations of Vulkan (e.g. https://swiftshader.googlesource.com/SwiftShader ) can be used to test if a model can be run using PyTorch Vulkan Backend (e.g. check if all model operators are supported).
