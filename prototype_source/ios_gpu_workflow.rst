(Prototype) Use iOS GPU in PyTorch
==================================

**Author**: `Tao Xu <https://github.com/xta0>`_

Introduction
------------

This tutorial introduces the steps to run your models on iOS GPU. We'll be using the mobilenetv2 model as an example. Since the mobile GPU features are currently in the prototype stage, you'll need to build a custom pytorch binary from source. For the time being, only a limited number of operators are supported, and certain client side APIs are subject to change in the future versions.

Model Preparation
-------------------

Since GPUs consume weights in a different order, the first step we need to do is to convert our TorchScript model to a GPU compatible model. This step is also known as "prepacking".

PyTorch with Metal
^^^^^^^^^^^^^^^^^^
To do that, we'll install a pytorch nightly binary that includes the Metal backend. Go ahead run the command below

.. code:: shell

    conda install pytorch -c pytorch-nightly
    // or
    pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

Also, you can build a custom pytorch binary from source that includes the Metal backend. Just checkout the pytorch source code from github and run the command below

.. code:: shell

    cd PYTORCH_ROOT
    USE_PYTORCH_METAL_EXPORT=ON python setup.py install --cmake

The command above will build a custom pytorch binary from master. The ``install`` argument simply tells ``setup.py`` to override the existing PyTorch on your desktop. Once the build finished, open another terminal to check the PyTorch version to see if the installation was successful. As the time of writing of this recipe, the version is ``1.8.0a0+41237a4``. You might be seeing different numbers depending on when you check out the code from master, but it should be greater than 1.7.0.

.. code:: python

    import torch
    torch.__version__ #1.8.0a0+41237a4

Metal Compatible Model
^^^^^^^^^^^^^^^^^^^^^^

The next step is going to be converting the mobilenetv2 torchscript model to a Metal compatible model. We'll be leveraging the ``optimize_for_mobile`` API from the ``torch.utils`` module. As shown below

.. code:: python

    import torch
    import torchvision
    from torch.utils.mobile_optimizer import optimize_for_mobile

    model = torchvision.models.mobilenet_v2(pretrained=True)
    scripted_model = torch.jit.script(model)
    optimized_model = optimize_for_mobile(scripted_model, backend='metal')
    print(torch.jit.export_opnames(optimized_model))
    optimized_model._save_for_lite_interpreter('./mobilenetv2_metal.pt')

Note that the ``torch.jit.export_opnames(optimized_model)`` is going to dump all the optimized operators from the ``optimized_mobile``. If everything works well, you should be able to see the following ops being printed out from the console


.. code:: shell

    ['aten::adaptive_avg_pool2d',
    'aten::add.Tensor',
    'aten::addmm',
    'aten::reshape',
    'aten::size.int',
    'metal::copy_to_host',
    'metal_prepack::conv2d_run']

Those are all the ops we need to run the mobilenetv2 model on iOS GPU. Cool! Now that you have the ``mobilenetv2_metal.pt`` saved on your disk, let's move on to the iOS part.


Use PyTorch iOS library with Metal
----------------------------------
The PyTorch iOS library with Metal support ``LibTorch-Lite-Nightly`` is available in Cocoapods. You can read the `Using the Nightly PyTorch iOS Libraries in CocoaPods <https://pytorch.org/mobile/ios/#using-the-nightly-pytorch-ios-libraries-in-cocoapods>`_ section from the iOS tutorial for more detail about its usage. 

We also have the `HelloWorld-Metal example <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld-Metal>`_ that shows how to conect all pieces together.  

Note that if you run the HelloWorld-Metal example, you may notice that the results are slighly different from the `results <https://pytorch.org/mobile/ios/#install-libtorch-via-cocoapods>`_ we got from the CPU model as shown in the iOS tutorial.

.. code:: shell

    - timber wolf, grey wolf, gray wolf, Canis lupus
    - malamute, malemute, Alaskan malamute
    - Eskimo dog, husky

This is because by default Metal uses fp16 rather than fp32 to compute. The precision loss is expected. 


Use LibTorch-Lite Built from Source
-----------------------------------

You can also build a custom LibTorch-Lite from Source and use it to run GPU models on iOS Metal. In this section, we'll be using the `HelloWorld example <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld>`_ to demonstrate this process. 

First, make sure you have deleted the **build** folder from the "Model Preparation" step in PyTorch root directory. Then run the command below

.. code:: shell

    IOS_ARCH=arm64 USE_PYTORCH_METAL=1 ./scripts/build_ios.sh

Note ``IOS_ARCH`` tells the script to build a arm64 version of Libtorch-Lite. This is because in PyTorch, Metal is only available for the iOS devices that support the Apple A9 chip or above. Once the build finished, follow the `Build PyTorch iOS libraries from source <https://pytorch.org/mobile/ios/#build-pytorch-ios-libraries-from-source>`_ section from the iOS tutorial to setup the XCode settings properly. Don't forget to copy the ``./mobilenetv2_metal.pt`` to your XCode project and modify the model file path accordingly.

Next we need to make some changes in ``TorchModule.mm``

.. code:: objective-c

    ...
    // #import <Libtorch-Lite/Libtorch-Lite.h>
    // If it's built from source with Xcode, comment out the line above
    // and use following headers
    #include <torch/csrc/jit/mobile/import.h>
    #include <torch/csrc/jit/mobile/module.h>
    #include <torch/script.h>
    ...

    - (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
      c10::InferenceMode mode;
      at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat).metal();
      auto outputTensor = _impl.forward({tensor}).toTensor().cpu();
      ...
    }
    ...

As you can see, we simply just call ``.metal()`` to move our input tensor from CPU to GPU, and then call ``.cpu()`` to move the result back. Internally, ``.metal()`` will copy the input data from the CPU buffer to a GPU buffer with a GPU compatible memory format. When ``.cpu()`` is invoked, the GPU command buffer will be flushed and synced. After `forward` finished, the final result will then be copied back from the GPU buffer back to a CPU buffer.

The last step we have to do is to add the ``Accelerate.framework`` and the ``MetalPerformanceShaders.framework`` to your xcode project (Open your project via XCode, go to your project target’s "General" tab, locate the "Frameworks, Libraries and Embedded Content" section and click the "+" button).

If everything works fine, you should be able to see the inference results on your phone. 


Conclusion
----------

In this tutorial, we demonstrated how to convert a mobilenetv2 model to a GPU compatible model. We walked through a HelloWorld example to show how to use the C++ APIs to run models on iOS GPU. Please be aware of that GPU feature is still under development, new operators will continue to be added. APIs are subject to change in the future versions.

Thanks for reading! As always, we welcome any feedback, so please create an issue `here <https://github.com/pytorch/pytorch/issues>`_ if you have any.

Learn More
----------

- The `Mobilenetv2 <https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>`_ from Torchvision
- To learn more about how to use ``optimize_for_mobile``, please refer to the `Mobile Perf Recipe <https://pytorch.org/tutorials/recipes/mobile_perf.html>`_
