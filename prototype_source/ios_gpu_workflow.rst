(Prototype) Use iOS GPU in PyTorch
==================================

**Author**: `Tao Xu <https://github.com/xta0>`_

Introduction
------------

This tutorial introduces the steps to run your models on iOS GPU. We'll be using the mobilenetv2 model as an example. Since the mobile GPU features are currently in the prototype stage, you'll need to build a custom pytorch binary from source. For the time being, only a limited number of operators are supported, and certain client side APIs are subject to change in the future versions.

Mobile Preparation
-------------------

Since GPUs consume weights in a different order, the first step we have to do is to convert our torchscript mobile to a GPU compatible model. This step is also known as "prepacking". To do that, we'll build a custom pytorch binary from source that includes the Metal backend. Go ahead checkout the pytorch source code from github and run the command below

.. code:: shell

    cd PYTORCH_ROOT
    USE_PYTORCH_METAL=ON python setup.py install --cmake

The command above will build a custom pytorch binary from master. The ``install`` argument simply tells ``setup.py`` to override the existing PyTorch on your desktop. Once the build finished, open another terminal to check the PyTorch version to see if the installation was successful. As the time of writing of this recipe, the version is ``1.8.0a0+41237a4``. You might be seeing different numbers depending on when you check out the code from master, but it should be greater than 1.7.0.

.. code:: python

    import torch
    torch.__version__ #1.8.0a0+41237a4


The next step is going to be converting the mobilenetv2 torchscript model to a Metal compatible model. We'll be leveraging the ``optimize_for_mobile`` API from the ``torch.utils`` module. As shown below

.. code:: python

    import torch
    from torch.utils.mobile_optimizer import optimize_for_mobile

    scripted_model = torch.jit.load('./mobilenetv2.pt')
    optimized_model = optimize_for_mobile(scripted_model, backend='metal')
    torch.jit.export_opnames(optimized_model)
    torch.jit.save(optimized_model, './mobilenetv2_metal.pt')

Note that the ``torch.jit.export_opnames(optimized_model)`` is going to dump all the optimized operators from the ``optimized_mobile``. If everything works well, you should be able to see the following ops being printed out from the console


.. code:: shell

    ['aten::adaptive_avg_pool2d', 
    'aten::add.Tensor', 
    'aten::addmm', 
    'aten::reshape', 
    'aten::size.int', 
    'metal::copy_to_host', 
    'metal_prepack::conv2d_run']

Those are all the ops we need to run the mobilenetv2 model on iOS GPU. Cool! Now you have the ``mobilenetv2_metal.pt`` saved on your disk, let's move on to the iOS part.


Use C++ APIs
---------------------

In this section, we'll be using the `HelloWorld example <https://github.com/pytorch/ios-demo-app>`_ to demonstrate how to use the C++ APIs. The first thing we need to do is to build a custom LibTorch from Source. Make sure you have deleted the **build** folder from the previous step in PyTorch root directory. Then run the command below

.. code:: shell
    
    IOS_ARCH=arm64 ./scripts/build_ios.sh -DUSE_PYTORCH_METAL=ON

Note ``IOS_ARCH`` tells the script to build a arm64 version of Libtorch. This is because in PyTorch, Metal is only available for the iOS devices that support the Apple A9 chip or above. Once the build finished, follow the `Build PyTorch iOS libraries from source <https://pytorch.org/mobile/ios/#build-pytorch-ios-libraries-from-source>`_ section from the iOS tutorial to setup the XCode settings properly. Don't forget to copy the `./mobilenetv2_metal.pt` to your XCode project.

The last step is to change the ``predictImage`` method in ``TorchModule.mm``

.. code:: objective-cd
    
    - (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
      torch::jit::GraphOptimizerEnabledGuard opguard(false);
      at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat).metal();
      auto outputTensor = _impl.forward({tensor}).toTensor().cpu();
      ...
      return nil;
    }

As you can see, we simply just call ``.metal()`` to move our input tensor from CPU to GPU, and then call ``.cpu()`` to move the result back. Internally, ``.metal()`` will copy the input data from the CPU buffer to a GPU buffer and convert the memory format. When `.cpu()` is invoked, the GPU command buffer will be flushed and synced. The final result will then be copied back from the GPU buffer back to a CPU buffer.

If everything works fine, you should see the inference results being showed on the screen. The result below is captured from a iPhone11 device

.. code:: shell

    - timber wolf, grey wolf, gray wolf, Canis lupus
    - malamute, malemute, Alaskan malamute
    - Eskimo dog, husky

You may notice that the results are slighly different from the `results <https://pytorch.org/mobile/ios/#install-libtorch-via-cocoapods>`_ we got from the CPU model as shown in the iOS tutorial. This because by default Metal uses fp16 rather than fp32 to compute. The precision loss from the result is expected.


Conclusion
----------

In this tutorial, we demonstrated how to convert the existing torchscript model to a GPU compatible model for iOS. We walked through the HelloWorld example to show how to use the C++ APIs. Please be aware of that GPU feature is still under development, new operators will continue be added. APIs are subject to change in the future versions.

Thanks for reading! As always, we welcome any feedback, so please create an issue `here <https://github.com/pytorch/pytorch/issues>`_ if you have any.




