(Prototype) Introduce lite interpreter workflow in Android and iOS

**Author**: `Chen Lai <https://github.com/cccclai>`_, `Martin Yuan <https://github.com/iseeyuan>`_

Introduction
------------

This tutorial introduces the steps to use lite interpreter on iOS and Android. We'll be using the ImageSegmentation demo app as an example. Since lite interpreter is currently in the prototype stage, a custom pytorch binary from source is required.

Android
-------------------

1. **Prepare model**: Prepare the lite interpreter version of model by run the script below to generate the scripted model deeplabv3_scripted.pt and deeplabv3_scripted.ptl

.. code:: python

    import torch

    model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    scripted_module = torch.jit.script(model)
    # Export full jit version model (not compatible lite interpreter), leave it here for comparison
    scripted_module.save("deeplabv3_scripted.pt")
    # Export lite interpreter version model (compatible with lite interpreter)
    scripted_module._save_for_lite_interpreter("deeplabv3_scripted.ptl")

2. **Build libtorch lite for android**: Build libtorch for android for all 4 android abis (``armeabi-v7a``, ``arm64-v8a``, ``x86``, ``x86_64``) ``BUILD_LITE_INTERPRETER=1 ./scripts/build_pytorch_android.sh``. For example, if it will be tested on Pixel 4 emulator with ``x86``, use cmd ``BUILD_LITE_INTERPRETER=1 ./scripts/build_pytorch_android.sh x86`` to specify abi to save built time. After the build finish, it will show the library path:


.. code-block:: bash

   BUILD SUCCESSFUL in 55s
   134 actionable tasks: 22 executed, 112 up-to-date
   + find /Users/chenlai/pytorch/android -type f -name '*aar'
   + xargs ls -lah
   -rw-r--r--  1 chenlai  staff    13M Feb 11 11:48 /Users/chenlai/pytorch/android/pytorch_android/build/outputs/aar/pytorch_android-release.aar
   -rw-r--r--  1 chenlai  staff    36K Feb  9 16:45 /Users/chenlai/pytorch/android/pytorch_android_torchvision/build/outputs/aar/pytorch_android_torchvision-release.aar

3. **Use the PyTorch Android libraries built from source in the ImageSegmentation app**: Create a folder `libs` in the path, the path from repository root will be `ImageSegmentation/app/libs`. Copy `pytorch_android-release` to the path `ImageSegmentation/app/libs/pytorch_android-release.aar`. Copy `pytorch_android_torchvision` (downloaded from `Pytorch Android Torchvision Nightly <https://oss.sonatype.org/#nexus-search;quick~torchvision_android/>`_) to the path `ImageSegmentation/app/libs/pytorch_android_torchvision.aar`. Update the `dependencies` part of `ImageSegmentation/app/build.gradle` to

.. code:: gradle

   dependencies {
       implementation 'androidx.appcompat:appcompat:1.2.0'
       implementation 'androidx.constraintlayout:constraintlayout:2.0.2'
       testImplementation 'junit:junit:4.12'
       androidTestImplementation 'androidx.test.ext:junit:1.1.2'
       androidTestImplementation 'androidx.test.espresso:espresso-core:3.3.0'


       implementation(name:'pytorch_android-release', ext:'aar')
       implementation(name:'pytorch_android_torchvision', ext:'aar')

       implementation 'com.android.support:appcompat-v7:28.0.0'
       implementation 'com.facebook.fbjni:fbjni-java-only:0.0.3'
   }

Update `all projects` part in `ImageSegmentation/build.gradle` to


.. code:: gradle

    allprojects {
        repositories {
            google()
            jcenter()
            flatDir {
                dirs 'libs'
            }
        }
    }

Those are all the ops we need to run the mobilenetv2 model on iOS GPU. Cool! Now that you have the ``mobilenetv2_metal.pt`` saved on your disk, let's move on to the iOS part.

4. **Update model loader api**: Update `ImageSegmentation/app/src/main/java/org/pytorch/imagesegmentation/MainActivity.java` by

  4.1 Add new import: `import org.pytorch.LiteModuleLoader`

  4.2 Replace the way to load pytorch lite model

.. code:: java

    // mModule = Module.load(MainActivity.assetFilePath(getApplicationContext(), "deeplabv3_scripted.pt"));
    mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "deeplabv3_scripted.ptl"));

Use C++ APIs
---------------------

In this section, we'll be using the `HelloWorld example <https://github.com/pytorch/ios-demo-app>`_ to demonstrate how to use the C++ APIs. The first thing we need to do is to build a custom LibTorch from Source. Make sure you have deleted the **build** folder from the previous step in PyTorch root directory. Then run the command below

.. code:: shell

    IOS_ARCH=arm64 USE_PYTORCH_METAL=1 ./scripts/build_ios.sh

Note ``IOS_ARCH`` tells the script to build a arm64 version of Libtorch. This is because in PyTorch, Metal is only available for the iOS devices that support the Apple A9 chip or above. Once the build finished, follow the `Build PyTorch iOS libraries from source <https://pytorch.org/mobile/ios/#build-pytorch-ios-libraries-from-source>`_ section from the iOS tutorial to setup the XCode settings properly. Don't forget to copy the `./mobilenetv2_metal.pt` to your XCode project.

Next we need to make some changes in ``TorchModule.mm``

.. code:: objective-c
    //#import <LibTorch/LibTorch.h>
    #import <torch/script.h>

    - (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
      torch::jit::GraphOptimizerEnabledGuard opguard(false);
      at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat).metal();
      auto outputTensor = _impl.forward({tensor}).toTensor().cpu();
      ...
    }

As you can see, we simply just call ``.metal()`` to move our input tensor from CPU to GPU, and then call ``.cpu()`` to move the result back. Internally, ``.metal()`` will copy the input data from the CPU buffer to a GPU buffer with a GPU compatible memory format. When `.cpu()` is invoked, the GPU command buffer will be flushed and synced. After `forward` finished, the final result will then be copied back from the GPU buffer back to a CPU buffer.

The last step we have to do is to add the `Accelerate.framework` and the `MetalShaderPerformance.framework` to your xcode project.

If everything works fine, you should be able to see the inference results on your phone. The result below was captured from an iPhone11 device

.. code:: shell

    - timber wolf, grey wolf, gray wolf, Canis lupus
    - malamute, malemute, Alaskan malamute
    - Eskimo dog, husky

You may notice that the results are slighly different from the `results <https://pytorch.org/mobile/ios/#install-libtorch-via-cocoapods>`_ we got from the CPU model as shown in the iOS tutorial. This is because by default Metal uses fp16 rather than fp32 to compute. The precision loss is expected.


Conclusion
----------

In this tutorial, we demonstrated how to convert a mobilenetv2 model to a GPU compatible model. We walked through a HelloWorld example to show how to use the C++ APIs to run models on iOS GPU. Please be aware of that GPU feature is still under development, new operators will continue to be added. APIs are subject to change in the future versions.

Thanks for reading! As always, we welcome any feedback, so please create an issue `here <https://github.com/pytorch/pytorch/issues>`_ if you have any.

Learn More
----------

- The `Mobilenetv2 <https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>`_ from Torchvision
- To learn more about how to use ``optimize_for_mobile``, please refer to the `Mobile Perf Recipe <https://pytorch.org/tutorials/recipes/mobile_perf.html>`_
=======
(Prototype) Use Lite Interpreter in PyTorch
==================================


Introduction
------------
