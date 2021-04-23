Model Preparation for Android Recipe
=====================================

This recipe demonstrates how to prepare a PyTorch MobileNet v2 image classification model for Android apps, and how to set up Android projects to use the mobile-ready model file.

Introduction
-----------------

After a PyTorch model is trained or a pre-trained model is made available, it is normally not ready to be used in mobile apps yet. It needs to be quantized (see the `Quantization Recipe <quantization.html>`_), converted to TorchScript so Android apps can load it, and optimized for mobile apps. Furthermore, Android apps need to be set up correctly to enable the use of PyTorch Mobile libraries, before they can load and use the model for inference.

Pre-requisites
-----------------

PyTorch 1.6.0 or 1.7.0

torchvision 0.6.0 or 0.7.0

Android Studio 3.5.1 or above with NDK installed

Steps
-----------------

1. Get Pretrained and Quantized MobileNet v2 Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get the MobileNet v2 quantized model, simply do:

::

    import torchvision

    model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)

2. Script and Optimize the Model for Mobile Apps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use either the `script` or `trace` method to convert the quantized model to the TorchScript format:

::

    import torch

    dummy_input = torch.rand(1, 3, 224, 224)
    torchscript_model = torch.jit.trace(model_quantized, dummy_input)

or

::

    torchscript_model = torch.jit.script(model_quantized)


.. warning::
    The `trace` method only scripts the code path executed during the trace, so it will not work properly for models that include decision branches. See the `Script and Optimize for Mobile Recipe <script_optimized.html>`_ for more details.

Then optimize the TorchScript formatted model for mobile and save it:

::

    from torch.utils.mobile_optimizer import optimize_for_mobile
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized, "mobilenetv2_quantized.pt")

With the total 7 or 8 (depending on if the `script` or `trace` method is called to get the TorchScript format of the model) lines of code in the two steps above, we have a model ready to be added to mobile apps.

3. Add the Model and PyTorch Library on Android
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* In your current or a new Android Studio project, open the build.gradle file, and add the following two lines (the second one is required only if you plan to use a TorchVision model):

::

    implementation 'org.pytorch:pytorch_android:1.6.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.6.0'

* Drag and drop the model file `mobilenetv2_quantized.pt` to your project's assets folder.

That's it! Now you can build your Android app with the PyTorch library and the model ready to use. To actually write code to use the model, refer to the PyTorch Mobile `Android Quickstart with a HelloWorld Example <https://pytorch.org/mobile/android/#quickstart-with-a-helloworld-example>`_ and `Android Hackathon Example <https://github.com/pytorch/workshops/tree/master/PTMobileWalkthruAndroid>`_.

Learn More
-----------------

1. `PyTorch Mobile site <https://pytorch.org/mobile>`_

2. `Introduction to TorchScript <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_
