Model Preparation for iOS Recipe
=====================================

This recipe demonstrates how to prepare a PyTorch MobileNet v2 image classification model for iOS apps, and how to set up an iOS project to use the mobile-ready model file.

Introduction
-----------------

After a PyTorch model is trained or a pre-trained model is made available, it is normally not ready to be used in mobile apps yet. It needs to be quantized (see `Quantization Recipe <quantization.html>`_ for more details), converted to TorchScript so iOS apps can load it and optimized for mobile apps (see `Script and Optimize for Mobile Recipe <script_optimized.html>`_). Furthermore, iOS apps need to be set up correctly to enable the use of PyTorch Mobile libraries, before they can load and use the model for inference.

Pre-requisites
-----------------

PyTorch 1.6.0 or 1.7.0

torchvision 0.6.0 or 0.7.0

Xcode 11 or 12

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

Use either the script or trace method to convert the quantized model to the TorchScript format:

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

With the total 7 or 8 (depending on if the script or trace method is called to get the TorchScript format of the model) lines of code in the two steps above, we have a model ready to be added to mobile apps.

3. Add the Model and PyTorch Library on iOS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the mobile-ready model `mobilenetv2_quantized.pt` in an iOS app, either create a new Xcode project or in your existing Xcode project, then follow the steps below:

* Open a Mac Terminal, cd to your iOS app's project folder;

* If your iOS app does not use Cocoapods yet, run `pod init` first to generate the `Podfile` file.

* Edit `Podfile` either from Xcode or any editor, and add the following line under the target:

::

    pod 'LibTorch', '~>1.6.1'

* Run `pod install` from the Terminal and then open your project's xcworkspace file;

* Save the two files `TorchModule.h` and `TorchModule.mm` from `here <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld/HelloWorld/HelloWorld/TorchBridge>`_ and drag and drop them to your project. If your project is Swift based, a message box with the title "Would you like to configure an Objective-C bridging header?" will show up; click the "Create Bridging Header" button to create a Swift to Objective-c bridging header file, and add `#import "TorchModule.h"` to the header file `<your_project_name>-Bridging-Header.h`;

* Drag and drop the model file `mobilenetv2_quantized.pt` to the project.

After these steps, you can successfully build and run your Xcode project. To actually write code to use the model, refer to the PyTorch Mobile `iOS Code Walkthrough <https://pytorch.org/mobile/ios/#code-walkthrough>`_ and two complete ready-to-run sample iOS apps `HelloWorld <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld>`_ and `iOS Hackathon Example <https://github.com/pytorch/workshops/tree/master/PTMobileWalkthruIOS>`_.


Learn More
-----------------

1. `PyTorch Mobile site <https://pytorch.org/mobile>`_

2. `Introduction to TorchScript <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_
