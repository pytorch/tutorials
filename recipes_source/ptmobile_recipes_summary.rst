Summary of PyTorch Mobile Recipes
=====================================

This summary provides a top level overview of recipes for PyTorch Mobile to help developers choose which recipes to follow for their PyTorch-powered mobile app development.

Introduction
----------------

When a PyTorch model is trained or retrained, or when a pre-trained model is available, for mobile deployment, follow the the recipes outlined in this summary so mobile apps can successfully use the model.

Pre-requisites
----------------

PyTorch 1.6.0 or 1.7.0

(Optional) torchvision 0.6.0 or 0.7.0

For iOS development: Xcode 11 or 12

For Android development: Android Studio 3.5.1 or above (with NDK installed); or Android SDK, NDK, Gradle, JDK.

New Recipes for PyTorch Mobile
--------------------------------

* (Recommended) To fuse a list of PyTorch modules into a single module to reduce the model size before quantization, read the `Fuse Modules recipe <fuse.html>`_.

* (Recommended) To reduce the model size and make it run faster without losing much on accuracy, read the `Quantization Recipe <quantization.html>`_.

* (Must) To convert the model to TorchScipt and (optional) optimize it for mobile apps, read the `Script and Optimize for Mobile Recipe <script_optimized.html>`_.

* (Must for iOS development) To add the model in an iOS project and use PyTorch pod for iOS, read the `Model preparation for iOS Recipe <model_preparation_ios.html>`_.

* (Must for Android development) To add the model in an Android project and use the PyTorch library for Android, read the `Model preparation for Android Recipe <model_preparation_android.html>`_.


Learn More
-----------------

1. `PyTorch Mobile site <https://pytorch.org/mobile>`_
2. `PyTorch Mobile Performance Recipes <https://pytorch.org/tutorials/recipes/mobile_perf.html>`_
