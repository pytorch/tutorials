(beta) Efficient mobile interpreter in Android and iOS
==================================================================

**Author**: `Chen Lai <https://github.com/cccclai>`_, `Martin Yuan <https://github.com/iseeyuan>`_

Introduction
------------

This tutorial introduces the steps to use PyTorch's efficient interpreter on iOS and Android. We will be using an  Image Segmentation demo application as an example.

This application will take advantage of the pre-built interpreter libraries available for Android and iOS, which can be used directly with Maven (Android) and CocoaPods (iOS). It is important to note that the pre-built libraries are the available for simplicity, but further size optimization can be achieved with by utilizing PyTorch's custom build capabilities.

.. note:: If you see the error message: `PytorchStreamReader failed locating file bytecode.pkl: file not found ()`, likely you are using a torch script model that requires the use of the PyTorch JIT interpreter (a version of our PyTorch interpreter that is not as size-efficient). In order to leverage our efficient interpreter, please regenerate the model by running: `module._save_for_lite_interpreter(${model_path})`.

   - If `bytecode.pkl` is missing, likely the model is generated with the api: `module.save(${model_psth})`.
   - The api `_load_for_lite_interpreter(${model_psth})` can be helpful to validate model with the efficient mobile interpreter.

Android
-------------------
Get the Image Segmentation demo app in Android: https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation

1. **Prepare model**: Prepare the mobile interpreter version of model by run the script below to generate the scripted model `deeplabv3_scripted.pt` and `deeplabv3_scripted.ptl`

.. code:: python

    import torch
    from torch.utils.mobile_optimizer import optimize_for_mobile
    model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    scripted_module = torch.jit.script(model)
    # Export full jit version model (not compatible mobile interpreter), leave it here for comparison
    scripted_module.save("deeplabv3_scripted.pt")
    # Export mobile interpreter version model (compatible with mobile interpreter)
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter("deeplabv3_scripted.ptl")

2. **Use the PyTorch Android library in the ImageSegmentation app**: Update the `dependencies` part of ``ImageSegmentation/app/build.gradle`` to

.. code:: gradle

    repositories {
        maven {
            url "https://oss.sonatype.org/content/repositories/snapshots"
        }
    }

    dependencies {
        implementation 'androidx.appcompat:appcompat:1.2.0'
        implementation 'androidx.constraintlayout:constraintlayout:2.0.2'
        testImplementation 'junit:junit:4.12'
        androidTestImplementation 'androidx.test.ext:junit:1.1.2'
        androidTestImplementation 'androidx.test.espresso:espresso-core:3.3.0'
        implementation 'org.pytorch:pytorch_android_lite:1.9.0'
        implementation 'org.pytorch:pytorch_android_torchvision:1.9.0'

        implementation 'com.facebook.fbjni:fbjni-java-only:0.0.3'
    }



3. **Update model loader api**: Update ``ImageSegmentation/app/src/main/java/org/pytorch/imagesegmentation/MainActivity.java`` by

  4.1 Add new import: `import org.pytorch.LiteModuleLoader`

  4.2 Replace the way to load pytorch lite model

.. code:: java

    // mModule = Module.load(MainActivity.assetFilePath(getApplicationContext(), "deeplabv3_scripted.pt"));
    mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "deeplabv3_scripted.ptl"));

4. **Test app**: Build and run the `ImageSegmentation` app in Android Studio

iOS
-------------------
Get ImageSegmentation demo app in iOS: https://github.com/pytorch/ios-demo-app/tree/master/ImageSegmentation

1. **Prepare model**: Same as Android.

2. **Build the project with Cocoapods and prebuilt interpreter** Update the `PodFile` and run `pod install`:

.. code-block:: podfile

    target 'ImageSegmentation' do
    # Comment the next line if you don't want to use dynamic frameworks
    use_frameworks!

    # Pods for ImageSegmentation
    pod 'LibTorch_Lite', '~>1.9.0'
    end

3. **Update library and API**

  3.1 Update ``TorchModule.mm``: To use the custom built libraries project, use `<Libtorch-Lite.h>` (in ``TorchModule.mm``):

.. code-block:: swift

    #import <Libtorch-Lite.h>
    // If it's built from source with xcode, comment out the line above
    // and use following headers
    // #include <torch/csrc/jit/mobile/import.h>
    // #include <torch/csrc/jit/mobile/module.h>
    // #include <torch/script.h>

.. code-block:: swift

    @implementation TorchModule {
    @protected
    // torch::jit::script::Module _impl;
     torch::jit::mobile::Module _impl;
    }

    - (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
      self = [super init];
      if (self) {
          try {
              _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
             //  _impl = torch::jit::load(filePath.UTF8String);
             //  _impl.eval();
            } catch (const std::exception& exception) {
                NSLog(@"%s", exception.what());
                return nil;
            }
        }
        return self;
    }

3.2 Update ``ViewController.swift``

.. code-block:: swift

    //  if let filePath = Bundle.main.path(forResource:
    //      "deeplabv3_scripted", ofType: "pt"),
    //      let module = TorchModule(fileAtPath: filePath) {
    //      return module
    //  } else {
    //      fatalError("Can't find the model file!")
    //  }
    if let filePath = Bundle.main.path(forResource:
        "deeplabv3_scripted", ofType: "ptl"),
        let module = TorchModule(fileAtPath: filePath) {
        return module
    } else {
        fatalError("Can't find the model file!")
    }

4. Build and test the app in Xcode.

How to use mobile interpreter + custom build
---------------------------------------------
A custom PyTorch interpreter library can be created to reduce binary size, by only containing the operators needed by the model. In order to do that follow these steps:

1. To dump the operators in your model, say `deeplabv3_scripted`, run the following lines of Python code:

.. code-block:: python

    # Dump list of operators used by deeplabv3_scripted:
    import torch, yaml
    model = torch.jit.load('deeplabv3_scripted.ptl')
    ops = torch.jit.export_opnames(model)
    with open('deeplabv3_scripted.yaml', 'w') as output:
        yaml.dump(ops, output)

In the snippet above, you first need to load the ScriptModule. Then, use export_opnames to return a list of operator names of the ScriptModule and its submodules. Lastly, save the result in a yaml file. The yaml file can be generated for any PyTorch 1.4.0 or above version. You can do that by checking the value of `torch.__version__`.

2. To run the build script locally with the prepared yaml list of operators, pass in the yaml file generate from the last step into the environment variable SELECTED_OP_LIST. Also in the arguments, specify BUILD_PYTORCH_MOBILE=1 as well as the platform/architechture type.

**iOS**: Take the simulator build for example, the command should be:

.. code-block:: bash

   SELECTED_OP_LIST=deeplabv3_scripted.yaml BUILD_PYTORCH_MOBILE=1 IOS_PLATFORM=SIMULATOR ./scripts/build_ios.sh

**Android**: Take the x86 build for example, the command should be:

.. code-block:: bash

   SELECTED_OP_LIST=deeplabv3_scripted.yaml ./scripts/build_pytorch_android.sh x86



Conclusion
----------

In this tutorial, we demonstrated how to use PyTorch's efficient mobile interpreter, in an Android and iOS app.

We walked through an Image Segmentation example to show how to dump the model, build a custom torch library from source and use the new api to run model.

Our efficient mobile interpreter is still under development, and we will continue improving its size in the future. Note, however, that the APIs are subject to change in future versions.

Thanks for reading! As always, we welcome any feedback, so please create an issue `here <https://github.com/pytorch/pytorch/issues>` - if you have any.

Learn More
----------

- To learn more about PyTorch Mobile, please refer to `PyTorch Mobile Home Page <https://pytorch.org/mobile/home/>`_
- To learn more about Image Segmentation, please refer to the `Image Segmentation DeepLabV3 on Android Recipe <https://pytorch.org/tutorials/beginner/deeplabv3_on_android.html>`_
