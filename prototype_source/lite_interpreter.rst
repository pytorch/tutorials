(Prototype) Introduce lite interpreter workflow in Android and iOS
==================================================================

**Author**: `Chen Lai <https://github.com/cccclai>`_, `Martin Yuan <https://github.com/iseeyuan>`_

Introduction
------------

This tutorial introduces the steps to use lite interpreter on iOS and Android. We'll be using the ImageSegmentation demo app as an example. Since lite interpreter is currently in the prototype stage, a custom pytorch binary from source is required.


Android
-------------------
Get ImageSegmentation demo app in Android: https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation

1. **Prepare model**: Prepare the lite interpreter version of model by run the script below to generate the scripted model `deeplabv3_scripted.pt` and `deeplabv3_scripted.ptl`

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

3. **Use the PyTorch Android libraries built from source in the ImageSegmentation app**: Create a folder `libs` in the path, the path from repository root will be `ImageSegmentation/app/libs`. Copy `pytorch_android-release` to the path ``ImageSegmentation/app/libs/pytorch_android-release.aar``. Copy `pytorch_android_torchvision` (downloaded from `Pytorch Android Torchvision Nightly <https://oss.sonatype.org/#nexus-search;quick~torchvision_android/>`_) to the path ``ImageSegmentation/app/libs/pytorch_android_torchvision.aar``. Update the `dependencies` part of ``ImageSegmentation/app/build.gradle`` to

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

Update `all projects` part in ``ImageSegmentation/build.gradle`` to


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

4. **Update model loader api**: Update ``ImageSegmentation/app/src/main/java/org/pytorch/imagesegmentation/MainActivity.java`` by

  4.1 Add new import: `import org.pytorch.LiteModuleLoader`

  4.2 Replace the way to load pytorch lite model

.. code:: java

    // mModule = Module.load(MainActivity.assetFilePath(getApplicationContext(), "deeplabv3_scripted.pt"));
    mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "deeplabv3_scripted.ptl"));

5. **Test app**: Build and run the `ImageSegmentation` app in Android Studio

iOS
-------------------
Get ImageSegmentation demo app in iOS: https://github.com/pytorch/ios-demo-app/tree/master/ImageSegmentation

1. **Prepare model**: Same as Android.

2. **Build libtorch lite for iOS**:

.. code-block:: bash

   BUILD_PYTORCH_MOBILE=1 IOS_PLATFORM=SIMULATOR BUILD_LITE_INTERPRETER=1 ./scripts/build_ios.sh


3. **Remove Cocoapods from the project** (this step is only needed if you ran `pod install`):

.. code-block:: bash

   pod deintegrate

4. **Link ImageSegmentation demo app with the custom built library**:
Open your project in XCode, go to your project Target’s **Build Phases - Link Binaries With Libraries**, click the **+** sign and add all the library files located in `build_ios/install/lib`. Navigate to the project **Build Settings**, set the value **Header Search Paths** to `build_ios/install/include` and **Library Search Paths** to `build_ios/install/lib`.
In the build settings, search for **other linker flags**. Add a custom linker flag below
```
-all_load
```
Finally, disable bitcode for your target by selecting the Build Settings, searching for Enable Bitcode, and set the value to **No**.

5. **Update library and api**

  5.1 Update ``TorchModule.mm``: To use the custom built libraries the project, replace `#import <LibTorch/LibTorch.h>` (in ``TorchModule.mm``) which is needed when using LibTorch via Cocoapods with the code below:

.. code-block:: swift

    //#import <LibTorch/LibTorch.h>
    #include "ATen/ATen.h"
    #include "caffe2/core/timer.h"
    #include "caffe2/utils/string_utils.h"
    #include "torch/csrc/autograd/grad_mode.h"
    #include "torch/script.h"
    #include <torch/csrc/jit/mobile/function.h>
    #include <torch/csrc/jit/mobile/import.h>
    #include <torch/csrc/jit/mobile/interpreter.h>
    #include <torch/csrc/jit/mobile/module.h>
    #include <torch/csrc/jit/mobile/observer.h>

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


5.2 Update ``ViewController.swift``

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

6. Build and test the app in Xcode.

How to use lite interpreter + custom build
------------------------------------------
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

   SELECTED_OP_LIST=deeplabv3_scripted.yaml BUILD_PYTORCH_MOBILE=1 IOS_PLATFORM=SIMULATOR BUILD_LITE_INTERPRETER=1 ./scripts/build_ios.sh

**Android**: Take the x86 build for example, the command should be:

.. code-block:: bash

   SELECTED_OP_LIST=deeplabv3_scripted.yaml BUILD_LITE_INTERPRETER=1 ./scripts/build_pytorch_android.sh x86


Conclusion
----------

In this tutorial, we demonstrated how to use lite interpreter in Android and iOS app. We walked through an Image Segmentation example to show how to dump the model, build torch library from source and use the new api to run model. Please be aware of that lite interpreter is still under development, more library size reduction will be introduced in the future. APIs are subject to change in the future versions.

Thanks for reading! As always, we welcome any feedback, so please create an issue `here <https://github.com/pytorch/pytorch/issues>`_ if you have any.

Learn More
----------

- To learn more about PyTorch Mobile, please refer to `PyTorch Mobile Home Page <https://pytorch.org/mobile/home/>`_
- To learn more about Image Segmentation, please refer to the `Image Segmentation DeepLabV3 on Android Recipe <https://pytorch.org/tutorials/beginner/deeplabv3_on_android.html>`_
