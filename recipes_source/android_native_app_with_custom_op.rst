Making Native Android Application that uses PyTorch prebuilt libraries
======================================================================

**Author**: `Ivan Kobzarev <https://github.com/IvanKobzarev>`_

In this recipe, you will learn:

 - How to make an Android Application that uses LibTorch API from native code (C++).

 - How to use within this application TorchScript models with custom operators.

The full setup of this app you can find in `PyTorch Android Demo Application Repository <https://github.com/pytorch/android-demo-app/tree/master/NativeApp>`_.


Setup
~~~~~

You will need a Python 3 environment with the following packages (and their dependencies) installed:

- PyTorch 1.6

For Android development, you will need to install:

- Android NDK

::

  wget https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip
  unzip android-ndk-r19c-linux-x86_64.zip
  export ANDROID_NDK=$(pwd)/android-ndk-r19c


- Android SDK

::

  wget https://dl.google.com/android/repository/sdk-tools-linux-3859397.zip
  unzip sdk-tools-linux-3859397.zip -d android_sdk
  export ANDROID_HOME=$(pwd)/android_sdk



- Gradle 4.10.3

Gradle is the most widely used build system for android applications, and we will need it to build our application. Download it and add to the path to use ``gradle`` in the command line.

.. code-block:: shell

  wget https://services.gradle.org/distributions/gradle-4.10.3-bin.zip
  unzip gradle-4.10.3-bin.zip
  export GRADLE_HOME=$(pwd)/gradle-4.10.3
  export PATH="${GRADLE_HOME}/bin/:${PATH}"

- JDK

Gradle requires JDK, you need to install it and set environment variable ``JAVA_HOME`` to point to it.
For example you can install OpenJDK, following `instructions <https://openjdk.java.net/install/>`_.

- OpenCV SDK for Android

Our custom operator will be implemented using the OpenCV library. To use it for Android, we need to download OpenCV SDK for Android with prebuilt libraries.
Download from `OpenCV releases page <https://opencv.org/releases/>`_. Unzip it and set the environment variable ``OPENCV_ANDROID_SDK`` to it.


Preparing TorchScript Model With Custom C++ Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TorchScript allows using custom C++ operators, to read about it with details you can read 
`the dedicated tutorial <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_.

As a result, you can script the model that uses custom op, that uses OpenCV ``cv::warpPerspective`` function.

.. code-block:: python

  import torch
  import torch.utils.cpp_extension

  print(torch.version.__version__)
  op_source = """
  #include <opencv2/opencv.hpp>
  #include <torch/script.h>

  torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
    cv::Mat image_mat(/*rows=*/image.size(0),
                      /*cols=*/image.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/image.data_ptr<float>());
    cv::Mat warp_mat(/*rows=*/warp.size(0),
                     /*cols=*/warp.size(1),
                     /*type=*/CV_32FC1,
                     /*data=*/warp.data_ptr<float>());

    cv::Mat output_mat;
    cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{64, 64});

    torch::Tensor output =
      torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{64, 64});
    return output.clone();
  }

  static auto registry =
    torch::RegisterOperators("my_ops::warp_perspective", &warp_perspective);
  """

  torch.utils.cpp_extension.load_inline(
      name="warp_perspective",
      cpp_sources=op_source,
      extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
      is_python_module=False,
      verbose=True,
  )

  print(torch.ops.my_ops.warp_perspective)


  @torch.jit.script
  def compute(x, y):
      if bool(x[0][0] == 42):
          z = 5
      else:
          z = 10
      x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
      return x.matmul(y) + z


  compute.save("compute.pt")


This snippet generates ``compute.pt`` file which is TorchScript model that uses custom op ``my_ops.warp_perspective``.

You need to have installed OpenCV for development to run it.
For Linux systems that can be done using the next commands:
CentOS:

.. code-block:: shell

  yum install opencv-devel

Ubuntu:

.. code-block:: shell

  apt-get install libopencv-dev

Making Android Application
~~~~~~~~~~~~~~~~~~~~~~~~~~

After we succeeded in having ``compute.pt``, we want to use this TorchScript model within Android application. Using general TorchScript models (without custom operators) on Android, using Java API, you can find `here <https://pytorch.org/mobile/android/>`_. We can not use this approach for our case, as our model uses a custom operator(``my_ops.warp_perspective``), default TorchScript execution will fail to find it.

Registration of ops is not exposed to PyTorch Java API, thus we need to build Android Application with native part (C++) and using LibTorch C++ API to implement and register the same custom operator for Android. As our operator uses the OpenCV library - we will use prebuilt OpenCV Android libraries and use the same functions from OpenCV.

Let's start creating Android application in ``NativeApp`` folder.

.. code-block:: shell
  
  mkdir NativeApp
  cd NativeApp

Android Application Build Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Android Application build consists of the main gradle part and native build CMake part.
All the listings here are the full file listing, that if to recreate the whole structure,
you will be able to build and install the result Android Application without any code additions.

Gradle Build Setup
------------------
We will need to add gradle setup files: build.gradle, gradle.properties, settings.gradle.
More about Android Gradle build configurations you can find `here <https://developer.android.com/studio/build>`_.

``NativeApp/settings.gradle``

.. code-block:: gradle

  include ':app'


``NativeApp/gradle.properties``

.. code-block:: gradle

  android.useAndroidX=true
  android.enableJetifier=true


``NativeApp/build.gradle``

.. code-block:: gradle

  buildscript {
      repositories {
          google()
          jcenter()
      }
      dependencies {
          classpath 'com.android.tools.build:gradle:3.5.0'
      }
  }

  allprojects {
      repositories {
          google()
          jcenter()
      }
  }


In ``NativeApp/build.gradle`` we specify Android gradle plugin version `3.5.0`. This version is not recent. Still, we use it as PyTorch android gradle builds use this version.

``NativeApp/settings.gradle`` shows that out project contains only one module - ``app``, which will be our Android Application.

.. code-block:: shell

    mkdir app
    cd app


``NativeApp/app/build.gradle``

.. code-block:: gradle
  
  apply plugin: 'com.android.application'

  repositories {
    jcenter()
    maven {
      url "https://oss.sonatype.org/content/repositories/snapshots"
    }
  }

  android {
    configurations {
      extractForNativeBuild
    }
    compileSdkVersion 28
    buildToolsVersion "29.0.2"
    defaultConfig {
      applicationId "org.pytorch.nativeapp"
      minSdkVersion 21
      targetSdkVersion 28
      versionCode 1
      versionName "1.0"
      externalNativeBuild {
        cmake {
          arguments "-DANDROID_STL=c++_shared"
        }
      }
    }
    buildTypes {
      release {
        minifyEnabled false
      }
    }
    externalNativeBuild {
      cmake {
        path "CMakeLists.txt"
      }
    }
    sourceSets {
      main {
        jniLibs.srcDirs = ['src/main/jniLibs']
      }
    }
  }

  dependencies {
    implementation 'com.android.support:appcompat-v7:28.0.0'

    implementation 'org.pytorch:pytorch_android:1.6.0-SNAPSHOT'
    extractForNativeBuild 'org.pytorch:pytorch_android:1.6.0-SNAPSHOT'
  }

  task extractAARForNativeBuild {
    doLast {
      configurations.extractForNativeBuild.files.each {
        def file = it.absoluteFile
        copy {
          from zipTree(file)
          into "$buildDir/$file.name"
          include "headers/**"
          include "jni/**"
        }
      }
    }
  }

  tasks.whenTaskAdded { task ->
    if (task.name.contains('externalNativeBuild')) {
      task.dependsOn(extractAARForNativeBuild)
    }
  }

This gradle build script registers dependencies on pytorch_android snapshots,
that are published on nightly channels.

As they are published to nexus sonatype repository - we need to register that repository:
``https://oss.sonatype.org/content/repositories/snapshots``.

In our application we need to use LibTorch C++ API in our application native build part. For this, we need access to prebuilt binaries and headers. They are prepacked in PyTorch Android builds, which is published in Maven repositories.

To use PyTorch Android prebuilt libraries from gradle dependencies (which is aar files) - 
we should add registration for configuration ``extractForNativeBuild``,
add this configuration in dependencies and put its definition in the end.

``extractForNativeBuild`` task will call ``extractAARForNativeBuild`` task that unpacks pytorch_android aar
to gradle build directory.

Pytorch_android aar contains LibTorch headers in ``headers`` folder
and prebuilt libraries for different Android abis in ``jni`` folder:
``$ANDROID_ABI/libpytorch_jni.so``, ``$ANDROID_ABI/libfbjni.so``.
We will use them for our native build.

The native build is registered in this ``build.gradle`` with lines:

.. code-block:: gradle

  android {
    ...
    externalNativeBuild {
      cmake {
        path "CMakeLists.txt"
      }
  }
  ...
  defaultConfig {
    externalNativeBuild {
      cmake {
        arguments "-DANDROID_STL=c++_shared"
      }
    }
  }

We will use ``CMake`` configuration for a native build. Here we also specify that we will dynamically link with STL, as we have several libraries. More about this, you can find `here <https://developer.android.com/ndk/guides/cpp-support>`_.


Native Build CMake Setup
------------------------

The native build will be configured in ``NativeApp/app/CMakeLists.txt``:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.4.1)
  set(TARGET pytorch_nativeapp)
  project(${TARGET} CXX)
  set(CMAKE_CXX_STANDARD 14)

  set(build_DIR ${CMAKE_SOURCE_DIR}/build)

  set(pytorch_testapp_cpp_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)
  file(GLOB pytorch_testapp_SOURCES
    ${pytorch_testapp_cpp_DIR}/pytorch_nativeapp.cpp
  )

  add_library(${TARGET} SHARED
      ${pytorch_testapp_SOURCES}
  )

  file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
  file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

  target_compile_options(${TARGET} PRIVATE
    -fexceptions
  )

  set(BUILD_SUBDIR ${ANDROID_ABI})

  find_library(PYTORCH_LIBRARY pytorch_jni
    PATHS ${PYTORCH_LINK_DIRS}
    NO_CMAKE_FIND_ROOT_PATH)
  find_library(FBJNI_LIBRARY fbjni
    PATHS ${PYTORCH_LINK_DIRS}
    NO_CMAKE_FIND_ROOT_PATH)

  # OpenCV
  if(NOT DEFINED ENV{OPENCV_ANDROID_SDK})
    message(FATAL_ERROR "Environment var OPENCV_ANDROID_SDK is not set")
  endif()

  set(OPENCV_INCLUDE_DIR "$ENV{OPENCV_ANDROID_SDK}/sdk/native/jni/include")

  target_include_directories(${TARGET} PRIVATE
   "${OPENCV_INCLUDE_DIR}"
    ${PYTORCH_INCLUDE_DIRS})

  set(OPENCV_LIB_DIR "$ENV{OPENCV_ANDROID_SDK}/sdk/native/libs/${ANDROID_ABI}")

  find_library(OPENCV_LIBRARY opencv_java4
    PATHS ${OPENCV_LIB_DIR}
    NO_CMAKE_FIND_ROOT_PATH)

  target_link_libraries(${TARGET}
    ${PYTORCH_LIBRARY}
    ${FBJNI_LIBRARY}
    ${OPENCV_LIBRARY}
    log)

Here we register only one source file ``pytorch_nativeapp.cpp``.

On the previous step in ``NativeApp/app/build.gradle``, the task ``extractAARForNativeBuild`` extracts headers and native libraries to build directory.  We set ``PYTORCH_INCLUDE_DIRS`` and ``PYTORCH_LINK_DIRS`` to them.

After that, we find libraries ``libpytorch_jni.so`` and ``libfbjni.so`` and add them to the linking of our target.

As we plan to use OpenCV functions to implement our custom operator ``my_ops::warp_perspective`` - we need to link to ``libopencv_java4.so``. It is packaged in OpenCV SDK for Android, that was downloaded on the Setup step.
In this configuration, we find it by environment variable ``OPENCV_ANDROID_SDK``.

We also link with ``log`` library to be able to log our results to Android LogCat.

As we link to OpenCV Android SDK's ``libopencv_java4.so``, we should copy it to ``NativeApp/app/src/main/jniLibs/${ANDROID_ABI}``

.. code-block:: shell

  cp -R $OPENCV_ANDROID_SDK/sdk/native/libs/* NativeApp/app/src/main/jniLibs/


Adding the model file to the application
----------------------------------------

To package the TorschScript model ``compute.pt`` within our application we should copy it to assets folder:

.. code-block:: shell

  mkdir -p NativeApp/app/src/main/assets
  cp compute.pt NativeApp/app/src/main/assets


Android Application Manifest
----------------------------

Every Android application has a manifest. 
Here we specify the application name, package, main activity.

``NativeApp/app/src/main/AndroidManifest.xml``:

.. code-block:: default

  <?xml version="1.0" encoding="utf-8"?>
  <manifest xmlns:android="http://schemas.android.com/apk/res/android"
      package="org.pytorch.nativeapp">

      <application
          android:allowBackup="true"
          android:label="PyTorchNativeApp"
          android:supportsRtl="true"
          android:theme="@style/Theme.AppCompat.Light.DarkActionBar">

          <activity android:name=".MainActivity">
              <intent-filter>
                  <action android:name="android.intent.action.MAIN" />
                  <category android:name="android.intent.category.LAUNCHER" />
              </intent-filter>
          </activity>
      </application>
  </manifest>


Sources
-------

Java Code
---------

Now we are ready to implement our MainActivity in

``NativeApp/app/src/main/java/org/pytorch/nativeapp/MainActivity.java``:

.. code-block:: java

  package org.pytorch.nativeapp;

  import android.content.Context;
  import android.os.Bundle;
  import android.util.Log;
  import androidx.appcompat.app.AppCompatActivity;
  import java.io.File;
  import java.io.FileOutputStream;
  import java.io.IOException;
  import java.io.InputStream;
  import java.io.OutputStream;

  public class MainActivity extends AppCompatActivity {

    private static final String TAG = "PyTorchNativeApp";

    public static String assetFilePath(Context context, String assetName) {
      File file = new File(context.getFilesDir(), assetName);
      if (file.exists() && file.length() > 0) {
        return file.getAbsolutePath();
      }

      try (InputStream is = context.getAssets().open(assetName)) {
        try (OutputStream os = new FileOutputStream(file)) {
          byte[] buffer = new byte[4 * 1024];
          int read;
          while ((read = is.read(buffer)) != -1) {
            os.write(buffer, 0, read);
          }
          os.flush();
        }
        return file.getAbsolutePath();
      } catch (IOException e) {
        Log.e(TAG, "Error process asset " + assetName + " to file path");
      }
      return null;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
      super.onCreate(savedInstanceState);
      final String modelFileAbsoluteFilePath =
          new File(assetFilePath(this, "compute.pt")).getAbsolutePath();
      NativeClient.loadAndForwardModel(modelFileAbsoluteFilePath);
    }
  }


In the previous step, when we copied our ``compute.pt`` to ``NativeApp/app/src/main/assets`` that file became an Android application asset, which will be packed in application. Android system provides only stream access to it.
To use this module from LibTorch, we need to materialize it as a file on the disk. ``assetFilePath`` function copies data from the asset input stream, writes it on the disk, and returns absolute file path for it.

``OnCreate`` method of Activity is called just after Activity creation. In this method, we call ``assertFilePath`` and call ``NativeClient`` class that will dispatch it to native code through JNI call.

``NativeClient`` is a helper class with an internal private class ``NativePeer``, which is responsible for working with the native part of our application. It has a static block that will load ``libpytorch_nativeapp.so``, that is build with ``CMakeLists.txt`` that we added on the previous step. The static block will be executed with the first reference of ``NativePeer`` class. It happens in ``NativeClient#loadAndForwardModel``.

``NativeApp/app/src/main/java/org/pytorch/nativeapp/NativeClient.java``:

.. code-block:: java

  package org.pytorch.nativeapp;

  public final class NativeClient {

    public static void loadAndForwardModel(final String modelPath) {
      NativePeer.loadAndForwardModel(modelPath);
    }

    private static class NativePeer {
      static {
        System.loadLibrary("pytorch_nativeapp");
      }

      private static native void loadAndForwardModel(final String modelPath);
    }
  }

``NativePeer#loadAndForwardModel`` is declared as ``native``, it does not have definition in Java. Call to this method will be re-dispatched through JNI to C++ method in our ``libpytorch_nativeapp.so``, in ``NativeApp/app/src/main/cpp/pytorch_nativeapp.cpp``.

Native code
-----------

Now we are ready to write a native part of our application.

``NativeApp/app/src/main/cpp/pytorch_nativeapp.cpp``:

.. code-block:: cpp

  #include <android/log.h>
  #include <cassert>
  #include <cmath>
  #include <pthread.h>
  #include <unistd.h>
  #include <vector>
  #define ALOGI(...)                                                             \
    __android_log_print(ANDROID_LOG_INFO, "PyTorchNativeApp", __VA_ARGS__)
  #define ALOGE(...)                                                             \
    __android_log_print(ANDROID_LOG_ERROR, "PyTorchNativeApp", __VA_ARGS__)

  #include "jni.h"

  #include <opencv2/opencv.hpp>
  #include <torch/script.h>

  namespace pytorch_nativeapp {
  namespace {
  torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
    cv::Mat image_mat(/*rows=*/image.size(0),
                      /*cols=*/image.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/image.data_ptr<float>());
    cv::Mat warp_mat(/*rows=*/warp.size(0),
                     /*cols=*/warp.size(1),
                     /*type=*/CV_32FC1,
                     /*data=*/warp.data_ptr<float>());

    cv::Mat output_mat;
    cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{8, 8});

    torch::Tensor output =
        torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{8, 8});
    return output.clone();
  }

  static auto registry =
      torch::RegisterOperators("my_ops::warp_perspective", &warp_perspective);

  template <typename T> void log(const char *m, T t) {
    std::ostringstream os;
    os << t << std::endl;
    ALOGI("%s %s", m, os.str().c_str());
  }

  struct JITCallGuard {
    torch::autograd::AutoGradMode no_autograd_guard{false};
    torch::AutoNonVariableTypeMode non_var_guard{true};
    torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
  };
  } // namespace

  static void loadAndForwardModel(JNIEnv *env, jclass, jstring jModelPath) {
    const char *modelPath = env->GetStringUTFChars(jModelPath, 0);
    assert(modelPath);
    JITCallGuard guard;
    torch::jit::Module module = torch::jit::load(modelPath);
    module.eval();
    torch::Tensor x = torch::randn({4, 8});
    torch::Tensor y = torch::randn({8, 5});
    log("x:", x);
    log("y:", y);
    c10::IValue t_out = module.forward({x, y});
    log("result:", t_out);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
  }
  } // namespace pytorch_nativeapp

  JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
    JNIEnv *env;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
      return JNI_ERR;
    }

    jclass c = env->FindClass("org/pytorch/nativeapp/NativeClient$NativePeer");
    if (c == nullptr) {
      return JNI_ERR;
    }

    static const JNINativeMethod methods[] = {
        {"loadAndForwardModel", "(Ljava/lang/String;)V",
         (void *)pytorch_nativeapp::loadAndForwardModel},
    };
    int rc = env->RegisterNatives(c, methods,
                                  sizeof(methods) / sizeof(JNINativeMethod));

    if (rc != JNI_OK) {
      return rc;
    }

    return JNI_VERSION_1_6;
  }


This listing is quite long, and a few things intermixed here, we will follow control flow to understand how this code works.
The first place where the control flow arrives is ``JNI_OnLoad``.
This function is called after loading the library. It is responsible for registering native method, which is called when ``NativePeer#loadAndForwardModel`` called, here it is ``pytorch_nativeapp::loadAndForwardModel`` function.

``pytorch_nativeapp::loadAndForwardModel`` takes as an argument model path.
First, we extract its ``const char*`` value and loading the module with ``torch::jit::load``.

To load TorchScript model for mobile, we need to set these guards, because mobile build doesn't support 
features like autograd for smaller build size, placed in ``struct JITCallGuard`` in this example.
It may change in the future. You can track the latest changes keeping an eye on the 
`source in PyTorch GitHub <https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/cpp/pytorch_jni_jit.cpp>`_.

Implementation of method ``warp_perspective`` and registration of it is entirely the same as
in `tutorial for desktop build <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_.

Building the app
----------------

To specify to gradle where is Android SDK and Android NDK, we need to fill ``NativeApp/local.properties``.

.. code-block:: shell

  cd NativeApp
  echo "sdk.dir=$ANDROID_HOME" >> NativeApp/local.properties
  echo "ndk.dir=$ANDROID_NDK" >> NativeApp/local.properties


To build the result ``apk`` file we run:

.. code-block:: shell

  cd NativeApp
  gradle app:assembleDebug

To install the app on the connected device:

.. code-block:: shell

  cd NativeApp
  gradle app::installDebug

After that, you can run the app on the device by clicking on PyTorchNativeApp icon.
Or you can do it from the command line:

.. code-block:: shell

  adb shell am start -n org.pytorch.nativeapp/.MainActivity

If you check the android logcat:

.. code-block:: shell

  adb logcat -v brief | grep PyTorchNativeApp


You should see logs with tag 'PyTorchNativeApp' that prints x, y, and the result of the model forward, which we print with ``log`` function in ``NativeApp/app/src/main/cpp/pytorch_nativeapp.cpp``.

::

  I/PyTorchNativeApp(26968): x: -0.9484 -1.1757 -0.5832  0.9144  0.8867  1.0933 -0.4004 -0.3389
  I/PyTorchNativeApp(26968): -1.0343  1.5200 -0.7625 -1.5724 -1.2073  0.4613  0.2730 -0.6789
  I/PyTorchNativeApp(26968): -0.2247 -1.2790  1.0067 -0.9266  0.6034 -0.1941  0.7021 -1.5368
  I/PyTorchNativeApp(26968): -0.3803 -0.0188  0.2021 -0.7412 -0.2257  0.5044  0.6592  0.0826
  I/PyTorchNativeApp(26968): [ CPUFloatType{4,8} ]
  I/PyTorchNativeApp(26968): y: -1.0084  1.8733  0.5435  0.1087 -1.1066
  I/PyTorchNativeApp(26968): -1.9926  1.1047  0.5311 -0.4944  1.9178
  I/PyTorchNativeApp(26968): -1.5451  0.8867  1.0473 -1.7571  0.3909
  I/PyTorchNativeApp(26968):  0.4039  0.5085 -0.2776  0.4080  0.9203
  I/PyTorchNativeApp(26968):  0.3655  1.4395 -1.4467 -0.9837  0.3335
  I/PyTorchNativeApp(26968): -0.0445  0.8039 -0.2512 -1.3122  0.6543
  I/PyTorchNativeApp(26968): -1.5819  0.0525  1.5680 -0.6442 -1.3090
  I/PyTorchNativeApp(26968): -1.6197 -0.0773 -0.5967 -0.1105 -0.3122
  I/PyTorchNativeApp(26968): [ CPUFloatType{8,5} ]
  I/PyTorchNativeApp(26968): result:  16.0274   9.0330   6.0124   9.8644  11.0493
  I/PyTorchNativeApp(26968):   8.7633   6.9657  12.3469  10.3159  12.0683
  I/PyTorchNativeApp(26968):  12.4529   9.4559  11.7038   7.8396   6.9716
  I/PyTorchNativeApp(26968):   8.5279   9.1780  11.3849   8.4368   9.1480
  I/PyTorchNativeApp(26968):  10.0000  10.0000  10.0000  10.0000  10.0000
  I/PyTorchNativeApp(26968):  10.0000  10.0000  10.0000  10.0000  10.0000
  I/PyTorchNativeApp(26968):  10.0000  10.0000  10.0000  10.0000  10.0000
  I/PyTorchNativeApp(26968):  10.0000  10.0000  10.0000  10.0000  10.0000
  I/PyTorchNativeApp(26968): [ CPUFloatType{8,5} ]



The full setup of this app you can find in `PyTorch Android Demo Application Repository <https://github.com/pytorch/android-demo-app/tree/master/NativeApp>`_.
