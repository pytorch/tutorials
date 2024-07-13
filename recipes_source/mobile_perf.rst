Pytorch Mobile Performance Recipes
==================================

Introduction
----------------
Performance (aka latency) is crucial to most, if not all,
applications and use-cases of ML model inference on mobile devices.

Today, PyTorch executes the models on the CPU backend pending availability
of other hardware backends such as GPU, DSP, and NPU.

In this recipe, you will learn:

- How to optimize your model to help decrease execution time (higher performance, lower latency) on the mobile device.
- How to benchmark (to check if optimizations helped your use case).


Model preparation
-----------------

We will start with preparing to optimize your model to help decrease execution time
(higher performance, lower latency) on the mobile device.


Setup
^^^^^^^

First we need to installed pytorch using conda or pip with version at least 1.5.0.

::

   conda install pytorch torchvision -c pytorch

or

::

   pip install torch torchvision

Code your model:

::

  import torch
  from torch.utils.mobile_optimizer import optimize_for_mobile

  class AnnotatedConvBnReLUModel(torch.nn.Module):
      def __init__(self):
          super(AnnotatedConvBnReLUModel, self).__init__()
          self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
          self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
          self.relu = torch.nn.ReLU(inplace=True)
          self.quant = torch.quantization.QuantStub()
          self.dequant = torch.quantization.DeQuantStub()

      def forward(self, x):
          x = x.contiguous(memory_format=torch.channels_last)
          x = self.quant(x)
          x = self.conv(x)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dequant(x)
          return x

  model = AnnotatedConvBnReLUModel()


``torch.quantization.QuantStub`` and ``torch.quantization.DeQuantStub()`` are no-op stubs, which will be used for quantization step.


1. Fuse operators using ``torch.quantization.fuse_modules``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do not be confused that fuse_modules is in the quantization package.
It works for all ``torch.nn.Module``.

``torch.quantization.fuse_modules`` fuses a list of modules into a single module.
It fuses only the following sequence of modules:

- Convolution, Batch normalization
- Convolution, Batch normalization, Relu
- Convolution, Relu
- Linear, Relu

This script will fuse Convolution, Batch Normalization and Relu in previously declared model.

::

  torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)


2. Quantize your model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can find more about PyTorch quantization in
`the dedicated tutorial <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/>`_.

Quantization of the model not only moves computation to int8,
but also reduces the size of your model on a disk.
That size reduction helps to reduce disk read operations during the first load of the model and decreases the amount of RAM.
Both of those resources can be crucial for the performance of mobile applications.
This code does quantization, using stub for model calibration function, you can find more about it `here <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization>`__.

::

  model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
  torch.quantization.prepare(model, inplace=True)
  # Calibrate your model
  def calibrate(model, calibration_data):
      # Your calibration code here
      return
  calibrate(model, [])
  torch.quantization.convert(model, inplace=True)



3. Use torch.utils.mobile_optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Torch mobile_optimizer package does several optimizations with the scripted model,
which will help to conv2d and linear operations.
It pre-packs model weights in an optimized format and fuses ops above with relu
if it is the next operation.

First we script the result model from previous step:

::

  torchscript_model = torch.jit.script(model)

Next we call ``optimize_for_mobile`` and save model on the disk.

::

  torchscript_model_optimized = optimize_for_mobile(torchscript_model)
  torch.jit.save(torchscript_model_optimized, "model.pt")

4. Prefer Using Channels Last Tensor memory format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Channels Last(NHWC) memory format was introduced in PyTorch 1.4.0. It is supported only for four-dimensional tensors. This memory format gives a better memory locality for most operators, especially convolution. Our measurements showed a 3x speedup of MobileNetV2 model compared with the default Channels First(NCHW) format.

At the moment of writing this recipe, PyTorch Android java API does not support using inputs in Channels Last memory format. But it can be used on the TorchScript model level, by adding the conversion to it for model inputs.

.. code-block:: python

  def forward(self, x):
      x = x.contiguous(memory_format=torch.channels_last)
      ...


This conversion is zero cost if your input is already in Channels Last memory format. After it, all operators will work preserving ChannelsLast memory format.

5. Android - Reusing tensors for forward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This part of the recipe is Android only.

Memory is a critical resource for android performance, especially on old devices.
Tensors can need a significant amount of memory.
For example, standard computer vision tensor contains 1*3*224*224 elements,
assuming that data type is float and will need 588Kb of memory.

::

  FloatBuffer buffer = Tensor.allocateFloatBuffer(1*3*224*224);
  Tensor tensor = Tensor.fromBlob(buffer, new long[]{1, 3, 224, 224});


Here we allocate native memory as ``java.nio.FloatBuffer`` and creating ``org.pytorch.Tensor`` which storage will be pointing to the memory of the allocated buffer.

For most of the use cases, we do not do model forward only once, repeating it with some frequency or as fast as possible.

If we are doing new memory allocation for every module forward - that will be suboptimal.
Instead of this, we can reuse the same memory that we allocated on the previous step, fill it with new data, and run module forward again on the same tensor object.

You can check how it looks in code in `pytorch android application example <https://github.com/pytorch/android-demo-app/blob/master/PyTorchDemoApp/app/src/main/java/org/pytorch/demo/vision/ImageClassificationActivity.java#L174>`_.

::

  protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
    if (mModule == null) {
      mModule = Module.load(moduleFileAbsoluteFilePath);
      mInputTensorBuffer =
      Tensor.allocateFloatBuffer(3 * 224 * 224);
      mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3, 224, 224});
    }

    TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
        image.getImage(), rotationDegrees,
        224, 224,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
        TensorImageUtils.TORCHVISION_NORM_STD_RGB,
        mInputTensorBuffer, 0);

    Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
  }

Member fields ``mModule``, ``mInputTensorBuffer`` and ``mInputTensor`` are initialized only once
and buffer is refilled using ``org.pytorch.torchvision.TensorImageUtils.imageYUV420CenterCropToFloatBuffer``.

6. Load time optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Available since Pytorch 1.13**

PyTorch Mobile also supports a FlatBuffer-based file format that is faster
to load. Both flatbuffer and pickle-based model file can be load with the
same ``_load_for_lite_interpreter`` (Python) or ``_load_for_mobile``(C++) API.

To use the FlatBuffer format, instead of creating the model file with
``model._save_for_lite_interpreter('path/to/file.ptl')``, you can run the following command:


One can save using

::

  model._save_for_lite_interpreter('path/to/file.ptl', _use_flatbuffer=True)


The extra argument ``_use_flatbuffer`` makes a FlatBuffer file instead of a
zip file. The created file will be faster to load.

For example, using ResNet-50 and running the following script:

::

  import torch
  from torch.jit import mobile
  import time
  model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
  model.eval()
  jit_model = torch.jit.script(model)

  jit_model._save_for_lite_interpreter('/tmp/jit_model.ptl')
  jit_model._save_for_lite_interpreter('/tmp/jit_model.ff', _use_flatbuffer=True)

  import timeit
  print('Load ptl file:')
  print(timeit.timeit('from torch.jit import mobile; mobile._load_for_lite_interpreter("/tmp/jit_model.ptl")',
                         number=20))
  print('Load flatbuffer file:')
  print(timeit.timeit('from torch.jit import mobile; mobile._load_for_lite_interpreter("/tmp/jit_model.ff")',
                         number=20))



you would get the following result: 

::

  Load ptl file:
  0.5387594579999999
  Load flatbuffer file:
  0.038842832999999466

While speed ups on actual mobile devices will be smaller, you can still expect
3x - 6x load time reductions.

### Reasons to avoid using a FlatBuffer-based mobile model

However, FlatBuffer format also has some limitations that you might want to consider:

* It is only available in PyTorch 1.13 or later. Therefore, client devices compiled
  with earlier PyTorch versions might not be able to load it.
* The Flatbuffer library imposes a 4GB limit for file sizes. So it is not suitable
  for large models.

Benchmarking
------------

The best way to benchmark (to check if optimizations helped your use case) - is to measure your particular use case that you want to optimize, as performance behavior can vary in different environments.

PyTorch distribution provides a way to benchmark naked binary that runs the model forward,
this approach can give more stable measurements rather than testing inside the application.


Android - Benchmarking Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This part of the recipe is Android only.

For this you first need to build benchmark binary:

::

    <from-your-root-pytorch-dir>
    rm -rf build_android
    BUILD_PYTORCH_MOBILE=1 ANDROID_ABI=arm64-v8a ./scripts/build_android.sh -DBUILD_BINARY=ON

You should have arm64 binary at: ``build_android/bin/speed_benchmark_torch``.
This binary takes ``--model=<path-to-model>``, ``--input_dim="1,3,224,224"`` as dimension information for the input and ``--input_type="float"`` as the type of the input as arguments.

Once you have your android device connected,
push speedbenchark_torch binary and your model to the phone:

::

  adb push <speedbenchmark-torch> /data/local/tmp
  adb push <path-to-scripted-model> /data/local/tmp


Now we are ready to benchmark your model:

::

  adb shell "/data/local/tmp/speed_benchmark_torch --model=/data/local/tmp/model.pt" --input_dims="1,3,224,224" --input_type="float"
  ----- output -----
  Starting benchmark.
  Running warmup runs.
  Main runs.
  Main run finished. Microseconds per iter: 121318. Iters per second: 8.24281


iOS - Benchmarking Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For iOS, we'll be using our `TestApp <https://github.com/pytorch/pytorch/tree/master/ios/TestApp>`_ as the benchmarking tool.

To begin with, let's apply the ``optimize_for_mobile`` method to our python script located at `TestApp/benchmark/trace_model.py <https://github.com/pytorch/pytorch/blob/master/ios/TestApp/benchmark/trace_model.py>`_. Simply modify the code as below.

::

  import torch
  import torchvision
  from torch.utils.mobile_optimizer import optimize_for_mobile

  model = torchvision.models.mobilenet_v2(pretrained=True)
  model.eval()
  example = torch.rand(1, 3, 224, 224)
  traced_script_module = torch.jit.trace(model, example)
  torchscript_model_optimized = optimize_for_mobile(traced_script_module)
  torch.jit.save(torchscript_model_optimized, "model.pt")

Now let's run ``python trace_model.py``. If everything works well, we should be able to generate our optimized model in the benchmark directory.

Next, we're going to build the PyTorch libraries from source.

::

  BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 ./scripts/build_ios.sh

Now that we have the optimized model and PyTorch ready, it's time to generate our XCode project and do benchmarking. To do that, we'll be using a ruby script - `setup.rb` which does the heavy lifting jobs of setting up the XCode project.

::

  ruby setup.rb

Now open the `TestApp.xcodeproj` and plug in your iPhone, you're ready to go. Below is an example result from iPhoneX

::

  TestApp[2121:722447] Main runs
  TestApp[2121:722447] Main run finished. Milliseconds per iter: 28.767
  TestApp[2121:722447] Iters per second: : 34.762
  TestApp[2121:722447] Done.
