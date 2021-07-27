Fuse Modules Recipe
=====================================

This recipe demonstrates how to fuse a list of PyTorch modules into a single module and how to do the performance test to compare the fused model with its non-fused version.

Introduction
------------

Before quantization is applied to a model to reduce its size and memory footprint (see `Quantization Recipe <quantization.html>`_ for details on quantization), the list of modules in the model may be fused first into a single module. Fusion is optional, but it may save on memory access, make the model run faster, and improve its accuracy.


Pre-requisites
--------------

PyTorch 1.6.0 or 1.7.0

Steps
--------------

Follow the steps below to fuse an example model, quantize it, script it, optimize it for mobile, save it and test it with the Android benchmark tool.

1. Define the Example Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the same example model defined in the `PyTorch Mobile Performance Recipes <https://pytorch.org/tutorials/recipes/mobile_perf.html>`_:

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


2. Generate Two Models with and without `fuse_modules`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the following code below the model definition above and run the script:

::

    model = AnnotatedConvBnReLUModel()
    print(model)

    def prepare_save(model, fused):
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        torchscript_model = torch.jit.script(model)
        torchscript_model_optimized = optimize_for_mobile(torchscript_model)
        torch.jit.save(torchscript_model_optimized, "model.pt" if not fused else "model_fused.pt")

    prepare_save(model, False)

    model = AnnotatedConvBnReLUModel()
    model_fused = torch.quantization.fuse_modules(model, [['bn', 'relu']], inplace=False)
    print(model_fused)

    prepare_save(model_fused, True)


The graphs of the original model and its fused version will be printed as follows:

::

    AnnotatedConvBnReLUModel(
      (conv): Conv2d(3, 5, kernel_size=(3, 3), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (quant): QuantStub()
      (dequant): DeQuantStub()
    )

    AnnotatedConvBnReLUModel(
      (conv): Conv2d(3, 5, kernel_size=(3, 3), stride=(1, 1), bias=False)
      (bn): BNReLU2d(
        (0): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): ReLU(inplace=True)
      )
      (relu): Identity()
      (quant): QuantStub()
      (dequant): DeQuantStub()
    )

In the second fused model output, the first item `bn` in the list is replaced with the fused module, and the rest of the modules (`relu` in this example) is replaced with identity. In addition, the non-fused and fused versions of the model `model.pt` and `model_fused.pt` are generated.

3. Build the Android benchmark Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the PyTorch source and build the Android benchmark tool as follows:

::

    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    git submodule update --init --recursive
    BUILD_PYTORCH_MOBILE=1 ANDROID_ABI=arm64-v8a ./scripts/build_android.sh -DBUILD_BINARY=ON


This will generate the Android benchmark binary `speed_benchmark_torch` in the `build_android/bin` folder.

4. Test Compare the Fused and Non-Fused Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Connect your Android device, then copy `speed_benchmark_torch` and the model files and run the benchmark tool on them:

::

    adb push build_android/bin/speed_benchmark_torch /data/local/tmp
    adb push model.pt /data/local/tmp
    adb push model_fused.pt /data/local/tmp
    adb shell "/data/local/tmp/speed_benchmark_torch --model=/data/local/tmp/model.pt" --input_dims="1,3,224,224" --input_type="float"
    adb shell "/data/local/tmp/speed_benchmark_torch --model=/data/local/tmp/model_fused.pt" --input_dims="1,3,224,224" --input_type="float"


The results from the last two commands should be like:

::

    Main run finished. Microseconds per iter: 6189.07. Iters per second: 161.575

and

::

    Main run finished. Microseconds per iter: 6216.65. Iters per second: 160.858

For this example model, there is no much performance difference between the fused and non-fused models. But the similar steps can be used to fuse and prepare a real deep model and test to see the performance improvement. Keep in mind that currently `torch.quantization.fuse_modules` only fuses the following sequence of modules:

* conv, bn
* conv, bn, relu
* conv, relu
* linear, relu
* bn, relu

If any other sequence list is provided to the `fuse_modules` call, it will simply be ignored.

Learn More
---------------

See `here <https://pytorch.org/docs/stable/quantization.html#preparing-model-for-quantization>`_ for the official documentation of `torch.quantization.fuse_modules`.
