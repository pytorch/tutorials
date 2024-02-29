(Beta) Convert MobileNetV2 to NNAPI
========================================

Introduction
------------

This tutorial shows how to prepare a computer vision model to use
`Android's Neural Networks API (NNAPI) <https://developer.android.com/ndk/guides/neuralnetworks>`_.
NNAPI provides access to powerful and efficient computational cores
on many modern Android devices.

PyTorch's NNAPI is currently in the "prototype" phase and only supports
a limited range of operators, but we expect to solidify the integration
and expand our operator support over time.


Environment
-----------

Install PyTorch and torchvision.

``pip install torch==1.10.0 torchvision==0.11.1``


Model Preparation
-----------------

First, we must prepare our model to execute with NNAPI.
This step runs on your training server or laptop.
The key conversion function to call is
``torch.backends._nnapi.prepare.convert_model_to_nnapi``,
but some extra steps are required to ensure that
the model is properly structured.
Most notably, quantizing the model is required
in order to run the model on certain accelerators.

You can copy/paste this entire Python script and run it,
or make your own modifications.
By default, it will save the models to ``~/mobilenetv2-nnapi/``.
Please create that directory first.

.. code:: python

    #!/usr/bin/env python
    import sys
    import os
    import torch
    import torch.utils.bundled_inputs
    import torch.utils.mobile_optimizer
    import torch.backends._nnapi.prepare
    import torchvision.models.quantization.mobilenet
    from pathlib import Path


    # This script supports 3 modes of quantization:
    # - "none": Fully floating-point model.
    # - "core": Quantize the core of the model, but wrap it a
    #    quantizer/dequantizer pair, so the interface uses floating point.
    # - "full": Quantize the model, and use quantized tensors
    #   for input and output.
    #
    # "none" maintains maximum accuracy
    # "core" sacrifices some accuracy for performance,
    # but maintains the same interface.
    # "full" maximized performance (with the same accuracy as "core"),
    # but requires the application to use quantized tensors.
    #
    # There is a fourth option, not supported by this script,
    # where we include the quant/dequant steps as NNAPI operators.
    def make_mobilenetv2_nnapi(output_dir_path, quantize_mode):
        quantize_core, quantize_iface = {
            "none": (False, False),
            "core": (True, False),
            "full": (True, True),
        }[quantize_mode]

        model = torchvision.models.quantization.mobilenet.mobilenet_v2(pretrained=True, quantize=quantize_core)
        model.eval()

        # Fuse BatchNorm operators in the floating point model.
        # (Quantized models already have this done.)
        # Remove dropout for this inference-only use case.
        if not quantize_core:
            model.fuse_model()
        assert type(model.classifier[0]) == torch.nn.Dropout
        model.classifier[0] = torch.nn.Identity()

        input_float = torch.zeros(1, 3, 224, 224)
        input_tensor = input_float

        # If we're doing a quantized model, we need to trace only the quantized core.
        # So capture the quantizer and dequantizer, use them to prepare the input,
        # and replace them with identity modules so we can trace without them.
        if quantize_core:
            quantizer = model.quant
            dequantizer = model.dequant
            model.quant = torch.nn.Identity()
            model.dequant = torch.nn.Identity()
            input_tensor = quantizer(input_float)

        # Many NNAPI backends prefer NHWC tensors, so convert our input to channels_last,
        # and set the "nnapi_nhwc" attribute for the converter.
        input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
        input_tensor.nnapi_nhwc = True

        # Trace the model.  NNAPI conversion only works with TorchScript models,
        # and traced models are more likely to convert successfully than scripted.
        with torch.no_grad():
            traced = torch.jit.trace(model, input_tensor)
        nnapi_model = torch.backends._nnapi.prepare.convert_model_to_nnapi(traced, input_tensor)

        # If we're not using a quantized interface, wrap a quant/dequant around the core.
        if quantize_core and not quantize_iface:
            nnapi_model = torch.nn.Sequential(quantizer, nnapi_model, dequantizer)
            model.quant = quantizer
            model.dequant = dequantizer
            # Switch back to float input for benchmarking.
            input_tensor = input_float.contiguous(memory_format=torch.channels_last)

        # Optimize the CPU model to make CPU-vs-NNAPI benchmarks fair.
        model = torch.utils.mobile_optimizer.optimize_for_mobile(torch.jit.script(model))

        # Bundle sample inputs with the models for easier benchmarking.
        # This step is optional.
        class BundleWrapper(torch.nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.mod = mod
            def forward(self, arg):
                return self.mod(arg)
        nnapi_model = torch.jit.script(BundleWrapper(nnapi_model))
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            nnapi_model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])

        # Save both models.
        model._save_for_lite_interpreter(str(output_dir_path / ("mobilenetv2-quant_{}-cpu.pt".format(quantize_mode))))
        nnapi_model._save_for_lite_interpreter(str(output_dir_path / ("mobilenetv2-quant_{}-nnapi.pt".format(quantize_mode))))


    if __name__ == "__main__":
        for quantize_mode in ["none", "core", "full"]:
            make_mobilenetv2_nnapi(Path(os.environ["HOME"]) / "mobilenetv2-nnapi", quantize_mode)


Running Benchmarks
------------------

Now that the models are ready, we can benchmark them on our Android devices.
See `our performance recipe <https://pytorch.org/tutorials/recipes/mobile_perf.html#android-benchmarking-setup>`_ for details.
The best-performing models are likely to be the "fully-quantized" models:
``mobilenetv2-quant_full-cpu.pt`` and ``mobilenetv2-quant_full-nnapi.pt``.

Because these models have bundled inputs, we can run the benchmark as follows:

.. code:: shell

   ./speed_benchmark_torch --pthreadpool_size=1 --model=mobilenetv2-quant_full-nnapi.pt --use_bundled_input=0 --warmup=5 --iter=200

Adjusting increasing the thread pool size can can reduce latency,
at the cost of increased CPU usage.
Omitting that argument will use one thread per big core.
The CPU models can get improved performance (at the cost of memory usage)
by passing ``--use_caching_allocator=true``.


Running model on host
---------------------

We can now run models on your linux machine using the reference implementation
of NNAPI. You need to build the NNAPI library from Android source code:

* Make sure you have at least 200GB of disk space
* Follow `these instructions <https://source.android.com/setup/develop#installing-repo>`_ to install ``repo`` 

.. code:: shell

    mkdir ~/android-nnapi && cd ~/android-nnapi
    repo init -u https://android.googlesource.com/platform/manifest -b master
    repo sync --network-only -j 16
    repo sync -l
    . build/envsetup.sh
    lunch aosp_x86_64-eng
    mm -j16 out/host/linux-x86/lib64/libneuralnetworks.so


With the host build of ``libneuralnetworks.so`` you can run Pytorch NNAPI models on
your linux machine:

.. code:: python

    #!/usr/bin/env python
    import ctypes
    import torch
    from pathlib import Path

    ctypes.cdll.LoadLibrary(Path.home() / "android-nnapi/out/host/linux-x86/lib64/libneuralnetworks.so")
    model = torch.jit.load(Path.home() / "mobilenetv2-nnapi/mobilenetv2-quant_full-nnapi.pt")
    print(model(*model.get_all_bundled_inputs()[0]))


Integration
-----------

The converted models are ordinary TorchScript models.
You can use them in your app just like any other PyTorch model.
See `https://pytorch.org/mobile/android/ <https://pytorch.org/mobile/android/>`_
for an introduction to using PyTorch on Android.


Learn More
----------

- Learn more about optimization in our
  `Mobile Performance Recipe <https://pytorch.org/tutorials/recipes/mobile_perf.html>`_
- `MobileNetV2 <https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>`_ from torchvision
- Information about `NNAPI <https://developer.android.com/ndk/guides/neuralnetworks>`_
