(Prototype) Convert Mobilenetv2 to Core ML
==========================================

**Author**: `Tao Xu <https://github.com/xta0>`_

Introduction
------------

Core ML provides access to powerful and efficient NPUs(Neural Process Unit) on modern iPhone devices. This tutorial shows how to prepare a computer vision model (mobilenetv2) to use the PyTorch Core ML mobile backend. 

Note that this feature is currently in the “prototype” phase and only supports a limited numbers of operators, but we expect to solidify the integration and expand our operator support over time. The APIs are subject to change in the future.

Environment Setup (MacOS)
-------------------------

Let's start off by creating a new conda environment.

.. code:: shell

    conda create --name 1.10 python=3.8 --yes
    conda activate 1.10

Next, since the Core ML delegate is a prototype feature, let's install the PyTorch nightly build and coremltools

.. code:: shell

    pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

    pip3 install coremltools==5.0b5 protobuf==3.20.1


Model Preparation
-------------------

To convert a pre-trained mobilenetv2 model to be Core ML compatible, we're going to use the ``to_backend()`` API, which is a prototype feature for delegating model executions to some specific backends. The following python code shows how to use it to convert the mobilenetv2 torchscript model.

.. code:: python

    import torch
    import torchvision

    from torch.backends._coreml.preprocess import (
        CompileSpec,
        TensorSpec,
        CoreMLComputeUnit,
    )

    def mobilenetv2_spec():
        return {
            "forward": CompileSpec(
                inputs=(
                    TensorSpec(
                        shape=[1, 3, 224, 224],
                    ),
                ),
                outputs=(
                    TensorSpec(
                        shape=[1, 1000],
                    ),
                ),
                backend=CoreMLComputeUnit.ALL,
                allow_low_precision=True,
            ),
        }


    def main():
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.eval()
        example = torch.rand(1, 3, 224, 224)
        model = torch.jit.trace(model, example)
        compile_spec = mobilenetv2_spec()
        mlmodel = torch._C._jit_to_backend("coreml", model, compile_spec)
        mlmodel._save_for_lite_interpreter("./mobilenetv2_coreml.ptl")


    if __name__ == "__main__":
        main()


First, we need to call ``.eval()`` to set the model to inference mode. Secondly, we defined a ``mobilenetv2_spec()`` function to tell Core ML what the model looks like. Note that the ``CoreMLComputeUnit`` corresponds to `Apple's processing unit <https://developer.apple.com/documentation/coreml/mlcomputeunits>`_ whose value can be ``CPU``, ``CPUAndGPU`` and ``ALL``. In our example, we set the ``backend`` type to ``ALL`` which means Core ML will try to run the model on Neural Engine. Finally, we called the ``to_backend`` API to convert the torchscript model to a Core ML compatible model and save it to the disk.

Run the python script. If everything works well, you should see following outputs from coremltools

.. code:: shell

    Converting Frontend ==> MIL Ops: 100%|███████████████████████████████████████████████████████████████████████████████▊| 384/385 [00:00<00:00, 1496.98 ops/s]
    Running MIL Common passes:   0%|
    0/33 [00:00<?, ? passes/s]/Users/distill/anaconda3/envs/1.10/lib/python3.8/site-packages/coremltools/converters/mil/mil/passes/name_sanitization_utils.py:129: UserWarning: Output, '647', of the source model, has been renamed to 'var_647' in the Core ML model.
    warnings.warn(msg.format(var.name, new_name))
    Running MIL Common passes: 100%|███████████████████████████████████████████████████████████████████████████████████████| 33/33 [00:00<00:00, 84.16 passes/s]
    Running MIL Clean up passes: 100%|██████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 138.17 passes/s]
    Translating MIL ==> NeuralNetwork Ops: 100%|██████████████████████████████████████████████████████████████████████████| 495/495 [00:00<00:00, 1977.15 ops/s]
    [W backend_detail.cpp:376] Warning: Backend [coreml] is not available. Execution of this Module is still possible by saving and loading on a device where the backend is available. (function codegen_backend_module)

We can safely ignore the warning above, as we don't plan to run our model on desktop.

iOS app integration
---------------------

Now that the model is ready, we can integrate it to our app. We'll be using the pytorch nightly cocoapods which contains the code for executing the Core ML model. Simply add the following code to your Podfile

.. code:: shell

    pod LibTorch-Lite-Nightly

In this tutorial, we'll be reusing our `HelloWorld <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld-CoreML>`_ project. Feel free to walk through the code there.

To benchmark the latency, you can simply put the following code before and after the PyTorch ``forward`` function

.. code:: objective-c

    caffe2::Timer t;
    auto outputTensor = _impl.forward({tensor}).toTensor().cpu();
    std::cout << "forward took: " << t.MilliSeconds() << std::endl;

Conclusion
----------

In this tutorial, we demonstrated how to convert a mobilenetv2 model to a Core ML compatible model. Please be aware of that Core ML feature is still under development, new operators/models will continue to be added. APIs are subject to change in the future versions.

Thanks for reading! As always, we welcome any feedback, so please create an issue `here <https://github.com/pytorch/pytorch/issues>`_ if you have any.

Learn More
----------

- The `Mobilenetv2 <https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>`_ from Torchvision
- Information about `Core ML <https://developer.apple.com/documentation/coreml>`_
