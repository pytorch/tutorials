(prototype) PyTorch 2.0 Export Post Training Static Quantization
================================================================
**Author**: `Jerry Zhang <https://github.com/jerryzh168>`_

This tutorial introduces the steps to do post training static quantization in graph mode based on
`torch._export.export <https://pytorch.org/docs/main/export.html>`_. Compared to `FX Graph Mode Quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html>`, this flow is expected to have significantly higher model coverage (`88% on 14K models <https://github.com/pytorch/pytorch/issues/93667#issuecomment-1601171596>`), better programmability, and a simplified UX.

Exportable by `torch._export.export` is a prerequisite to use the flow, you can find what are the constructs that's supported in `Export DB <https://pytorch.org/docs/main/generated/exportdb/index.html>`_.

The high level architecture of quantization 2.0 with quantizer could look like this:

::

    float_model(Python)                               Input
        \                                              /
         \                                            /
    —-------------------------------------------------------
    |                    Dynamo Export                     |
    —-------------------------------------------------------
                                |
                        FX Graph in ATen     XNNPACKQuantizer,
                                |            or X86InductorQuantizer,
                                |            or <Other Backend Quantizer>
                                |                /
    —--------------------------------------------------------
    |                 prepare_pt2e                          |
    —--------------------------------------------------------
                                |
                         Calibrate/Train
                                |
    —--------------------------------------------------------
    |                      convert_pt2e                     |
    —--------------------------------------------------------
                                |
                    Reference Quantized Model
                                |
    —--------------------------------------------------------
    |                        Lowering                       |
    —--------------------------------------------------------
                                |
            Executorch, or Inductor, or <Other Backends>  


The PyTorch 2.0 export quantization API looks like this:

.. code:: python

  import torch
  class M(torch.nn.Module):
     def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)

     def forward(self, x):
        return self.linear(x)


  example_inputs = (torch.randn(1, 5),)
  m = M().eval()

  # Step 1. program capture
  m = torch._dynamo.export(m, *example_inputs, aten_graph=True)
  # we get a model with aten ops


  # Step 2. quantization
  from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
  )

  from torch.ao.quantization.quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
  )
  # backend developer will write their own Quantizer and expose methods to allow users to express how they
  # want the model to be quantized
  quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
  m = prepare_pt2e(m, quantizer)

  # calibration omitted

  m = convert_pt2e(m)
  # we have a model with aten ops doing integer computations when possible

            
1. Motivation of PyTorch 2.0 Export Quantization
------------------------------------------------

In PyTorch versions prior to 2.0, we have FX Graph Mode Quantization that uses `QConfigMapping <https://pytorch.org/docs/main/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html>`_ and `BackendConfig <https://pytorch.org/docs/stable/generated/torch.ao.quantization.backend_config.BackendConfig.html>`_ for customizations. ``QConfigMapping`` allows modeling users to specify how they want their model to be quantized, ``BackendConfig`` allows backend developers to specify the supported ways of quantization in their backend. While that API covers most use cases relatively well, it is not fully extensible. There are two main limitations for current API:

1. Limitation around expressing quantization intentions for complicated operator patterns (how an operator pattern should be observed/quantized) using existing objects: ``QConfig`` and ``QConfigMapping``.
2. Limited support on how user can express their intention of how they want their model to be quantized. For example, if users want to quantize the every other linear in the model, or the quantization behavior has some dependency on the actual shape of the Tensor (for example, only observe/quantize inputs and outputs when the linear has a 3D input), backend developer or modeling users need to change the core quantization api/flow.

A few improvements could make the existing flow better:
3. We use ``QConfigMapping`` and ``BackendConfig`` as separate objects, ``QConfigMapping`` describes user’s intention of how they want their model to be quantized, ``BackendConfig`` describes what kind of quantization a backend support. ``BackendConfig`` is backend specific, but ``QConfigMapping`` is not, and user can provide a ``QConfigMapping`` that is incompatible with a specific ``BackendConfig``, this is not a great UX. Ideally we can structure this better by making both configuration (``QConfigMapping``) and quantization capability (``BackendConfig``) backend specific, so there will be less confusion about incompatibilities.

4. In ``QConfig`` we are exposing observer/fake_quant observer classes as an object for user to configure quantization, this increases the things that user may need to care about, e.g. not only the dtype but also how the observation should happen, these could potentially be hidden from user so that the user interface is simpler.

Here is a summary of the benefits of the new API:
- Programmability (addressing 1. and 2.): When a user’s quantization needs are not covered by available quantizers, users can build their own quantizer and compose it with other quantizers as mentioned above.
- Simplified UX (addressing 3.): Provides a single instance with which both backend and users interact. Thus you no longer have 1) user facing quantization config mapping to map users intent and 2) a separate quantization config that backends interact with to configure what backend support. We will still have a method for users to query what is supported in a quantizer. With a single instance, composing different quantization capabilities also becomes more natural than previously. For example XNNPACK does not support embedding_byte and we have native support for this in ExecuTorch. Thus if we had ExecuTorchQuantizer that only quantized embedding_byte, then it can be composed with XNNPACKQuantizer. (Previously this will be concatenating the two ``BackendConfig`` together and since options in ``QConfigMapping``s are not backend specific, user also need to figure out how to specify the configurations by themselves that matches the quantization capabilities of the combined backend. with a single quantizer instance, we can compose two quantizers and query the composed quantizer for capabilities, which makes it less error prone and cleaner, e.g. composed_quantizer.quantization_capabilities())
- Separation of Concerns (addressing 4.): As we design the quantizer API, we also decouple specification of quantization, as expressed in terms of ``dtype``, min/max (# of bits), symmetric, and so on, from the observer concept. Currently, the observer captures both quantization specification and how to observe (Histogram vs MinMax observer). Modeling users are freed from interacting with observer and fake quant objects with this change.

2. Define Helper Functions and Prepare Dataset
----------------------------------------------

We’ll start by doing the necessary imports, defining some helper functions and prepare the data.
These steps are identitcal to `Static Quantization with Eager Mode in PyTorch <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.

To run the code in this tutorial using the entire ImageNet dataset, first download Imagenet by following the instructions at here `ImageNet Data <http://www.image-net.org/download>`_. Unzip the downloaded file into the ``data_path`` folder.

Download the `torchvision resnet18 model <https://download.pytorch.org/models/resnet18-f37072fd.pth>`_ and rename it to
``data/resnet18_pretrained_float.pth``.

.. code:: python

    import os
    import sys
    import time
    import numpy as np

    import torch
    from torch.ao.quantization import get_default_qconfig, QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
    import torch.nn as nn
    from torch.utils.data import DataLoader

    import torchvision
    from torchvision import datasets
    from torchvision.models.resnet import resnet18
    import torchvision.transforms as transforms

    # Set up warnings
    import warnings
    warnings.filterwarnings(
        action='ignore',
        category=DeprecationWarning,
        module=r'.*'
    )
    warnings.filterwarnings(
        action='default',
        module=r'torch.ao.quantization'
    )

    # Specify random seed for repeatable results
    _ = torch.manual_seed(191009)


    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)


    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    def evaluate(model, criterion, data_loader):
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        cnt = 0
        with torch.no_grad():
            for image, target in data_loader:
                output = model(image)
                loss = criterion(output, target)
                cnt += 1
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))
        print('')

        return top1, top5

    def load_model(model_file):
        model = resnet18(pretrained=False)
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        model.to("cpu")
        return model

    def print_size_of_model(model):
        if isinstance(model, torch.jit.RecursiveScriptModule):
            torch.jit.save(model, "temp.p")
        else:
            torch.jit.save(torch.jit.script(model), "temp.p")
        print("Size (MB):", os.path.getsize("temp.p")/1e6)
        os.remove("temp.p")

    def prepare_data_loaders(data_path):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset = torchvision.datasets.ImageNet(
            data_path, split="train", transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        dataset_test = torchvision.datasets.ImageNet(
            data_path, split="val", transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size,
            sampler=train_sampler)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=eval_batch_size,
            sampler=test_sampler)

        return data_loader, data_loader_test

    data_path = '~/.data/imagenet'
    saved_model_dir = 'data/'
    float_model_file = 'resnet18_pretrained_float.pth'

    train_batch_size = 30
    eval_batch_size = 50

    data_loader, data_loader_test = prepare_data_loaders(data_path)
    example_inputs = (next(iter(data_loader))[0])
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(saved_model_dir + float_model_file).to("cpu")
    float_model.eval()

    # create another instance of the model since
    # we need to keep the original model around
    model_to_quantize = load_model(saved_model_dir + float_model_file).to("cpu")

3. Set model to eval mode
-------------------------
For post training quantization, we'll need to set model to the eval mode.

.. code:: python

    model_to_quantize.eval()

4. Export the model with torch export
-------------------------------------

.. code:: python
    import torch._dynamo as torchdynamo

    example_inputs = (torch.rand(2, 3, 224, 224),)
    exported_model, _ = torchdynamo.export(model_to_quantize, *example_inputs, aten_graph=True, tracing_mode="symbolic")
    

5. Import the Backend Specific Quantizer and Configure how to Quantize the Model
--------------------------------------------------------------------------------

.. code:: python

  from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
  )
  quantizer = XNNPACKQuantizer()
  quantizer.set_globa(get_symmetric_quantization_config())

``Quantizer`` is backend specific, and each ``Quantizer`` will provide their own way to allow users to configure their model. Just as an example, here is the different configuration APIs supported by XNNPackQuantizer:

.. code:: python
  quantizer.set_global(qconfig_opt)  # qconfig_opt is an optional qconfig, either a valid qconfig or None
      .set_object_type(torch.nn.Conv2d, qconfig_opt) # can be a module type
      .set_object_type(torch.nn.functional.linear, qconfig_opt) # or torch functional op      
      .set_module_name("foo.bar", qconfig_opt)

We have another `tutorial <https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html>`_ that talks about how to write a new ``Quantizer``.

6. Prepare the Model for Post Training Static Quantization
----------------------------------------------------------

``prepare_pt2e`` folds ``BatchNorm`` operators into preceding ``Conv2d`` operators, and inserts observers
in appropriate places in the model.

.. code:: python

    prepared_model = prepare_pt2e(exported_model, quantizer)
    print(prepared_model.graph)

7. Calibration
--------------
The calibration function is run after the observers are inserted in the model.
The purpose for calibration is to run through some sample examples that is representative of the workload
(for example a sample of the training data set) so that the observers in the model are able to observe
the statistics of the Tensors and we can later use this information to calculate quantization parameters.

.. code:: python

    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)
    calibrate(prepared_model, data_loader_test)  # run calibration on sample data

8. Convert the Calibrated Model to a Quantized Model
----------------------------------------------------
``convert_pt2e`` takes a calibrated model and produces a quantized model.

.. code:: python

    quantized_model = convert_pt2e(prepared_model)
    print(quantized_model)

.. note::
   the model produced here also had some improvement upon the previous `representations <https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md>`_ in the FX graph mode quantizaiton, previously all quantized operators are represented as ``dequantize -> fp32_op -> qauntize``, in the new flow, we choose to represent some of the operators with integer computation so that it's closer to the computation happens in hardwares.
   For example, here is how we plan to represent a quantized linear operator:
   ```
   def quantized_linear(x_int8, x_scale, x_zero_point, weight_int8, weight_scale, weight_zero_point, bias_int32, bias_scale, bias_zero_point, output_scale, output_zero_point):
       x_int16 = x_int8.to(torch.int16)
       weight_int16 = weight_int8.to(torch.int16)
       acc_int32 = torch.ops.out_dtype(torch.mm, torch.int32, (x_int16 - x_zero_point), (weight_int16 - weight_zero_point))
       acc_rescaled_int32 = torch.ops.out_dtype(torch.ops.aten.mul.Scalar, torch.int32, acc_int32, x_scale * weight_scale / output_scale)
       bias_int32 = torch.ops.out_dtype(torch.ops.aten.mul.Scalar, bias_int32 - bias_zero_point, bias_scale / output_scale))
       out_int8 = torch.ops.aten.clamp(acc_rescaled_int32 + bias_int32 + output_zero_point, qmin, qmax).to(torch.int8)
       return out_int8
   ```
   
   For more details, please see: `Quantized Model Representation <https://docs.google.com/document/d/17h-OEtD4o_hoVuPqUFsdm5uo7psiNMY8ThN03F9ZZwg/edit>`_ (TODO: make this a public API doc/issue).
   

9. Checking Model Size and Accuracy Evaluation
----------------------------------------------
Now we can compare the size and model accuracy with baseline model.

.. code:: python

    # Baseline model size and accuracy
    scripted_float_model_file = "resnet18_scripted.pth"

    print("Size of baseline model")
    print_size_of_model(float_model)

    top1, top5 = evaluate(float_model, criterion, data_loader_test)
    print("Baseline Float Model Evaluation accuracy: %2.2f, %2.2f"%(top1.avg, top5.avg))

    # Quantized model size and accuracy
    print("Size of model after quantization")
    print_size_of_model(quantized_model)
    
    top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
    print("[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))


Note: we can't do performance evaluation now since the model is not lowered to target device, it's just a representation of quantized computation in aten operators. Each backend should have their tutorial about how to lower to their backend, for example, we'll have separate tutorials on how to do lowering in executorch for models that target edge devices.

If you want to get better accuracy or performance,  try configuring ``quantizer`` in different ways, and each ``quantizer`` will have its own way of configuration, so please consult the documentation for the
quantization you are using to learn more about how you can have more control over how to quantize a model.

10. Save and Load Quantized Model
---------------------------------
We'll show how to save and load the quantized model.

.. code:: python
    # 1. Save state_dict
    pt2e_quantized_model_file_path = saved_model_dir + "resnet18_pt2e_quantized.pth"
    torch.save(quantized_model.state_dict(), pt2e_quantized_model_file_path)

    # Get a reference output
    example_inputs = (next(iter(data_loader))[0],)
    ref = quantized_model(*example_inputs)

    # 2. Initialize the quantized model and Load state_dict
    # Rerun all steps to get a quantized model
    model_to_quantize = load_model(saved_model_dir + float_model_file).to("cpu")
    model_to_quantize.eval()
    import torch._dynamo as torchdynamo

    exported_model, _ = torchdynamo.export(model_to_quantize, *copy.deepcopy(example_inputs), aten_graph=True, tracing_mode="symbolic")
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
          XNNPACKQuantizer,
          get_symmetric_quantization_config,
    )

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model(*example_inputs)
    loaded_quantized_model = convert_pt2e(prepared_model)

    # load the state_dict from saved file to intialized model
    loaded_quantized_model.load_state_dict(torch.load(pt2e_quantized_model_file_path))

    # Sanity check with sample data
    res = loaded_quantized_model(*example_inputs)

    # 3. Evaluate the loaded quantized model
    top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
    print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

11. Debugging Quantized Model
----------------------------
We have `Numeric Suite <https://pytorch.org/docs/stable/quantization-accuracy-debugging.html#numerical-debugging-tooling-prototype>`_ that can help with debugging in eager mode and FX graph mode. The new version of Numeric Suite working with PyTorch 2.0 Export models is still in development.

12. Lowering and Performance Evaluation
---------------------------------------

The model produced at this point is not the final model that runs on device, it is a reference quantized model that captures the intended quantized computation from user, expressed as aten operators, to get a model that runs in real devices, we'll need to lower the model. For example for models that runs on edge devices, we can lower to executorch.

13. Conclusion
--------------
In this tutorial, we went through the overall quantization flow in PyTorch 2.0 Export Quantization using ``XNNPACKQuantizer`` and get a quantized model that could be further lowered to a backend that supports inference with XNNPACK backend. To use this for your own backend, please first follow the `tutorial <https://pytorch.org/tutorials/prototype/pt2e_quantizer.html>`__ and implement a ``Quantizer`` for your backend, and then quantize the model with that ``Quantizer``.
