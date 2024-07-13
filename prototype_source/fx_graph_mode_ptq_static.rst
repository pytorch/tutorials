(prototype) FX Graph Mode Post Training Static Quantization
===========================================================
**Author**: `Jerry Zhang <https://github.com/jerryzh168>`_ **Edited by**: `Charles Hernandez <https://github.com/HDCharles>`_

This tutorial introduces the steps to do post training static quantization in graph mode based on
`torch.fx <https://github.com/pytorch/pytorch/blob/master/torch/fx/__init__.py>`_.
The advantage of FX graph mode quantization is that we can perform quantization fully automatically on the model.
Although there might be some effort required to make the model compatible with FX Graph Mode Quantization (symbolically traceable with ``torch.fx``),
we'll have a separate tutorial to show how to make the part of the model we want to quantize compatible with FX Graph Mode Quantization.
We also have a tutorial for `FX Graph Mode Post Training Dynamic Quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html>`_.
tldr; The FX Graph Mode API looks like the following:

.. code:: python

  import torch
  from torch.ao.quantization import get_default_qconfig
  from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
  from torch.ao.quantization import QConfigMapping
  float_model.eval()
  # The old 'fbgemm' is still available but 'x86' is the recommended default.
  qconfig = get_default_qconfig("x86") 
  qconfig_mapping = QConfigMapping().set_global(qconfig)
  def calibrate(model, data_loader):
      model.eval()
      with torch.no_grad():
          for image, target in data_loader:
              model(image)
  example_inputs = (next(iter(data_loader))[0]) # get an example input
  prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)  # fuse modules and insert observers
  calibrate(prepared_model, data_loader_test)  # run calibration on sample data
  quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model



1. Motivation of FX Graph Mode Quantization
-------------------------------------------

Currently, PyTorch only has eager mode quantization as an alternative: `Static Quantization with Eager Mode in PyTorch <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.

We can see there are multiple manual steps involved in the eager mode quantization process, including:

- Explicitly quantize and dequantize activations-this is time consuming when floating point and quantized operations are mixed in a model.
- Explicitly fuse modules-this requires manually identifying the sequence of convolutions, batch norms and relus and other fusion patterns.
- Special handling is needed for pytorch tensor operations (like add, concat etc.)
- Functionals did not have first class support (functional.conv2d and functional.linear would not get quantized)

Most of these required modifications comes from the underlying limitations of eager mode quantization. Eager mode works in module level since it can not inspect the code that is actually run (in the forward function), quantization is achieved by module swapping, and we don’t know how the modules are used in forward function in eager mode, so it requires users to insert QuantStub and DeQuantStub manually to mark the points they want to quantize or dequantize.
In graph mode, we can inspect the actual code that’s been executed in forward function (e.g. aten function calls) and quantization is achieved by module and graph manipulations. Since graph mode has full visibility of the code that is run, our tool is able to automatically figure out things like which modules to fuse and where to insert observer calls, quantize/dequantize functions etc., we are able to automate the whole quantization process.

Advantages of FX Graph Mode Quantization are:

- Simple quantization flow, minimal manual steps
- Unlocks the possibility of doing higher level optimizations like automatic precision selection

2. Define Helper Functions and Prepare Dataset
----------------------------------------------

We’ll start by doing the necessary imports, defining some helper functions and prepare the data.
These steps are identitcal to `Static Quantization with Eager Mode in PyTorch <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.

To run the code in this tutorial using the entire ImageNet dataset, first download imagenet by following the instructions at here `ImageNet Data <http://www.image-net.org/download>`_. Unzip the downloaded file into the 'data_path' folder.

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
For post training quantization, we'll need to set model to eval mode.

.. code:: python

    model_to_quantize.eval()


4. Specify how to quantize the model with ``QConfigMapping``
------------------------------------------------------------

.. code:: python

  qconfig_mapping = QConfigMapping.set_global(default_qconfig)

We use the same qconfig used in eager mode quantization, ``qconfig`` is just a named tuple
of the observers for activation and weight. ``QConfigMapping`` contains mapping information from ops to qconfigs:

.. code:: python

  qconfig_mapping = (QConfigMapping()
      .set_global(qconfig_opt)  # qconfig_opt is an optional qconfig, either a valid qconfig or None
      .set_object_type(torch.nn.Conv2d, qconfig_opt)  # can be a callable...
      .set_object_type("reshape", qconfig_opt)  # ...or a string of the method
      .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig_opt) # matched in order, first match takes precedence
      .set_module_name("foo.bar", qconfig_opt)
      .set_module_name_object_type_order()
  )
      # priority (in increasing order): global, object_type, module_name_regex, module_name
      # qconfig == None means fusion and quantization should be skipped for anything
      # matching the rule (unless a higher priority match is found)


Utility functions related to ``qconfig`` can be found in the `qconfig <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/qconfig.py>`_ file
while those for ``QConfigMapping`` can be found in the `qconfig_mapping <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/qconfig_mapping.py>`

.. code:: python

    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    qconfig = get_default_qconfig("x86") 
    qconfig_mapping = QConfigMapping().set_global(qconfig)

5. Prepare the Model for Post Training Static Quantization
----------------------------------------------------------

.. code:: python

    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)

prepare_fx folds BatchNorm modules into previous Conv2d modules, and insert observers
in appropriate places in the model.

.. code:: python

    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    print(prepared_model.graph)

6. Calibration
--------------
Calibration function is run after the observers are inserted in the model.
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

7. Convert the Model to a Quantized Model
-----------------------------------------
``convert_fx`` takes a calibrated model and produces a quantized model.

.. code:: python

    quantized_model = convert_fx(prepared_model)
    print(quantized_model)

8. Evaluation
-------------
We can now print the size and accuracy of the quantized model.

.. code:: python

    print("Size of model before quantization")
    print_size_of_model(float_model)
    print("Size of model after quantization")
    print_size_of_model(quantized_model)
    top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
    print("[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

    fx_graph_mode_model_file_path = saved_model_dir + "resnet18_fx_graph_mode_quantized.pth"

    # this does not run due to some erros loading convrelu module:
    # ModuleAttributeError: 'ConvReLU2d' object has no attribute '_modules'
    # save the whole model directly
    # torch.save(quantized_model, fx_graph_mode_model_file_path)
    # loaded_quantized_model = torch.load(fx_graph_mode_model_file_path)

    # save with state_dict
    # torch.save(quantized_model.state_dict(), fx_graph_mode_model_file_path)
    # import copy
    # model_to_quantize = copy.deepcopy(float_model)
    # prepared_model = prepare_fx(model_to_quantize, {"": qconfig})
    # loaded_quantized_model = convert_fx(prepared_model)
    # loaded_quantized_model.load_state_dict(torch.load(fx_graph_mode_model_file_path))

    # save with script
    torch.jit.save(torch.jit.script(quantized_model), fx_graph_mode_model_file_path)
    loaded_quantized_model = torch.jit.load(fx_graph_mode_model_file_path)

    top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
    print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

If you want to get better accuracy or performance,  try changing the `qconfig_mapping`.
We plan to add support for graph mode in the Numerical Suite so that you can
easily determine the sensitivity towards quantization of different modules in a model. For more information, see `PyTorch Numeric Suite Tutorial <https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html>`_

9. Debugging Quantized Model
----------------------------
We can also print the weight for quantized a non-quantized convolution op to see the difference,
we'll first call fuse explicitly to fuse the convolution and batch norm in the model:
Note that ``fuse_fx`` only works in eval mode.

.. code:: python

    fused = fuse_fx(float_model)

    conv1_weight_after_fuse = fused.conv1[0].weight[0]
    conv1_weight_after_quant = quantized_model.conv1.weight().dequantize()[0]

    print(torch.max(abs(conv1_weight_after_fuse - conv1_weight_after_quant)))

10. Comparison with Baseline Float Model and Eager Mode Quantization
--------------------------------------------------------------------

.. code:: python

    scripted_float_model_file = "resnet18_scripted.pth"

    print("Size of baseline model")
    print_size_of_model(float_model)

    top1, top5 = evaluate(float_model, criterion, data_loader_test)
    print("Baseline Float Model Evaluation accuracy: %2.2f, %2.2f"%(top1.avg, top5.avg))
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

In this section, we compare the model quantized with FX graph mode quantization with the model
quantized in eager mode. FX graph mode and eager mode produce very similar quantized models,
so the expectation is that the accuracy and speedup are similar as well.

.. code:: python

    print("Size of Fx graph mode quantized model")
    print_size_of_model(quantized_model)
    top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
    print("FX graph mode quantized model Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

    from torchvision.models.quantization.resnet import resnet18
    eager_quantized_model = resnet18(pretrained=True, quantize=True).eval()
    print("Size of eager mode quantized model")
    eager_quantized_model = torch.jit.script(eager_quantized_model)
    print_size_of_model(eager_quantized_model)
    top1, top5 = evaluate(eager_quantized_model, criterion, data_loader_test)
    print("eager mode quantized model Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
    eager_mode_model_file = "resnet18_eager_mode_quantized.pth"
    torch.jit.save(eager_quantized_model, saved_model_dir + eager_mode_model_file)

We can see that the model size and accuracy of FX graph mode and eager mode quantized model are pretty similar.

Running the model in AIBench (with single threading) gives the following result:

.. code::

  Scripted Float Model:
  Self CPU time total: 192.48ms

  Scripted Eager Mode Quantized Model:
  Self CPU time total: 50.76ms

  Scripted FX Graph Mode Quantized Model:
  Self CPU time total: 50.63ms

As we can see for resnet18 both FX graph mode and eager mode quantized model get similar speedup over the floating point model,
which is around 2-4x faster than the floating point model. But the actual speedup over floating point model may vary
depending on model, device, build, input batch sizes, threading etc.
