(prototype) PyTorch 2 Export Post Training Quantization
================================================================
**Author**: `Jerry Zhang <https://github.com/jerryzh168>`_

This tutorial introduces the steps to do post training static quantization in
graph mode based on
`torch._export.export <https://pytorch.org/docs/main/export.html>`_. Compared
to `FX Graph Mode Quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html>`_,
this flow is expected to have significantly higher model coverage
(`88% on 14K models <https://github.com/pytorch/pytorch/issues/93667#issuecomment-1601171596>`_),
better programmability, and a simplified UX.

Exportable by `torch.export.export` is a prerequisite to use the flow, you can
find what are the constructs that's supported in `Export DB <https://pytorch.org/docs/main/generated/exportdb/index.html>`_.

The high level architecture of quantization 2 with quantizer could look like
this:

::

    float_model(Python)                          Example Input
        \                                              /
         \                                            /
    —-------------------------------------------------------
    |                        export                        |
    —-------------------------------------------------------
                                |
                        FX Graph in ATen     Backend Specific Quantizer
                                |                       /
    —--------------------------------------------------------
    |                     prepare_pt2e                      |
    —--------------------------------------------------------
                                |
                         Calibrate/Train
                                |
    —--------------------------------------------------------
    |                    convert_pt2e                       |
    —--------------------------------------------------------
                                |
                        Quantized Model
                                |
    —--------------------------------------------------------
    |                       Lowering                        |
    —--------------------------------------------------------
                                |
            Executorch, Inductor or <Other Backends>


The PyTorch 2 export quantization API looks like this:

.. code:: python

  import torch
  from torch._export import capture_pre_autograd_graph
  class M(torch.nn.Module):
     def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)

     def forward(self, x):
        return self.linear(x)


  example_inputs = (torch.randn(1, 5),)
  m = M().eval()

  # Step 1. program capture
  # NOTE: this API will be updated to torch.export API in the future, but the captured
  # result shoud mostly stay the same
  m = capture_pre_autograd_graph(m, *example_inputs)
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
  # backend developer will write their own Quantizer and expose methods to allow
  # users to express how they
  # want the model to be quantized
  quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
  m = prepare_pt2e(m, quantizer)

  # calibration omitted

  m = convert_pt2e(m)
  # we have a model with aten ops doing integer computations when possible


Motivation of PyTorch 2 Export Quantization
---------------------------------------------

In PyTorch versions prior to 2, we have FX Graph Mode Quantization that uses
`QConfigMapping <https://pytorch.org/docs/main/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html>`_
and `BackendConfig <https://pytorch.org/docs/stable/generated/torch.ao.quantization.backend_config.BackendConfig.html>`_
for customizations. ``QConfigMapping`` allows modeling users to specify how
they want their model to be quantized, ``BackendConfig`` allows backend
developers to specify the supported ways of quantization in their backend. While
that API covers most use cases relatively well, it is not fully extensible.
There are two main limitations for the current API:

* Limitation around expressing quantization intentions for complicated operator
  patterns (how an operator pattern should be observed/quantized) using existing
  objects: ``QConfig`` and ``QConfigMapping``.

* Limited support on how user can express their intention of how they want
  their model to be quantized. For example, if users want to quantize the every
  other linear in the model, or the quantization behavior has some dependency on
  the actual shape of the Tensor (for example, only observe/quantize inputs
  and outputs when the linear has a 3D input), backend developer or modeling
  users need to change the core quantization API/flow.

A few improvements could make the existing flow better:

* We use ``QConfigMapping`` and ``BackendConfig`` as separate objects,
  ``QConfigMapping`` describes user’s intention of how they want their model to
  be quantized, ``BackendConfig`` describes what kind of quantization a backend
  supports. ``BackendConfig`` is backend-specific, but ``QConfigMapping`` is not,
  and the user can provide a ``QConfigMapping`` that is incompatible with a specific
  ``BackendConfig``, this is not a great UX. Ideally, we can structure this better
  by making both configuration (``QConfigMapping``) and quantization capability
  (``BackendConfig``) backend-specific, so there will be less confusion about
  incompatibilities.
* In ``QConfig`` we are exposing observer/ ``fake_quant`` observer classes as an
  object for the user to configure quantization, this increases the things that
  the user may need to care about. For example, not only the ``dtype`` but also
  how the observation should happen, these could potentially be hidden from the
  user so that the user flow is simpler.

Here is a summary of the benefits of the new API:

- **Programmability** (addressing 1. and 2.): When a user’s quantization needs
  are not covered by available quantizers, users can build their own quantizer and
  compose it with other quantizers as mentioned above.
- **Simplified UX** (addressing 3.): Provides a single instance with which both
  backend and users interact. Thus you no longer have the user facing quantization
  config mapping to map users intent and a separate quantization config that
  backends interact with to configure what backend support. We will still have a
  method for users to query what is supported in a quantizer. With a single
  instance, composing different quantization capabilities also becomes more
  natural than previously.

  For example XNNPACK does not support ``embedding_byte``
  and we have natively support for this in ExecuTorch. Thus, if we had
  ``ExecuTorchQuantizer`` that only quantized ``embedding_byte``, then it can be
  composed with ``XNNPACKQuantizer``. (Previously, this used to be concatenating the
  two ``BackendConfig`` together and since options in ``QConfigMapping`` are not
  backend specific, user also need to figure out how to specify the configurations
  by themselves that matches the quantization capabilities of the combined
  backend. With a single quantizer instance, we can compose two quantizers and
  query the composed quantizer for capabilities, which makes it less error prone
  and cleaner, for example, ``composed_quantizer.quantization_capabilities())``.

- **Separation of concerns** (addressing 4.): As we design the quantizer API, we
  also decouple specification of quantization, as expressed in terms of ``dtype``,
  min/max (# of bits), symmetric, and so on, from the observer concept.
  Currently, the observer captures both quantization specification and how to
  observe (Histogram vs MinMax observer). Modeling users are freed from
  interacting with observer and fake quant objects with this change.

Define Helper Functions and Prepare Dataset
-------------------------------------------

We’ll start by doing the necessary imports, defining some helper functions and
prepare the data. These steps are identitcal to
`Static Quantization with Eager Mode in PyTorch <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.

To run the code in this tutorial using the entire ImageNet dataset, first
download Imagenet by following the instructions at here
`ImageNet Data <http://www.image-net.org/download>`_. Unzip the downloaded file
into the ``data_path`` folder.

Download the `torchvision resnet18 model <https://download.pytorch.org/models/resnet18-f37072fd.pth>`_
and rename it to ``data/resnet18_pretrained_float.pth``.

.. code:: python

    import os
    import sys
    import time
    import numpy as np

    import torch
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
        """
        Computes the accuracy over the k top predictions for the specified
        values of k.
        """
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

Set the model to eval mode
--------------------------

For post training quantization, we'll need to set the model to the eval mode.

.. code:: python

    model_to_quantize.eval()

Export the model with torch.export
----------------------------------

Here is how you can use ``torch.export`` to export the model:

.. code-block:: python

    from torch._export import capture_pre_autograd_graph

    example_inputs = (torch.rand(2, 3, 224, 224),)
    exported_model = capture_pre_autograd_graph(model_to_quantize, example_inputs)
    # or capture with dynamic dimensions
    # from torch._export import dynamic_dim
    # exported_model = capture_pre_autograd_graph(model_to_quantize, example_inputs, constraints=[dynamic_dim(example_inputs[0], 0)])


``capture_pre_autograd_graph`` is a short term API, it will be updated to use the offical ``torch.export`` API when that is ready.


Import the Backend Specific Quantizer and Configure how to Quantize the Model
-----------------------------------------------------------------------------

The following code snippets describes how to quantize the model:

.. code-block:: python

  from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
  )
  quantizer = XNNPACKQuantizer()
  quantizer.set_global(get_symmetric_quantization_config())

``Quantizer`` is backend specific, and each ``Quantizer`` will provide their
own way to allow users to configure their model. Just as an example, here is
the different configuration APIs supported by ``XNNPackQuantizer``:

.. code-block:: python

  quantizer.set_global(qconfig_opt)  # qconfig_opt is an optional quantization config
      .set_object_type(torch.nn.Conv2d, qconfig_opt) # can be a module type
      .set_object_type(torch.nn.functional.linear, qconfig_opt) # or torch functional op
      .set_module_name("foo.bar", qconfig_opt)

.. note::

   Check out our
   `tutorial <https://pytorch.org/tutorials/prototype/pt2e_quantizer.html>`_
   that describes how to write a new ``Quantizer``.

Prepare the Model for Post Training Quantization
----------------------------------------------------------

``prepare_pt2e`` folds ``BatchNorm`` operators into preceding ``Conv2d``
operators, and inserts observers in appropriate places in the model.

.. code-block:: python

    prepared_model = prepare_pt2e(exported_model, quantizer)
    print(prepared_model.graph)

Calibration
--------------

The calibration function is run after the observers are inserted in the model.
The purpose for calibration is to run through some sample examples that is
representative of the workload (for example a sample of the training data set)
so that the observers in themodel are able to observe the statistics of the
Tensors and we can later use this information to calculate quantization
parameters.

.. code-block:: python

    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)
    calibrate(prepared_model, data_loader_test)  # run calibration on sample data

Convert the Calibrated Model to a Quantized Model
-------------------------------------------------

``convert_pt2e`` takes a calibrated model and produces a quantized model.

.. code-block:: python

    quantized_model = convert_pt2e(prepared_model)
    print(quantized_model)

At this step, we currently have two representations that you can choose from, but exact representation
we offer in the long term might change based on feedback from PyTorch users.

* Q/DQ Representation (default)
      
  Previous documentation for `representations <https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md>`_ all quantized operators are represented as ``dequantize -> fp32_op -> qauntize``.

.. code-block:: python

   def quantized_linear(x_int8, x_scale, x_zero_point, weight_int8, weight_scale, weight_zero_point, bias_fp32, output_scale, output_zero_point):
       x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
       weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
                weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
       weight_permuted = torch.ops.aten.permute_copy.default(weight_fp32, [1, 0]);
       out_fp32 = torch.ops.aten.addmm.default(bias_fp32, x_fp32, weight_permuted)
       out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
       out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8)
       return out_i8
     
* Reference Quantized Model Representation (available in the nightly build)

  We will have a special representation for selected ops, for example, quantized linear. Other ops are represented as ``dq -> float32_op -> q`` and ``q/dq`` are decomposed into more primitive operators.
  You can get this representation by using ``convert_pt2e(..., use_reference_representation=True)``.

.. code-block:: python
   
  # Reference Quantized Pattern for quantized linear
  def quantized_linear(x_int8, x_scale, x_zero_point, weight_int8, weight_scale, weight_zero_point, bias_fp32, output_scale, output_zero_point):
      x_int16 = x_int8.to(torch.int16)
      weight_int16 = weight_int8.to(torch.int16)
      acc_int32 = torch.ops.out_dtype(torch.mm, torch.int32, (x_int16 - x_zero_point), (weight_int16 - weight_zero_point))
      bias_scale = x_scale * weight_scale
      bias_int32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)
      acc_int32 = acc_int32 + bias_int32
      acc_int32 = torch.ops.out_dtype(torch.ops.aten.mul.Scalar, torch.int32, acc_int32, x_scale * weight_scale / output_scale) + output_zero_point
      out_int8 = torch.ops.aten.clamp(acc_int32, qmin, qmax).to(torch.int8)
      return out_int8


See `here <https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/pt2e/representation/rewrite.py>`_ for the most up-to-date reference representations.


Checking Model Size and Accuracy Evaluation
----------------------------------------------

Now we can compare the size and model accuracy with baseline model.

.. code-block:: python

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


.. note::
   We can't do performance evaluation now since the model is not lowered to
   target device, it's just a representation of quantized computation in ATen
   operators.

.. note::
   The weights are still in fp32 right now, we may do constant propagation for quantize op to
   get integer weights in the future.

If you want to get better accuracy or performance,  try configuring
``quantizer`` in different ways, and each ``quantizer`` will have its own way
of configuration, so please consult the documentation for the
quantizer you are using to learn more about how you can have more control
over how to quantize a model.

Save and Load Quantized Model
---------------------------------

We'll show how to save and load the quantized model.


.. code-block:: python

    # 0. Store reference output, for example, inputs, and check evaluation accuracy:
    example_inputs = (next(iter(data_loader))[0],)
    ref = quantized_model(*example_inputs)
    top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
    print("[before serialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

    # 1. Export the model and Save ExportedProgram
    pt2e_quantized_model_file_path = saved_model_dir + "resnet18_pt2e_quantized.pth"
    # capture the model to get an ExportedProgram
    quantized_ep = torch.export.export(quantized_model, example_inputs)
    # use torch.export.save to save an ExportedProgram
    torch.export.save(quantized_ep, pt2e_quantized_model_file_path)


    # 2. Load the saved ExportedProgram
    loaded_quantized_ep = torch.export.load(pt2e_quantized_model_file_path)
    loaded_quantized_model = loaded_quantized_ep.module()

    # 3. Check results for example inputs and check evaluation accuracy again:
    res = loaded_quantized_model(*example_inputs)
    print("diff:", ref - res)
    
    top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
    print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))


Output:


.. code-block:: python
                
   [before serialization] Evaluation accuracy on test dataset: 79.82, 94.55
   diff: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.],
           ...,
           [0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.]])

   [after serialization/deserialization] Evaluation accuracy on test dataset: 79.82, 94.55


Debugging the Quantized Model
------------------------------

You can use `Numeric Suite <https://pytorch.org/docs/stable/quantization-accuracy-debugging.html#numerical-debugging-tooling-prototype>`_
that can help with debugging in eager mode and FX graph mode. The new version of
Numeric Suite working with PyTorch 2 Export models is still in development.

Lowering and Performance Evaluation
------------------------------------

The model produced at this point is not the final model that runs on the device,
it is a reference quantized model that captures the intended quantized computation
from the user, expressed as ATen operators and some additional quantize/dequantize operators,
to get a model that runs on real devices, we'll need to lower the model.
For example, for the models that run on edge devices, we can lower with delegation and ExecuTorch runtime
operators.

Conclusion
--------------

In this tutorial, we went through the overall quantization flow in PyTorch 2
Export Quantization using ``XNNPACKQuantizer`` and got a quantized model that
could be further lowered to a backend that supports inference with XNNPACK
backend. To use this for your own backend, please first follow the
`tutorial <https://pytorch.org/tutorials/prototype/pt2e_quantizer.html>`__ and
implement a ``Quantizer`` for your backend, and then quantize the model with
that ``Quantizer``.
