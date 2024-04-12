(prototype) PyTorch 2 Export Quantization-Aware Training (QAT)
================================================================
**Author**: `Andrew Or <https://github.com/andrewor14>`_

This tutorial shows how to perform quantization-aware training (QAT) in
graph mode based on `torch.export.export <https://pytorch.org/docs/main/export.html>`_.
For more details about PyTorch 2 Export Quantization in general, refer
to the `post training quantization tutorial <https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html>`_.

The PyTorch 2 Export QAT flow looks like the following—it is similar
to the post training quantization (PTQ) flow for the most part:

.. code:: python

  import torch
  from torch._export import capture_pre_autograd_graph
  from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
  )
  from torch.ao.quantization.quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
  )

  class M(torch.nn.Module):
     def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)

     def forward(self, x):
        return self.linear(x)


  example_inputs = (torch.randn(1, 5),)
  m = M()

  # Step 1. program capture
  # NOTE: this API will be updated to torch.export API in the future, but the captured
  # result shoud mostly stay the same
  m = capture_pre_autograd_graph(m, *example_inputs)
  # we get a model with aten ops

  # Step 2. quantization-aware training
  # backend developer will write their own Quantizer and expose methods to allow
  # users to express how they want the model to be quantized
  quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
  m = prepare_qat_pt2e(m, quantizer)

  # train omitted

  m = convert_pt2e(m)
  # we have a model with aten ops doing integer computations when possible

  # move the quantized model to eval mode, equivalent to `m.eval()`
  torch.ao.quantization.move_exported_model_to_eval(m)

Note that calling ``model.eval()`` or ``model.train()`` after program capture is
not allowed, because these methods no longer correctly change the behavior of
certain ops like dropout and batch normalization. Instead, please use
``torch.ao.quantization.move_exported_model_to_eval()`` and
``torch.ao.quantization.move_exported_model_to_train()`` (coming soon)
respectively.


Define Helper Functions and Prepare the Dataset
-----------------------------------------------

To run the code in this tutorial using the entire ImageNet dataset, first
download ImageNet by following the instructions in
`ImageNet Data <http://www.image-net.org/download>`_. Unzip the downloaded file
into the ``data_path`` folder.

Next, download the `torchvision resnet18 model <https://download.pytorch.org/models/resnet18-f37072fd.pth>`_
and rename it to ``data/resnet18_pretrained_float.pth``.

We’ll start by doing the necessary imports, defining some helper functions and
prepare the data. These steps are very similar to the ones defined in the
`static eager mode post training quantization tutorial <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_:

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

    def evaluate(model, criterion, data_loader, device):
        torch.ao.quantization.move_exported_model_to_eval(model)
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        cnt = 0
        with torch.no_grad():
            for image, target in data_loader:
                image = image.to(device)
                target = target.to(device)
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

    def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
        # Note: do not call model.train() here, since this doesn't work on an exported model.
        # Instead, call `torch.ao.quantization.move_exported_model_to_train(model)`, which will
        # be added in the near future
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        avgloss = AverageMeter('Loss', '1.5f')
    
        cnt = 0
        for image, target in data_loader:
            start_time = time.time()
            print('.', end = '')
            cnt += 1
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            avgloss.update(loss, image.size(0))
            if cnt >= ntrain_batches:
                print('Loss', avgloss.avg)
    
                print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5))
                return
    
        print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
              .format(top1=top1, top5=top5))
        return

    data_path = '~/.data/imagenet'
    saved_model_dir = 'data/'
    float_model_file = 'resnet18_pretrained_float.pth'

    train_batch_size = 32
    eval_batch_size = 32

    data_loader, data_loader_test = prepare_data_loaders(data_path)
    example_inputs = (next(iter(data_loader))[0])
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(saved_model_dir + float_model_file).to("cuda")


Export the model with torch.export
----------------------------------

Here is how you can use ``torch.export`` to export the model:

.. code:: python

    from torch._export import capture_pre_autograd_graph

    example_inputs = (torch.rand(2, 3, 224, 224),)
    exported_model = capture_pre_autograd_graph(float_model, example_inputs)


.. code:: python

    # or, to capture with dynamic dimensions:
    from torch._export import dynamic_dim

    example_inputs = (torch.rand(2, 3, 224, 224),)
    exported_model = capture_pre_autograd_graph(
        float_model,
        example_inputs,
        constraints=[dynamic_dim(example_inputs[0], 0)],
    )
.. note::

   ``capture_pre_autograd_graph`` is a short term API, it will be updated to use the offical ``torch.export`` API when that is ready.


Import the Backend Specific Quantizer and Configure how to Quantize the Model
-----------------------------------------------------------------------------

The following code snippets describe how to quantize the model:

.. code-block:: python

  from torch.ao.quantization.quantizer.xnnpack_quantizer import (
      XNNPACKQuantizer,
      get_symmetric_quantization_config,
  )
  quantizer = XNNPACKQuantizer()
  quantizer.set_global(get_symmetric_quantization_config(is_qat=True))

``Quantizer`` is backend specific, and each ``Quantizer`` will provide their
own way to allow users to configure their model.

.. note::

   Check out our
   `tutorial <https://pytorch.org/tutorials/prototype/pt2e_quantizer.html>`_
   that describes how to write a new ``Quantizer``.


Prepare the Model for Quantization-Aware Training
----------------------------------------------------------

``prepare_qat_pt2e`` inserts fake quantizes in appropriate places in the model
and performs the appropriate QAT "fusions", such as ``Conv2d`` + ``BatchNorm2d``,
for better training accuracies. The fused operations are represented as a subgraph
of ATen ops in the prepared graph.

.. code-block:: python

    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    print(prepared_model)

.. note::

    If your model contains batch normalization, the actual ATen ops you get
    in the graph depend on the model's device when you export the model.
    If the model is on CPU, then you'll get ``torch.ops.aten._native_batch_norm_legit``.
    If the model is on CUDA, then you'll get ``torch.ops.aten.cudnn_batch_norm``.
    However, this is not fundamental and may be subject to change in the future.

    Between these two ops, it has been shown that ``torch.ops.aten.cudnn_batch_norm``
    provides better numerics on models like MobileNetV2. To get this op, either
    call ``model.cuda()`` before export, or run the following after prepare to manually
    swap the ops:

    .. code:: python

        for n in prepared_model.graph.nodes:
            if n.target == torch.ops.aten._native_batch_norm_legit.default:
                n.target = torch.ops.aten.cudnn_batch_norm.default
        prepared_model.recompile()

    In the future, we plan to consolidate the batch normalization ops such that
    the above will no longer be necessary.

Training Loop
-----------------------------------------------------------------------------

The training loop is similar to the ones in previous versions of QAT. To achieve
better accuracies, you may optionally disable observers and updating batch
normalization statistics after a certain number of epochs, or evaluate the QAT
or the quantized model trained so far every ``N`` epochs.

.. code:: python

    num_epochs = 10
    num_train_batches = 20
    num_eval_batches = 20
    num_observer_update_epochs = 4
    num_batch_norm_update_epochs = 3
    num_epochs_between_evals = 2
    
    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    for nepoch in range(num_epochs):
        train_one_epoch(prepared_model, criterion, optimizer, data_loader, "cuda", num_train_batches)

        # Optionally disable observer/batchnorm stats after certain number of epochs
        if epoch >= num_observer_update_epochs:
            print("Disabling observer for subseq epochs, epoch = ", epoch)
            prepared_model.apply(torch.ao.quantization.disable_observer)
        if epoch >= num_batch_norm_update_epochs:
            print("Freezing BN for subseq epochs, epoch = ", epoch)
            for n in prepared_model.graph.nodes:
                # Args: input, weight, bias, running_mean, running_var, training, momentum, eps
                # We set the `training` flag to False here to freeze BN stats
                if n.target in [
                    torch.ops.aten._native_batch_norm_legit.default,
                    torch.ops.aten.cudnn_batch_norm.default,
                ]:
                    new_args = list(n.args)
                    new_args[5] = False
                    n.args = new_args
            prepared_model.recompile()
    
        # Check the quantized accuracy every N epochs
        # Note: If you wish to just evaluate the QAT model (not the quantized model),
        # then you can just call `torch.ao.quantization.move_exported_model_to_eval/train`.
        # However, the latter API is not ready yet and will be available in the near future.
        if (nepoch + 1) % num_epochs_between_evals == 0:
            prepared_model_copy = copy.deepcopy(prepared_model)
            quantized_model = convert_pt2e(prepared_model_copy)
            top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
            print('Epoch %d: Evaluation accuracy on %d images, %2.2f' % (nepoch, num_eval_batches * eval_batch_size, top1.avg))


Saving and Loading Model Checkpoints
----------------------------------------------------------

Model checkpoints for the PyTorch 2 Export QAT flow are
the same as in any other training flow. They are useful for
pausing training and resuming it later, recovering from
failed training runs, and performing inference on different
machines at a later time. You can save model checkpoints
during or after training as follows:

.. code:: python

    checkpoint_path = "/path/to/my/checkpoint_%s.pth" % nepoch
    torch.save(prepared_model.state_dict(), "checkpoint_path")

To load the checkpoints, you must export and prepare the
model the exact same way it was initially exported and
prepared. For example:

.. code:: python

    from torch._export import capture_pre_autograd_graph
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchvision.models.resnet import resnet18

    example_inputs = (torch.rand(2, 3, 224, 224),)
    float_model = resnet18(pretrained=False)
    exported_model = capture_pre_autograd_graph(float_model, example_inputs)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    prepared_model.load_state_dict(torch.load(checkpoint_path))

    # resume training or perform inference


Convert the Trained Model to a Quantized Model
----------------------------------------------------------

``convert_pt2e`` takes a calibrated model and produces a quantized model.
Note that, before inference, you must first call
``torch.ao.quantization.move_exported_model_to_eval()`` to ensure certain ops
like dropout behave correctly in the eval graph. Otherwise, we would continue
to incorrectly apply dropout in the forward pass during inference, for example.

.. code-block:: python

    quantized_model = convert_pt2e(prepared_model)

    # move certain ops like dropout to eval mode, equivalent to `m.eval()`
    torch.ao.quantization.move_exported_model_to_eval(m)

    print(quantized_model)

    top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Final evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))

.. TODO: add results here


Conclusion
--------------

In this tutorial, we demonstrated how to run Quantization-Aware Training (QAT)
flow in PyTorch 2 Export Quantization. After convert, the rest of the flow
is the same as Post-Training Quantization (PTQ); the user can
serialize/deserialize the model and further lower it to a backend that supports
inference with XNNPACK backend. For more detail, follow the
`PTQ tutorial <https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html>`_.
