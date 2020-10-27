"""
(prototype) Graph Mode Post Training Static Quantization in PyTorch
=========================================================

**Author**: `Jerry Zhang <https://github.com/jerryzh168>`_

This tutorial introduces the steps to do post training static quantization in graph mode. 
The advantage of graph mode quantization is that as long as the model can be scripted or traced, 
we can perform quantization fully automatically on the model. 
Right now we can do post training static and post training dynamic quantization 
and quantization aware training support will come later. 
We have a separate tutorial for `Graph Mode Post Training Dynamic Quantization <https://pytorch.org/tutorials/prototype_source/graph_mode_dynamic_bert_tutorial.html>`_.

tldr; The graph mode API looks like the following:

.. code:: python

    import torch
    from torch.quantization import get_default_qconfig, quantize_jit
    
    ts_model = torch.jit.script(float_model.eval()) # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig('fbgemm')
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)
    quantized_model = quantize_jit(
        ts_model, # TorchScript model
        {'': qconfig}, # qconfig dict
        calibrate, # calibration function
        [data_loader_test]) # positional arguments to calibration function, typically some sample dataset

"""
######################################################################
# 1. Motivation of Graph Mode Quantization
# ---------------------
# Currently PyTorch only has eager mode quantization: `Static Quantization with Eager Mode in PyTorch <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.
#
# We can see there are multiple manual steps involved in the process, including:
#
# - Explicitly quantize and dequantize activations, this is time consuming when floating point and quantized operations are mixed in a model.
# - Explicitly fuse modules, this requires manually identifying the sequence of convolutions, batch norms and relus and other fusion patterns.
# - Special handling is needed for pytorch tensor operations (like add, concat etc.)
# - Functionals did not have first class support (functional.conv2d and functional.linear would not get quantized)
#
# Most of these required modifications comes from the underlying limitations of eager mode quantization. Eager mode works in module level since it can not inspect the code that is actually run (in the forward function), quantization is achieved by module swapping, and we don’t know how the modules are used in forward function in eager mode, so it requires users to insert QuantStub and DeQuantStub manually to mark the points they want to quantize or dequantize. 
# In graph mode, we can inspect the actual code that’s been executed in forward function (e.g. aten function calls) and quantization is achieved by module and graph manipulations. Since graph mode has full visibility of the code that is run, our tool is able to automatically figure out things like which modules to fuse and where to insert observer calls, quantize/dequantize functions etc., we are able to automate the whole quantization process.
#
# Advantages of graph mode quantization are:
# 
# - Simple quantization flow, minimal manual steps
# - Unlocks the possibility of doing higher level optimizations like automatic precision selection
#
# Limitations of graph mode quantization is that quantization is configurable only at the level of module and the set of operators that are quantized is not configurable by user currently.
#
# 2. Define Helper Functions and Prepare Dataset
# ---------------------
# We’ll start by doing the necessary imports, defining some helper functions and prepare the data. 
# These steps are identitcal to `Static Quantization with Eager Mode in PyTorch <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.    
#
# Download dataset:
#
# .. code::
#
#     wget https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip
#
# and unzip to `data` folder.
# Download the `torchvision resnet18 model <https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L12>`_ and rename it to
# ``data/resnet18_pretrained_float.pth``.


import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization

# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)


from torchvision.models.resnet import resnet18
from torch.quantization import get_default_qconfig, quantize_jit

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
    model.to('cpu')
    return model

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def prepare_data_loaders(data_path):

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
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

data_path = 'data/imagenet_1k'
saved_model_dir = 'data/'
float_model_file = 'resnet18_pretrained_float.pth'

train_batch_size = 30
eval_batch_size = 30

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')
float_model.eval();


######################################################################
# 3. Script/Trace the model
# --------------------------
# The input for graph mode quantization is a TorchScript model, so we'll need to either script or trace the model first.
# 

ts_model = torch.jit.script(float_model).eval() # ts_model = torch.jit.trace(float_model, input)

######################################################################
# 4. Specify how to quantize the model with ``qconfig_dict``
# -------------------------
#
# .. code:: python
#
#   qconfig_dict = {'' : default_qconfig}
#
# We use the same ``qconfig`` used in eager mode quantization, ``qconfig`` is just a named tuple of the observers for ``activation`` and ``weight``. `qconfig_dict` is a dictionary with names of sub modules as key and qconfig for that module as value, empty key means the qconfig will be applied to whole model unless it’s overwritten by more specific configurations, the qconfig for each module is either found in the dictionary or fallback to the qconfig of parent module.
#
# Right now ``qconfig_dict`` is the only way to configure how the model is quantized, and it is done in the granularity of module, that is, we only support one type of ``qconfig`` for each ``torch.nn.Module``, for example, if we have:
#
# .. code:: python
#
#   qconfig = {
#         '' : qconfig_global,
#        'sub' : qconfig_sub,
#         'sub.fc' : qconfig_fc,
#        'sub.conv': None
#   }
#
# Module ``sub.fc`` will be configured with ``qconfig_fc``, and all other child modules in ``sub`` will be configured with ``qconfig_sub`` and ``sub.conv`` will not be quantized. All other modules in the model will be quantized with ``qconfig_global``
# Utility functions related to ``qconfig`` can be found in https://github.com/pytorch/pytorch/blob/master/torch/quantization/qconfig.py.

qconfig = get_default_qconfig('fbgemm')
qconfig_dict = {'': qconfig}


######################################################################
# 5. Define Calibration Function
# -------------------------
#
# .. code:: python
#
#   def calibrate(model, sample_data, ...):
#       model(sample_data, ...)
#
#
# Calibration function is run after the observers are inserted in the model. 
# The purpose for calibration is to run through some sample examples that is representative of the workload 
# (for example a sample of the training data set) so that the observers in the model are able to observe
# the statistics of the Tensors and we can later use this information to calculate quantization parameters.
#

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)


######################################################################
# 6. Quantize
# ---------------------
#
# .. code:: python
#
#     quantized_model = quantize_jit(
#         ts_model, # TorchScript model
#         {'': qconfig}, # qconfig dict
#         calibrate, # calibration function
#         [data_loader_test], # positional arguments to calibration function, typically some sample dataset
#         inplace=False, # whether to modify the model inplace or not
#         debug=True) # whether to prduce a debug friendly model or not
#
# There are three things we do in ``quantize_jit``:
#
# 1. ``prepare_jit`` folds BatchNorm modules into previous Conv2d modules, and insert observers in appropriate places in the Torchscript model.
# 2. Run calibrate function on the provided sample dataset.
# 3. ``convert_jit`` takes a calibrated model and produces a quantized model.
#
# If ``debug`` is False (default option), ``convert_jit`` will:
#
# - Calculate quantization parameters using the observers in the model
# - Ifnsert quantization ops like ``aten::quantize_per_tensor`` and ``aten::dequantize`` to the model, and remove the observer modules after that.
# - Replace floating point ops with quantized ops
# - Freeze the model (remove constant attributes and make them as Constant node in the graph).
# - Fold the quantize and prepack ops like ``quantized::conv2d_prepack`` into an attribute, so we don't need to quantize and prepack the weight everytime we run the model.
#
# If ``debug`` is set to ``True``:
# 
# - We can still access the attributes of the quantized model the same way as the original floating point model, e.g. ``model.conv1.weight`` (might be harder if you use a module list or sequential)
# - The arithmetic operations all occur in floating point with the numerics being identical to the final quantized model, allowing for debugging.

quantized_model = quantize_jit(
    ts_model,
    {'': qconfig},
    calibrate,
    [data_loader_test])

print(quantized_model.graph)

######################################################################
# As we can see ``aten::conv2d`` is changed to ``quantized::conv2d`` and the floating point weight has been quantized 
# and packed into an attribute (``quantized._jit_pass_packed_weight_30``), so we don't need to quantize/pack in runtime.
# Also we can't access the weight attributes anymore after the debug option since they are frozen.
#
# 7. Evaluation
# --------------
# We can now print the size and accuracy of the quantized model.

print('Size of model before quantization')
print_size_of_model(ts_model)
print('Size of model after quantization')
print_size_of_model(quantized_model)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print('[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f'%(top1.avg, top5.avg))

graph_mode_model_file = 'resnet18_graph_mode_quantized.pth'
torch.jit.save(quantized_model, saved_model_dir + graph_mode_model_file)
quantized_model = torch.jit.load(saved_model_dir + graph_mode_model_file)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print('[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f'%(top1.avg, top5.avg))

######################################################################
# If you want to get better accuracy or performance,  try changing the `qconfig_dict`. 
# We plan to add support for graph mode in the Numerical Suite so that you can 
# easily determine the sensitivity towards quantization of different modules in a model: `PyTorch Numeric Suite Tutorial <https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html>`_
#
# 8. Debugging Quantized Model
# ---------------------------
# We can also use debug option:

quantized_debug_model = quantize_jit(
    ts_model,
    {'': qconfig},
    calibrate,
    [data_loader_test],
    debug=True)

top1, top5 = evaluate(quantized_debug_model, criterion, data_loader_test)
print('[debug=True] quantized model Evaluation accuracy on test dataset: %2.2f, %2.2f'%(top1.avg, top5.avg))

######################################################################
# Note that the accuracy of the debug version is close to, but not exactly the same as the non-debug 
# version as the debug version uses floating point ops to emulate quantized ops and the numerics match 
# is approximate. We are working on making this even more exact.
#

print(quantized_debug_model.graph)

######################################################################
# We can see that there is no ``quantized::conv2d`` in the model, but the numerically equivalent pattern 
# of ``aten::dequnatize - aten::conv2d - aten::quantize_per_tensor``.

print_size_of_model(quantized_debug_model)

######################################################################
# Size of the debug model is the close to the floating point model because all the weights are 
# in float and not yet quantized and frozen, this allows people to inspect the weight. 
# You may access the weight attributes directly in the torchscript model, except for batch norm as
# it is fused into the preceding convolutions. We will also develop graph mode ``Numeric Suite`` 
# to allow easier inspection of weights in the future. Accessing the weight in the debug model is 
# the same as accessing the weight in a TorchScript model:

def get_first_conv_weight(model):
    return model.conv1.weight
w1 = get_first_conv_weight(ts_model)
w2 = get_first_conv_weight(quantized_debug_model)
print('first conv weight for input model:', str(w1)[:200])
print('first conv weight for quantized model:', str(w2)[:200])

######################################################################
# The weights are different because we fold the weights of BatchNorm to the previous conv before we quantize the model.
# More instructions on how to debug TorchScript model can be found `here <https://pytorch.org/docs/stable/jit.html#debugging>`_.
#
#
# As we can see, this is not as straightforward as eager mode, that's why we also plan to support graph mode ``Numeric Suite``,
# and it will probably be the primary tool people use to debug numerical issues.
#
# 9. Comparison with Baseline Float Model and Eager Mode Quantization
# ---------------------------

scripted_float_model_file = 'resnet18_scripted.pth'

print('Size of baseline model')
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test)
print('Baseline Float Model Evaluation accuracy: %2.2f, %2.2f'%(top1.avg, top5.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

######################################################################
# In this section we compare the model quantized with graph mode quantization with the model 
# quantized in eager mode. Graph mode and eager mode produce very similar quantized models, 
# so the expectation is that the accuracy and speedup are similar as well.

print('Size of graph mode quantized model')
print_size_of_model(quantized_model)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print('graph mode quantized model Evaluation accuracy on test dataset: %2.2f, %2.2f'%(top1.avg, top5.avg))

from torchvision.models.quantization.resnet import resnet18
eager_quantized_model = resnet18(pretrained=True, quantize=True).eval()
print('Size of eager mode quantized model')
eager_quantized_model = torch.jit.script(eager_quantized_model)
print_size_of_model(eager_quantized_model)
top1, top5 = evaluate(eager_quantized_model, criterion, data_loader_test)
print('eager mode quantized model Evaluation accuracy on test dataset: %2.2f, %2.2f'%(top1.avg, top5.avg))
eager_mode_model_file = 'resnet18_eager_mode_quantized.pth'
torch.jit.save(eager_quantized_model, saved_model_dir + eager_mode_model_file)

######################################################################
# We can see that the model size and accuracy of graph mode and eager mode quantized model are pretty similar.
#
# Running the model in AIBench (with single threading) gives the following result:
#
# .. code::
#
#   Scripted Float Model:
#   Self CPU time total: 418.472ms
#
#   Scripted Eager Mode Quantized Model:
#   Self CPU time total: 177.768ms
#
#   Graph Mode Quantized Model:
#   Self CPU time total: 157.256ms
#
# As we can see for resnet18 both graph mode and eager mode quantized model get similar speed up over the floating point model,
# which is around 2-3x faster than the floating point model. But the actual speedup over floating point model may vary 
# depending on model, device, build, input batch sizes, threading etc.
#


