# -*- coding: utf-8 -*-
"""
PyTorch FX Numeric Suite Core APIs Tutorial
===========================================

Introduction
------------

Quantization is good when it works, but it is difficult to know what is wrong
when it does not satisfy the accuracy we expect. Debugging the accuracy issue
of quantization is not easy and time-consuming.

One important step of debugging is to measure the statistics of the float model
and its corresponding quantized model to know where they differ most.
We built a suite of numeric tools called PyTorch FX Numeric Suite Core APIs in
PyTorch quantization to enable the measurement of the statistics between
quantized module and float module to support quantization debugging efforts.
Even for the quantized model with good accuracy, PyTorch FX Numeric Suite Core
APIs can still be used as the profiling tool to better understand the
quantization error within the model and provide the guidance for further
optimization.

PyTorch FX Numeric Suite Core APIs currently supports models quantized through
both static quantization and dynamic quantization with unified APIs.

In this tutorial we will use MobileNetV2 as an example to show how to use
PyTorch FX Numeric Suite Core APIs to measure the statistics between static
quantized model and float model.

Setup
^^^^^
We’ll start by doing the necessary imports:
"""

##############################################################################

# Imports and util functions

import copy
import torch
import torchvision
import torch.quantization
import torch.ao.ns._numeric_suite_fx as ns
import torch.quantization.quantize_fx as quantize_fx

import matplotlib.pyplot as plt
from tabulate import tabulate

torch.manual_seed(0)
plt.style.use('seaborn-whitegrid')


# a simple line graph
def plot(xdata, ydata, xlabel, ylabel, title):
    _ = plt.figure(figsize=(10, 5), dpi=100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax = plt.axes()
    ax.plot(xdata, ydata)
    plt.show()

##############################################################################
# Then we load the pretrained float MobileNetV2 model, and quantize it.


# create float model
mobilenetv2_float = torchvision.models.quantization.mobilenet_v2(
    pretrained=True, quantize=False).eval()

# create quantized model
qconfig_dict = {
    '': torch.quantization.get_default_qconfig('fbgemm'),
    # adjust the qconfig to make the results more interesting to explore
    'module_name': [
        # turn off quantization for the first couple of layers
        ('features.0', None),
        ('features.1', None),
        # use MinMaxObserver for `features.17`, this should lead to worse
        # weight SQNR
        ('features.17', torch.quantization.default_qconfig),
    ]
}
# Note: quantization APIs are inplace, so we save a copy of the float model for
# later comparison to the quantized model. This is done throughout the
# tutorial.
datum = torch.randn(1, 3, 224, 224)
mobilenetv2_prepared = quantize_fx.prepare_fx(
    copy.deepcopy(mobilenetv2_float), qconfig_dict, (datum,))
mobilenetv2_prepared(datum)
# Note: there is a long standing issue that we cannot copy.deepcopy a
# quantized model. Since quantization APIs are inplace and we need to use
# different copies of the quantized model throughout this tutorial, we call
# `convert_fx` on a copy, so we have access to the original `prepared_model`
# later. This is done throughout the tutorial.
mobilenetv2_quantized = quantize_fx.convert_fx(
    copy.deepcopy(mobilenetv2_prepared))

##############################################################################
# 1. Compare the weights of float and quantized models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The first analysis we can do is comparing the weights of the fp32 model and
# the int8 model by calculating the SQNR between each pair of weights.
#
# The `extract_weights` API can be used to extract weights from linear,
# convolution and LSTM layers. It works for dynamic quantization as well as
# PTQ/QAT.

# Note: when comparing weights in models with Conv-BN for PTQ, we need to
# compare weights after Conv-BN fusion for a proper comparison.  Because of
# this, we use `prepared_model` instead of `float_model` when comparing
# weights.

# Extract conv and linear weights from corresponding parts of two models, and
# save them in `wt_compare_dict`.
mobilenetv2_wt_compare_dict = ns.extract_weights(
    'fp32',  # string name for model A
    mobilenetv2_prepared,  # model A
    'int8',  # string name for model B
    mobilenetv2_quantized,  # model B
)

# calculate SQNR between each pair of weights
ns.extend_logger_results_with_comparison(
    mobilenetv2_wt_compare_dict,  # results object to modify inplace
    'fp32',  # string name of model A (from previous step)
    'int8',  # string name of model B (from previous step)
    torch.ao.ns.fx.utils.compute_sqnr,  # tensor comparison function
    'sqnr',  # the name to use to store the results under
)

# massage the data into a format easy to graph and print
mobilenetv2_wt_to_print = []
for idx, (layer_name, v) in enumerate(mobilenetv2_wt_compare_dict.items()):
    mobilenetv2_wt_to_print.append([
        idx,
        layer_name,
        v['weight']['int8'][0]['prev_node_target_type'],
        v['weight']['int8'][0]['values'][0].shape,
        v['weight']['int8'][0]['sqnr'][0],
    ])

# plot the SQNR between fp32 and int8 weights for each layer
plot(
    [x[0] for x in mobilenetv2_wt_to_print],
    [x[4] for x in mobilenetv2_wt_to_print],
    'idx',
    'sqnr',
    'weights, idx to sqnr'
)

##############################################################################
# Also print out the SQNR, so we can inspect the layer name and type:

print(tabulate(
    mobilenetv2_wt_to_print,
    headers=['idx', 'layer_name', 'type', 'shape', 'sqnr']
))

##############################################################################
# 2. Compare activations API
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# The second tool allows for comparison of activations between float and
# quantized models at corresponding locations for the same input.
#
# .. figure:: /_static/img/compare_output.png
#
# The `add_loggers`/`extract_logger_info` API can be used to to extract
# activations from any layer with a `torch.Tensor` return type. It works for
# dynamic quantization as well as PTQ/QAT.

# Compare unshadowed activations

# Create a new copy of the quantized model, because we cannot `copy.deepcopy`
# a quantized model.
mobilenetv2_quantized = quantize_fx.convert_fx(
    copy.deepcopy(mobilenetv2_prepared))
mobilenetv2_float_ns, mobilenetv2_quantized_ns = ns.add_loggers(
    'fp32',  # string name for model A
    copy.deepcopy(mobilenetv2_prepared),  # model A
    'int8',  # string name for model B
    mobilenetv2_quantized,  # model B
    ns.OutputLogger,  # logger class to use
)

# feed data through network to capture intermediate activations
mobilenetv2_float_ns(datum)
mobilenetv2_quantized_ns(datum)

# extract intermediate activations
mobilenetv2_act_compare_dict = ns.extract_logger_info(
    mobilenetv2_float_ns,  # model A, with loggers (from previous step)
    mobilenetv2_quantized_ns,  # model B, with loggers (from previous step)
    ns.OutputLogger,  # logger class to extract data from
    'int8',  # string name of model to use for layer names for the output
)

# add SQNR comparison
ns.extend_logger_results_with_comparison(
    mobilenetv2_act_compare_dict,  # results object to modify inplace
    'fp32',  # string name of model A (from previous step)
    'int8',  # string name of model B (from previous step)
    torch.ao.ns.fx.utils.compute_sqnr,  # tensor comparison function
    'sqnr',  # the name to use to store the results under
)

# massage the data into a format easy to graph and print
mobilenet_v2_act_to_print = []
for idx, (layer_name, v) in enumerate(mobilenetv2_act_compare_dict.items()):
    mobilenet_v2_act_to_print.append([
        idx,
        layer_name,
        v['node_output']['int8'][0]['prev_node_target_type'],
        v['node_output']['int8'][0]['values'][0].shape,
        v['node_output']['int8'][0]['sqnr'][0]])

# plot the SQNR between fp32 and int8 activations for each layer
plot(
    [x[0] for x in mobilenet_v2_act_to_print],
    [x[4] for x in mobilenet_v2_act_to_print],
    'idx',
    'sqnr',
    'unshadowed activations, idx to sqnr',
)

##############################################################################
# Also print out the SQNR, so we can inspect the layer name and type:
print(tabulate(
    mobilenet_v2_act_to_print,
    headers=['idx', 'layer_name', 'type', 'shape', 'sqnr']
))
