# -*- coding: utf-8 -*-

"""
(Beta) ``torch.export`` AOTInductor Tutorial for Python runtime
===================================================
**Author:** Ankith Gunapal
"""

######################################################################
#
# .. warning::
#
#     ``torch._export.aot_compile`` and ``torch._export.aot_load`` are in Beta status and are subject to backwards compatibility
#     breaking changes. This tutorial provides an example of how to use these APIs for model deployment using Python runtime.
#
# It has been shown `previously <https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html#>`__ how AOTInductor can be used 
# to do Ahead-of-Time compilation of PyTorch exported models by creating
# a shared library that can be run in a non-Python environment.
#
#
# In this tutorial, you will learn an end-to-end example of how to use AOTInductor for python runtime.
# We will look at how  to use :func:`torch._export.aot_compile` to generate a shared library.
# Additionally, we will examine how to execute the shared library in Python runtime using :func:`torch._export.aot_load`.
#
# **Contents**
#
# .. contents::
#     :local:

######################################################################
# Prerequisites
# -------------
#   * PyTorch 2.4 or later
#   * Basic understanding of ``torch._export`` and AOTInductor
#   * Complete the `AOTInductor: Ahead-Of-Time Compilation for Torch.Export-ed Models <https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html#>`_ tutorial

######################################################################
# What you will learn
# ----------------------
# * How to use AOTInductor for python runtime.
# * How  to use :func:`torch._export.aot_compile` to generate a shared library
# * How to run a shared library in Python runtime using :func:`torch._export.aot_load`.
# * When do you use AOTInductor for python runtime

######################################################################
# Model Compilation
# ------------
#
# We will use TorchVision's pretrained `ResNet18` model and TorchInductor on the 
# exported PyTorch program using :func:`torch._export.aot_compile`.
#
#  .. note::
#
#       This API also supports :func:`torch.compile` options like ``mode``
#       This means that if used on a CUDA enabled device, you can, for example, set ``"max_autotune": True``
#       which leverages Triton based matrix multiplications & convolutions, and enables CUDA graphs by default.
#
# We also specify ``dynamic_shapes`` for the batch dimension. In this example, ``min=2`` is not a bug and is 
# explained in `The 0/1 Specialization Problem <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk>`__


import os
import torch
from torchvision.models import ResNet18_Weights, resnet18

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

with torch.inference_mode():

    # Specify the generated shared library path
    aot_compile_options = {
            "aot_inductor.output_path": os.path.join(os.getcwd(), "resnet18_pt2.so"),
    }
    if torch.cuda.is_available():
        device = "cuda"
        aot_compile_options.update({"max_autotune": True})
    else:
        device = "cpu"
        # We need to turn off the below optimizations to support batch_size = 16,
        # which is treated like a special case
        # https://github.com/pytorch/pytorch/pull/116152
        torch.backends.mkldnn.set_flags(False)
        torch.backends.nnpack.set_flags(False)

    model = model.to(device=device)
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)

    # min=2 is not a bug and is explained in the 0/1 Specialization Problem
    batch_dim = torch.export.Dim("batch", min=2, max=32)
    so_path = torch._export.aot_compile(
        model,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        dynamic_shapes={"x": {0: batch_dim}},
        # Specify the generated shared library path
        options=aot_compile_options
    )


######################################################################
# Model Inference in Python
# ------------
#
# Typically, the shared object generated above is used in a non-Python environment. In PyTorch 2.3, 
# we added a new API called :func:`torch._export.aot_load` to load the shared library in the Python runtime.
# The API follows a structure similar to the :func:`torch.jit.load` API . You need to specify the path 
# of the shared library and the device where it should be loaded.
#  .. note::
#
#      In the example above, we specified ``batch_size=1`` for inference and  it still functions correctly even though we specified ``min=2`` in 
#      :func:`torch._export.aot_compile`.


import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_so_path = os.path.join(os.getcwd(), "resnet18_pt2.so")

model = torch._export.aot_load(model_so_path, device)
example_inputs = (torch.randn(1, 3, 224, 224, device=device),)

with torch.inference_mode():
    output = model(example_inputs)

######################################################################
# When to use AOTInductor for Python Runtime
# ---------------------------------------
#
# One of the requirements for using AOTInductor is that the model shouldn't have any graph breaks.
# Once this requirement is met, the primary use case for using AOTInductor Python Runtime is for
# model deployment using Python.
# There are mainly two reasons why you would use AOTInductor Python Runtime:
#
# -  ``torch._export.aot_compile`` generates a shared library. This is useful for model
#    versioning for deployments and tracking model performance over time.
# -  With :func:`torch.compile` being a JIT compiler, there is a warmup
#    cost associated with the first compilation. Your deployment needs to account for the
#    compilation time taken for the first inference. With AOTInductor, the compilation is
#    done offline using ``torch._export.aot_compile``. The deployment would only load the
#    shared library using ``torch._export.aot_load`` and run inference.
#
#
# The section below shows the speedup achieved with AOTInductor for first inference
#
# We define a utility function ``timed`` to measure the time taken for inference
#

import time
def timed(fn):
    # Returns the result of running `fn()` and the time it took for `fn()` to run,
    # in seconds. We use CUDA events and synchronization for accurate
    # measurement on CUDA enabled devices.
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.time()

    result = fn()
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
    else:
        end = time.time()

    # Measure time taken to execute the function in miliseconds
    if torch.cuda.is_available():
        duration = start.elapsed_time(end)
    else:
        duration = (end - start) * 1000

    return result, duration


######################################################################
# Lets measure the time for first inference using AOTInductor

torch._dynamo.reset()

model = torch._export.aot_load(model_so_path, device)
example_inputs = (torch.randn(1, 3, 224, 224, device=device),)

with torch.inference_mode():
    _, time_taken = timed(lambda: model(example_inputs))
    print(f"Time taken for first inference for AOTInductor is {time_taken:.2f} ms")


######################################################################
# Lets measure the time for first inference using ``torch.compile``

torch._dynamo.reset()

model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
model.eval()

model = torch.compile(model)
example_inputs = torch.randn(1, 3, 224, 224, device=device)

with torch.inference_mode():
    _, time_taken = timed(lambda: model(example_inputs))
    print(f"Time taken for first inference for torch.compile is {time_taken:.2f} ms")

######################################################################
# We see that there is a drastic speedup in first inference time using AOTInductor compared
# to ``torch.compile``

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we have learned how to effectively use the AOTInductor for Python runtime by 
# compiling and loading a pretrained ``ResNet18`` model using the ``torch._export.aot_compile``
# and ``torch._export.aot_load`` APIs. This process demonstrates the practical application of 
# generating a shared library and running it within a Python environment, even with dynamic shape
# considerations and device-specific optimizations. We also looked at the advantage of using 
# AOTInductor in model deployments, with regards to speed up in first inference time.
