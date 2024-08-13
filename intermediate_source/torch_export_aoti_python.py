# -*- coding: utf-8 -*-

"""
(Beta) ``torch.export`` AOT Inductor Tutorial for Python runtime
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
# Model Compilation
# ------------
#
# We will use TorchVision's pretrained `ResNet18` model and TorchInductor on the 
# exported PyTorch program using :func:`torch._export.aot_compile`.
#
#  .. note::
#
#       This API also supports :func:`torch.compile` options like `mode`
#       As an example, if used on a CUDA enabled device, we can set `"max_autotune": True`
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
# The API follows a similar structure to the :func:`torch.jit.load` API . We specify the path 
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