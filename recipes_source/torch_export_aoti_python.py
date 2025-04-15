# -*- coding: utf-8 -*-

"""
.. meta::
   :description: An end-to-end example of how to use AOTInductor for Python runtime.
   :keywords: torch.export, AOTInductor, torch._inductor.aoti_compile_and_package, aot_compile, torch._export.aoti_load_package

``torch.export`` AOTInductor Tutorial for Python runtime (Beta)
===============================================================
**Author:** Ankith Gunapal, Bin Bao, Angela Yi
"""

######################################################################
#
# .. warning::
#
#     ``torch._inductor.aoti_compile_and_package`` and
#     ``torch._inductor.aoti_load_package`` are in Beta status and are subject
#     to backwards compatibility breaking changes. This tutorial provides an
#     example of how to use these APIs for model deployment using Python
#     runtime.
#
# It has been shown `previously
# <https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html#>`__ how
# AOTInductor can be used to do Ahead-of-Time compilation of PyTorch exported
# models by creating an artifact that can be run in a non-Python environment.
# In this tutorial, you will learn an end-to-end example of how to use
# AOTInductor for Python runtime.
#
# **Contents**
#
# .. contents::
#     :local:

######################################################################
# Prerequisites
# -------------
# * PyTorch 2.6 or later
# * Basic understanding of ``torch.export`` and AOTInductor
# * Complete the `AOTInductor: Ahead-Of-Time Compilation for Torch.Export-ed Models <https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html#>`_ tutorial

######################################################################
# What you will learn
# ----------------------
# * How to use AOTInductor for Python runtime.
# * How to use :func:`torch._inductor.aoti_compile_and_package` along with :func:`torch.export.export` to generate a compiled artifact
# * How to load and run the artifact in a Python runtime using :func:`torch._export.aot_load`.
# * When to you use AOTInductor with a Python runtime

######################################################################
# Model Compilation
# -----------------
#
# We will use the TorchVision pretrained ``ResNet18`` model as an example.
#
# The first step is to export the model to a graph representation using
# :func:`torch.export.export`. To learn more about using this function, you can
# check out the `docs <https://pytorch.org/docs/main/export.html>`_ or the
# `tutorial <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`_.
#
# Once we have exported the PyTorch model and obtained an ``ExportedProgram``,
# we can apply :func:`torch._inductor.aoti_compile_and_package` to AOTInductor
# to compile the program to a specified device, and save the generated contents
# into a ".pt2" artifact.
#
# .. note::
#
#       This API supports the same available options that :func:`torch.compile`
#       has, such as ``mode`` and ``max_autotune`` (for those who want to enable
#       CUDA graphs and leverage Triton based matrix multiplications and
#       convolutions)

import os
import torch
import torch._inductor
from torchvision.models import ResNet18_Weights, resnet18

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

with torch.inference_mode():
    inductor_configs = {}

    if torch.cuda.is_available():
        device = "cuda"
        inductor_configs["max_autotune"] = True
    else:
        device = "cpu"

    model = model.to(device=device)
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)

    exported_program = torch.export.export(
        model,
        example_inputs,
    )
    path = torch._inductor.aoti_compile_and_package(
        exported_program,
        package_path=os.path.join(os.getcwd(), "resnet18.pt2"),
        inductor_configs=inductor_configs
    )

######################################################################
# The result of :func:`aoti_compile_and_package` is an artifact "resnet18.pt2"
# which can be loaded and executed in Python and C++.
# 
# The artifact itself contains a bunch of AOTInductor generated code, such as
# a generated C++ runner file, a shared library compiled from the C++ file, and
# CUDA binary files, aka cubin files, if optimizing for CUDA.
# 
# Structure-wise, the artifact is a structured ``.zip`` file, with the following 
# specification:
#
# .. code::
#    .
#    ├── archive_format
#    ├── version
#    ├── data
#    │   ├── aotinductor
#    │   │   └── model
#    │   │       ├── xxx.cpp            # AOTInductor generated cpp file
#    │   │       ├── xxx.so             # AOTInductor generated shared library
#    │   │       ├── xxx.cubin          # Cubin files (if running on CUDA)
#    │   │       └── xxx_metadata.json  # Additional metadata to save
#    │   ├── weights
#    │   │  └── TBD
#    │   └── constants
#    │      └── TBD
#    └── extra
#        └── metadata.json
#
# We can use the following command to inspect the artifact contents:
#
# .. code:: bash
#
#    $ unzip -l resnet18.pt2
#
# .. code::
#
#    Archive:  resnet18.pt2
#      Length      Date    Time    Name
#    ---------  ---------- -----   ----
#            1  01-08-2025 16:40   version
#            3  01-08-2025 16:40   archive_format
#        10088  01-08-2025 16:40   data/aotinductor/model/cagzt6akdaczvxwtbvqe34otfe5jlorktbqlojbzqjqvbfsjlge4.cubin
#        17160  01-08-2025 16:40   data/aotinductor/model/c6oytfjmt5w4c7onvtm6fray7clirxt7q5xjbwx3hdydclmwoujz.cubin
#        16616  01-08-2025 16:40   data/aotinductor/model/c7ydp7nocyz323hij4tmlf2kcedmwlyg6r57gaqzcsy3huneamu6.cubin
#        17776  01-08-2025 16:40   data/aotinductor/model/cyqdf46ordevqhiddvpdpp3uzwatfbzdpl3auj2nx23uxvplnne2.cubin
#        10856  01-08-2025 16:40   data/aotinductor/model/cpzfebfgrusqslui7fxsuoo4tvwulmrxirc5tmrpa4mvrbdno7kn.cubin
#        14608  01-08-2025 16:40   data/aotinductor/model/c5ukeoz5wmaszd7vczdz2qhtt6n7tdbl3b6wuy4rb2se24fjwfoy.cubin
#        11376  01-08-2025 16:40   data/aotinductor/model/csu3nstcp56tsjfycygaqsewpu64l5s6zavvz7537cm4s4cv2k3r.cubin
#        10984  01-08-2025 16:40   data/aotinductor/model/cp76lez4glmgq7gedf2u25zvvv6rksv5lav4q22dibd2zicbgwj3.cubin
#        14736  01-08-2025 16:40   data/aotinductor/model/c2bb5p6tnwz4elgujqelsrp3unvkgsyiv7xqxmpvuxcm4jfl7pc2.cubin
#        11376  01-08-2025 16:40   data/aotinductor/model/c6eopmb2b4ngodwsayae4r5q6ni3jlfogfbdk3ypg56tgpzhubfy.cubin
#        11624  01-08-2025 16:40   data/aotinductor/model/chmwe6lvoekzfowdbiizitm3haiiuad5kdm6sd2m6mv6dkn2zk32.cubin
#        15632  01-08-2025 16:40   data/aotinductor/model/c3jop5g344hj3ztsu4qm6ibxyaaerlhkzh2e6emak23rxfje6jam.cubin
#        25472  01-08-2025 16:40   data/aotinductor/model/chaiixybeiuuitm2nmqnxzijzwgnn2n7uuss4qmsupgblfh3h5hk.cubin
#       139389  01-08-2025 16:40   data/aotinductor/model/cvk6qzuybruhwxtfblzxiov3rlrziv5fkqc4mdhbmantfu3lmd6t.cpp
#           27  01-08-2025 16:40   data/aotinductor/model/cvk6qzuybruhwxtfblzxiov3rlrziv5fkqc4mdhbmantfu3lmd6t_metadata.json
#     47195424  01-08-2025 16:40   data/aotinductor/model/cvk6qzuybruhwxtfblzxiov3rlrziv5fkqc4mdhbmantfu3lmd6t.so
#    ---------                     -------
#     47523148                     18 files


######################################################################
# Model Inference in Python
# -------------------------
#
# To load and run the artifact in Python, we can use :func:`torch._inductor.aoti_load_package`.
#

import os
import torch
import torch._inductor

model_path = os.path.join(os.getcwd(), "resnet18.pt2")

compiled_model = torch._inductor.aoti_load_package(model_path)
example_inputs = torch.randn(2, 3, 224, 224, device=device)

with torch.inference_mode():
    output = compiled_model(example_inputs)


######################################################################
# When to use AOTInductor with a Python Runtime
# ---------------------------------------------
#
# There are mainly two reasons why one would use AOTInductor with a Python Runtime:
#
# -  ``torch._inductor.aoti_compile_and_package`` generates a singular
#    serialized artifact. This is useful for model versioning for deployments
#    and tracking model performance over time.
# -  With :func:`torch.compile` being a JIT compiler, there is a warmup
#    cost associated with the first compilation. Your deployment needs to
#    account for the compilation time taken for the first inference. With
#    AOTInductor, the compilation is done ahead of time using
#    ``torch.export.export`` and ``torch._inductor.aoti_compile_and_package``.
#    At deployment time, after loading the model, running inference does not
#    have any additional cost.
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

compiled_model = torch._inductor.aoti_load_package(model_path)
example_inputs = torch.randn(1, 3, 224, 224, device=device)

with torch.inference_mode():
    _, time_taken = timed(lambda: compiled_model(example_inputs))
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
# In this recipe, we have learned how to effectively use the AOTInductor for Python runtime by 
# compiling and loading a pretrained ``ResNet18`` model. This process
# demonstrates the practical application of generating a compiled artifact and
# running it within a Python environment. We also looked at the advantage of using 
# AOTInductor in model deployments, with regards to speed up in first inference time.
