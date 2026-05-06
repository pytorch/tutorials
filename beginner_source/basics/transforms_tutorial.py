"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
**Transforms** ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Transforms
===================

Data does not always come in its final processed form that is required for
training machine learning algorithms. We use **transforms** to perform some
manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters -``transform`` to modify the features and
``target_transform`` to modify the labels - that accept callables containing the transformation logic.
The `torchvision.transforms <https://pytorch.org/vision/stable/transforms.html>`_ module offers
several commonly-used transforms out of the box.

The FashionMNIST features are in PIL Image format, and the labels are integers.
For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors.
To make these transformations, we use the ``torchvision.transforms.v2`` API along with ``torch.nn.functional.one_hot``.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
)

#################################################
# ToImage() and ToDtype()
# -------------------------------
#
# The ``torchvision.transforms.v2`` API replaces the legacy ``ToTensor`` transform with a two-step pipeline.
# `v2.ToImage <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToImage.html>`_
# converts a PIL image or NumPy ``ndarray`` into a ``torchvision.tv_tensors.Image`` tensor, and
# `v2.ToDtype <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html>`_
# with ``scale=True`` casts it to ``float32`` and scales the pixel intensity values to the range [0., 1.].
#

##############################################
# Lambda Transforms
# -------------------------------
#
# Lambda transforms apply any user-defined lambda function. Here, we use
# `torch.nn.functional.one_hot <https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html>`_
# to turn the integer label into a one-hot encoded tensor of size 10 (the number of labels in our dataset),
# then cast it to ``float`` to match the expected dtype.

target_transform = v2.Lambda(
    lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
)

######################################################################
# --------------
#

#################################################################
# Further Reading
# ~~~~~~~~~~~~~~~~~
# - `Getting started with transforms v2 <https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html>`_
# - `torchvision.transforms.v2 API <https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended>`_
