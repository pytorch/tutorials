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
To make these transformations, we use ``ToTensor`` and ``Lambda``.
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

target_transform = Lambda(
    lambda y: torch.zeros(10, dtype=torch.float32).scatter_(
        dim=0, index=torch.tensor(y), value=1
    )
)

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=target_transform,
)

#################################################
# ToTensor()
# -------------------------------
#
# `ToTensor <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor>`_
# converts a PIL image or NumPy ``ndarray`` into a ``FloatTensor`` and scales
# the image's pixel intensity values in the range [0., 1.]
#

##############################################
# Lambda Transforms
# -------------------------------
#
# Lambda transforms apply any user-defined callable. Above, ``target_transform`` wraps a
# small lambda that turns each label ``y`` into a one-hot encoded tensor: it allocates a
# length-10 zero vector and uses
# `scatter_ <https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html>`_ to write
# ``value=1`` at index ``y``.

######################################################################
# --------------
#

#################################################################
# Further Reading
# ~~~~~~~~~~~~~~~~~
# - `torchvision.transforms API <https://pytorch.org/vision/stable/transforms.html>`_
