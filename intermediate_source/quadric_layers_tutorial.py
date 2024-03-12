"""
(beta) Quadric Layers
==================================================================

**Author**: `Dirk Roeckmann <https://github.com/diro5t>`_

Introduction
------------

Quadric layers introduce quadratic functions with second-order decision boundaries (quadric hypersurfaces)
and can be used as 100% drop-in for linear layers (torch.nn.Linear) and present a high-level means
to reduce overall model size.

In comparison to linear layers with n weights and 1 bias (if needed) per neuron, a quadric neuron consists of
2n weights (n quadratic weights and n linear weights) and 1 bias (if needed).
Although this means a doubling in weights per neuron, the more powerful decision boundaries per neuron lead 
in many applications to significantly less neurons per layer or even less layers and in total to less model parameters.

In this tutorial, a simple classification application on the MNIST dataset using a non convolutional model is presented.
"""

# imports
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim, Tensor
from torch.nn.parameter import Parameter, UninitializedParameter

import numpy as np
import matplotlib.pyplot as plt
import math

######################################################################
# 1. Load MNIST data
# ------------------
transf = transforms.Compose([transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,)),])

batch_size = 128

train_set = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transf)
test_set = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transf)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

img_iter = iter(train_loader)
images, labels = next(img_iter)

figure = plt.figure()
img_num = 60
for index in range(1, img_num + 1):
    plt.subplot(10, 6, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

######################################################################
# 2. Introduce quadric layer 
# --------------------------
# NOTE: If quadric layers are part of torch.nn in the future, this definition is not necessary anymore

class Quadric(nn.Module):
    r"""Applies a quadric transformation to the incoming data: :math:`y = x^2A^T + xB^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        qweight: the learnable quadratic weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        weight: the learnable linear weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Quadric(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    qweight: Tensor
    lweight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qweight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.lweight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.qweight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lweight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in_q, _ = nn.init._calculate_fan_in_and_fan_out(self.qweight)
            fan_in_l, _ = nn.init._calculate_fan_in_and_fan_out(self.lweight)
            bound = 1 / math.sqrt(fan_in_l) if fan_in_l > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        input_sqr = torch.mul(input, input)
        q = nn.functional.linear(input_sqr, self.qweight, None)
        lb = nn.functional.linear(input, self.lweight, self.bias)
        return torch.add(q, lb)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

######################################################################
# 3. Define the model
# -------------------
#
# NOTE: If quadric layers are part of torch.nn in the future, the model
# can be defined like this:
# model = nn.Sequential(nn.Quadric(784, 16),
#                       nn.ReLU(),
#                       nn.Quadric(16, 10),
#                       nn.LogSoftmax(dim=1))

model = nn.Sequential(Quadric(784, 16),
                      nn.ReLU(),
                      Quadric(16, 10),
                      nn.LogSoftmax(dim=1))

# total number of model parameters
sum(p.numel() for p in model.parameters())

######################################################################
# 4. Train the model
# ------------------

loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
epochs = 20
for e in range(epochs):
    train_loss = 0
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    else:
        print(f"Epoch {e} - Training loss: {train_loss/len(train_loader)}")

######################################################################
# 5. Evaluate the model
# ---------------------

# training data
corrects, all = 0, 0
for images,labels in train_loader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.inference_mode():
        logps = model(img)
    ps = torch.exp(logps)
    prob = list(ps.numpy()[0])
    inf_label = prob.index(max(prob))
    true_label = labels.numpy()[i]
    if(true_label == inf_label):
      corrects += 1
    all += 1

print(f"Number of training images = {all}")
print(f"\nModel training accuracy = {corrects / all}")

# test data
corrects, all = 0, 0
for images,labels in test_loader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.inference_mode():
        logps = model(img)
    ps = torch.exp(logps)
    prob = list(ps.numpy()[0])
    inf_label = prob.index(max(prob))
    true_label = labels.numpy()[i]
    if(true_label == inf_label):
      corrects += 1
    all += 1

print(f"Number of test images = {all}")
print(f"\nModel test accuracy = {corrects / all}")

######################################################################
# 6. Conclusion
# -------------
#
# Quadric layers can easily be used to reduce model size in many applications just by replacing linear layers.
#
# It is hard to quantify model size reduction because of several factors including number of epochs, training and test accuracy etc.
#
# In this example, with the same number of epochs and identical training and test data and virtually the same accuracy 
# a model reduction size of ~75% could be achieved
#
# quadric model:
# --------------
# model = nn.Sequential(Quadric(784, 16),
#                      nn.ReLU(),
#                      Quadric(16, 10),
#                      nn.LogSoftmax(dim=1)
# model size: 25434 parameters
#
# comparable linear  model:
# -------------------------
# model = nn.Sequential(nn.Linear(784, 128),
#                      nn.ReLU(),
#                      nn.Linear(128, 10),
#                      nn.LogSoftmax(dim=1))
# model size: 101770 parameters
#
# Thanks for reading! Any feedback is highly appreciated. Just create an issue
# `here <https://github.com/pytorch/pytorch/issues>` if you have any.


