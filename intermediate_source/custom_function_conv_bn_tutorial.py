# -*- coding: utf-8 -*-
"""
Fusing Convolution and Batch Norm using Custom Function
=======================================================

Fusing adjacent convolution and batch norm layers together is typically an
inference-time optimization to improve run-time. It is typically achieved
by eliminating the batch norm layer entirely and updating the weight
and bias of the preceding convolution [0]. This technique is not applicable
for training models, however.

In this tutorial, we will show a different technique to fuse the two layers
that can be applied during training. Rather than improved runtime, the
objective of this optimization is to reduce memory usage.

The idea behind this optimization is to see that both convolution and
batch norm (as well as many other ops) need to save a copy of their input
during forward for the backward pass. For large
batch sizes, these saved inputs are responsible for much of your memory usage,
so being able to avoid allocating another input tensor for every
convolution batch norm pair can be a significant reduction.

In this tutorial, we avoid this extra allocation by combining convolution
and batch norm into a single layer (as a custom function). In the forward
of this combined layer, we perform normal convolution and batch norm as-is,
with the only difference being that we will only save the inputs to the convolution.
To obtain the input of batch norm, which is necessary to backward through
it, we recompute convolution forward again during the backward pass.

For simplicity, in this tutorial we will not support the bias, stride, dilation, or
group parameters for convolutions.

[0] https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
"""

######################################################################
# Implementing a custom function requires us to implement the backward
# ourselves. In this case, we need both the backward formulas for Conv2D
# and BatchNorm2D. Eventually we'd chain them together in our unified
# backward function, but below we first implement them as their own
# custom functions so we can validate their correctness individually
import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

class Conv2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight):
        ctx.save_for_backward(X, weight)
        return F.conv2d(X, weight)

    # Use @once_differentiable by default unless we intend to double
    # backward
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        X, weight = ctx.saved_tensors
        grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
        grad_X = F.conv_transpose2d(grad_out, weight)
        return grad_X, grad_input

######################################################################
# When testing with gradcheck, it is important to use double precision
weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(10, 3, 7, 7, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Conv2D.apply, (X, weight))

######################################################################
# Batch Norm
def expand(t):
    # Helper function to unsqueeze the dimensions that we reduce over
    return t[None, :, None, None]

def batch_norm_backward(grad_out, X, sum, sqrt_var, N, eps):
    # To simplify our derivation, we follow the chain rule and compute
    # the following gradients before accumulating them all into a final grad_input
    #  1) X of the numerator
    #  2) mean of the numerator
    #  3) var of the denominator
    # d_denom = -num / denom**2
    tmp = -((X - expand(sum) / N) * grad_out).sum(dim=(0, 2, 3))
    d_denom = tmp / (sqrt_var + eps)**2
    # It is useful to delete tensors when you no longer need them
    # It's not a big difference here though because tmp only has size of (C,)
    # The important thing is avoid allocating NCHW-sized tensors unnecessarily
    del tmp
    grad_input = grad_out / expand(sqrt_var + eps)  # = d_numerator
    grad_input -= expand(grad_input.sum(dim=(0, 2, 3))) / N  # mean = X.sum(dim=(0, 2, 3)) / N
    # denom = torch.sqrt(var) + eps
    d_var = d_denom / (2 * sqrt_var)
    del d_denom
    # unbiased_var(x) = ((X - expand(mean))**2).sum(dim=(0, 2, 3)) / (N - 1)
    grad_input += (X * expand(d_var) * N + expand(-d_var * sum)) * 2 / ((N - 1) * N)
    return grad_input

class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, eps=1e-6):
        # Don't save keepdim'd values for backward
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.save_for_backward(X)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        return (X - expand(mean)) / expand(denom)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        X, = ctx.saved_tensors
        return batch_norm_backward(grad_out, X, ctx.sum, ctx.sqrt_var, ctx.N, ctx.eps)

######################################################################
# Testing with gradcheck
a = torch.rand(1, 2, 3, 4, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(BatchNorm.apply, (a,), fast_mode=False)

######################################################################
# Now that the bulk of the work has been done, we can combine
# them together. As seen in (1) of the forward pass below, we only need to
# save a single buffer for backward, but we need to recompute
# Conv2D again in (2).
class FusedConvBN2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, conv_weight, running_mean, running_var, training, exp_avg_factor=0.1, eps=1e-6):
        assert not running_mean.requires_grad and not running_var.requires_grad
        assert X.ndim == 4
        ctx.save_for_backward(X, conv_weight)  # (1) Only need to save this single buffer for backward!
        X = F.conv2d(X, conv_weight)

        if training:
            sum = X.sum(dim=(0, 2, 3), keepdim=False)  # Save squeezed statistics
            N = X.numel() / X.size(1)
            mean = sum / N
            var = X.var(dim=(0, 2, 3), unbiased=True, keepdim=False)
            running_mean = exp_avg_factor * mean + (1 - exp_avg_factor) * running_mean
            running_var = exp_avg_factor * var + (1 - exp_avg_factor) * running_var
            ctx.sum = sum
            ctx.N = N
        else:
            mean = running_mean
            var = running_var

        sqrt_var = torch.sqrt(var)
        ctx.sqrt_var = sqrt_var
        ctx.eps = eps
        ctx.training = training
        ctx.mark_non_differentiable(running_mean, running_var)
        out = (X - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None]) + eps)
        return out, running_mean, running_var

    @staticmethod
    def backward(ctx, grad_out, _grad_running_mean, _grad_running_var):
        X, conv_weight, = ctx.saved_tensors
        # Batch norm backward
        if ctx.training:
            X_conv_out = F.conv2d(X, conv_weight)  # (2) We need to recompute conv
            grad_out = batch_norm_backward(grad_out, X_conv_out, ctx.sum, ctx.sqrt_var,
                                           ctx.N, ctx.eps)
        else:
            # If not training, we use running mean and var, which are constant wrt x
            grad_out = grad_out / (ctx.sqrt_var[None, :, None, None] + ctx.eps)
        # Conv2d backward
        grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
        grad_X = F.conv_transpose2d(grad_out, conv_weight)
        return grad_X, grad_input, None, None, None, None, None

######################################################################
# Now that the bulk of the work has been done, it is time to combine
# them together. As seen in (1) of the forward pass below, we only need to
# save a single buffer for backward, but we need to recompute
# Conv2D again in (2).
import torch.nn as nn
import math

class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_avg_factor=0.1, eps=1e-6,
                 device=None, dtype=None):
        super(FusedConvBN, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # conv2d parameters
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **factory_kwargs))

        # batch norm parameters
        num_features = out_channels
        self.num_features = num_features
        self.register_buffer('running_var', torch.zeros(num_features, **factory_kwargs))
        self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
        self.eps = eps
        self.exp_avg_factor = exp_avg_factor

        self.reset_parameters()

    def forward(self, X):
        out, running_mean, running_var = FusedConvBN2DFunction.apply(X, self.conv_weight,
                                                                     self.running_mean, self.running_var,
                                                                     self.training, self.exp_avg_factor, self.eps)
        self.running_mean = running_mean
        self.running_var = running_var
        return out

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

######################################################################
# Use gradcheck to validate the correctness of our backward formula
weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(2, 3, 4, 4, requires_grad=True, dtype=torch.double)

with torch.no_grad():
    X_bn = F.conv2d(X, weight)
    mean = X_bn.mean(dim=(0, 2, 3))
    var = X_bn.var(dim=(0, 2, 3), unbiased=True)

for training in (True, False):
    torch.autograd.gradcheck(FusedConvBN2DFunction.apply, (X, weight, mean, var, training))

######################################################################
# Use FusedConvBN to train a basic network
# The code below is after some light modifications to the example here:
# https://github.com/pytorch/examples/tree/master/mnist
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self, fused=True):
        super(Net, self).__init__()
        self.fused = fused
        if fused:
            self.convbn1 = FusedConvBN(1, 32, 3)
            self.convbn2 = FusedConvBN(32, 64, 3)
            self.convbn3 = FusedConvBN(64, 64, 3)
        else:
            self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if self.fused:
            x = self.convbn1(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
        x = F.relu(x)
        if self.fused:
            x = self.convbn2(x)
        else:
            x = self.conv2(x)
            x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # Use inference mode instead of no_grad, for free improved test-time performance
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

use_cuda = torch.cuda.is_available()

torch.manual_seed(123456)

device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': 4096}
test_kwargs = {'batch_size': 1024}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = Net(fused=True).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

for epoch in range(14):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
