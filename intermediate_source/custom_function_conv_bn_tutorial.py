# -*- coding: utf-8 -*-
"""
Fusing Convolution and Batch Norm using Custom Function
=======================================================

Fusing adjacent convolution and batch norm layers together is typically an
inference-time optimization to improve run-time. It is usually achieved
by eliminating the batch norm layer entirely and updating the weight
and bias of the preceding convolution [0]. However, this technique is not
applicable for training models.

In this tutorial, we will show a different technique to fuse the two layers
that can be applied during training. Rather than improved runtime, the
objective of this optimization is to reduce memory usage.

The idea behind this optimization is to see that both convolution and
batch norm (as well as many other ops) need to save a copy of their input
during forward for the backward pass. For large
batch sizes, these saved inputs are responsible for most of your memory usage,
so being able to avoid allocating another input tensor for every
convolution batch norm pair can be a significant reduction.

In this tutorial, we avoid this extra allocation by combining convolution
and batch norm into a single layer (as a custom function). In the forward
of this combined layer, we perform normal convolution and batch norm as-is,
with the only difference being that we will only save the inputs to the convolution.
To obtain the input of batch norm, which is necessary to backward through
it, we recompute convolution forward again during the backward pass.

It is important to note that the usage of this optimization is situational.
Though (by avoiding one buffer saved) we always reduce the memory allocated at
the end of the forward pass, there are cases when the *peak* memory allocated
may not actually be reduced. See the final section for more details.

For simplicity, in this tutorial we hardcode `bias=False`, `stride=1`, `padding=0`, `dilation=1`,
and `groups=1` for Conv2D. For BatchNorm2D, we hardcode `eps=1e-3`, `momentum=0.1`,
`affine=False`, and `track_running_statistics=False`. Another small difference
is that we add epsilon in the denominator outside of the square root in the computation
of batch norm.

[0] https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
"""

######################################################################
# Backward Formula Implementation for Convolution
# -------------------------------------------------------------------
# Implementing a custom function requires us to implement the backward
# ourselves. In this case, we need both the backward formulas for Conv2D
# and BatchNorm2D. Eventually we'd chain them together in our unified
# backward function, but below we first implement them as their own
# custom functions so we can validate their correctness individually
import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

def convolution_backward(grad_out, X, weight):
    grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight)
    return grad_X, grad_input

class Conv2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight):
        ctx.save_for_backward(X, weight)
        return F.conv2d(X, weight)

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        X, weight = ctx.saved_tensors
        return convolution_backward(grad_out, X, weight)

######################################################################
# When testing with ``gradcheck``, it is important to use double precision
weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(10, 3, 7, 7, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Conv2D.apply, (X, weight))

######################################################################
# Backward Formula Implementation for Batch Norm
# -------------------------------------------------------------------
# Batch Norm has two modes: training and ``eval`` mode. In training mode
# the sample statistics are a function of the inputs. In ``eval`` mode,
# we use the saved running statistics, which are not a function of the inputs.
# This makes non-training mode's backward significantly simpler. Below
# we implement and test only the training mode case.
def unsqueeze_all(t):
    # Helper function to ``unsqueeze`` all the dimensions that we reduce over
    return t[None, :, None, None]

def batch_norm_backward(grad_out, X, sum, sqrt_var, N, eps):
    # We use the formula: ``out = (X - mean(X)) / (sqrt(var(X)) + eps)``
    # in batch norm 2D forward. To simplify our derivation, we follow the
    # chain rule and compute the gradients as follows before accumulating
    # them all into a final grad_input.
    #  1) ``grad of out wrt var(X)`` * ``grad of var(X) wrt X``
    #  2) ``grad of out wrt mean(X)`` * ``grad of mean(X) wrt X``
    #  3) ``grad of out wrt X in the numerator`` * ``grad of X wrt X``
    # We then rewrite the formulas to use as few extra buffers as possible
    tmp = ((X - unsqueeze_all(sum) / N) * grad_out).sum(dim=(0, 2, 3))
    tmp *= -1
    d_denom = tmp / (sqrt_var + eps)**2  # ``d_denom = -num / denom**2``
    # It is useful to delete tensors when you no longer need them with ``del``
    # For example, we could've done ``del tmp`` here because we won't use it later
    # In this case, it's not a big difference because ``tmp`` only has size of (C,)
    # The important thing is avoid allocating NCHW-sized tensors unnecessarily
    d_var = d_denom / (2 * sqrt_var)  # ``denom = torch.sqrt(var) + eps``
    # Compute ``d_mean_dx`` before allocating the final NCHW-sized grad_input buffer
    d_mean_dx = grad_out / unsqueeze_all(sqrt_var + eps)
    d_mean_dx = unsqueeze_all(-d_mean_dx.sum(dim=(0, 2, 3)) / N)
    # ``d_mean_dx`` has already been reassigned to a C-sized buffer so no need to worry

    # ``(1) unbiased_var(x) = ((X - unsqueeze_all(mean))**2).sum(dim=(0, 2, 3)) / (N - 1)``
    grad_input = X * unsqueeze_all(d_var * N)
    grad_input += unsqueeze_all(-d_var * sum)
    grad_input *= 2 / ((N - 1) * N)
    # (2) mean (see above)
    grad_input += d_mean_dx
    # (3) Add 'grad_out / <factor>' without allocating an extra buffer
    grad_input *= unsqueeze_all(sqrt_var + eps)
    grad_input += grad_out
    grad_input /= unsqueeze_all(sqrt_var + eps)  # ``sqrt_var + eps > 0!``
    return grad_input

class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, eps=1e-3):
        # Don't save ``keepdim`` values for backward
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
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        X, = ctx.saved_tensors
        return batch_norm_backward(grad_out, X, ctx.sum, ctx.sqrt_var, ctx.N, ctx.eps)

######################################################################
# Testing with ``gradcheck``
a = torch.rand(1, 2, 3, 4, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(BatchNorm.apply, (a,), fast_mode=False)

######################################################################
# Fusing Convolution and BatchNorm
# -------------------------------------------------------------------
# Now that the bulk of the work has been done, we can combine
# them together. Note that in (1) we only save a single buffer
# for backward, but this also means we recompute convolution forward
# in (5). Also see that in (2), (3), (4), and (6), it's the same
# exact code as the examples above.
class FusedConvBN2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, conv_weight, eps=1e-3):
        assert X.ndim == 4  # N, C, H, W
        # (1) Only need to save this single buffer for backward!
        ctx.save_for_backward(X, conv_weight)

        # (2) Exact same Conv2D forward from example above
        X = F.conv2d(X, conv_weight)
        # (3) Exact same BatchNorm2D forward from example above
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        # Try to do as many things in-place as possible
        # Instead of `out = (X - a) / b`, doing `out = X - a; out /= b`
        # avoids allocating one extra NCHW-sized buffer here
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X, conv_weight, = ctx.saved_tensors
        # (4) Batch norm backward
        # (5) We need to recompute conv
        X_conv_out = F.conv2d(X, conv_weight)
        grad_out = batch_norm_backward(grad_out, X_conv_out, ctx.sum, ctx.sqrt_var,
                                       ctx.N, ctx.eps)
        # (6) Conv2d backward
        grad_X, grad_input = convolution_backward(grad_out, X, conv_weight)
        return grad_X, grad_input, None, None, None, None, None

######################################################################
# The next step is to wrap our functional variant in a stateful
# `nn.Module`
import torch.nn as nn
import math

class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_avg_factor=0.1,
                 eps=1e-3, device=None, dtype=None):
        super(FusedConvBN, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Conv parameters
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **factory_kwargs))
        # Batch norm parameters
        num_features = out_channels
        self.num_features = num_features
        self.eps = eps
        # Initialize
        self.reset_parameters()

    def forward(self, X):
        return FusedConvBN2DFunction.apply(X, self.conv_weight, self.eps)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

######################################################################
# Use ``gradcheck`` to validate the correctness of our backward formula
weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
X = torch.rand(2, 3, 4, 4, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(FusedConvBN2DFunction.apply, (X, weight))

######################################################################
# Testing out our new Layer
# -------------------------------------------------------------------
# Use ``FusedConvBN`` to train a basic network
# The code below is after some light modifications to the example here:
# https://github.com/pytorch/examples/tree/master/mnist
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Record memory allocated at the end of the forward pass
memory_allocated = [[],[]]

class Net(nn.Module):
    def __init__(self, fused=True):
        super(Net, self).__init__()
        self.fused = fused
        if fused:
            self.convbn1 = FusedConvBN(1, 32, 3)
            self.convbn2 = FusedConvBN(32, 64, 3)
        else:
            self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(32, affine=False, track_running_stats=False)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if self.fused:
            x = self.convbn1(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
        F.relu_(x)
        if self.fused:
            x = self.convbn2(x)
        else:
            x = self.conv2(x)
            x = self.bn2(x)
        F.relu_(x)
        x = F.max_pool2d(x, 2)
        F.relu_(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.dropout(x)
        F.relu_(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        if fused:
            memory_allocated[0].append(torch.cuda.memory_allocated())
        else:
            memory_allocated[1].append(torch.cuda.memory_allocated())
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
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
train_kwargs = {'batch_size': 2048}
test_kwargs = {'batch_size': 2048}

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

######################################################################
# A Comparison of Memory Usage
# -------------------------------------------------------------------
# If CUDA is enabled, print out memory usage for both `fused=True` and `fused=False`
# For an example run on NVIDIA GeForce RTX 3070, NVIDIA CUDA® Deep Neural Network library (cuDNN) 8.0.5: fused peak memory: 1.56GB,
# unfused peak memory: 2.68GB
#
# It is important to note that the *peak* memory usage for this model may vary depending
# the specific cuDNN convolution algorithm used. For shallower models, it
# may be possible for the peak memory allocated of the fused model to exceed
# that of the unfused model! This is because the memory allocated to compute
# certain cuDNN convolution algorithms can be high enough to "hide" the typical peak
# you would expect to be near the start of the backward pass.
#
# For this reason, we also record and display the memory allocated at the end
# of the forward pass as an approximation, and to demonstrate that we indeed
# allocate one fewer buffer per fused ``conv-bn`` pair.
from statistics import mean

torch.backends.cudnn.enabled = True

if use_cuda:
    peak_memory_allocated = []

    for fused in (True, False):
        torch.manual_seed(123456)

        model = Net(fused=fused).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

        for epoch in range(1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()
        peak_memory_allocated.append(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    print("cuDNN version:", torch.backends.cudnn.version())
    print()
    print("Peak memory allocated:")
    print(f"fused: {peak_memory_allocated[0]/1024**3:.2f}GB, unfused: {peak_memory_allocated[1]/1024**3:.2f}GB")
    print("Memory allocated at end of forward pass:")
    print(f"fused: {mean(memory_allocated[0])/1024**3:.2f}GB, unfused: {mean(memory_allocated[1])/1024**3:.2f}GB")


