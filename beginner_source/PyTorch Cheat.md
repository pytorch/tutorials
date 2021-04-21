---
Title: PyTorch Cheat Sheet
PyTorch version: 1.0Pre
Date updated: 7/30/18

---

# Imports
---------------
### General

```
import torch                                        # root package
from torch.utils.data import Dataset, DataLoader    # dataset representation and loading
```

### Neural Network API

```
import torch.autograd as autograd         # computation graph
from torch.autograd import Variable       # variable node in computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.jit import script, trace       # hybrid frontend decorator and tracing jit
```
See [autograd](https://pytorch.org/docs/stable/autograd.html), [nn](https://pytorch.org/docs/stable/nn.html), [functional](https://pytorch.org/docs/stable/nn.html#torch-nn-functional) and [optim](https://pytorch.org/docs/stable/optim.html)

### Torchscript and JIT

```
torch.jit.trace()         # takes your module or function and an example data input, and traces the computational steps that the data encounters as it progresses through the model
@script                   # decorator used to indicate data-dependent control flow within the code being traced
```
See [Torchscript](https://pytorch.org/docs/stable/jit.html)

### ONNX

```
torch.onnx.export(model, dummy data, xxxx.proto)       # exports an ONNX formatted model using a trained model, dummy data and the desired file name
model = onnx.load("alexnet.proto")                     # load an ONNX model
onnx.checker.check_model(model)                        # check that the model IR is well formed
onnx.helper.printable_graph(model.graph)               # print a human readable representation of the graph
```
See [onnx](https://pytorch.org/docs/stable/onnx.html)

### Vision

```
from torchvision import datasets, models, transforms     # vision datasets, architectures & transforms
import torchvision.transforms as transforms              # composable transforms
```
See [torchvision](https://pytorch.org/vision/stable/index.html)

### Distributed Training

```
import torch.distributed as dist          # distributed communication
from multiprocessing import Process       # memory sharing processes
```
See [distributed](https://pytorch.org/docs/stable/distributed.html) and [multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html)


# Tensors
--------------------

### Creation

```
torch.randn(*size)              # tensor with independent N(0,1) entries
torch.[ones|zeros](*size)       # tensor with all 1's [or 0's]
torch.Tensor(L)                 # create tensor from [nested] list or ndarray L
x.clone()                       # clone of x
with torch.no_grad():           # code wrap that stops autograd from tracking tensor history
requires_grad=True              # arg, when set to True, tracks computation history for future derivative calculations
```
See [tensor](https://pytorch.org/docs/stable/tensors.html)

### Dimensionality

```
x.size()                              # return tuple-like object of dimensions
torch.cat(tensor_seq, dim=0)          # concatenates tensors along dim
x.view(a,b,...)                       # reshapes x into size (a,b,...)
x.view(-1,a)                          # reshapes x into size (b,a) for some b
x.transpose(a,b)                      # swaps dimensions a and b
x.permute(*dims)                      # permutes dimensions
x.unsqueeze(dim)                      # tensor with added axis
x.unsqueeze(dim=2)                    # (a,b,c) tensor -> (a,b,1,c) tensor
```
See [tensor](https://pytorch.org/docs/stable/tensors.html)

### Algebra

```
A.mm(B)       # matrix multiplication
A.mv(x)       # matrix-vector multiplication
x.t()         # matrix transpose
```
See [math operations](https://pytorch.org/docs/stable/torch.html?highlight=mm#math-operations)

### GPU Usage

```
torch.cuda.is_available                                 # check for cuda
x.cuda()                                                # move x's data from CPU to GPU and return new object
x.cpu()                                                 # move x's data from GPU to CPU and return new object

if not args.disable_cuda and torch.cuda.is_available(): # device agnostic code and modularity
    args.device = torch.device('cuda')                  #
else:                                                   #
    args.device = torch.device('cpu')                   #

net.to(device)                                          # recursively convert their parameters and buffers to device specific tensors
mytensor.to(device)                                     # copy your tensors to a device (gpu, cpu)
```
See [cuda](https://pytorch.org/docs/stable/cuda.html)


# Deep Learning
```
nn.Linear(m,n)                                # fully connected layer from m to n units
nn.ConvXd(m,n,s)                              # X dimensional conv layer from m to n channels where X⍷{1,2,3} and the kernel size is s
nn.MaxPoolXd(s)                               # X dimension pooling layer (notation as above)
nn.BatchNorm                                  # batch norm layer
nn.RNN/LSTM/GRU                               # recurrent layers
nn.Dropout(p=0.5, inplace=False)              # dropout layer for any dimensional input
nn.Dropout2d(p=0.5, inplace=False)            # 2-dimensional channel-wise dropout
nn.Embedding(num_embeddings, embedding_dim)   # (tensor-wise) mapping from indices to embedding vectors
```
See [nn](https://pytorch.org/docs/stable/nn.html)

### Loss Functions

```
nn.X                                        # where X is BCELoss, CrossEntropyLoss, L1Loss, MSELoss, NLLLoss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, KLDivLoss, MarginRankingLoss, HingeEmbeddingLoss or CosineEmbeddingLoss
```
See [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)

### Activation Functions

```
nn.X                                  # where X is ReLU, ReLU6, ELU, SELU, PReLU, LeakyReLU, Threshold, HardTanh, Sigmoid, Tanh, LogSigmoid, Softplus, SoftShrink, Softsign, TanhShrink, Softmin, Softmax, Softmax2d or LogSoftmax
```
See [activation functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

### Optimizers

```
opt = optim.x(model.parameters(), ...)      # create optimizer
opt.step()                                  # update weights
optim.X                                     # where X is SGD, Adadelta, Adagrad, Adam, SparseAdam, Adamax, ASGD, LBFGS, RMSProp or Rprop
```
See [optimizers](https://pytorch.org/docs/stable/optim.html)

### Learning rate scheduling

```
scheduler = optim.X(optimizer,...)      # create lr scheduler
scheduler.step()                        # update lr at start of epoch
optim.lr_scheduler.X                    # where X is LambdaLR, StepLR, MultiStepLR, ExponentialLR or ReduceLROnPLateau
```
See [learning rate scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)


# Data Utilities

### Datasets

```
Dataset                    # abstract class representing dataset
TensorDataset              # labelled dataset in the form of tensors
ConcatDataset              # concatenation of Datasets
```
See [datasets](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)

### Dataloaders and DataSamplers

```
DataLoader(dataset, batch_size=1, ...)      # loads data batches agnostic of structure of individual data points
sampler.Sampler(dataset,...)                # abstract class dealing with ways to sample from dataset
sampler.XSampler                            # where X is Sequential, Random, Subset, WeightedRandom or Distributed
```
See [dataloader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)


## Also see

* [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) _(pytorch.org)_
* [PyTorch Forums](https://discuss.pytorch.org/) _(discuss.pytorch.org)_
* [PyTorch for Numpy users](https://github.com/wkentaro/pytorch-for-numpy-users) _(github.com/wkentaro/pytorch-for-numpy-users)_
