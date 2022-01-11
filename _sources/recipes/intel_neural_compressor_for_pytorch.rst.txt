Ease-of-use quantization for PyTorch with Intel® Neural Compressor
==================================================================

Overview
--------

Most deep learning applications are using 32-bits of floating-point precision
for inference. But low precision data types, especially int8, are getting more
focus due to significant performance boost. One of the essential concerns on
adopting low precision is how to easily mitigate the possible accuracy loss
and reach predefined accuracy requirement.

Intel® Neural Compressor aims to address the aforementioned concern by extending
PyTorch with accuracy-driven automatic tuning strategies to help user quickly find
out the best quantized model on Intel hardware, including Intel Deep Learning
Boost (`Intel DL Boost <https://www.intel.com/content/www/us/en/artificial-intelligence/deep-learning-boost.html>`_)
and Intel Advanced Matrix Extensions (`Intel AMX <https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-amx-instructions/intrinsics-for-amx-tile-instructions.html>`_).

Intel® Neural Compressor has been released as an open-source project
at `Github <https://github.com/intel/neural-compressor>`_.

Features
--------

- **Ease-of-use Python API:** Intel® Neural Compressor provides simple frontend
  Python APIs and utilities for users to do neural network compression with few
  line code changes.
  Typically, only 5 to 6 clauses are required to be added to the original code.

- **Quantization:** Intel® Neural Compressor supports accuracy-driven automatic
  tuning process on post-training static quantization, post-training dynamic
  quantization, and quantization-aware training on PyTorch fx graph mode and
  eager model.

*This tutorial mainly focuses on the quantization part. As for how to use Intel®
Neural Compressor to do pruning and distillation, please refer to corresponding
documents in the Intel® Neural Compressor github repo.*

Getting Started
---------------

Installation
~~~~~~~~~~~~

.. code:: bash

    # install stable version from pip
    pip install neural-compressor

    # install nightly version from pip
    pip install -i https://test.pypi.org/simple/ neural-compressor

    # install stable version from from conda
    conda install neural-compressor -c conda-forge -c intel

*Supported python versions are 3.6 or 3.7 or 3.8 or 3.9*

Usages
~~~~~~

Minor code changes are required for users to get started with Intel® Neural Compressor
quantization API. Both PyTorch fx graph mode and eager mode are supported.

Intel® Neural Compressor takes a FP32 model and a yaml configuration file as inputs.
To construct the quantization process, users can either specify the below settings via
the yaml configuration file or python APIs:

1. Calibration Dataloader (Needed for static quantization)
2. Evaluation Dataloader
3. Evaluation Metric

Intel® Neural Compressor supports some popular dataloaders and evaluation metrics. For
how to configure them in yaml configuration file, user could refer to `Built-in Datasets
<https://github.com/intel/neural-compressor/blob/master/docs/dataset.md>`_.

If users want to use a self-developed dataloader or evaluation metric, Intel® Neural
Compressor supports this by the registration of customized dataloader/metric using python code.

For the yaml configuration file format please refer to `yaml template
<https://github.com/intel/neural-compressor/blob/master/neural_compressor/template/ptq.yaml>`_.

The code changes that are required for *Intel® Neural Compressor* are highlighted with
comments in the line above.

Model
^^^^^

In this tutorial, the LeNet model is used to demonstrate how to deal with *Intel® Neural Compressor*.

.. code-block:: python3

    # main.py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # LeNet Model definition
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc1_drop = nn.Dropout()
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.reshape(-1, 320)
            x = F.relu(self.fc1(x))
            x = self.fc1_drop(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net()
    model.load_state_dict(torch.load('./lenet_mnist_model.pth'))

The pretrained model weight `lenet_mnist_model.pth` comes from
`here <https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h?usp=sharing>`_.

Accuracy driven quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intel® Neural Compressor supports accuracy-driven automatic tuning to generate the optimal
int8 model which meets a predefined accuracy goal.

Below is an example of how to quantize a simple network on PyTorch
`FX graph mode <https://pytorch.org/docs/stable/fx.html>`_ by auto-tuning.

.. code-block:: yaml

    # conf.yaml
    model:
        name: LeNet
        framework: pytorch_fx

    evaluation:
        accuracy:
            metric:
                topk: 1

    tuning:
      accuracy_criterion:
        relative: 0.01

.. code-block:: python3

    # main.py
    model.eval()

    from torchvision import datasets, transforms
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=1)

    # launch code for Intel® Neural Compressor
    from neural_compressor.experimental import Quantization
    quantizer = Quantization("./conf.yaml")
    quantizer.model = model
    quantizer.calib_dataloader = test_loader
    quantizer.eval_dataloader = test_loader
    q_model = quantizer()
    q_model.save('./output')

In the `conf.yaml` file, the built-in metric `top1` of Intel® Neural Compressor is specified as
the evaluation method, and `1%` relative accuracy loss is set as the accuracy target for auto-tuning.
Intel® Neural Compressor will traverse all possible quantization config combinations on per-op level
to find out the optimal int8 model that reaches the predefined accuracy target.

Besides those built-in metrics, Intel® Neural Compressor also supports customized metric through
python code:

.. code-block:: yaml

    # conf.yaml
    model:
        name: LeNet
        framework: pytorch_fx

    tuning:
        accuracy_criterion:
            relative: 0.01

.. code-block:: python3

    # main.py
    model.eval()

    from torchvision import datasets, transforms
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=1)

    # define a customized metric
    class Top1Metric(object):
        def __init__(self):
            self.correct = 0
        def update(self, output, label):
            pred = output.argmax(dim=1, keepdim=True)
            self.correct += pred.eq(label.view_as(pred)).sum().item()
        def reset(self):
            self.correct = 0
        def result(self):
            return 100. * self.correct / len(test_loader.dataset)

    # launch code for Intel® Neural Compressor
    from neural_compressor.experimental import Quantization
    quantizer = Quantization("./conf.yaml")
    quantizer.model = model
    quantizer.calib_dataloader = test_loader
    quantizer.eval_dataloader = test_loader
    quantizer.metric = Top1Metric()
    q_model = quantizer()
    q_model.save('./output')

In the above example, a `class` which contains `update()` and `result()` function is implemented
to record per mini-batch result and calculate final accuracy at the end.

Quantization aware training
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides post-training static quantization and post-training dynamic quantization, Intel® Neural
Compressor supports quantization-aware training with an accuracy-driven automatic tuning mechanism.

Below is an example of how to do quantization aware training on a simple network on PyTorch
`FX graph mode <https://pytorch.org/docs/stable/fx.html>`_.

.. code-block:: yaml

    # conf.yaml
    model:
        name: LeNet
        framework: pytorch_fx

    quantization:
        approach: quant_aware_training

    evaluation:
        accuracy:
            metric:
                topk: 1

    tuning:
        accuracy_criterion:
            relative: 0.01

.. code-block:: python3

    # main.py
    model.eval()

    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1)

    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.1)

    def training_func(model):
        model.train()
        for epoch in range(1, 3):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), loss.item()))

    # launch code for Intel® Neural Compressor
    from neural_compressor.experimental import Quantization
    quantizer = Quantization("./conf.yaml")
    quantizer.model = model
    quantizer.q_func = training_func
    quantizer.eval_dataloader = test_loader
    q_model = quantizer()
    q_model.save('./output')

Performance only quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intel® Neural Compressor supports directly yielding int8 model with dummy dataset for the
performance benchmarking purpose.

Below is an example of how to quantize a simple network on PyTorch
`FX graph mode <https://pytorch.org/docs/stable/fx.html>`_ with a dummy dataset.

.. code-block:: yaml

    # conf.yaml
    model:
        name: lenet
        framework: pytorch_fx

.. code-block:: python3

    # main.py
    model.eval()

    # launch code for Intel® Neural Compressor
    from neural_compressor.experimental import Quantization, common
    from neural_compressor.experimental.data.datasets.dummy_dataset import DummyDataset
    quantizer = Quantization("./conf.yaml")
    quantizer.model = model
    quantizer.calib_dataloader = common.DataLoader(DummyDataset([(1, 1, 28, 28)]))
    q_model = quantizer()
    q_model.save('./output')

Quantization outputs
~~~~~~~~~~~~~~~~~~~~

Users could know how many ops get quantized from log printed by Intel® Neural Compressor
like below:

::

    2021-12-08 14:58:35 [INFO] |********Mixed Precision Statistics*******|
    2021-12-08 14:58:35 [INFO] +------------------------+--------+-------+
    2021-12-08 14:58:35 [INFO] |        Op Type         | Total  |  INT8 |
    2021-12-08 14:58:35 [INFO] +------------------------+--------+-------+
    2021-12-08 14:58:35 [INFO] |  quantize_per_tensor   |   2    |   2   |
    2021-12-08 14:58:35 [INFO] |         Conv2d         |   2    |   2   |
    2021-12-08 14:58:35 [INFO] |       max_pool2d       |   1    |   1   |
    2021-12-08 14:58:35 [INFO] |          relu          |   1    |   1   |
    2021-12-08 14:58:35 [INFO] |       dequantize       |   2    |   2   |
    2021-12-08 14:58:35 [INFO] |       LinearReLU       |   1    |   1   |
    2021-12-08 14:58:35 [INFO] |         Linear         |   1    |   1   |
    2021-12-08 14:58:35 [INFO] +------------------------+--------+-------+

The quantized model will be generated under `./output` directory, in which there are two files:
1. best_configure.yaml
2. best_model_weights.pt

The first file contains the quantization configurations of each op, the second file contains
int8 weights and zero point and scale info of activations.

Deployment
~~~~~~~~~~

Users could use the below code to load quantized model and then do inference or performance benchmark.

.. code-block:: python3

    from neural_compressor.utils.pytorch import load
    int8_model = load('./output', model)

Tutorials
---------

Please visit `Intel® Neural Compressor Github repo <https://github.com/intel/neural-compressor>`_
for more tutorials.
