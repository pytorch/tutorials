"""
Hyperparameter tuning with Ray Tune
===================================

Hyperparameter tuning can make the difference between an average model
and a highly accurate one. Often, simple decisions like choosing a
different learning rate or changing a network layer size can
dramatically impact model performance.

Fortunately, there are tools that help with finding the best combination
of parameters. `Ray Tune <https://docs.ray.io/en/latest/tune.html>`__ is
an industry standard tool for distributed hyperparameter tuning. Ray
Tune includes the latest hyperparameter search algorithms, integrates
with various analysis libraries, and natively supports distributed
training through `Ray’s distributed machine learning
engine <https://ray.io/>`__.

In this tutorial, we will show you how to integrate Ray Tune into your
PyTorch training workflow. We will extend `this tutorial from the
PyTorch
documentation <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`__
for training a CIFAR10 image classifier.

We only need to make minor modifications:

1. wrap data loading and training in functions,
2. make some network parameters configurable,
3. add checkpointing (optional),
4. define the search space for the model tuning

To run this tutorial, please make sure the following packages are
installed:

- ``ray[tune]``: Distributed hyperparameter tuning library
- ``torchvision``: For the data transformers

Setup / Imports
---------------

Let’s start with the imports:

"""

from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler

######################################################################
# Most of the imports are needed for building the PyTorch model. Only the
# last few are specific to Ray Tune.
#
# Data loaders
# ------------
#
# We wrap the data loaders in a function and pass a global data directory.
# This allows us to share a data directory across different trials.

def load_data(data_dir="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset

######################################################################
# Configurable neural network
# ---------------------------
#
# We can only tune parameters that are configurable. In this example, we
# specify the layer sizes of the fully connected layers:

class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

######################################################################
# The train function
# ------------------
#
# Now it gets interesting, because we introduce some changes to the
# example `from the PyTorch
# documentation <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`__.
#
# We wrap the training script in a function
# ``train_cifar(config, data_dir=None)``. The ``config`` parameter
# receives the hyperparameters we want to train with. The ``data_dir``
# specifies the directory where we load and store the data, allowing
# multiple runs to share the same data source. This is especially useful
# in cluster environments where you can mount a shared storage (e.g. NFS)
# to this directory, preventing the data from being downloaded to each
# node separately. We also load the model and optimizer state at the start
# of the run if a checkpoint is provided. Further down in this tutorial,
# you will find information on how to save the checkpoint and what it is
# used for.
#
# .. code-block:: python
#
#    net = Net(config["l1"], config["l2"])
#
#    checkpoint = tune.get_checkpoint()
#    if checkpoint:
#        with checkpoint.as_directory() as checkpoint_dir:
#            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
#            checkpoint_state = torch.load(checkpoint_path)
#            start_epoch = checkpoint_state["epoch"]
#            net.load_state_dict(checkpoint_state["net_state_dict"])
#            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
#    else:
#        start_epoch = 0
#
# The learning rate of the optimizer is made configurable, too:
#
# .. code-block:: python
#
#    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
#
# We also split the training data into a training and validation subset.
# We thus train on 80% of the data and calculate the validation loss on
# the remaining 20%. The batch sizes with which we iterate through the
# training and test sets are configurable as well.
#
# Adding (multi) GPU support with DataParallel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Image classification benefits largely from GPUs. Luckily, we can
# continue to use PyTorch’s abstractions in Ray Tune. Thus, we can wrap
# our model in ``nn.DataParallel`` to support data parallel training on
# multiple GPUs:
#
# .. code-block:: python
#
#    device = "cpu"
#    if torch.cuda.is_available():
#        device = "cuda:0"
#        if torch.cuda.device_count() > 1:
#            net = nn.DataParallel(net)
#    net.to(device)
#
# By using a ``device`` variable, we ensure that training works even
# without a GPU. PyTorch requires us to send our data to the GPU memory
# explicitly:
#
# .. code-block:: python
#
#    for i, data in enumerate(trainloader, 0):
#        inputs, labels = data
#        inputs, labels = inputs.to(device), labels.to(device)
#
# The code now supports training on CPUs, on a single GPU, and on multiple
# GPUs. Notably, Ray also supports `fractional
# GPUs <https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#fractional-accelerators>`__
# so we can share GPUs among trials, as long as the model still fits on
# the GPU memory. We’ll come back to that later.
#
# Communicating with Ray Tune
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The most interesting part is the communication with Ray Tune. As you’ll
# see, integrating Ray Tune into your training code requires only a few
# additional lines:
#
# .. code-block:: python
#
#    checkpoint_data = {
#        "epoch": epoch,
#        "net_state_dict": net.state_dict(),
#        "optimizer_state_dict": optimizer.state_dict(),
#    }
#    with tempfile.TemporaryDirectory() as checkpoint_dir:
#        checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
#        torch.save(checkpoint_data, checkpoint_path)
#
#        checkpoint = Checkpoint.from_directory(checkpoint_dir)
#        tune.report(
#            {"loss": val_loss / val_steps, "accuracy": correct / total},
#            checkpoint=checkpoint,
#        )
#
# Here we first save a checkpoint and then report some metrics back to Ray
# Tune. Specifically, we send the validation loss and accuracy back to Ray
# Tune. Ray Tune uses these metrics to determine the best hyperparameter
# configuration and to stop underperforming trials early, saving
# resources.
#
# The checkpoint saving is optional, however, it is necessary if we wanted
# to use advanced schedulers like `Population Based
# Training <https://docs.ray.io/en/latest/tune/examples/pbt_guide.html>`__.
# Saving the checkpoint also allows us to later load the trained models
# for validation on a test set. Lastly, it provides fault tolerance,
# enabling us to pause and resume training.
#
# To summarize, integrating Ray Tune into your PyTorch training requires
# just a few key additions:
#
# - ``tune.report()`` to report metrics (and optionally checkpoints) to
#   Ray Tune
# - ``tune.get_checkpoint()`` to load a model from a checkpoint
# - ``Checkpoint.from_directory()`` to create a checkpoint object from
#   saved state
#
# The rest of your training code remains standard PyTorch!
#
# Full training function
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The full code example looks like this:

def train_cifar(config, data_dir=None):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # Load checkpoint if resuming training
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            checkpoint_state = torch.load(checkpoint_path)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=2
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Save checkpoint and report metrics
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            torch.save(checkpoint_data, checkpoint_path)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            tune.report(
                {"loss": val_loss / val_steps, "accuracy": correct / total},
                checkpoint=checkpoint,
            )

    print("Finished Training")

######################################################################
# As you can see, most of the code is adapted directly from the original
# example.
#
# Test set accuracy
# -----------------
#
# Commonly the performance of a machine learning model is tested on a
# hold-out test set with data that has not been used for training the
# model. We also wrap this in a function:

def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

######################################################################
# The function also expects a ``device`` parameter, so we can do the test
# set validation on a GPU.
#
# Configuring the search space
# ----------------------------
#
# Lastly, we need to define Ray Tune’s search space. Here is an example:
#
# .. code-block:: python
#
#    config = {
#        "l1": tune.choice([2 ** i for i in range(9)]),
#        "l2": tune.choice([2 ** i for i in range(9)]),
#        "lr": tune.loguniform(1e-4, 1e-1),
#        "batch_size": tune.choice([2, 4, 8, 16])
#    }
#
# The ``tune.choice()`` accepts a list of values that are uniformly
# sampled from. In this example, the ``l1`` and ``l2`` parameters should
# be powers of 2 between 4 and 256, so either 4, 8, 16, 32, 64, 128, or
# 256. The ``lr`` (learning rate) should be uniformly sampled between
# 0.0001 and 0.1. Lastly, the batch size is a choice between 2, 4, 8, and
# 16.
#
# For each trial, Ray Tune samples a combination of parameters from these
# search spaces according to the search space configuration and search
# strategy. It then trains multiple models in parallel to identify the
# best performing one.
#
# By default, Ray Tune uses random search to pick the next hyperparameter
# configuration to try. However, Ray Tune also provides more sophisticated
# search algorithms that can more efficiently navigate the search space,
# such as
# `Optuna <https://docs.ray.io/en/latest/tune/api/suggestion.html#optuna>`__,
# `HyperOpt <https://docs.ray.io/en/latest/tune/api/suggestion.html#hyperopt>`__,
# and `Bayesian
# Optimization <https://docs.ray.io/en/latest/tune/api/suggestion.html#bayesopt>`__.
#
# We use the ``ASHAScheduler`` to terminate underperforming trials early.
#
# We wrap the ``train_cifar`` function with ``functools.partial`` to set
# the constant ``data_dir`` parameter. We can also tell Ray Tune what
# resources should be available for each trial using
# ``tune.with_resources``:
#
# .. code-block:: python
#
#    gpus_per_trial = 2
#    # ...
#    tuner = tune.Tuner(
#        tune.with_resources(
#            partial(train_cifar, data_dir=data_dir),
#            resources={"cpu": 8, "gpu": gpus_per_trial}
#        ),
#        tune_config=tune.TuneConfig(
#            metric="loss",
#            mode="min",
#            scheduler=scheduler,
#            num_samples=num_samples,
#        ),
#        param_space=config,
#    )
#    results = tuner.fit()
#
# You can specify the number of CPUs, which are then available e.g. to
# increase the ``num_workers`` of the PyTorch ``DataLoader`` instances.
# The selected number of GPUs are made visible to PyTorch in each trial.
# Trials do not have access to GPUs that haven’t been requested, so you
# don’t need to worry about resource contention.
#
# You can also specify fractional GPUs (e.g., ``gpus_per_trial=0.5``),
# which allows trials to share a GPU. Just ensure that the models fit
# within the GPU memory.
#
# After training the models, we will find the best performing one and load
# the trained network from the checkpoint file. We then obtain the test
# set accuracy and report everything by printing.
#
# The full main function looks like this. Note that the
# ``if __name__ == "__main__":`` block is configured for a quick run (1
# trial, 1 epoch, CPU only) to verify that everything works. You should
# increase these values to perform an actual hyperparameter tuning search.

def main(num_trials=10, max_num_epochs=10, gpus_per_trial=2):
    print("Starting hyperparameter tuning.")
    ray.init()
    
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    
    tuner = tune.Tuner(
        tune.with_resources(
            partial(train_cifar, data_dir=data_dir),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_trials,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_result.metrics['accuracy']}")

    best_trained_model = Net(best_result.config["l1"], best_result.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_result.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
        best_checkpoint_data = torch.load(checkpoint_path)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_acc = test_accuracy(best_trained_model, device)
        print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # Set the number of trials, epochs, and GPUs per trial here:
    main(num_trials=1, max_num_epochs=1, gpus_per_trial=0)

######################################################################
# If you run the code, an example output could look like this:
#
# .. code-block:: bash
#
#    Number of trials: 10/10 (10 TERMINATED)
#    +-----+--------------+------+------+-------------+--------+---------+------------+
#    | ... |   batch_size |   l1 |   l2 |          lr |   iter |    loss |   accuracy |
#    |-----+--------------+------+------+-------------+--------+---------+------------|
#    | ... |            2 |    1 |  256 | 0.000668163 |      1 | 2.31479 |     0.0977 |
#    | ... |            4 |   64 |    8 | 0.0331514   |      1 | 2.31605 |     0.0983 |
#    | ... |            4 |    2 |    1 | 0.000150295 |      1 | 2.30755 |     0.1023 |
#    | ... |           16 |   32 |   32 | 0.0128248   |     10 | 1.66912 |     0.4391 |
#    | ... |            4 |    8 |  128 | 0.00464561  |      2 | 1.7316  |     0.3463 |
#    | ... |            8 |  256 |    8 | 0.00031556  |      1 | 2.19409 |     0.1736 |
#    | ... |            4 |   16 |  256 | 0.00574329  |      2 | 1.85679 |     0.3368 |
#    | ... |            8 |    2 |    2 | 0.00325652  |      1 | 2.30272 |     0.0984 |
#    | ... |            2 |    2 |    2 | 0.000342987 |      2 | 1.76044 |     0.292  |
#    | ... |            4 |   64 |   32 | 0.003734    |      8 | 1.53101 |     0.4761 |
#    +-----+--------------+------+------+-------------+--------+---------+------------+
#
#    Best trial config: {'l1': 64, 'l2': 32, 'lr': 0.0037339984519545164, 'batch_size': 4}
#    Best trial final validation loss: 1.5310075663924216
#    Best trial final validation accuracy: 0.4761
#    Best trial test set accuracy: 0.4737
#
# Most trials were stopped early to conserve resources. The best
# performing trial achieved a validation accuracy of approximately 47%,
# which could be confirmed on the test set.
#
# So that’s it! You can now tune the parameters of your PyTorch models.
