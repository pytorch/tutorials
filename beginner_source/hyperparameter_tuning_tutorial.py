"""
Hyperparameter tuning using Ray Tune
====================================

**Author:** `Ricardo Decal <https://github.com/crypdick>`__

This tutorial shows how to integrate Ray Tune into your PyTorch training
workflow to perform scalable and efficient hyperparameter tuning.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to modify a PyTorch training loop for Ray Tune
       * How to scale a hyperparameter sweep to multiple nodes and GPUs without code changes
       * How to define a hyperparameter search space and run a sweep with ``tune.Tuner``
       * How to use an early-stopping scheduler (ASHA) and report metrics/checkpoints
       * How to use checkpointing to resume training and load the best model

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.9+ and ``torchvision``
       * Ray Tune (``ray[tune]``) v2.52.1+
       * GPU(s) are optional, but recommended for faster training

`Ray <https://docs.ray.io/en/latest/index.html>`__, a project of the
PyTorch Foundation, is an open source unified framework for scaling AI
and Python applications. It helps run distributed jobs by handling the
complexity of distributed computing. `Ray
Tune <https://docs.ray.io/en/latest/tune/index.html>`__ is a library
built on Ray for hyperparameter tuning that enables you to scale a
hyperparameter sweep from your machine to a large cluster with no code
changes.

This tutorial adapts the `PyTorch tutorial for training a CIFAR10
classifier <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`__
to run multi-GPU hyperparameter sweeps with Ray Tune.

Setup
-----

To run this tutorial, install the following dependencies:

.. code-block:: bash

   pip install "ray[tune]" torchvision

"""

######################################################################
# Then start with the imports:

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
# New: imports for Ray Tune
import ray
from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler

######################################################################
# Data loading
# ============
#
# Wrap the data loaders in a constructor function. In this tutorial, a
# global data directory is passed to the function to enable reusing the
# dataset across different trials. In a cluster environment, you can use
# shared storage, such as network file systems, to prevent each node from
# downloading the data separately.

def load_data(data_dir="./data"):
    # Mean and standard deviation of the CIFAR10 training subset.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.48216, 0.44653), (0.2022, 0.19932, 0.20086))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset

######################################################################
# Model architecture
# ==================
#
# This tutorial searches for the best sizes for the fully connected layers
# and the learning rate. To enable this, the ``Net`` class exposes the
# layer sizes ``l1`` and ``l2`` as configurable parameters that Ray Tune
# can search over:

class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super().__init__()
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
# Define the search space
# =======================
#
# Next, define the hyperparameters to tune and how Ray Tune samples them.
# Ray Tune offers a variety of `search space
# distributions <https://docs.ray.io/en/latest/tune/api/search_space.html>`__
# to suit different parameter types: ``loguniform``, ``uniform``,
# ``choice``, ``randint``, ``grid``, and more. You can also express
# complex dependencies between parameters with `conditional search
# spaces <https://docs.ray.io/en/latest/tune/tutorials/tune-search-spaces.html#how-to-use-custom-and-conditional-search-spaces-in-tune>`__
# or sample from arbitrary functions.
#
# Here is the search space for this tutorial:
#
# .. code-block:: python
#
#    config = {
#        "l1": tune.choice([2**i for i in range(9)]),
#        "l2": tune.choice([2**i for i in range(9)]),
#        "lr": tune.loguniform(1e-4, 1e-1),
#        "batch_size": tune.choice([2, 4, 8, 16]),
#    }
#
# The ``tune.choice()`` accepts a list of values that are uniformly
# sampled from. In this example, the ``l1`` and ``l2`` parameter values
# are powers of 2 between 1 and 256, and the learning rate samples on a
# log scale between 0.0001 and 0.1. Sampling on a log scale enables
# exploration across a range of magnitudes on a relative scale, rather
# than an absolute scale.
#
# Training function
# =================
#
# Ray Tune requires a training function that accepts a configuration
# dictionary and runs the main training loop. As Ray Tune runs different
# trials, it updates the configuration dictionary for each trial.
#
# Here is the full training function, followed by explanations of the key
# Ray Tune integration points:

def train_cifar(config, data_dir=None):
    net = Net(config["l1"], config["l2"])
    device = config["device"]

    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

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

    trainset, _testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
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
# Key integration points
# ----------------------
#
# Using hyperparameters from the configuration dictionary
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Ray Tune updates the ``config`` dictionary with the hyperparameters for
# each trial. In this example, the model architecture and optimizer
# receive the hyperparameters from the ``config`` dictionary:
#
# .. code-block:: python
#
#    net = Net(config["l1"], config["l2"])
#    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
#
# Reporting metrics and saving checkpoints
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The most important integration is communicating with Ray Tune. Ray Tune
# uses the validation metrics to determine the best hyperparameter
# configuration and to stop underperforming trials early, saving
# resources.
#
# Checkpointing enables you to later load the trained models, resume
# hyperparameter searches, and provides fault tolerance. It’s also
# required for some Ray Tune schedulers like `Population Based
# Training <https://docs.ray.io/en/latest/tune/examples/pbt_guide.html>`__
# that pause and resume trials during the search.
#
# This code from the training function loads model and optimizer state at
# the start if a checkpoint exists:
#
# .. code-block:: python
#
#    checkpoint = tune.get_checkpoint()
#    if checkpoint:
#        with checkpoint.as_directory() as checkpoint_dir:
#            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
#            checkpoint_state = torch.load(checkpoint_path)
#            start_epoch = checkpoint_state["epoch"]
#            net.load_state_dict(checkpoint_state["net_state_dict"])
#            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
#
# At the end of each epoch, save a checkpoint and report the validation
# metrics:
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
# Ray Tune checkpointing supports local file systems, cloud storage, and
# distributed file systems. For more information, see the `Ray Tune
# storage
# documentation <https://docs.ray.io/en/latest/tune/tutorials/tune-storage.html>`__.
#
# Multi-GPU support
# ~~~~~~~~~~~~~~~~~
#
# Image classification models can be greatly accelerated by using GPUs.
# The training function supports multi-GPU training by wrapping the model
# in ``nn.DataParallel``:
#
# .. code-block:: python
#
#    if torch.cuda.device_count() > 1:
#        net = nn.DataParallel(net)
#
# This training function supports training on CPUs, a single GPU, multiple GPUs, or
# multiple nodes without code changes. Ray Tune automatically distributes the trials
# across the nodes according to the available resources. Ray Tune also supports `fractional
# GPUs <https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#fractional-accelerators>`__
# so that one GPU can be shared among multiple trials, provided that the
# models, optimizers, and data batches fit into the GPU memory.
#
# Validation split
# ~~~~~~~~~~~~~~~~
#
# The original CIFAR10 dataset only has train and test subsets. This is
# sufficient for training a single model, however for hyperparameter
# tuning a validation subset is required. The training function creates a
# validation subset by reserving 20% of the training subset. The test
# subset is used to evaluate the best model’s generalization error after
# the search completes.
#
# Evaluation function
# ===================
#
# After finding the optimal hyperparameters, test the model on a held-out
# test set to estimate the generalization error:

def test_accuracy(net, device="cpu", data_dir=None):
    _trainset, testset = load_data(data_dir)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            image_batch, labels = data
            image_batch, labels = image_batch.to(device), labels.to(device)
            outputs = net(image_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

######################################################################
# Configure and run Ray Tune
# ==========================
#
# With the training and evaluation functions defined, configure Ray Tune
# to run the hyperparameter search.
#
# Scheduler for early stopping
# ----------------------------
#
# Ray Tune provides schedulers to improve the efficiency of the
# hyperparameter search by detecting underperforming trials and stopping
# them early. The ``ASHAScheduler`` uses the Asynchronous Successive
# Halving Algorithm (ASHA) to aggressively terminate low-performing
# trials:
#
# .. code-block:: python
#
#    scheduler = ASHAScheduler(
#        max_t=max_num_epochs,
#        grace_period=1,
#        reduction_factor=2,
#    )
#
# Ray Tune also provides `advanced search
# algorithms <https://docs.ray.io/en/latest/tune/api/suggestion.html>`__
# to smartly pick the next set of hyperparameters based on previous
# results, instead of relying only on random or grid search. Examples
# include
# `Optuna <https://docs.ray.io/en/latest/tune/api/suggestion.html#optuna>`__
# and
# `BayesOpt <https://docs.ray.io/en/latest/tune/api/suggestion.html#bayesopt>`__.
#
# Resource allocation
# -------------------
#
# Tell Ray Tune what resources to allocate for each trial by passing a
# ``resources`` dictionary to ``tune.with_resources``:
#
# .. code-block:: python
#
#    tune.with_resources(
#        partial(train_cifar, data_dir=data_dir),
#        resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
#    )
#
# Ray Tune automatically manages the placement of these trials and ensures
# that the trials run in isolation, so you don’t need to manually assign
# GPUs to processes.
#
# For example, if you are running this experiment on a cluster of 20
# machines, each with 8 GPUs, you can set ``gpus_per_trial = 0.5`` to
# schedule two concurrent trials per GPU. This configuration runs 320
# trials in parallel across the cluster.
#
#    **Note**: To run this tutorial without GPUs, set ``gpus_per_trial=0``
#    and expect significantly longer runtimes.
#
#    To avoid long runtimes during development, start with a small number
#    of trials and epochs.
#
# Creating the Tuner
# ------------------
#
# The Ray Tune API is modular and composable. Pass your configuration to
# the ``tune.Tuner`` class to create a tuner object, then run
# ``tuner.fit()`` to start training:
#
# .. code-block:: python
#
#    tuner = tune.Tuner(
#        tune.with_resources(
#            partial(train_cifar, data_dir=data_dir),
#            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
#        ),
#        tune_config=tune.TuneConfig(
#            metric="loss",
#            mode="min",
#            scheduler=scheduler,
#            num_samples=num_trials,
#        ),
#        param_space=config,
#    )
#    results = tuner.fit()
#
# After training completes, retrieve the best performing trial, load its
# checkpoint, and evaluate on the test set.
#
# Putting it all together
# -----------------------

def main(num_trials=10, max_num_epochs=10, gpus_per_trial=0, cpus_per_trial=2):
    print("Starting hyperparameter tuning.")
    ray.init(include_dashboard=False)
    
    data_dir = os.path.abspath("./data")
    load_data(data_dir)  # Pre-download the dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "device": device,
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    
    tuner = tune.Tuner(
        tune.with_resources(
            partial(train_cifar, data_dir=data_dir),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
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
    best_trained_model = best_trained_model.to(device)
    if gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)

    best_checkpoint = best_result.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
        best_checkpoint_data = torch.load(checkpoint_path)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_acc = test_accuracy(best_trained_model, device, data_dir)
        print(f"Best trial test set accuracy: {test_acc}")


if __name__ == "__main__":
    # Set the number of trials, epochs, and GPUs per trial here:
    main(num_trials=10, max_num_epochs=10, gpus_per_trial=1)

######################################################################
# Results
# =======
#
# Your Ray Tune trial summary output looks something like this. The text
# table summarizes the validation performance of the trials and highlights
# the best hyperparameter configuration:
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
# Most trials stopped early to conserve resources. The best performing
# trial achieved a validation accuracy of approximately 47%, which the
# test set confirms.
#
# Observability
# =============
#
# Monitoring is critical when running large-scale experiments. Ray
# provides a
# `dashboard <https://docs.ray.io/en/latest/ray-observability/getting-started.html>`__
# that lets you view the status of your trials, check cluster resource
# use, and inspect logs in real time.
#
# For debugging, Ray also offers `distributed debugging
# tools <https://docs.ray.io/en/latest/ray-observability/index.html>`__
# that let you attach a debugger to running trials across the cluster.
#
# Conclusion
# ==========
#
# In this tutorial, you learned how to tune the hyperparameters of a
# PyTorch model using Ray Tune. You saw how to integrate Ray Tune into
# your PyTorch training loop, define a search space for your
# hyperparameters, use an efficient scheduler like ASHAScheduler to
# terminate low-performing trials early, save checkpoints and report
# metrics to Ray Tune, and run the hyperparameter search and analyze the
# results.
#
# Ray Tune makes it straightforward to scale your experiments from a
# single machine to a large cluster, helping you find the best model
# configuration efficiently.
#
# Further reading
# ===============
#
# - `Ray Tune
#   documentation <https://docs.ray.io/en/latest/tune/index.html>`__
# - `Ray Tune
#   examples <https://docs.ray.io/en/latest/tune/examples/index.html>`__
