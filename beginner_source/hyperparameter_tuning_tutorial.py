"""
Hyperparameter tuning with Ray Tune
===================================

This tutorial shows how to integrate Ray Tune into your PyTorch training
workflow to perform scalable and efficient hyperparameter tuning.

`Ray <https://docs.ray.io/en/latest/index.html>`__, a project of the
PyTorch Foundation, is an open-source unified framework for scaling AI
and Python applications. It helps run distributed workloads by handling
the complexity of distributed computing. `Ray
Tune <https://docs.ray.io/en/latest/tune/index.html>`__ is a library
built on Ray for hyperparameter tuning that enables you to scale a
hyperparameter sweep from your machine to a large cluster with no code
changes.

This tutorial extends the PyTorch tutorial for training a CIFAR10 image
classifier in the `CIFAR10 tutorial (PyTorch
documentation) <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`__.
Only minor modifications are needed to adapt the PyTorch tutorial for
Ray Tune. Specifically, this tutorial wraps the data loading and
training in functions, makes some network parameters configurable, adds
optional checkpointing, and defines the search space for model tuning.

Setup
-----

To run this tutorial, install the dependencies:

"""

# %%bash
# pip install "ray[tune]" torchvision

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
# How to use PyTorch data loaders with Ray Tune
# ---------------------------------------------
#
# Wrap the data loaders in a constructor function. Pass a global data
# directory here to reuse the dataset across different trials.

def load_data(data_dir="./data"):
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
# Configure the hyperparameters
# -----------------------------
#
# In this example, we specify the layer sizes of the fully connected
# layers.

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
# Use a train function with Ray Tune
# ----------------------------------
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
# in cluster environments where you can mount shared storage (for example
# NFS) to prevent the data from being downloaded to each node separately.
# We also load the model and optimizer state at the start of the run if a
# checkpoint is provided. Further down in this tutorial, you will find
# information on how to save the checkpoint and how it is used.
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
# Add multi-GPU support with DataParallel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Image classification benefits largely from GPUs. Luckily, you can
# continue to use PyTorch tools in Ray Tune. Thus, you can wrap the model
# in ``nn.DataParallel`` to support data-parallel training on multiple
# GPUs:
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
# the GPU memory. We will return to that later.
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
# The checkpoint saving is optional. However, it is necessary if we wanted
# to use advanced schedulers like `Population Based
# Training <https://docs.ray.io/en/latest/tune/examples/pbt_guide.html>`__.
# Saving the checkpoint also allows us to later load the trained models
# for validation on a test set. Lastly, it provides fault tolerance,
# enabling us to pause and resume training.
#
# To summarize, integrating Ray Tune into your PyTorch training requires
# just a few key additions: use ``tune.report()`` to report metrics (and
# optionally checkpoints) to Ray Tune, ``tune.get_checkpoint()`` to load a
# model from a checkpoint, and ``Checkpoint.from_directory()`` to create a
# checkpoint object from saved state. The rest of your training code
# remains standard PyTorch.
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
# Compute test set accuracy
# -------------------------
#
# Commonly the performance of a machine learning model is tested on a
# held-out test set with data that has not been used for training the
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
# The function also expects a ``device`` parameter so you can run the test
# set validation on a GPU.
#
# Configure the search space
# --------------------------
#
# Lastly, we need to define Ray Tune’s search space. Ray Tune offers a
# variety of `search space
# distributions <https://docs.ray.io/en/latest/tune/api/search_space.html>`__
# to suit different parameter types: ``loguniform``, ``uniform``,
# ``choice``, ``randint``, ``grid``, and more. It also lets you express
# complex dependencies between parameters with `conditional search
# spaces <https://docs.ray.io/en/latest/tune/tutorials/tune-search-spaces.html#how-to-use-custom-and-conditional-search-spaces-in-tune>`__.
#
# Here is an example:
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
# will be powers of 2 between 1 and 256. The learning rate is sampled on a
# log scale between 0.0001 and 0.1. Sampling on a log scale ensures that
# the search space is explored efficiently across different magnitudes.
#
# Smarter sampling and scheduling
# -------------------------------
#
# To make the hyperparameter search process efficient, Ray Tune provides
# two main controls:
#
# 1. It can intelligently pick the next set of hyperparameters to test
#    based on previous results using `advanced search
#    algorithms <https://docs.ray.io/en/latest/tune/api/suggestion.html>`__
#    such as
#    `Optuna <https://docs.ray.io/en/latest/tune/api/suggestion.html#optuna>`__
#    or
#    ```bayesopt`` <https://docs.ray.io/en/latest/tune/api/suggestion.html#bayesopt>`__,
#    instead of relying only on random or grid search.
# 2. It can detect underperforming trials and stop them early using
#    `schedulers <https://docs.ray.io/en/latest/tune/key-concepts.html#tune-schedulers>`__,
#    enabling you to explore the parameter space more on the same compute
#    budget.
#
# In this tutorial, we use the ``ASHAScheduler``, which aggressively
# terminates low-performing trials to save computational resources.
#
# Configure the resources
# -----------------------
#
# Tell Ray Tune what resources should be available for each trial using
# ``tune.with_resources``:
#
# .. code-block:: python
#
#    tune.with_resources(
#        partial(train_cifar, data_dir=data_dir),
#        resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
#    )
#
# This tells Ray Tune to allocate ``cpus_per_trial`` CPUs and
# ``gpus_per_trial`` GPUs for each trial. Ray Tune automatically manages
# the placement of these trials and ensures they are isolated, so you
# don’t need to manually assign GPUs to processes.
#
# For example, if you are running this experiment on a cluster of 20
# machines, each with 8 GPUs, you can set ``gpus_per_trial = 0.5`` to
# schedule 2 concurrent trials per GPU. This configuration runs 320 trials
# in parallel across the cluster.
#
# Putting it together
# -------------------
#
# The Ray Tune API is designed to be modular and composable: you pass your
# configurations to the ``tune.Tuner`` class to create a tuner object,
# then execute ``tuner.fit()`` to start training:
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
#            num_samples=num_samples,
#        ),
#        param_space=config,
#    )
#    results = tuner.fit()
#
# After training the models, we will find the best performing one and load
# the trained network from the checkpoint file. We then obtain the test
# set accuracy and report the results.

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
    # The following configuration is for a quick run (1 trial, 1 epoch, CPU only) for demonstration purposes.
    main(num_trials=1, max_num_epochs=1, gpus_per_trial=0)

######################################################################
# Your Ray Tune trial summary output will look something like this:
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
# You can now tune the parameters of your PyTorch models.
#
# Observability
# -------------
#
# When running large-scale experiments, monitoring is crucial. Ray
# provides a
# `Dashboard <https://docs.ray.io/en/latest/ray-observability/getting-started.html>`__
# that lets you view the status of your trials, check cluster resource
# utilization, and inspect logs in real-time.
#
# For debugging, Ray also offers `Distributed
# Debugging <https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugger.html>`__
# tools that let you attach a debugger to running trials across the
# cluster.
#
# Conclusion
# ----------
#
# In this tutorial, you learned how to tune the hyperparameters of a
# PyTorch model using Ray Tune. You saw how to integrate Ray Tune into
# your PyTorch training loop, define a search space for your
# hyperparameters, use an efficient scheduler like ASHA to terminate bad
# trials early, save checkpoints and report metrics to Ray Tune, and run
# the hyperparameter search and analyze the results.
#
# Ray Tune makes it easy to scale your experiments from a single machine
# to a large cluster, helping you find the best model configuration
# efficiently.
#
# Further reading
# ---------------
#
# - `Ray Tune
#   documentation <https://docs.ray.io/en/latest/tune/index.html>`__
# - `Ray Tune
#   examples <https://docs.ray.io/en/latest/tune/examples/index.html>`__
