# -*- coding: utf-8 -*-


"""
Implementing a Keras Progress Bar
=================================

**Author:** `Logan Thomas <https://github.com/loganthomas>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * How to implement Keras's model training progress bar in PyTorch
"""

#########################################################################
# Overview
# --------
#
# Keras implements a progress bar that shows a nice snapshot of what is occurring during model training.
# This can be more visually appealing and less burden some for a developer to write the code to report
# key metrics. Adding a Keras progress bar only takes a few lines of code.
#

#########################################################################
# Traditional Approach
# --------------------
#
# Let's start with a simple example borrowing from the below `Learn the Basics Tutorials <https://pytorch.org/tutorials/beginner/basics/intro.html>`__:
#
# - `Datasets & DataLoaders <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`__
# - `Build the Neural Network  <https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html>`__
# - `Optimizing Model Parameters <https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>`__
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 64

training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
print(model)


######################################################################
# Define a Traditional Training Loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_loop(dataloader, model, batch_size, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode
    # Important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


######################################################################
# Define a Traditional Testing Loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    # Important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad()
    # ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations
    # and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


######################################################################
# Train the Model
# ~~~~~~~~~~~~~~~
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 10
batch_size = BATCH_SIZE
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, batch_size, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

######################################################################
# Setup a Keras Progress Bar
# --------------------------
#
# The same train loop and test loop can be performed using Keras's `ProgBar class <https://keras.io/api/utils/python_utils/#progbar-class>`__.
# This can cut down on the code needed to report metrics during training and be more visual appealing.
#
# Define a Training Loop with a Progress Bar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras


def train_loop_pbar(dataloader, model, loss_fn, optimizer, epochs):
    # Move the epoch looping into the train loop
    # so epoch and pbar reported together
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}")

        # Set the target of the pbar to
        # the number of iterations (batches) in the dataloader
        n_batches = len(dataloader)
        pbar = keras.utils.Progbar(target=n_batches)

        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= len(y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update the pbar with each batch
            pbar.update(batch, values=[("loss", loss), ("acc", correct)])

            # Finish the progress bar with a final update
            # This ensures the progress bar reaches the end
            # (i.e. the target of n_batches is met on the last batch)
            if batch + 1 == n_batches:
                pbar.update(n_batches, values=None)


######################################################################
# Define a Testing Loop with a Progress Bar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_loop_pbar(dataloader, model, loss_fn):
    n_batches = len(dataloader)
    pbar = keras.utils.Progbar(target=n_batches)

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= len(y)

            pbar.update(batch, values=[("loss", loss), ("acc", correct)])

            if batch + 1 == n_batches:
                pbar.update(n_batches, values=None)


######################################################################
# Train the Model with a Progress Bar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Re-instantiate model
model = NeuralNetwork()
######################################################################
# Train the model with progress bar
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 10
train_loop_pbar(train_dataloader, model, loss_fn, optimizer, epochs)
######################################################################
# Evaluate the model with progress bar
test_loop_pbar(test_dataloader, model, loss_fn)

######################################################################
# Including a Validation Dataset
# ------------------------------
#
# The progress bar can handle validation datasets as well.
# Start by splitting the existing training data into a new training set and validation set using an 80%-20% split.
# That is, 48,000 new training and 12,000 validation datapoints.

new_train_set, val_set = torch.utils.data.random_split(
    training_data, [int(len(training_data) * 0.8), int(len(training_data) * 0.2)]
)

new_train_dataloader = DataLoader(new_train_set, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE)

print(len(train_dataloader), len(train_dataloader.dataset))
print()
print(len(new_train_dataloader), len(new_train_dataloader.dataset))
print(len(val_dataloader), len(val_dataloader.dataset))


######################################################################
# Define a Training Loop with a Progress Bar Including Validation Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_loop_pbar_with_val(
    train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs
):
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}")

        n_batches = len(train_dataloader)
        pbar = keras.utils.Progbar(target=n_batches)

        # Train step
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= len(y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update the pbar with each batch
            pbar.update(batch, values=[("loss", loss), ("acc", correct)])

        # Validation step
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for X, y in val_dataloader:
                val_pred = model(X)
                val_loss += loss_fn(val_pred, y).item()
                val_correct += (val_pred.argmax(1) == y).type(torch.float).sum().item()

            val_loss /= len(val_dataloader)
            val_correct /= len(val_dataloader.dataset)

            # Final update now belongs to the validation data
            pbar.update(
                n_batches, values=[("val_loss", val_loss), ("val_acc", val_correct)]
            )


######################################################################
# Train the Model with a Progress Bar Including Validation Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Re-instantiate model
model = NeuralNetwork()
######################################################################
# Train the model with progress bar
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 10
train_loop_pbar_with_val(
    new_train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs
)
######################################################################
# As a sniff test, run the original `test_loop()` with the validation dataset and
# verify the output is the same as the last epoch in the `train_loop_pbar_with_val()` loop:
test_loop(val_dataloader, model, loss_fn)

######################################################################
# Conclusion
# ----------
#
# Writing training loops and test loops can be verbose, especially when gathering important metrics to report.
# Utilizing Keras's progress bar not only makes for a more visually appealing report, but can
# also simplify the amount of code needed for gathering and reporting metrics when training a model.
#
# This recipe tutorial explored how to replace the traditional manual reporting in a
# training and test loop with an approach that uses Keras's `ProgBar class <https://keras.io/api/utils/python_utils/#progbar-class>`__.
# It was shown how to employ a progress bar with not just a training and testing set,
# but also with a validation set to be used during model training.
