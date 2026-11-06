Note

Go to the end
to download the full example code.

# What is a state_dict in PyTorch

In PyTorch, the learnable parameters (i.e. weights and biases) of a
`torch.nn.Module` model are contained in the model's parameters
(accessed with `model.parameters()`). A `state_dict` is simply a
Python dictionary object that maps each layer to its parameter tensor.

## Introduction

A `state_dict` is an integral entity if you are interested in saving
or loading models from PyTorch.
Because `state_dict` objects are Python dictionaries, they can be
easily saved, updated, altered, and restored, adding a great deal of
modularity to PyTorch models and optimizers.
Note that only layers with learnable parameters (convolutional layers,
linear layers, etc.) and registered buffers (batchnorm's running_mean)
have entries in the model's `state_dict`. Optimizer objects
(`torch.optim`) also have a `state_dict`, which contains information
about the optimizer's state, as well as the hyperparameters used.
In this recipe, we will see how `state_dict` is used with a simple
model.

## Setup

Before we begin, we need to install `torch` if it isn't already
available.

```
pip install torch
```

## Steps

1. Import all necessary libraries for loading our data
2. Define and initialize the neural network
3. Initialize the optimizer
4. Access the model and optimizer `state_dict`

### 1. Import necessary libraries for loading our data

For this recipe, we will use `torch` and its subsidiaries `torch.nn`
and `torch.optim`.

### 2. Define and initialize the neural network

For sake of example, we will create a neural network for training
images. To learn more see the Defining a Neural Network recipe.

### 3. Initialize the optimizer

We will use SGD with momentum.

### 4. Access the model and optimizer `state_dict`

Now that we have constructed our model and optimizer, we can understand
what is preserved in their respective `state_dict` properties.

```
# Print model's state_dict

# Print optimizer's state_dict
```

This information is relevant for saving and loading the model and
optimizers for future use.

Congratulations! You have successfully used `state_dict` in PyTorch.

## Learn More

Take a look at these other recipes to continue your learning:

- [Saving and loading models for inference in PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html)
- [Saving and loading a general checkpoint in PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: what_is_state_dict.ipynb`](../../_downloads/597dbaac5c207608e108534fea081ff9/what_is_state_dict.ipynb)

[`Download Python source code: what_is_state_dict.py`](../../_downloads/c087bb345bbf8da823696bf30d4cf850/what_is_state_dict.py)

[`Download zipped: what_is_state_dict.zip`](../../_downloads/7eec06ac2cdf546b53c11d8fa75af1da/what_is_state_dict.zip)