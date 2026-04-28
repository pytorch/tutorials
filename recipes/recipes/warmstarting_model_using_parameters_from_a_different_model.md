Note

Go to the end
to download the full example code.

# Warmstarting model using parameters from a different model in PyTorch

Partially loading a model or loading a partial model are common
scenarios when transfer learning or training a new complex model.
Leveraging trained parameters, even if only a few are usable, will help
to warmstart the training process and hopefully help your model converge
much faster than training from scratch.

## Introduction

Whether you are loading from a partial `state_dict`, which is missing
some keys, or loading a `state_dict` with more keys than the model
that you are loading into, you can set the strict argument to `False`
in the `load_state_dict()` function to ignore non-matching keys.
In this recipe, we will experiment with warmstarting a model using
parameters of a different model.

## Setup

Before we begin, we need to install `torch` if it isn't already
available.

```
pip install torch
```

## Steps

1. Import all necessary libraries for loading our data
2. Define and initialize the neural network A and B
3. Save model A
4. Load into model B

### 1. Import necessary libraries for loading our data

For this recipe, we will use `torch` and its subsidiaries `torch.nn`
and `torch.optim`.

### 2. Define and initialize the neural network A and B

For sake of example, we will create a neural network for training
images. To learn more see the Defining a Neural Network recipe. We will
create two neural networks for sake of loading one parameter of type A
into type B.

### 3. Save model A

```
# Specify a path to save to
```

### 4. Load into model B

If you want to load parameters from one layer to another, but some keys
do not match, simply change the name of the parameter keys in the
state_dict that you are loading to match the keys in the model that you
are loading into.

You can see that all keys matched successfully!

Congratulations! You have successfully warmstarted a model using
parameters from a different model in PyTorch.

## Learn More

Take a look at these other recipes to continue your learning:

- [Saving and loading multiple models in one file using PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_multiple_models_in_one_file.html)
- [Saving and loading models across devices in PyTorch](https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: warmstarting_model_using_parameters_from_a_different_model.ipynb`](../../_downloads/9a7fec783882a14243d1253bf1335d36/warmstarting_model_using_parameters_from_a_different_model.ipynb)

[`Download Python source code: warmstarting_model_using_parameters_from_a_different_model.py`](../../_downloads/2f650bced5c3cb530859bfc63b23681d/warmstarting_model_using_parameters_from_a_different_model.py)

[`Download zipped: warmstarting_model_using_parameters_from_a_different_model.zip`](../../_downloads/e632e04029e202469aefbb4fc1b771f7/warmstarting_model_using_parameters_from_a_different_model.zip)