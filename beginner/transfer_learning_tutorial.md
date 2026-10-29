Note

Go to the end
to download the full example code.

# Transfer Learning for Computer Vision Tutorial

**Author**: [Sasank Chilamkurthy](https://chsasank.github.io)

In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at [cs231n notes](https://cs231n.github.io/transfer-learning/)

Quoting these notes,

> In practice, very few people train an entire Convolutional Network
> from scratch (with random initialization), because it is relatively
> rare to have a dataset of sufficient size. Instead, it is common to
> pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
> contains 1.2 million images with 1000 categories), and then use the
> ConvNet either as an initialization or a fixed feature extractor for
> the task of interest.

These two major transfer learning scenarios look as follows:

- **Finetuning the ConvNet**: Instead of random initialization, we
initialize the network with a pretrained network, like the one that is
trained on imagenet 1000 dataset. Rest of the training looks as
usual.
- **ConvNet as fixed feature extractor**: Here, we will freeze the weights
for all of the network except that of the final fully connected
layer. This last fully connected layer is replaced with a new one
with random weights and only this layer is trained.

```
# License: BSD
# Author: Sasank Chilamkurthy
```

## Load Data

We will use torchvision and torch.utils.data packages for loading the
data.

The problem we're going to solve today is to train a model to classify
**ants** and **bees**. We have about 120 training images each for ants and bees.
There are 75 validation images for each class. Usually, this is a very
small dataset to generalize upon, if trained from scratch. Since we
are using transfer learning, we should be able to generalize reasonably
well.

This dataset is a very small subset of imagenet.

Note

Download the data from
[here](https://download.pytorch.org/tutorial/hymenoptera_data.zip)
and extract it to the current directory.

```
# Data augmentation and normalization for training
# Just normalization for validation

# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
```

### Visualize a few images

Let's visualize a few training images so as to understand the data
augmentations.

```
# Get a batch of training data

# Make a grid from batch
```

## Training the model

Now, let's write a general function to train a model. Here, we will
illustrate:

- Scheduling the learning rate
- Saving the best model

In the following, parameter `scheduler` is an LR scheduler object from
`torch.optim.lr_scheduler`.

### Visualizing the model predictions

Generic function to display predictions for a few images

## Finetuning the ConvNet

Load a pretrained model and reset final fully connected layer.

```
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.

# Observe that all parameters are being optimized

# Decay LR by a factor of 0.1 every 7 epochs
```

### Train and evaluate

It should take around 15-25 min on CPU. On GPU though, it takes less than a
minute.

## ConvNet as fixed feature extractor

Here, we need to freeze all the network except the final layer. We need
to set `requires_grad = False` to freeze the parameters so that the
gradients are not computed in `backward()`.

You can read more about this in the documentation
[here](https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward).

```
# Parameters of newly constructed modules have requires_grad=True by default

# Observe that only parameters of final layer are being optimized as
# opposed to before.

# Decay LR by a factor of 0.1 every 7 epochs
```

### Train and evaluate

On CPU this will take about half the time compared to previous scenario.
This is expected as gradients don't need to be computed for most of the
network. However, forward does need to be computed.

## Inference on custom images

Use the trained model to make predictions on custom images and visualize
the predicted class labels along with the images.

## Further Learning

If you would like to learn more about the applications of transfer learning,
checkout our [Quantized Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html).

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: transfer_learning_tutorial.ipynb`](../_downloads/74249e7f9f1f398f57ccd094a4f3021b/transfer_learning_tutorial.ipynb)

[`Download Python source code: transfer_learning_tutorial.py`](../_downloads/d923ca53b1bfbeb3c222ae46d65d485e/transfer_learning_tutorial.py)

[`Download zipped: transfer_learning_tutorial.zip`](../_downloads/1baf319766772c6431fef590c87c2d3f/transfer_learning_tutorial.zip)