Note

Go to the end
to download the full example code.

[Introduction](introyt1_tutorial.html) ||
[Tensors](tensors_deeper_tutorial.html) ||
[Autograd](autogradyt_tutorial.html) ||
[Building Models](modelsyt_tutorial.html) ||
[TensorBoard Support](tensorboardyt_tutorial.html) ||
**Training Models** ||
[Model Understanding](captumyt.html)

# Training with PyTorch

Follow along with the video below or on [youtube](https://www.youtube.com/watch?v=jF43_wj_DCQ).

## Introduction

In past videos, we've discussed and demonstrated:

- Building models with the neural network layers and functions of the torch.nn module
- The mechanics of automated gradient computation, which is central to
gradient-based model training
- Using TensorBoard to visualize training progress and other activities

In this video, we'll be adding some new tools to your inventory:

- We'll get familiar with the dataset and dataloader abstractions, and how
they ease the process of feeding data to your model during a training loop
- We'll discuss specific loss functions and when to use them
- We'll look at PyTorch optimizers, which implement algorithms to adjust
model weights based on the outcome of a loss function

Finally, we'll pull all of these together and see a full PyTorch
training loop in action.

## Dataset and DataLoader

The `Dataset` and `DataLoader` classes encapsulate the process of
pulling your data from storage and exposing it to your training loop in
batches.

The `Dataset` is responsible for accessing and processing single
instances of data.

The `DataLoader` pulls instances of data from the `Dataset` (either
automatically or with a sampler that you define), collects them in
batches, and returns them for consumption by your training loop. The
`DataLoader` works with all kinds of datasets, regardless of the type
of data they contain.

For this tutorial, we'll be using the Fashion-MNIST dataset provided by
TorchVision. We use `torchvision.transforms.v2.Normalize()` to
zero-center and normalize the distribution of the image tile content,
and download both training and validation data splits.

```
# PyTorch TensorBoard support

# Create datasets for training & validation, download if necessary

# Create data loaders for our datasets; shuffle for training, not for validation

# Class labels

# Report split sizes
```

As always, let's visualize the data as a sanity check:

```
# Helper function for inline image display

# Create a grid from the images and show them
```

## The Model

The model we'll use in this example is a variant of LeNet-5 - it should
be familiar if you've watched the previous videos in this series.

```
# PyTorch models inherit from torch.nn.Module
```

## Loss Function

For this example, we'll be using a cross-entropy loss. For demonstration
purposes, we'll create batches of dummy output and label values, run
them through the loss function, and examine the result.

```
# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input

# Represents the correct class among the 10 being tested
```

## Optimizer

For this example, we'll be using simple [stochastic gradient
descent](https://pytorch.org/docs/stable/optim.html) with momentum.

It can be instructive to try some variations on this optimization
scheme:

- Learning rate determines the size of the steps the optimizer
takes. What does a different learning rate do to the your training
results, in terms of accuracy and convergence time?
- Momentum nudges the optimizer in the direction of strongest gradient over
multiple steps. What does changing this value do to your results?
- Try some different optimization algorithms, such as averaged SGD, Adagrad, or
Adam. How do your results differ?

```
# Optimizers specified in the torch.optim package
```

## The Training Loop

Below, we have a function that performs one training epoch. It
enumerates data from the DataLoader, and on each pass of the loop does
the following:

- Gets a batch of training data from the DataLoader
- Zeros the optimizer's gradients
- Performs an inference - that is, gets predictions from the model for an input batch
- Calculates the loss for that set of predictions vs. the labels on the dataset
- Calculates the backward gradients over the learning weights
- Tells the optimizer to perform one learning step - that is, adjust the model's
learning weights based on the observed gradients for this batch, according to the
optimization algorithm we chose
- It reports on the loss for every 1000 batches.
- Finally, it reports the average per-batch loss for the last
1000 batches, for comparison with a validation run

### Per-Epoch Activity

There are a couple of things we'll want to do once per epoch:

- Perform validation by checking our relative loss on a set of data that was not
used for training, and report this
- Save a copy of the model

Here, we'll do our reporting in TensorBoard. This will require going to
the command line to start TensorBoard, and opening it in another browser
tab.

```
# Initializing in a separate cell so we can easily add more epochs to the same run
```

To load a saved version of the model:

```
saved_model = GarmentClassifier()
saved_model.load_state_dict(torch.load(PATH))
```

Once you've loaded the model, it's ready for whatever you need it for -
more training, inference, or analysis.

Note that if your model has constructor parameters that affect model
structure, you'll need to provide them and configure the model
identically to the state in which it was saved.

## Other Resources

- Docs on the [data
utilities](https://pytorch.org/docs/stable/data.html), including
Dataset and DataLoader, at pytorch.org
- A [note on the use of pinned
memory](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning)
for GPU training
- Documentation on the datasets available in
[TorchVision](https://pytorch.org/vision/stable/datasets.html),
[TorchText](https://pytorch.org/text/stable/datasets.html), and
[TorchAudio](https://pytorch.org/audio/stable/datasets.html)
- Documentation on the [loss
functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
available in PyTorch
- Documentation on the [torch.optim
package](https://pytorch.org/docs/stable/optim.html), which
includes optimizers and related tools, such as learning rate
scheduling
- A detailed [tutorial on saving and loading
models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- The [Tutorials section of
pytorch.org](https://pytorch.org/tutorials/) contains tutorials on
a broad variety of training tasks, including classification in
different domains, generative adversarial networks, reinforcement
learning, and more

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: trainingyt.ipynb`](../../_downloads/770632dd3941d2a51b831c52ded57aa2/trainingyt.ipynb)

[`Download Python source code: trainingyt.py`](../../_downloads/9f7a57e14d8a2ebf975344f34d6ef247/trainingyt.py)

[`Download zipped: trainingyt.zip`](../../_downloads/100140395067906afff547644ddab928/trainingyt.zip)