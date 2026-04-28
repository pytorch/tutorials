Note

Go to the end
to download the full example code.

[Introduction](introyt1_tutorial.html) ||
[Tensors](tensors_deeper_tutorial.html) ||
[Autograd](autogradyt_tutorial.html) ||
[Building Models](modelsyt_tutorial.html) ||
**TensorBoard Support** ||
[Training Models](trainingyt.html) ||
[Model Understanding](captumyt.html)

# PyTorch TensorBoard Support

Follow along with the video below or on [youtube](https://www.youtube.com/watch?v=6CEld3hZgqc).

## Before You Start

To run this tutorial, you'll need to install PyTorch, TorchVision,
Matplotlib, and TensorBoard.

With `conda`:

```
conda install pytorch torchvision -c pytorch
conda install matplotlib tensorboard
```

With `pip`:

```
pip install torch torchvision matplotlib tensorboard
```

Once the dependencies are installed, restart this notebook in the Python
environment where you installed them.

## Introduction

In this notebook, we'll be training a variant of LeNet-5 against the
Fashion-MNIST dataset. Fashion-MNIST is a set of image tiles depicting
various garments, with ten class labels indicating the type of garment
depicted.

```
# PyTorch model and training necessities

# Image datasets and image manipulation

# Image display

# PyTorch TensorBoard support

# In case you are using an environment that has TensorFlow installed,
# such as Google Colab, uncomment the following code to avoid
# a bug with saving embeddings to your TensorBoard directory

# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
```

## Showing Images in TensorBoard

Let's start by adding sample images from our dataset to TensorBoard:

```
# Gather datasets and prepare them for consumption

# Store separate training and validations splits in ./data

# Class labels

# Helper function for inline image display

# Extract a batch of 4 images

# Create a grid from the images and show them
```

Above, we used TorchVision and Matplotlib to create a visual grid of a
minibatch of our input data. Below, we use the `add_image()` call on
`SummaryWriter` to log the image for consumption by TensorBoard, and
we also call `flush()` to make sure it's written to disk right away.

```
# Default log_dir argument is "runs" - but it's good to be specific
# torch.utils.tensorboard.SummaryWriter is imported above

# Write image data to TensorBoard log dir

# To view, start TensorBoard on the command line with:
# tensorboard --logdir=runs
# ...and open a browser tab to http://localhost:6006/
```

If you start TensorBoard at the command line and open it in a new
browser tab (usually at [localhost:6006](localhost:6006)), you should
see the image grid under the IMAGES tab.

## Graphing Scalars to Visualize Training

TensorBoard is useful for tracking the progress and efficacy of your
training. Below, we'll run a training loop, track some metrics, and save
the data for TensorBoard's consumption.

Let's define a model to categorize our image tiles, and an optimizer and
loss function for training:

Now let's train a single epoch, and evaluate the training vs. validation
set losses every 1000 batches:

Switch to your open TensorBoard and have a look at the SCALARS tab.

## Visualizing Your Model

TensorBoard can also be used to examine the data flow within your model.
To do this, call the `add_graph()` method with a model and sample
input:

```
# Again, grab a single mini-batch of images

# add_graph() will trace the sample input through your model,
# and render it as a graph.
```

When you switch over to TensorBoard, you should see a GRAPHS tab.
Double-click the "NET" node to see the layers and data flow within your
model.

## Visualizing Your Dataset with Embeddings

The 28-by-28 image tiles we're using can be modeled as 784-dimensional
vectors (28 * 28 = 784). It can be instructive to project this to a
lower-dimensional representation. The `add_embedding()` method will
project a set of data onto the three dimensions with highest variance,
and display them as an interactive 3D chart. The `add_embedding()`
method does this automatically by projecting to the three dimensions
with highest variance.

Below, we'll take a sample of our data, and generate such an embedding:

```
# Select a random subset of data and corresponding labels

# Extract a random subset of data

# get the class labels for each image

# log embeddings
```

Now if you switch to TensorBoard and select the PROJECTOR tab, you
should see a 3D representation of the projection. You can rotate and
zoom the model. Examine it at large and small scales, and see whether
you can spot patterns in the projected data and the clustering of
labels.

For better visibility, it's recommended to:

- Select "label" from the "Color by" drop-down on the left.
- Toggle the Night Mode icon along the top to place the
light-colored images on a dark background.

## Other Resources

For more information, have a look at:

- PyTorch documentation on [torch.utils.tensorboard.SummaryWriter](https://pytorch.org/docs/stable/tensorboard.html?highlight=summarywriter)
- Tensorboard tutorial content in the [PyTorch.org Tutorials](https://pytorch.org/tutorials/)
- For more information about TensorBoard, see the [TensorBoard
documentation](https://www.tensorflow.org/tensorboard)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: tensorboardyt_tutorial.ipynb`](../../_downloads/e2e556f6b4693c2cef716dd7f40caaf6/tensorboardyt_tutorial.ipynb)

[`Download Python source code: tensorboardyt_tutorial.py`](../../_downloads/ba6d64f1f8bd0d6b3c21839705dc840a/tensorboardyt_tutorial.py)

[`Download zipped: tensorboardyt_tutorial.zip`](../../_downloads/cbd2d56d96f217c86d55a469995f619f/tensorboardyt_tutorial.zip)