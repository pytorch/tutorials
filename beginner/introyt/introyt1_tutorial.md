Note

Go to the end
to download the full example code.

**Introduction** ||
[Tensors](tensors_deeper_tutorial.html) ||
[Autograd](autogradyt_tutorial.html) ||
[Building Models](modelsyt_tutorial.html) ||
[TensorBoard Support](tensorboardyt_tutorial.html) ||
[Training Models](trainingyt.html) ||
[Model Understanding](captumyt.html)

# Introduction to PyTorch

Follow along with the video below or on [youtube](https://www.youtube.com/watch?v=IC0_FRiX-sw).

## PyTorch Tensors

Follow along with the video beginning at [03:50](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=230s).

First, we'll import pytorch.

Let's see a few basic tensor manipulations. First, just a few of the
ways to create tensors:

Above, we create a 5x3 matrix filled with zeros, and query its datatype
to find out that the zeros are 32-bit floating point numbers, which is
the default PyTorch.

What if you wanted integers instead? You can always override the
default:

You can see that when we do change the default, the tensor helpfully
reports this when printed.

It's common to initialize learning weights randomly, often with a
specific seed for the PRNG for reproducibility of results:

PyTorch tensors perform arithmetic operations intuitively. Tensors of
similar shapes may be added, multiplied, etc. Operations with scalars
are distributed over the tensor:

```
# uncomment this line to get a runtime error
# r3 = r1 + r2
```

Here's a small sample of the mathematical operations available:

```
# Common mathematical operations are supported:

# ...as are trigonometric functions:

# ...and linear algebra operations like determinant and singular value decomposition

# ...and statistical and aggregate operations:
```

There's a good deal more to know about the power of PyTorch tensors,
including how to set them up for parallel computations on GPU - we'll be
going into more depth in another video.

## PyTorch Models

Follow along with the video beginning at [10:00](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=600s).

Let's talk about how we can express models in PyTorch

![le-net-5 diagram](../../_images/mnist.png)

*Figure: LeNet-5*

Above is a diagram of LeNet-5, one of the earliest convolutional neural
nets, and one of the drivers of the explosion in Deep Learning. It was
built to read small images of handwritten numbers (the MNIST dataset),
and correctly classify which digit was represented in the image.

Here's the abridged version of how it works:

- Layer C1 is a convolutional layer, meaning that it scans the input
image for features it learned during training. It outputs a map of
where it saw each of its learned features in the image. This
"activation map" is downsampled in layer S2.
- Layer C3 is another convolutional layer, this time scanning C1's
activation map for *combinations* of features. It also puts out an
activation map describing the spatial locations of these feature
combinations, which is downsampled in layer S4.
- Finally, the fully-connected layers at the end, F5, F6, and OUTPUT,
are a *classifier* that takes the final activation map, and
classifies it into one of ten bins representing the 10 digits.

How do we express this simple neural network in code?

Looking over this code, you should be able to spot some structural
similarities with the diagram above.

This demonstrates the structure of a typical PyTorch model:

- It inherits from `torch.nn.Module` - modules may be nested - in fact,
even the `Conv2d` and `Linear` layer classes inherit from
`torch.nn.Module`.
- A model will have an `__init__()` function, where it instantiates
its layers, and loads any data artifacts it might
need (e.g., an NLP model might load a vocabulary).
- A model will have a `forward()` function. This is where the actual
computation happens: An input is passed through the network layers
and various functions to generate an output.
- Other than that, you can build out your model class like any other
Python class, adding whatever properties and methods you need to
support your model's computation.

Let's instantiate this object and run a sample input through it.

There are a few important things happening above:

First, we instantiate the `LeNet` class, and we print the `net`
object. A subclass of `torch.nn.Module` will report the layers it has
created and their shapes and parameters. This can provide a handy
overview of a model if you want to get the gist of its processing.

Below that, we create a dummy input representing a 32x32 image with 1
color channel. Normally, you would load an image tile and convert it to
a tensor of this shape.

You may have noticed an extra dimension to our tensor - the *batch
dimension.* PyTorch models assume they are working on *batches* of data
- for example, a batch of 16 of our image tiles would have the shape
`(16, 1, 32, 32)`. Since we're only using one image, we create a batch
of 1 with shape `(1, 1, 32, 32)`.

We ask the model for an inference by calling it like a function:
`net(input)`. The output of this call represents the model's
confidence that the input represents a particular digit. (Since this
instance of the model hasn't learned anything yet, we shouldn't expect
to see any signal in the output.) Looking at the shape of `output`, we
can see that it also has a batch dimension, the size of which should
always match the input batch dimension. If we had passed in an input
batch of 16 instances, `output` would have a shape of `(16, 10)`.

## Datasets and Dataloaders

Follow along with the video beginning at [14:00](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=840s).

Below, we're going to demonstrate using one of the ready-to-download,
open-access datasets from TorchVision, how to transform the images for
consumption by your model, and how to use the DataLoader to feed batches
of data to your model.

The first thing we need to do is transform our incoming images into a
PyTorch tensor.

```
#%matplotlib inline
```

Here, we specify two transformations for our input:

- `transforms.ToTensor()` converts images loaded by Pillow into
PyTorch tensors.
- `transforms.Normalize()` adjusts the values of the tensor so
that their average is zero and their standard deviation is 1.0. Most
activation functions have their strongest gradients around x = 0, so
centering our data there can speed learning.
The values passed to the transform are the means (first tuple) and the
standard deviations (second tuple) of the rgb values of the images in
the dataset. You can calculate these values yourself by running these
few lines of code:

```
from torch.utils.data import ConcatDataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
 download=True, transform=transform)

# stack all train images together into a tensor of shape
# (50000, 3, 32, 32)
x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])

# get the mean of each channel
mean = torch.mean(x, dim=(0,2,3)) # tensor([0.4914, 0.4822, 0.4465])
std = torch.std(x, dim=(0,2,3)) # tensor([0.2470, 0.2435, 0.2616])
```

There are many more transforms available, including cropping, centering,
rotation, and reflection.

Next, we'll create an instance of the CIFAR10 dataset. This is a set of
32x32 color image tiles representing 10 classes of objects: 6 of animals
(bird, cat, deer, dog, frog, horse) and 4 of vehicles (airplane,
automobile, ship, truck):

Note

When you run the cell above, it may take a little time for the
dataset to download.

This is an example of creating a dataset object in PyTorch. Downloadable
datasets (like CIFAR-10 above) are subclasses of
`torch.utils.data.Dataset`. `Dataset` classes in PyTorch include the
downloadable datasets in TorchVision, Torchtext, and TorchAudio, as well
as utility dataset classes such as `torchvision.datasets.ImageFolder`,
which will read a folder of labeled images. You can also create your own
subclasses of `Dataset`.

When we instantiate our dataset, we need to tell it a few things:

- The filesystem path to where we want the data to go.
- Whether or not we are using this set for training; most datasets
will be split into training and test subsets.
- Whether we would like to download the dataset if we haven't already.
- The transformations we want to apply to the data.

Once your dataset is ready, you can give it to the `DataLoader`:

A `Dataset` subclass wraps access to the data, and is specialized to
the type of data it's serving. The `DataLoader` knows *nothing* about
the data, but organizes the input tensors served by the `Dataset` into
batches with the parameters you specify.

In the example above, we've asked a `DataLoader` to give us batches of
4 images from `trainset`, randomizing their order (`shuffle=True`),
and we told it to spin up two workers to load data from disk.

It's good practice to visualize the batches your `DataLoader` serves:

```
# get some random training images

# show images

# print labels
```

Running the above cell should show you a strip of four images, and the
correct label for each.

## Training Your PyTorch Model

Follow along with the video beginning at [17:10](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=1030s).

Let's put all the pieces together, and train a model:

```
#%matplotlib inline
```

First, we'll need training and test datasets. If you haven't already,
run the cell below to make sure the dataset is downloaded. (It may take
a minute.)

We'll run our check on the output from `DataLoader`:

```
# functions to show an image

# get some random training images

# show images

# print labels
```

This is the model we'll train. If it looks familiar, that's because it's
a variant of LeNet - discussed earlier in this video - adapted for
3-color images.

The last ingredients we need are a loss function and an optimizer:

The loss function, as discussed earlier in this video, is a measure of
how far from our ideal output the model's prediction was. Cross-entropy
loss is a typical loss function for classification models like ours.

The **optimizer** is what drives the learning. Here we have created an
optimizer that implements *stochastic gradient descent,* one of the more
straightforward optimization algorithms. Besides parameters of the
algorithm, like the learning rate (`lr`) and momentum, we also pass in
`net.parameters()`, which is a collection of all the learning weights
in the model - which is what the optimizer adjusts.

Finally, all of this is assembled into the training loop. Go ahead and
run this cell, as it will likely take a few minutes to execute:

Here, we are doing only **2 training epochs** (line 1) - that is, two
passes over the training dataset. Each pass has an inner loop that
**iterates over the training data** (line 4), serving batches of
transformed input images and their correct labels.

**Zeroing the gradients** (line 9) is an important step. Gradients are
accumulated over a batch; if we do not reset them for every batch, they
will keep accumulating, which will provide incorrect gradient values,
making learning impossible.

In line 12, we **ask the model for its predictions** on this batch. In
the following line (13), we compute the loss - the difference between
`outputs` (the model prediction) and `labels` (the correct output).

In line 14, we do the `backward()` pass, and calculate the gradients
that will direct the learning.

In line 15, the optimizer performs one learning step - it uses the
gradients from the `backward()` call to nudge the learning weights in
the direction it thinks will reduce the loss.

The remainder of the loop does some light reporting on the epoch number,
how many training instances have been completed, and what the collected
loss is over the training loop.

**When you run the cell above,** you should see something like this:

```
[1, 2000] loss: 2.235
[1, 4000] loss: 1.940
[1, 6000] loss: 1.713
[1, 8000] loss: 1.573
[1, 10000] loss: 1.507
[1, 12000] loss: 1.442
[2, 2000] loss: 1.378
[2, 4000] loss: 1.364
[2, 6000] loss: 1.349
[2, 8000] loss: 1.319
[2, 10000] loss: 1.284
[2, 12000] loss: 1.267
Finished Training
```

Note that the loss is monotonically descending, indicating that our
model is continuing to improve its performance on the training dataset.

As a final step, we should check that the model is actually doing
*general* learning, and not simply "memorizing" the dataset. This is
called **overfitting,** and usually indicates that the dataset is too
small (not enough examples for general learning), or that the model has
more learning parameters than it needs to correctly model the dataset.

This is the reason datasets are split into training and test subsets -
to test the generality of the model, we ask it to make predictions on
data it hasn't trained on:

If you followed along, you should see that the model is roughly 50%
accurate at this point. That's not exactly state-of-the-art, but it's
far better than the 10% accuracy we'd expect from a random output. This
demonstrates that some general learning did happen in the model.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: introyt1_tutorial.ipynb`](../../_downloads/3195443a0ced3cabc0ad643537bdb5cd/introyt1_tutorial.ipynb)

[`Download Python source code: introyt1_tutorial.py`](../../_downloads/0e4c2becda3dfc54e1816634d49f8e73/introyt1_tutorial.py)

[`Download zipped: introyt1_tutorial.zip`](../../_downloads/de84aff475dd61dbf39a1efc4a9d638a/introyt1_tutorial.zip)