Note

Go to the end
to download the full example code.

# DCGAN Tutorial

**Author**: [Nathan Inkawhich](https://github.com/inkawhich)

## Introduction

This tutorial will give an introduction to DCGANs through an example. We
will train a generative adversarial network (GAN) to generate new
celebrities after showing it pictures of many real celebrities. Most of
the code here is from the DCGAN implementation in
[pytorch/examples](https://github.com/pytorch/examples), and this
document will give a thorough explanation of the implementation and shed
light on how and why this model works. But don't worry, no prior
knowledge of GANs is required, but it may require a first-timer to spend
some time reasoning about what is actually happening under the hood.
Also, for the sake of time it will help to have a GPU, or two. Lets
start from the beginning.

## Generative Adversarial Networks

### What is a GAN?

GANs are a framework for teaching a deep learning model to capture the training
data distribution so we can generate new data from that same
distribution. GANs were invented by Ian Goodfellow in 2014 and first
described in the paper [Generative Adversarial
Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf).
They are made of two distinct models, a *generator* and a
*discriminator*. The job of the generator is to spawn 'fake' images that
look like the training images. The job of the discriminator is to look
at an image and output whether or not it is a real training image or a
fake image from the generator. During training, the generator is
constantly trying to outsmart the discriminator by generating better and
better fakes, while the discriminator is working to become a better
detective and correctly classify the real and fake images. The
equilibrium of this game is when the generator is generating perfect
fakes that look as if they came directly from the training data, and the
discriminator is left to always guess at 50% confidence that the
generator output is real or fake.

Now, lets define some notation to be used throughout tutorial starting
with the discriminator. Let \(x\) be data representing an image.
\(D(x)\) is the discriminator network which outputs the (scalar)
probability that \(x\) came from training data rather than the
generator. Here, since we are dealing with images, the input to
\(D(x)\) is an image of CHW size 3x64x64. Intuitively, \(D(x)\)
should be HIGH when \(x\) comes from training data and LOW when
\(x\) comes from the generator. \(D(x)\) can also be thought of
as a traditional binary classifier.

For the generator's notation, let \(z\) be a latent space vector
sampled from a standard normal distribution. \(G(z)\) represents the
generator function which maps the latent vector \(z\) to data-space.
The goal of \(G\) is to estimate the distribution that the training
data comes from (\(p_{data}\)) so it can generate fake samples from
that estimated distribution (\(p_g\)).

So, \(D(G(z))\) is the probability (scalar) that the output of the
generator \(G\) is a real image. As described in [Goodfellow's
paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf),
\(D\) and \(G\) play a minimax game in which \(D\) tries to
maximize the probability it correctly classifies reals and fakes
(\(logD(x)\)), and \(G\) tries to minimize the probability that
\(D\) will predict its outputs are fake (\(log(1-D(G(z)))\)).
From the paper, the GAN loss function is

\[\underset{G}{\text{min}} \underset{D}{\text{max}}V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}\big[logD(x)\big] + \mathbb{E}_{z\sim p_{z}(z)}\big[log(1-D(G(z)))\big]

\]

In theory, the solution to this minimax game is where
\(p_g = p_{data}\), and the discriminator guesses randomly if the
inputs are real or fake. However, the convergence theory of GANs is
still being actively researched and in reality models do not always
train to this point.

### What is a DCGAN?

A DCGAN is a direct extension of the GAN described above, except that it
explicitly uses convolutional and convolutional-transpose layers in the
discriminator and generator, respectively. It was first described by
Radford et. al. in the paper [Unsupervised Representation Learning With
Deep Convolutional Generative Adversarial
Networks](https://arxiv.org/pdf/1511.06434.pdf). The discriminator
is made up of strided
[convolution](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)
layers, [batch
norm](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d)
layers, and
[LeakyReLU](https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU)
activations. The input is a 3x64x64 input image and the output is a
scalar probability that the input is from the real data distribution.
The generator is comprised of
[convolutional-transpose](https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d)
layers, batch norm layers, and
[ReLU](https://pytorch.org/docs/stable/nn.html#relu) activations. The
input is a latent vector, \(z\), that is drawn from a standard
normal distribution and the output is a 3x64x64 RGB image. The strided
conv-transpose layers allow the latent vector to be transformed into a
volume with the same shape as an image. In the paper, the authors also
give some tips about how to setup the optimizers, how to calculate the
loss functions, and how to initialize the model weights, all of which
will be explained in the coming sections.

```
#%matplotlib inline

# Set random seed for reproducibility

#manualSeed = random.randint(1, 10000) # use if you want new results
```

## Inputs

Let's define some inputs for the run:

- `dataroot` - the path to the root of the dataset folder. We will
talk more about the dataset in the next section.
- `workers` - the number of worker threads for loading the data with
the `DataLoader`.
- `batch_size` - the batch size used in training. The DCGAN paper
uses a batch size of 128.
- `image_size` - the spatial size of the images used for training.
This implementation defaults to 64x64. If another size is desired,
the structures of D and G must be changed. See
[here](https://github.com/pytorch/examples/issues/70) for more
details.
- `nc` - number of color channels in the input images. For color
images this is 3.
- `nz` - length of latent vector.
- `ngf` - relates to the depth of feature maps carried through the
generator.
- `ndf` - sets the depth of feature maps propagated through the
discriminator.
- `num_epochs` - number of training epochs to run. Training for
longer will probably lead to better results but will also take much
longer.
- `lr` - learning rate for training. As described in the DCGAN paper,
this number should be 0.0002.
- `beta1` - beta1 hyperparameter for Adam optimizers. As described in
paper, this number should be 0.5.
- `ngpu` - number of GPUs available. If this is 0, code will run in
CPU mode. If this number is greater than 0 it will run on that number
of GPUs.

```
# Root directory for dataset

# Number of workers for dataloader

# Batch size during training

# Spatial size of training images. All images will be resized to this
# size using a transformer.

# Number of channels in the training images. For color images this is 3

# Size of z latent vector (i.e. size of generator input)

# Size of feature maps in generator

# Size of feature maps in discriminator

# Number of training epochs

# Learning rate for optimizers

# Beta1 hyperparameter for Adam optimizers

# Number of GPUs available. Use 0 for CPU mode.
```

## Data

In this tutorial we will use the [Celeb-A Faces
dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) which can
be downloaded at the linked site, or in [Google
Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg).
The dataset will download as a file named `img_align_celeba.zip`. Once
downloaded, create a directory named `celeba` and extract the zip file
into that directory. Then, set the `dataroot` input for this notebook to
the `celeba` directory you just created. The resulting directory
structure should be:

```
/path/to/celeba
 -> img_align_celeba
 -> 188242.jpg
 -> 173822.jpg
 -> 284702.jpg
 -> 537394.jpg
 ...
```

This is an important step because we will be using the `ImageFolder`
dataset class, which requires there to be subdirectories in the
dataset root folder. Now, we can create the dataset, create the
dataloader, set the device to run on, and finally visualize some of the
training data.

```
# We can use an image folder dataset the way we have it setup.
# Create the dataset

# Create the dataloader

# Decide which device we want to run on

# Plot some training images
```

## Implementation

With our input parameters set and the dataset prepared, we can now get
into the implementation. We will start with the weight initialization
strategy, then talk about the generator, discriminator, loss functions,
and training loop in detail.

### Weight Initialization

From the DCGAN paper, the authors specify that all model weights shall
be randomly initialized from a Normal distribution with `mean=0`,
`stdev=0.02`. The `weights_init` function takes an initialized model as
input and reinitializes all convolutional, convolutional-transpose, and
batch normalization layers to meet this criteria. This function is
applied to the models immediately after initialization.

```
# custom weights initialization called on ``netG`` and ``netD``
```

### Generator

The generator, \(G\), is designed to map the latent space vector
(\(z\)) to data-space. Since our data are images, converting
\(z\) to data-space means ultimately creating a RGB image with the
same size as the training images (i.e. 3x64x64). In practice, this is
accomplished through a series of strided two dimensional convolutional
transpose layers, each paired with a 2d batch norm layer and a relu
activation. The output of the generator is fed through a tanh function
to return it to the input data range of \([-1,1]\). It is worth
noting the existence of the batch norm functions after the
conv-transpose layers, as this is a critical contribution of the DCGAN
paper. These layers help with the flow of gradients during training. An
image of the generator from the DCGAN paper is shown below.

![dcgan_generator](../_images/dcgan_generator.png)

Notice, how the inputs we set in the input section (`nz`, `ngf`, and
`nc`) influence the generator architecture in code. `nz` is the length
of the z input vector, `ngf` relates to the size of the feature maps
that are propagated through the generator, and `nc` is the number of
channels in the output image (set to 3 for RGB images). Below is the
code for the generator.

```
# Generator Code
```

Now, we can instantiate the generator and apply the `weights_init`
function. Check out the printed model to see how the generator object is
structured.

```
# Create the generator

# Handle multi-GPU if desired

# Apply the ``weights_init`` function to randomly initialize all weights
# to ``mean=0``, ``stdev=0.02``.

# Print the model
```

### Discriminator

As mentioned, the discriminator, \(D\), is a binary classification
network that takes an image as input and outputs a scalar probability
that the input image is real (as opposed to fake). Here, \(D\) takes
a 3x64x64 input image, processes it through a series of Conv2d,
BatchNorm2d, and LeakyReLU layers, and outputs the final probability
through a Sigmoid activation function. This architecture can be extended
with more layers if necessary for the problem, but there is significance
to the use of the strided convolution, BatchNorm, and LeakyReLUs. The
DCGAN paper mentions it is a good practice to use strided convolution
rather than pooling to downsample because it lets the network learn its
own pooling function. Also batch norm and leaky relu functions promote
healthy gradient flow which is critical for the learning process of both
\(G\) and \(D\).

Discriminator Code

Now, as with the generator, we can create the discriminator, apply the
`weights_init` function, and print the model's structure.

```
# Create the Discriminator

# Handle multi-GPU if desired

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.

# Print the model
```

### Loss Functions and Optimizers

With \(D\) and \(G\) setup, we can specify how they learn
through the loss functions and optimizers. We will use the Binary Cross
Entropy loss
([BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss))
function which is defined in PyTorch as:

\[\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = - \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

\]

Notice how this function provides the calculation of both log components
in the objective function (i.e. \(log(D(x))\) and
\(log(1-D(G(z)))\)). We can specify what part of the BCE equation to
use with the \(y\) input. This is accomplished in the training loop
which is coming up soon, but it is important to understand how we can
choose which component we wish to calculate just by changing \(y\)
(i.e. GT labels).

Next, we define our real label as 1 and the fake label as 0. These
labels will be used when calculating the losses of \(D\) and
\(G\), and this is also the convention used in the original GAN
paper. Finally, we set up two separate optimizers, one for \(D\) and
one for \(G\). As specified in the DCGAN paper, both are Adam
optimizers with learning rate 0.0002 and Beta1 = 0.5. For keeping track
of the generator's learning progression, we will generate a fixed batch
of latent vectors that are drawn from a Gaussian distribution
(i.e. fixed_noise) . In the training loop, we will periodically input
this fixed_noise into \(G\), and over the iterations we will see
images form out of the noise.

```
# Initialize the ``BCELoss`` function

# Create batch of latent vectors that we will use to visualize
# the progression of the generator

# Establish convention for real and fake labels during training

# Setup Adam optimizers for both G and D
```

### Training

Finally, now that we have all of the parts of the GAN framework defined,
we can train it. Be mindful that training GANs is somewhat of an art
form, as incorrect hyperparameter settings lead to mode collapse with
little explanation of what went wrong. Here, we will closely follow
Algorithm 1 from the [Goodfellow's paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf),
while abiding by some of the best
practices shown in [ganhacks](https://github.com/soumith/ganhacks).
Namely, we will "construct different mini-batches for real and fake"
images, and also adjust G's objective function to maximize
\(log(D(G(z)))\). Training is split up into two main parts. Part 1
updates the Discriminator and Part 2 updates the Generator.

**Part 1 - Train the Discriminator**

Recall, the goal of training the discriminator is to maximize the
probability of correctly classifying a given input as real or fake. In
terms of Goodfellow, we wish to "update the discriminator by ascending
its stochastic gradient". Practically, we want to maximize
\(log(D(x)) + log(1-D(G(z)))\). Due to the separate mini-batch
suggestion from [ganhacks](https://github.com/soumith/ganhacks),
we will calculate this in two steps. First, we
will construct a batch of real samples from the training set, forward
pass through \(D\), calculate the loss (\(log(D(x))\)), then
calculate the gradients in a backward pass. Secondly, we will construct
a batch of fake samples with the current generator, forward pass this
batch through \(D\), calculate the loss (\(log(1-D(G(z)))\)),
and *accumulate* the gradients with a backward pass. Now, with the
gradients accumulated from both the all-real and all-fake batches, we
call a step of the Discriminator's optimizer.

**Part 2 - Train the Generator**

As stated in the original paper, we want to train the Generator by
minimizing \(log(1-D(G(z)))\) in an effort to generate better fakes.
As mentioned, this was shown by Goodfellow to not provide sufficient
gradients, especially early in the learning process. As a fix, we
instead wish to maximize \(log(D(G(z)))\). In the code we accomplish
this by: classifying the Generator output from Part 1 with the
Discriminator, computing G's loss *using real labels as GT*, computing
G's gradients in a backward pass, and finally updating G's parameters
with an optimizer step. It may seem counter-intuitive to use the real
labels as GT labels for the loss function, but this allows us to use the
\(log(x)\) part of the `BCELoss` (rather than the \(log(1-x)\)
part) which is exactly what we want.

Finally, we will do some statistic reporting and at the end of each
epoch we will push our fixed_noise batch through the generator to
visually track the progress of G's training. The training statistics
reported are:

- **Loss_D** - discriminator loss calculated as the sum of losses for
the all real and all fake batches (\(log(D(x)) + log(1 - D(G(z)))\)).
- **Loss_G** - generator loss calculated as \(log(D(G(z)))\)
- **D(x)** - the average output (across the batch) of the discriminator
for the all real batch. This should start close to 1 then
theoretically converge to 0.5 when G gets better. Think about why
this is.
- **D(G(z))** - average discriminator outputs for the all fake batch.
The first number is before D is updated and the second number is
after D is updated. These numbers should start near 0 and converge to
0.5 as G gets better. Think about why this is.

**Note:** This step might take a while, depending on how many epochs you
run and if you removed some data from the dataset.

```
# Training Loop

# Lists to keep track of progress

# For each epoch
```

## Results

Finally, lets check out how we did. Here, we will look at three
different results. First, we will see how D and G's losses changed
during training. Second, we will visualize G's output on the fixed_noise
batch for every epoch. And third, we will look at a batch of real data
next to a batch of fake data from G.

**Loss versus training iteration**

Below is a plot of D & G's losses versus training iterations.

**Visualization of G's progression**

Remember how we saved the generator's output on the fixed_noise batch
after every epoch of training. Now, we can visualize the training
progression of G with an animation. Press the play button to start the
animation.

**Real Images vs. Fake Images**

Finally, lets take a look at some real images and fake images side by
side.

```
# Grab a batch of real images from the dataloader

# Plot the real images

# Plot the fake images from the last epoch
```

## Where to Go Next

We have reached the end of our journey, but there are several places you
could go from here. You could:

- Train for longer to see how good the results get
- Modify this model to take a different dataset and possibly change the
size of the images and the model architecture
- Check out some other cool GAN projects
[here](https://github.com/nashory/gans-awesome-applications)
- Create GANs that generate
[music](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio/)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: dcgan_faces_tutorial.ipynb`](../_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb)

[`Download Python source code: dcgan_faces_tutorial.py`](../_downloads/176a0eb17083f2a53b7e388098a15704/dcgan_faces_tutorial.py)

[`Download zipped: dcgan_faces_tutorial.zip`](../_downloads/1171400bfac7e45d5ecb874319f1a261/dcgan_faces_tutorial.zip)