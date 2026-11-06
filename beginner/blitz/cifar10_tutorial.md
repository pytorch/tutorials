Note

Go to the end
to download the full example code.

# Training a Classifier

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

## What about data?

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a `torch.*Tensor`.

- For images, packages such as Pillow, OpenCV are useful
- For audio, packages such as scipy and librosa
- For text, either raw Python or Cython based loading, or NLTK and
SpaCy are useful

Specifically for vision, we have created a package called
`torchvision`, that has data loaders for common datasets such as
ImageNet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
`torchvision.datasets` and `torch.utils.data.DataLoader`.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: 'airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck'. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

![cifar10](../../_images/cifar10.png)

cifar10

## Training an image classifier

We will do the following steps in order:

1. Load and normalize the CIFAR10 training and test datasets using
`torchvision`
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

### 1. Load and normalize CIFAR10

Using `torchvision`, it's extremely easy to load CIFAR10.

The output of torchvision datasets are PILImage images of range [0, 1].
We transform them to Tensors of normalized range [-1, 1].

Note

If you are running this tutorial on Windows or MacOS and encounter a
BrokenPipeError or RuntimeError related to multiprocessing, try setting
the num_worker of torch.utils.data.DataLoader() to 0.

Let us show some of the training images, for fun.

```
# functions to show an image

# get some random training images

# show images

# print labels
```

### 2. Define a Convolutional Neural Network

Copy the neural network from the Neural Networks section before and modify it to
take 3-channel images (instead of 1-channel images as it was defined).

### 3. Define a Loss function and optimizer

Let's use a Classification Cross-Entropy loss and SGD with momentum.

### 4. Train the network

This is when things start to get interesting.
We simply have to loop over our data iterator, and feed the inputs to the
network and optimize.

Let's quickly save our trained model:

See [here](https://pytorch.org/docs/stable/notes/serialization.html)
for more details on saving PyTorch models.

### 5. Test the network on the test data

We have trained the network for 2 passes over the training dataset.
But we need to check if the network has learnt anything at all.

We will check this by predicting the class label that the neural network
outputs, and checking it against the ground-truth. If the prediction is
correct, we add the sample to the list of correct predictions.

Okay, first step. Let us display an image from the test set to get familiar.

```
# print images
```

Next, let's load back in our saved model (note: saving and re-loading the model
wasn't necessary here, we only did it to illustrate how to do so):

Okay, now let us see what the neural network thinks these examples above are:

The outputs are energies for the 10 classes.
The higher the energy for a class, the more the network
thinks that the image is of the particular class.
So, let's get the index of the highest energy:

The results seem pretty good.

Let us look at how the network performs on the whole dataset.

```
# since we're not training, we don't need to calculate the gradients for our outputs
```

That looks way better than chance, which is 10% accuracy (randomly picking
a class out of 10 classes).
Seems like the network learnt something.

Hmmm, what are the classes that performed well, and the classes that did
not perform well:

```
# prepare to count predictions for each class

# again no gradients needed

# print accuracy for each class
```

Okay, so what next?

How do we run these neural networks on the GPU?

## Training on GPU

Just like how you transfer a Tensor onto the GPU, you transfer the neural
net onto the GPU.

Let's first define our device as the first visible cuda device if we have
CUDA available:

```
# Assuming that we are on a CUDA machine, this should print a CUDA device:
```

The rest of this section assumes that `device` is a CUDA device.

Then these methods will recursively go over all modules and convert their
parameters and buffers to CUDA tensors:

```
net.to(device)
```

Remember that you will have to send the inputs and targets at every step
to the GPU too:

```
inputs, labels = data[0].to(device), data[1].to(device)
```

Why don't I notice MASSIVE speedup compared to CPU? Because your network
is really small.

**Exercise:** Try increasing the width of your network (argument 2 of
the first `nn.Conv2d`, and argument 1 of the second `nn.Conv2d` -
they need to be the same number), see what kind of speedup you get.

**Goals achieved**:

- Understanding PyTorch's Tensor library and neural networks at a high level.
- Train a small neural network to classify images

## Training on multiple GPUs

If you want to see even more MASSIVE speedup using all of your GPUs,
please check out [Optional: Data Parallelism](data_parallel_tutorial.html).

## Where do I go next?

- [Train neural nets to play video games](../../intermediate/reinforcement_q_learning.html)
- [Train a state-of-the-art ResNet network on imagenet](https://github.com/pytorch/examples/tree/main/imagenet)
- [Train a face generator using Generative Adversarial Networks](https://github.com/pytorch/examples/tree/main/dcgan)
- [Train a word-level language model using Recurrent LSTM networks](https://github.com/pytorch/examples/tree/main/word_language_model)
- [More examples](https://github.com/pytorch/examples)
- [More tutorials](https://github.com/pytorch/tutorials)
- [Discuss PyTorch on the Forums](https://discuss.pytorch.org/)
- [Chat with other users on Slack](https://pytorch.slack.com/messages/beginner/)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: cifar10_tutorial.ipynb`](../../_downloads/4e865243430a47a00d551ca0579a6f6c/cifar10_tutorial.ipynb)

[`Download Python source code: cifar10_tutorial.py`](../../_downloads/c51fcdf96d93a8e4b3f2943cb36bab19/cifar10_tutorial.py)

[`Download zipped: cifar10_tutorial.zip`](../../_downloads/3b5f5aeb255cdb504d5e38213ac8a112/cifar10_tutorial.zip)