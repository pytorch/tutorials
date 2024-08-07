"""
Autoencoder: A Deep Dive
=========================

Introduction
~~~~~~~~~~~~

Autoencoder represent a class of artificial neural networks  
utilized for unsupervised learning tasks. They are engineered  
to learn efficient data encodings by mapping input data into  
a lower-dimensional latent space, subsequently reconstructing  
the original data from this latent representation. This  
methodology compels the autoencoder to encapsulate the most  
salient features of the input data, thereby enhancing the  
efficiency and effectiveness of data compression and feature  
extraction.

Architecture of an Autoencoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A typical autoencoder consists of two main components:

-  **Encoder:** This part of the network maps the input data to a latent
   space representation.
-  **Decoder:** This part reconstructs the original data from the latent
   space representation.

The goal of training is to minimize the reconstruction error between the
input and the reconstructed output.

Types of Autoencoder
~~~~~~~~~~~~~~~~~~~~~

There are several variations of autoencoder:

- **Denoising Autoencoder:** These are trained on corrupted 
  input data, learning to reconstruct the original clean data.

- **Variational Autoencoder:** These introduce 
  probabilistic elements into the encoding process, allowing 
  for generating new data samples.

- **Convolutional Autoencoder (CAE):** These use convolutional 
  layers, making them suitable for image data.

Applications of Autoencoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autoencoder have a wide range of applications:

-  **Dimensionality Reduction:** By projecting data into a
   lower-dimensional space, autoencoder can be used for visualization
   and feature extraction.
-  **Image Denoising:** Denoising autoencoder can effectively remove
   noise from images.
-  **Anomaly Detection:** Autoencoder can be used to identify unusual
   data points by measuring reconstruction errors.
-  **Image Generation:** Variational Autoencoder can generate new, realistic images based
   on the learned latent space distribution.
-  **Data Compression:** Undercomplete autoencoder can be used for data
   compression.

PyTorch Implementation
~~~~~~~~~~~~~~~~~~~~~~

Let’s implement a basic autoencoder using PyTorch for image compression:

"""

import torch
import torchvision

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torchvision import transforms
from IPython.display import clear_output


######################################################################
# Define the needed Functions.
# 

def make_dataloader(data_, batch_size: int):
    """Helper function to convert datasets to batches."""
    batch_size = 32

    # Make the Data loader Object
    train_loader = torch.utils.data.DataLoader(
        data_, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader


def make_transforms():
    """Helper function to make the transforms for datasets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return transform


def load_data_general(data_name: str):
    """Helper function to load the data."""
    transform_ = make_transforms()

    if data_name == "mnist":
        data_ = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_
        )
    elif data_name == "cifar":
        data_ = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_
        )

    return data_


def load_batch_data(dataset_name: str):
    # Load data
    train_data = load_data_general(dataset_name)

    # Make batches of data
    data_loader = make_dataloader(data_=train_data, batch_size=32)

    return data_loader


def load_mnist_data():
    """Load the MNIST dataset and covert it to batches."""

    return load_batch_data("mnist")


def load_cifar_data():
    """Load the CIFAR10 dataset and covert it to batches."""

    return load_batch_data("cifar")

def make_model(model_object, lr_rate=0.001, compress_=None):
    """Make all of the needed objects for training.

    Args:
        model_object:
            The class which we want to derive the model from.
        lr_rate:
            elarning rate for the optimizer
        compress_:
            the number of neurons at the heart of autoencoder which defines
            how much we are going to compress the data. We use this with linear
            autoencoder.

    Returns:
        A tuple cotanining the initiated model, optimizer and loss function.
    """
    if not compress_:
        model_ = model_object()
    else:
        model_ = model_object(compress_)
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=lr_rate)
    loss_ = nn.MSELoss()

    return model_, optimizer_, loss_


def test_model(loader_obj, model_, linear=True) -> None:
    """Test the output of the autoencoder model by showing the images.

    Args:
        loader_obj:
            The object of the loader for data batches.
        model_:
            The model which we want to test the output.
        linear:
            If te model is linear or CNN.
    """
    batch_iter = iter(loader_obj)
    batch_images = next(batch_iter)
    tmp_image = batch_images[0][0, 0, :, :]
    plt.imshow(tmp_image)
    plt.title("Original Image")
    plt.show()

    plt.figure()
    if linear:
        model_input = tmp_image.reshape(28 * 28)
    else:
        model_input = tmp_image.reshape(1, 1, 28, 28)

    model_.eval()
    with torch.inference_mode():
        output = model_(model_input)
    plt.imshow(output.detach().numpy().reshape(28, 28))
    plt.title("Model's Regenerated Picture")
    plt.show()

    return


def train_model(
    model_obj: nn.Module,
    optimizer_obj,
    loss_obj,
    loader_obj,
    batch_s: int,
    epoch_num: int = 1,
    model_linear=True,
) -> nn.Module:
    """Train the input model with optimizer and loss function."""
    train_loss = []

    for epoch in range(epoch_num):
        for i, data_ in enumerate(loader_obj, 0):
            batches, targets = data_
            if model_linear:
                batches = batches.reshape([batch_s, 28 * 28])

            # zero the parameter gradients
            optimizer_obj.zero_grad()

            # Find the output of the Neural Net
            # Forward Pass
            logits = model_obj(batches)

            # Calculate the loss
            loss = loss_obj(logits, batches)

            # Update the neural net and gradients
            # Backward Propagation
            loss.backward()
            optimizer_obj.step()

            # print(f"{loss.item():0.5f}")
            # Append the loss of training
            train_loss.append(loss.item())

    plt.plot(train_loss)
    plt.title("Training loss")
    plt.show()

    return model_obj


def add_noise(img_, noise_int: float) -> torch.Tensor:
    """Add noise to the given image.

    Args:
        img_:
            The given image.
        noise_int:
            The intensity of the noise, varies between 0 and 1.

    Returns:
        A tensor of the noisy image.
    """
    noise = np.random.normal(loc=0, scale=1, size=img_.shape)

    # noise overlaid over image
    noisy = np.clip((img_.numpy() + noise * noise_int), 0, 1)
    noisy_tensor = torch.tensor(noisy, dtype=torch.float).reshape(1, 1, 28, 28)

    return noisy_tensor


def noisy_test(
    loader_obj, model_: nn.Module, linear: bool = True, noise_intensity: float = 0.2
):
    """Test the model by adding noise to the image."""
    batch_iter = iter(loader_obj)
    batch_images = next(batch_iter)
    tmp_image = batch_images[0][0, 0, :, :]
    plt.imshow(tmp_image)
    plt.title("Original Image")
    plt.show()

    noisy_img = add_noise(tmp_image, noise_intensity)
    plt.figure()
    plt.imshow(noisy_img.reshape(28, 28).numpy())
    plt.title("Noisy Image")
    plt.show()

    plt.figure()
    if linear:
        model_input = noisy_img.reshape(28 * 28)
    else:
        model_input = noisy_img.reshape(1, 1, 28, 28)

    model_.eval()
    with torch.inference_mode():
        output = model_(model_input)
    plt.imshow(output.detach().numpy().reshape(28, 28))
    plt.title("Model's Regenerated Image")
    plt.show()

    return


def image_show(img_: torch.tensor, img_title: str):
    """Convert the batches to grids and show image."""
    img_ = torchvision.utils.make_grid(img_)
    npimg = img_.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(img_title)
    plt.show()

    return


def test_cifar(cifar_model, data_loader_):
    """Test the CIFAR model"""
    # get some random training images
    dataiter = iter(data_loader_)
    images, labels = next(dataiter)

    # show images by converting batches to grids
    image_show(images, "Original Image")

    cifar_model.eval()
    with torch.inference_mode():
        out_batch = cifar_model(images)
    image_show(out_batch, "Model's Regenerated Image")

    return



######################################################################
# Load Fashion MNIST Dataset
# --------------------------
# 
# **Breakdown:**
# 
# 1. **Import necessary libraries:**
# 
#    -  ``torchvision.transforms``: For image transformations.
#    -  ``torchvision.datasets``: For loading the MNIST dataset.
# 
# 2. **Define image transformations:**
# 
#    -  ``transforms.ToTensor()``: Converts PIL images to PyTorch tensors.
#    -  ``transforms.Normalize()``: Normalizes tensor images with mean and
#       standard deviation of 0.5.
# 
# 3. **Load training data:**
# 
#    -  ``torchvision.datasets.FashionMNIST``: Loads the Fashion MNIST
#       training dataset.
#    -  ``root``: Specifies the data directory.
#    -  ``train``: Set to ``True`` for training data.
#    -  ``download``: Downloads the dataset if not present.
#    -  ``transform``: Applies the defined transformations to the images.
# 
# 4. **Load testing data:**
# 
#    -  Similar to loading training data, but with ``train=False`` to load
#       the test set.
# 

batch_size = 32

# This gives us the loader object which is iterable and also has batches of data
train_loader = load_mnist_data()

clear_output()


######################################################################
# Let’s explore a little about the size of data and also what is included
# in it.
# 

for i, data_ in enumerate(train_loader):
    print(i, data_[0].shape)
    if i==10:
        break

for i, data_ in enumerate(train_loader):
    plt.imshow(data_[0][i, :, :].view(28, 28))
    plt.show()
    if i==3:
        break


######################################################################
# Autoencoder Definition
# ======================
# 


######################################################################
# Define Model
# ============
# 
# Here we define our model which us based on Autoencoder class and use the
# optimizer based on ``Adam Optimizer``.
# 
# First, We start with a simple Model which only uses Linear layers with
# Leaky Relu activations.
# 

class AutoencoderLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_en_1 = nn.Linear(in_features=28*28, out_features=196)
        self.linear_en_2 = nn.Linear(in_features=196, out_features=98)
        self.linear_de_1 = nn.Linear(in_features=98, out_features=196)
        self.linear_de_2 = nn.Linear(in_features=196, out_features=28*28)

    def forward(self, x):
        encode_1 = F.leaky_relu(self.linear_en_1(x))
        encode_2 = F.leaky_relu(self.linear_en_2(encode_1))
        decode_1 = F.leaky_relu(self.linear_de_1(encode_2))
        decode_2 = F.sigmoid(self.linear_de_2(decode_1))
        return decode_2

model_aal, optimizer, loss_fn = make_model(AutoencoderLinear)


######################################################################
# Is our model even working???
# 

test_model(train_loader, model_aal)


######################################################################
# So we are just getting noise, Let’s see what happens after a brief
# training.
# 

model_aal = train_model(model_aal, optimizer, loss_fn, train_loader, batch_size, epoch_num=10)


######################################################################
# Let’s test that how much our model has learned to implicate the exact
# input by seeing the real images
# 

test_model(train_loader, model_aal)


######################################################################
# We can See that After 10 epochs of training our model is learning the
# general shape of the given input. so we are on the right track.
# 
# Let’s make the linear model a bit dynamic. We add two linear layers
# which we could adjust the size of compression.
# 

class AutoencoderLinearA(nn.Module):
    def __init__(self, compress_nodes):
        super().__init__()
        self.linear_en_1 = nn.Linear(in_features=28*28, out_features=196)
        self.linear_en_2 = nn.Linear(in_features=196, out_features=98)
        self.linear_de_1 = nn.Linear(in_features=98, out_features=196)
        self.linear_de_2 = nn.Linear(in_features=196, out_features=28*28)
        self.linear_en_c = nn.Linear(in_features=98, out_features=compress_nodes)
        self.linear_de_c = nn.Linear(in_features=compress_nodes, out_features=98)

    def forward(self, x):
        encode_1 = F.leaky_relu(self.linear_en_1(x))
        encode_2 = F.leaky_relu(self.linear_en_2(encode_1))
        encode_c = self.linear_en_c(encode_2)
        decode_c = self.linear_de_c(encode_c)
        decode_1 = F.leaky_relu(self.linear_de_1(decode_c))
        decode_2 = F.sigmoid(self.linear_de_2(decode_1))
        return decode_2

model_aala, optimizer, loss_fn = make_model(AutoencoderLinearA, compress_=10)
model_aala = train_model(model_aala, optimizer, loss_fn, train_loader, batch_size, epoch_num=10)


######################################################################
# With a compression level which we compress all the 784 pixels to 10
# nodes and then rescale them we still learning the general shape of the
# item after 10 epochs.
# 
# We might get better results if we try and use higher epochs, These
# results are just for 10 epochs. (Although the chart is showing that the
# training loss might not increase a lot and we might need to make the
# model more complicated to decrease the error)
# 
# Although we can see that when we decrease the compression nodes we are
# loosing some data which increases the training error, as you can compare
# the charts before.
# 

test_model(model_=model_aala, loader_obj=train_loader)


######################################################################
# Autoencoder with CNN
# ====================
# 
# This Python code defines a convolutional autoencoder class using
# PyTorch. The autoencoder consists of an encoder and a decoder network.
# 
# **Encoder:** \* Takes a 1-Channel image as input. \* Applies a series of
# convolutional layers with LeakyReLU activations to extract features. \*
# Uses a flattening layer to convert the feature maps into a linear
# vector. \* Finally, projects the vector into a latent space
# representation.
# 
# **Decoder:** \* Takes the latent space representation as input. \*
# Projects it back to the original feature map size using a linear
# and unflatten layer. \* Applies a series of transposed convolutional layers
# with LeakyReLU activations to reconstruct the image. \* Uses an
# activation function to output the reconstructed image with pixel values
# between 0 and 1.
# 
# **Forward Pass:** \* Encodes the input image using the encoder. \*
# Decodes the encoded representation using the decoder. \* Returns the
# reconstructed image.
# 

# Build the Autoencoder with CNN using the sequential method from pytorch
class AutoencoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5,5), stride=(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=128*10*10, out_features=144)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=144, out_features=128*10*10),
            nn.Unflatten(1, (128, 10, 10)),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(5,5), stride=(1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(5,5), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model_cnn, optimizer, loss_fn = make_model(AutoencoderCNN)
model_cnn = train_model(model_cnn, optimizer, loss_fn, train_loader, batch_size, epoch_num=5, model_linear=False)

test_model(train_loader, model_cnn, linear=False)


######################################################################
# Looks like we have achieved a better result using a CNN Autoencoder with
# just only 5 epochs but a longer training time. (10 minutes instead of 30
# seconds training time)
# 
# Let’s take a look at the output of the ``code layer`` which compresses
# the data, and convert it to a picture to see if there is anything
# meaningful in there.
# 

batch_iter = iter(train_loader)
batch_images = next(batch_iter)
tmp_image = batch_images[0][0, 0, :, :]
enc_output = model_cnn.encoder(tmp_image.reshape(1, 1, 28, 28))

# We have 144 Nodes so we can derive a 12*12 picture from it.
plt.imshow(enc_output.detach().numpy().reshape(12, 12))
plt.title("Model's Encoder Output")
plt.show()


######################################################################
# Autoencoder for Data Noise Reduction
# -------------------------------------
# 
# Autoencoder have emerged as a powerful tool for mitigating noise in
# various data modalities. By training a neural network to reconstruct
# clean data from noisy inputs, these models effectively learn to filter
# out unwanted disturbances.
# 
# A key advantage of autoencoder lies in their ability to capture
# complex, non-linear relationships within data. This enables them to
# effectively remove noise while preserving essential features. Moreover,
# autoencoder are unsupervised learning models, requiring only unlabeled
# data for training, making them versatile for a wide range of
# applications.
# 
# By effectively removing noise, autoencoder can significantly enhance
# the performance of downstream machine learning models, leading to
# improved accuracy and robustness.
# 
# Let's introduce some noise to the image and evaluate how our model
# performs in reconstructing the output without noise.
# 

noisy_test(train_loader, model_cnn, linear=False, noise_intensity=0.3)


######################################################################
# We have added a lot of noise to our input data and our model was able to
# reduce many of them and find the general shape of our original image.
# 


######################################################################
# CIFAR 10
# ========
# 
# We will try to use the autoencoder with CIFAR10 dataset. This dataset
# consists of color images with 3 channels and 32*32 size.
# 
# Since the images in this dataset has more variety and also has colors in
# them we need to use a bigger model to be able to distinguish between
# pattern and also reproduce the given image with a low loss.
# 

# Load data and make it into chunks
cifar_loader = load_cifar_data()


######################################################################
# Let’s check the size of chunks
# 

cifar_loader.dataset.data.shape


######################################################################
# A quick peek at the images.
# 

# get some random training images
dataiter = iter(cifar_loader)
images, labels = next(dataiter)

# show images by converting batches to grids
image_show(images, "Original image")

# We use a similar architecture as before just tweaking some numbers for a bigger model
# since these pictures has 3 channels and we need to compress more data in our model
# We also add some padding to take into account the information that is stored on the edges of the pictures.
class AutoencoderCNNCIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4,4), stride=(1,1), padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3,3), stride=(2,2)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=512*6*6, out_features=100)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=100, out_features=512*6*6),
            nn.Unflatten(1, (512, 6, 6)),
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(3,3), stride=(2,2)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5,5), stride=(1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4,4), stride=(1,1), padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model_cifar, optimizer_cifar, loss_cifar = make_model(AutoencoderCNNCIF, .001)
model_cifar = train_model(model_cifar, optimizer_cifar, loss_cifar, cifar_loader, 32, 3, False)

# Test the output model by feeding random batches to it and get the output
test_cifar(model_cifar, cifar_loader)


######################################################################
# Our CNN model has successfully reconstructed many details 
# of the images, though the outputs remain somewhat blurry.
# 
# We should consider adding additional layers to the model 
# to enhance its ability to detect and preserve patterns in 
# the data during compression.
# 
# Another potential cause of the blurry images is the size 
# of the "code layer." If it is too small for this type of 
# data, it may lose crucial details, making it difficult to 
# recover specific information during reconstruction.