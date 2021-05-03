# -*- coding: utf-8 -*-
"""
SimCLR Tutorial
==============

**Author**: `Aritra Roy Gosthipaty <https://twitter.com/ariG23498>`__

"""


######################################################################
# Introduction
# ============
#
# This tutorial will give an introduction to self supervised learning
# through the minimal implementation of
# `SimCLR <https://arxiv.org/abs/2002.05709>`__. We will train SimCLR on
# the STL10 dataset and use the learned representations to classify
# images. This document will give a thorough explanation of the
# implementation and shed light on how and why this algorithm works.
#
# Self Supervised Learning
# ------------------------
#
# Yann LeCun in his `Le Cake <https://youtu.be/7I0Qt7GALVk?t=2773>`__
# analogy says > "If intelligence is a cake, the bulk of the cake is
# self-supervised learning, the icing on the cake is supervised learning,
# and the cherry on the cake is reinforcement learning (RL)."
#
# On the one hand problem with supervised learning is the unavailability
# of labeled data, and on the other convertig unlabeled data takes a lot
# of time, effort and human speculation.
#
# Researchers were always intrigued with the unsupervised paradigm of
# learning which could make use of the unlimited unlabeled data lying
# around. However vast the unlabaled data space might be, deep learning
# has gone a long way in supervised learning. This calls for a method that
# can not only consume the vast unlabeled data space but also use
# supervision to help achieve intelligence. A subset of unsupervised
# learning later came into existence known as the self-supervised
# learning.
#
# In the self-supervised learning method we try formulating a task where
# the labels are obtained from the data itself. This leads to the
# consumption of unlabeled data with a supervised approach. The end goal
# of self-supervision is to obtain rich general represenation of the data
# which can be later used for down stream tasks like classification. The
# intuition here is to obtain a generic representation from a huge amount
# of unlabeled data which would be better to train a downstream task than
# using the raw data itself.
#
# SimCLR
# ~~~~~~
#
# .. figure:: https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s640/image4.gif
#    :alt: SimCLR GIF
#
#    SimCLR GIF
#
# `Source: Google AI
# Blog <https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html>`__
#
# The beauty of every self supervised learning algorithm lies in the
# pretext task. The pretext task here is to maximise similarity between
# two augmented views of the same obhject while minimse the same for
# different objects. The intuition here is to view an object with
# different perspectives and retian the different representation from the
# same object.
#
# SimCLR is made of the following modules: - A stochastic data
# augmentation module. - A neural network base encoder :math:`f(.)`. - A
# neural network projection head :math:`g(.)` - A contrastive loss
# function.
#
######################################################################
# Imports
# =======
# 
# The following imports are used in the tutorial.
# 

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

import torchvision
from torchvision.transforms import (
    CenterCrop,
    Resize,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale,
    RandomApply,
    Compose,
    GaussianBlur,
    ToTensor,
)
import torchvision.models as models
from torchvision.datasets import STL10

import os
import time
import random
import matplotlib.pyplot as plt

# %matplotlib inline

print(f'Torch-Version {torch.__version__}')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {DEVICE}')


######################################################################
# Stochastic Data Augmentation Module
# ===================================
# 
# The authors suggest that a **strong** data augmentation is useful for
# self supervised learning.
# 
# The following augmentation are used in the order provided:
# 
# - Random Crop with Resize 
# - Random Horizontal Flip with 50% probability 
# - Random Color Distortion 
#   - Random Color Jitter with 80% probability 
#   - Random Color Drop with 20% probability
# - Random Gaussian Blur with 50% probability
# 
# .. figure:: https://amitness.com/images/simclr-random-transformation-function.gif
#    :alt: Augmentation
# 
#    Augmentation
# 
# `Source: Amitness' take on
# SimCLR <https://amitness.com/2020/03/illustrated-simclr/>`__
# 
# The data pipeline accepts an image and provides with two randomly
# augmented views of the images. In this section we build a custom data
# transformation module that will help with augmenting the images, and
# also a data loader that helps with providing two views of the same
# image.
# 

def get_baseline_transform(output_shape, kernel_size, s=1.0):
    """
    Method that creates a Compose of all the data pipeline
    transforms in the paper. This provides the baseline augmentation
    module which needs to applied on images.
    
    Args:
        output_shape: The output shape of images. 
        kernel_size: The gaussian kernel size.
        s: Strength parameter for color distortion.
    
    Returns:
        The complete baseline transform.
    """
    rnd_crop = RandomResizedCrop(output_shape)
    rnd_flip = RandomHorizontalFlip(p=0.5)
    
    color_jitter = ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = RandomApply([color_jitter], p=0.8)
    
    rnd_gray = RandomGrayscale(p=0.2)
    gaussian_blur = GaussianBlur(kernel_size=kernel_size)
    rnd_gaussian_blur = RandomApply([gaussian_blur], p=0.5)
    to_tensor = ToTensor()
    image_transform = Compose([
        to_tensor,
        rnd_crop,
        rnd_flip,
        rnd_color_jitter,
        rnd_gray,
        rnd_gaussian_blur,
    ])
    return image_transform


class ContrastiveLearningViewGenerator(object):
    """
    The data pipeline.

    Takes an image and returns two augmented views of
    the original image.
    """

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        views = [self.base_transform(x) for i in range(self.n_views)]
        views = [torch.clip(view, 0.0, 1.0) for view in views]
        return views

# The size of the images
output_shape = [224,224]
kernel_size = [21,21] # 10% of the output_shape

# The custom transform
base_transforms = get_baseline_transform(
    output_shape=output_shape, 
    kernel_size=kernel_size,
    s=1.0
)
custom_transform = ContrastiveLearningViewGenerator(
    base_transform=base_transforms
)


######################################################################
# Dataset
# -------
# 
# We will be using the ``unlabeled`` split of the STL10 dataset for our
# training purposes. There are 100000 data points to train on. This might
# be very good for the training purpose but would eventually need a lot of
# time to train on. For the sake of minimal implementation I have used
# 1,000 data points from the unlabelled training split.
# 
# After we train the SimCLR model we will use the ``train`` split of STL10
# to train a linear classifier on the representations learned. This
# becomes the down stream task where we use the representations learned
# from our SimCLR model.
# 
# We then test to see how rich the SimCLR represenations are as we
# performs a classification test on the ``test`` split of the STL10
# dataset.
# 

# SimCLR training data
unlabeled_ds = STL10(
    root="/content/drive/MyDrive/Colab Notebooks/SimCLR",
    split="unlabeled",
    transform=custom_transform,
    download=True)


######################################################################
# View the data
# -------------
# 
# Visulaization of the data is a great part of training. We need to see
# whether the data pipeline works as expected. The ``view_data`` method
# takes the dataset and an index as input and provides ``6`` pairs of
# augmented views of an object. Every pair has to be different thus
# proving that out augmentation module is indeed stochastic in nature.
# 
# Feel free to change the index and view another data point.
# 

plt.figure(figsize=(10,20))
def view_data(ds, index):
    """
    Visulaisation of the data pipeline.

    Args:
        ds: The dataset
        index: The index of the dataset that
            we want to view.
    """
    for i in range(1,6):
        images, _ = ds[index]
        view1, view2 = images
        plt.subplot(5,2,2*i-1)
        plt.imshow(view1.permute(1,2,0))
        plt.subplot(5,2,2*i)
        plt.imshow(view2.permute(1,2,0))

view_data(unlabeled_ds, random.randint(0,100000))


######################################################################
# Build the SimCLR dataloader
# ---------------------------
# 

BATCH_SIZE = 128

train, _ = random_split(unlabeled_ds, [1_000, 99_000])
del _
del unlabeled_ds
train_dl = torch.utils.data.DataLoader(
    train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)


######################################################################
# SimCLR model
# ============
# 
# In this section I take you throught the architecture of the SimCLR
# model. The model consists of three individual parts. - An encoder
# :math:`f(.)` - A projection head :math:`g(.)` - A contrastive loss
# function :math:`J(\theta)`
# 
# Encoder :math:`f(.)`
# --------------------
# 
# The authors do not stress on the choice of the encoder. They let reader
# choose the encoder's architecture. The main motive of the encoder is to
# extract representation from the input image fed to it. For this tutorial
# we will be using the ``ResNet18`` architecture.
# 
# .. math::
# 
# 
#    f(x) = h
# 
#  Where:
# 
# - :math:`x` is an image, :math:`x \in \mathcal{R}^{h,w,c}`
# - :math:`h` is the encoded representation, :math:`h \in \mathcal{R}^{d_{e}}`
# 
# Projection head :math:`g(.)`
# ----------------------------
# 
# The sole purpose of the projection head is to transform the encoded
# representations into a better latent space. The authors have noted that
# a richer understanding is extracted upon using the projection embeddings
# to compute the contrastive loss.
# 
# For the projection head we will be using ``linear layers`` with ``ReLU``
# activation units.
# 
# .. math::
# 
# 
#    g(h) = z
# 
#  where: 
# - :math:`z` is the projection of the encoder embeddings, :math:`z \in \mathcal{R}^{d_{p}}`
# 

class SimCLR(nn.Module):
    def __init__(self,
                 resnet_dim=1024,
                 hidden_dim = 512,
                 out_dim=256,
                 downstream_flag=False):
        
        super().__init__()
        # Flag to check when
        # the model is used
        # in downstream task
        self.downstream_flag = downstream_flag
        
        resnet18 = models.resnet18(
            pretrained=False,
            num_classes=resnet_dim)
        self.encoder = resnet18
        self.projection = nn.Sequential(
            nn.Linear(resnet_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        if not self.downstream_flag:
            # SimCLR training
            view1, view2 = x
            enc1 = self.encoder(view1)
            enc2 = self.encoder(view2)

            z1 = self.projection(enc1)
            z2 = self.projection(enc2)
            return z1, z2
        else:
            # Downstream training
            enc = self.encoder(x)
            return enc


######################################################################
# Contrastive loss
# ----------------
# 
# The contrastive loss function is the heart of the pretext task presented
# in SimCLR. The loss function needs to be broken down into three
# submodules to be understood better: 
# - Similarity function: A function that quantifies how similar two vectors are.
# - Softmax function: A function that transforms a distribution into a normalised probability disctribution. 
# - NT-Xent loss: The contrastive loss function.
# 
# Similarity
# ~~~~~~~~~~
# 
# The authors went ahead with using the ``cosine`` similarity function to
# measure how similar two views are.
# 
# .. math::
# 
# 
#    sim(u,v)=\frac{u_{t}v}{||u||\space||v||}
# 
# .. figure:: https://amitness.com/images/image-similarity.png
#    :alt: Similarity
# 
#    Similarity
# 
# `Source: Amitness' take on
# SimCLR <https://amitness.com/2020/03/illustrated-simclr/>`__
# 
# Softmax
# ~~~~~~~
# 
# .. math::
# 
# 
#    softmax(x_{i})=\frac{\exp{x_{i}}}{\sum_{k=1}^{N}\exp{x_{k}}}
# 
#     What happens if we feed the similarity scores into the softmax
#     function?
# 
# The answer is quite simple to visualize, we will obtain a probability
# distribution of the similarity scores. This means we will know how
# probable is one view going to be similar to another.
# 
# NT-Xent Loss
# ~~~~~~~~~~~~
# 
# The normalized temperature scaled cross entropy loss. The loss function
# will calculate how much does the prediction of the most similar views
# miss by the real similar views. The use of the temperature term is
# described really well in this `reddit
# thread <https://www.reddit.com/r/MachineLearning/comments/n1qk8w/d_temperature_term_in_simclr_or_moco_papers/gwex2ap?utm_source=share&utm_medium=web2x&context=3>`__.
# 
# .. math::
# 
# 
#    l(i,j)=-\log{\frac{\exp(sim(i,j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{k\neq i}\exp{sim(i,k)/\tau}}}\\
#    \mathcal{L}=\frac{1}{2N}\sum_{k}^{N}[l(2k-1,k)+l(2k,2k-1)]
# 
#     The following implementation of the contrastive loss is a modified
#     version of sthalles's contrastive loss from his implementation of
#     `SimCLR <https://github.com/sthalles/SimCLR>`__
# 

def contrastive_loss(features, temp, LABELS, MASK):
    """
    Contrastive loss function.

    Args:
        features: list of both the encoded views, [z1, z2]
        temp: temperature term used in the loss
        LABLES: positive and negative labels
        MASK: the mask used to remove the diagonal elements
    """
    SHAPE = BATCH_SIZE*2
    features = torch.cat(features, dim=0)

    features = nn.functional.normalize(features, p=2, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    similarity_matrix = similarity_matrix[~MASK].view(SHAPE,-1)
    LABELS = LABELS[~MASK].view(SHAPE,-1)

    positives = similarity_matrix[LABELS.bool()].view(SHAPE, -1)
    negatives = similarity_matrix[~LABELS.bool()].view(SHAPE, -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

    logits = logits / temp
    return logits, labels

# GLOBALS
EYE_SHAPE = BATCH_SIZE*2
MASK = torch.eye(EYE_SHAPE, dtype=torch.bool).to(DEVICE)
LABELS = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() # Creates a one-hot with broadcasting
LABELS = LABELS.to(DEVICE)

EPOCHS = 50

simclr_model = SimCLR().to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(
    params=simclr_model.parameters(),
    lr=1e-1,
)


######################################################################
# Training the model
# ------------------
# 
# We will be training the model in this section. The metrics to note here
# is the NT-Xent loss. We will plot the loss in the next section to
# understand how well our model learns.
# 

epoch_loss = []
for epoch in range(EPOCHS):
    t0 = time.time()
    batch_loss = []
    running_loss = 0.0
    for i, elements in enumerate(train_dl):
        views, _ = elements
        z1, z2 = simclr_model([view.to(DEVICE) for view in views])
        logits, labels = contrastive_loss(
            [z1, z2],
            temp=1e-1,
            LABELS=LABELS,
            MASK=MASK,
        )

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print(f'EPOCH: {epoch+1} BATCH: {i+1} BATCH_LOSS: {(running_loss/10):.3f}')
            running_loss = 0.0
        batch_loss.append(loss.item())
    time_taken = (time.time()-t0)/60
    total_loss = sum(batch_loss)/len(batch_loss)
    epoch_loss.append(total_loss)
    print(f'EPOCH: {epoch+1} LOSS: {(total_loss):.3f} Time taken: {time_taken:.2f} mins\n')

plt.plot(epoch_loss)
plt.title("SimCLR Training")
plt.xlabel("Epochs")
plt.ylabel("NT-Xent Loss")
plt.show()


######################################################################
# Down Stream Task
# ================
# 
# The self-supervised learning algorithm has helped us
# 

resize = Resize(255)
ccrop = CenterCrop(224)
ttensor = ToTensor()

custom_transform = Compose([
    resize,
    ccrop,
    ttensor,
])

# Linear Classifier training data
labeled_ds = STL10(
    root="/content/drive/MyDrive/Colab Notebooks/SimCLR",
    split="train",
    transform=custom_transform,
    download=True)

# Linear Classifier testing data
test_ds = STL10(
    root="/content/drive/MyDrive/Colab Notebooks/SimCLR",
    split="test",
    transform=custom_transform,
    download=True)

nu_classes = len(labeled_ds.classes)
BATCH_SIZE = 128

# Building the data loader
train_dl = DataLoader(
    labeled_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)

# Building the data loader
test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)


######################################################################
# Classification
# --------------
# 
# All that is important to us is the base encoder of the trained SimCLR
# model. We will need the representation of the images from the trained
# SimCLR model and then use it in the down stream task, here
# classification. We will replace the projection head and swap it with a
# linear layer. The linear layer will output the probability of the
# classes the input belongs to.
# 

class Classification(nn.Module):
    def __init__(self, model, nu_classes, simclr_dim=1024):
        super().__init__()
        
        simclr = model
        simclr.downstream_flag = True
        
        self.simclr = simclr
        # Freeze the simcle model
        # Do not learn in the downstram task
        for param in self.simclr.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(simclr_dim, nu_classes)

    def forward(self, x):
        encoding = self.simclr(x)
        pred = self.linear(encoding)
        pred = nn.functional.normalize(pred, p=2, dim=1)
        return pred

simclr_dim=1024
classification_model = Classification(
    simclr_model,
    nu_classes,
    simclr_dim=simclr_dim
    ).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(classification_model.parameters())

EPOCHS = 30
epoch_loss = []
for epoch in range(EPOCHS):
    t0 = time.time()
    batch_loss = []
    running_loss = 0.0
    for i, element in enumerate(train_dl):
        images, labels = element
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        preds = classification_model(images)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print(f'EPOCH: {epoch+1} BATCH: {i+1} LOSS: {(running_loss/10):.4f} ')
            running_loss = 0.0
        batch_loss.append(loss.item())
    time_taken = (time.time()-t0)/60
    total_loss = sum(batch_loss)/len(batch_loss)
    epoch_loss.append(total_loss)
    print(f'EPOCH: {epoch+1} LOSS: {(total_loss):.3f} Time taken: {time_taken:.2f} mins\n')

plt.plot(epoch_loss)
plt.title("Classification with SimCLR representation")
plt.xlabel("Epochs")
plt.ylabel("Classification Loss")
plt.show()


######################################################################
# Testing
# =======
# 

correct = 0
total = 0
with torch.no_grad():
    for data in test_dl:
        images, labels = data
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        preds = classification_model(images)
        _, predicted = torch.max(preds, 1)
        correct += (predicted == labels).sum().item()
        total += 1

print(f'Accuracy: {(correct/(BATCH_SIZE*total)*100):.2f}%')

######################################################################
# Conclusion and Final Thoughts:
# -----------------------------
# 
# While the accuracy of our model looks really bad
# we need to think about the way the classification takes place. For an
# ablation why not you try classifying images using a single linear layer?
# This will itself unroll how rich the SimCLR representations are.
# 
# I have taken help from the following sources: 
# 
# - `Amitness' take on SimCLR <https://amitness.com/2020/03/illustrated-simclr/>`__
# - `Sayak Paul's take on SimCLR <https://amitness.com/2020/03/illustrated-simclr/>`__
# - `sthalles pytorch code implementation on SimCLR <https://amitness.com/2020/03/illustrated-simclr/>`__
#
# What are the changes one could do?
# 
# - Train for longer to see how good the results get.
# - Use a large number of datapoints for SimCLR.
# - Use a different optimiser to stabilize the training (LARS, as used in the paper)
# 