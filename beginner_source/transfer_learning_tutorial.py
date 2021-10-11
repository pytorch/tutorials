# -*- coding: utf-8 -*-
"""
Transfer Learning for Computer Vision Tutorial
==============================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_ and `Kaichao You <https://youkaichao.github.io>`_

In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at `cs231n notes <https://cs231n.github.io/transfer-learning/>`__

Quoting these notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

To be specific, we will show three transfer learning scenarios as follows:

-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.
-  **Finetuning the convnet**: Instead of random initialization, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **Finetuning the convnet with an advanced method**: Besides vanilla
   finetuning, we will show how to use an advanced finetuning method named 
   Co-Tuning to further improve transfer learning almost for free (without
   any additional data)!
"""
# License: BSD
# Author: Sasank Chilamkurthy and Kaichao You

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

seed = 0 # set seed for reproducibility
torch.manual_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

plt.ion()   # interactive mode

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# fine-grained bird species. The dataset is a subset of the CUB-200-2011
# dataset with 200 classes. There are about 1200 images for training 
# (~6 images per class). If trained from scratch, neural networks with
# such a small dataset would have a difficult time generalizing to the
# validation data. Since we are using transfer learning, we should be able
# to generalize reasonably well.
# 
# .. Note ::
#    Download the data from
#    `here <https://cloud.tsinghua.edu.cn/f/d222ed46a3064dbe8a48/?dl=1>`_
#    and extract it to the current directory.

#
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/cubsub'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                              shuffle=(x == 'train'), num_workers=4, worker_init_fn=seed_worker)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from the first 4 images in the batch
out = torchvision.utils.make_grid(inputs[:4])

imshow(out, title=[class_names[x] for x in classes[:4]])


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # compute gradients if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    target_outputs = outputs[1] if isinstance(outputs, tuple) else outputs
                    _, preds = torch.max(target_outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            target_outputs = outputs[1] if isinstance(outputs, tuple) else outputs
            _, preds = torch.max(target_outputs, 1)

            for j in range(inputs.size()[0]):
                label = labels[j].item()
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('label:{}; predicted: {}'.format(class_names[label], class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


######################################################################
# 1. ConvNet as fixed feature extractor
# ----------------------------------
#
# First, we show the simplest way of transfer learning: use pre-trained 
# network as an fixed feature extractor. Here we need to freeze all the
# network except the final layer. We need
# to set ``requires_grad == False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
#
# You can read more about this in the documentation
# `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
#

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

######################################################################
#

visualize_model(model_conv)

plt.ioff()
plt.show()

######################################################################
# On my computer, using pre-trained ConvNet as fixed feature extractor
# achieves 33.29% accuracy. Considering there are 200 classes, the accuracy
# is decent enough (much better than random guess accuracy 0.5%). But
# actually, freezing the feature extractor is not a common practice. Keep
# reading the following to see how to improve.
# 

        
######################################################################
# Finetuning the convnet
# ----------------------
# 
# To train a decent classifier, typically we need to finetune the pre-trained
# network to better fit the target dataset. Let's load a pre-trained model
# and reset final fully connected layer, finetuning the feature extractor.
#

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

######################################################################
# Since pre-trained models have been trained, it is a common practice to set
# a smaller learning rate for pre-trained parameters compared to newly created
# parameters. 
# 
# You can read more on how to set parameter groups in the documentation
# `here <https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer>`__.
#

params = [{"params": [p for name, p in model_ft.named_parameters() if 'fc' in name], "lr": 1e-2},
               {"params": [p for name, p in model_ft.named_parameters() if 'fc' not in name], "lr": 1e-3}]

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

######################################################################
#

visualize_model(model_ft)

######################################################################
# After finetuning, the classification accuracy is 38.44%, five percents
# higher than fixing the feature extractor. That's why people often prefer
# to finetune the pre-trained model. As the task becomes more and more
# complex, the performance gain of finetuning over fixing ConvNet can be
# even larger.
# 

######################################################################
# Finetuning the convnet with an advanced method
# ----------------------
#
# Can we do better than finetuning? Researchers have explored a lot and 
# here we present one conceptually simple method named Co-Tuning.
#

from torch import nn
from torch.functional import F


class CoTuningHead(nn.Module):
    """
    Implements the Co-Tuning algorithm as described in the NeurIPS 2020 paper `Co-Tuning for Transfer Learning <https://papers.nips.cc/paper/2020/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf>`_.
    """
    def __init__(self, old_head: nn.Linear, num_class: int):
        """
        :param old_head: the last module that transforms features into logits
        :param num_class: number of classes for the target task
        """
        super(CoTuningHead, self).__init__()
        self.old_head = old_head
        self.new_head = nn.Linear(old_head.in_features, num_class)

    def forward(self, features):
        old_output = self.old_head(features)
        new_output = self.new_head(features)
        return old_output, new_output

    def deploy(self):
        """
        When training finishes, convert the CoTuningHead to an ordinary Linear layer
        """
        return copy.deepcopy(self.new_head)


class CoTuningLoss(nn.Module):
    def __init__(self, trade_off: float):
        super().__init__()
        self.trade_off = trade_off
        self.fitted = False
        self.relationship = torch.Tensor([0, 0, 0]) # placeholder

    def to(self, device):
        self.relationship = self.relationship.to(device=device)
        return self

    def fit(self, old_logits: torch.Tensor, new_labels: torch.Tensor):
        """
        :param old_logits: shape of [N, Cs], where Cs is the number of classes in the pre-training task
        :param new_labels: shape of [N], where each element is an integer indicating the class id starting from 0
        """
        Ct = new_labels.max().item() + 1
        old_prob = F.softmax(old_logits, dim=-1)
        self.relationship = torch.stack([torch.mean(old_prob[new_labels == i], dim=0) for i in range(Ct)])
        self.fitted = True

    def forward(self, inputs, target):
        if not self.fitted:
            raise Exception("please call fit() first!")
        old_output, new_output = inputs
        target_loss = F.cross_entropy(new_output, target)
        old_softlabel = self.relationship[target]
        cotuning_loss = - (F.log_softmax(old_output, dim=-1) * old_softlabel).sum(dim=-1).mean(dim=0)
        return target_loss + self.trade_off * cotuning_loss


model_ct = models.resnet18(pretrained=True).to(device)
criterion = CoTuningLoss(trade_off = 1.0)

with torch.no_grad():
    was_training = model_ct.training
    model_ct.eval()
    old_logits = []
    new_labels = []
    for i, (inputs, labels) in enumerate(dataloaders['train']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_ct(inputs)
        target_outputs = outputs[1] if isinstance(outputs, tuple) else outputs
        old_logits.append(target_outputs.cpu())
        new_labels.append(labels.cpu())
    old_logits = torch.cat(old_logits)
    new_labels = torch.cat(new_labels)
    criterion.fit(old_logits, new_labels)
    model_ct.train(mode=was_training)

model_ct.fc = CoTuningHead(model_ct.fc, len(class_names))

model_ct = model_ct.to(device)
criterion = criterion.to(device)

params = [{"params": [p for name, p in model_ct.named_parameters() if 'new' in name], "lr": 1e-2},
               {"params": [p for name, p in model_ct.named_parameters() if 'new' not in name], "lr": 1e-3}]

# Observe that all parameters are being optimized
optimizer_ct = optim.SGD(params, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ct, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#

model_ct = train_model(model_ct, criterion, optimizer_ct, exp_lr_scheduler,
                       num_epochs=25)

######################################################################
#

visualize_model(model_ct)

######################################################################
# Co-Tuning achieves 41.03% accuracy, outperforming finetuning and fixing
# ConvNet.
# 

model_ct.fc = model_ct.fc.deploy()

######################################################################
# When the training finishes, convert the CoTuningHead to original Linear
# layer, so that inference procedure is unchanged.
# 
# What's next if the accuracy cannot satisfy your application? Then it is
# time to collect more data! That's the loop of transfer learning: try fixed
# ConvNet --> try finetuning ConvNet --> try some advanced finetuning methods
# (if not satisfied with the performance) --> collect more data and back to
# the first step.
# 

######################################################################
# Further Learning
# -----------------
#
# If you would like to learn more about the applications of transfer learning,
# checkout our `Quantized Transfer Learning for Computer Vision Tutorial <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.
#
#

