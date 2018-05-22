# -*- coding: utf-8 -*-
"""
Classifying an imbalanced dataset
*********************************
**Author**: `Piotr Bialecki <https://github.com/ptrblck>`_

This tutorial show how to use the `WeightedRandomSampler` and a weighted loss
function to classify an imbalanced dataset.

We will load the CIFAR10 dataset and resample some classes creating an
artificially imbalanced dataset. Using this dataset we will train a CNN and
see how poorly the model performs on the undersampled class instances.

In the next step we will create a `WeightedRandomSampler` and oversample the
minority classes, so that our `DataLoader` will yield samples from nearly
uniformly distributed classes.

Finally, we will use a weighted criterion and train our model again on the
imbalanced dataset.

Have a look at the `CIFAR10 tutorial 
<https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_ if you
would like to get more information about the dataset.

Since we have to train the model for a few epochs, the usage of a GPU is 
recommended.

Let's start by importing some libs.
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Loading the data and creating a model
# =====================================
#
# In this tutorial we will use the CIFAR10 dataset and a standard CNN model.
# Both can be loaded using ``torchvision``, but feel free to use your own
# dataset or model.

SEED = 2809
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
batch_size = 64
epochs = 25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Training dataset
train_dataset = datasets.CIFAR10(
    root='.', train=True, download=True, transform=transform)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available())

test_dataset = datasets.CIFAR10(
    root='.', train=False, download=True, transform=transform)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available())

class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
nb_classes = len(class_names)

###############################################################################
# Let's have a look at the class distribution in the datasets.


def get_labels_and_class_counts(labels_list):
    '''
    Calculates the counts of all unique classes.
    '''
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)
    return labels, class_counts


def plot_class_distributions(class_names, train_class_counts,
                             test_class_counts):
    '''
    Plots the class distributions for the training and test set asa barplot.
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 6))
    ax1.bar(class_names, train_class_counts)
    ax1.set_title('Training dataset distribution')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Class counts')
    ax2.bar(class_names, test_class_counts)
    ax2.set_title('Test dataset distribution')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Class counts')


# Get all training targets and count the number of class instances
train_targets, train_class_counts = get_labels_and_class_counts(
    train_dataset.train_labels)
test_targets, test_class_counts = get_labels_and_class_counts(
    test_dataset.test_labels)

plot_class_distributions(class_names, train_class_counts, test_class_counts)

###############################################################################
# The classes are balanced in the original CIFAR10 dataset.
# We have 5000 and 1000 instanced of each class in the training and test
# dataset, respectively.
#
# Building a CNN and training with the original dataset
# -----------------------------------------------------
#
# Now let's build a model, setup the criterion and start the training.


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


model = Net()
model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


def train(epoch, model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Push data and target to GPU, if available
        data = data.to(device)
        target = target.to(device)

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Weight update
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:  # print every 100 mini-batches
            print('[{}, {}] loss: {:.3f}'.format(epoch + 1, batch_idx + 1,
                                                 running_loss / 100))
            running_loss = 0.0


def evaluate(model, test_loader, plot_confusion_mat=False):
    model.eval()
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for data, target in test_loader:
            # Push data and target to GPU, if available
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)

            correct += (preds == target).sum().item()
            total += target.size(0)
            for t, p in zip(target, preds):
                confusion_matrix[t, p] += 1

    # Calculate global accuracy
    accuracy = 100 * correct / total

    # Calculate class accuracies
    class_correct = confusion_matrix.diag()
    class_total = confusion_matrix.sum(1)
    class_accuracies = class_correct / class_total

    # Normalize confusion matrix
    for i in range(nb_classes):
        confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

    # Print statistics
    print('Accuracy of the model {:.3f}%'.format(accuracy))
    for i in range(nb_classes):
        print('Accuracy for {}: {:.3f}%'.format(class_names[i], 100 *
                                                class_accuracies[i]))
    print('Mean per class accuracy: {:.3f}%'.format(100 *
                                                    class_accuracies.mean()))

    if plot_confusion_mat:
        # Plot confusion matrix
        f = plt.figure()
        ax = f.add_subplot(111)
        cax = ax.imshow(confusion_matrix.numpy(), interpolation='nearest')
        f.colorbar(cax)
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.yticks(range(len(class_names)), class_names)


###############################################################################
# Let's have a look at the performance of our model on the test dataset.
# It's also interesting to see the confusion matrix, where each row represents
# the target instances while each column represents the predicted instances.

for epoch in range(epochs):
    train(epoch, model, train_loader, optimizer, criterion)
evaluate(model, test_loader, plot_confusion_mat=True)

###############################################################################
# As expected the network performs quite good and the accuracies for each class
# are very similar.
#
# Creating an imbalanced dataset
# =================================
#
# Now let's simulate an imbalanced dataset.
# We will resample the training and test dataset, so that some
# classes will be undersampled. Let's reduce the amount of instances for the
# first 5 classes by 90% leaving only the first 10% in the dataset.
# The last 5 classes will keep their samples.


class ImbalancedCIFAR10(Dataset):
    def __init__(self, imbal_class_prop, root, train, download, transform):
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform)
        self.train = train
        self.imbal_class_prop = imbal_class_prop
        self.idxs = self.resample()

    def get_labels_and_class_counts(self):
        return self.labels, self.imbal_class_counts

    def resample(self):
        '''
        Resample the indices to create an artificially imbalanced dataset.
        '''
        if self.train:
            targets, class_counts = get_labels_and_class_counts(
                self.dataset.train_labels)
        else:
            targets, class_counts = get_labels_and_class_counts(
                self.dataset.test_labels)
        # Get class indices for resampling
        class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
        # Reduce class count by proportion
        self.imbal_class_counts = [
            int(count * prop)
            for count, prop in zip(class_counts, self.imbal_class_prop)
        ]
        # Get class indices for reduced class count
        idxs = []
        for c in range(nb_classes):
            imbal_class_count = self.imbal_class_counts[c]
            idxs.append(class_indices[c][:imbal_class_count])
        idxs = np.hstack(idxs)
        self.labels = targets[idxs]
        return idxs

    def __getitem__(self, index):
        img, target = self.dataset[self.idxs[index]]
        return img, target

    def __len__(self):
        return len(self.idxs)


# Create class proportions
imbal_class_prop = np.hstack(([0.1] * 5, [1.0] * 5))
train_dataset_imbalanced = ImbalancedCIFAR10(
    imbal_class_prop, root='.', train=True, download=True, transform=transform)
test_dataset_imbalanced = ImbalancedCIFAR10(
    imbal_class_prop,
    root='.',
    train=False,
    download=True,
    transform=transform)

_, train_class_counts = train_dataset_imbalanced.get_labels_and_class_counts()
_, test_class_counts = test_dataset_imbalanced.get_labels_and_class_counts()

# Visualize imbalanced class distribution
plot_class_distributions(class_names, train_class_counts, test_class_counts)

# Create new DataLoaders, since the datasets have changed
train_loader_imbalanced = DataLoader(
    dataset=train_dataset_imbalanced,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available())

test_loader_imbalanced = DataLoader(
    dataset=test_dataset_imbalanced,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available())

###############################################################################
# Before starting the training, let's have a look at some batches and its class
# distribution.

train_iter_imbalanced = iter(train_loader_imbalanced)
for _ in range(5):
    _, target = train_iter_imbalanced.next()
    print('Classes {}, counts: {}'.format(*np.unique(
        target.numpy(), return_counts=True)))

###############################################################################
# As we can see some (minority) classes are completely missing in some batches.
# Our model will thus get a stronger signal from the majority classes and could
# neglect the undersampled classes.
#
# Training on the imbalanced dataset
# ----------------------------------
#
# Let's train a new model on the imbalanced dataset and have a look at its
# performance.

model = Net()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
for epoch in range(epochs):
    train(epoch, model, train_loader_imbalanced, optimizer, criterion)
evaluate(model, test_loader_imbalanced, plot_confusion_mat=True)

###############################################################################
# As you can see the overall accuracy of a classifier for an imbalanced dataset
# might be misleading. Have a look at the Wikipedia article on the
# `Accuracy paradox <https://en.wikipedia.org/wiki/Accuracy_paradox>`_.
#
# Calculating the mean per class accuracy we see that the model performs pretty
# poorly on the imbalanced dataset.
# To tackle this issue, we can use a ``WeightedRandomSample`` to oversample
# the minority classes while training the model.
#
# Using a WeightedRandomSampler
# =============================
#
# To create a ``WeightedRandomSampler`` we need to specify the ``weights``,
# which can be seen as probabilities (although they don't need to sum to one).
#
# First, we have to get the current class counts for the imbalanced dataset.
# Since we usually don't know the targets of the test data, let's just use
# the targets from the training dataset.

train_stats = train_dataset_imbalanced.get_labels_and_class_counts()
train_targets, train_class_counts = train_stats[0], train_stats[1]

###############################################################################
# We will calculate the ``weights`` as the reciprocal of the class counts,
# so that frequent classes will get a lower weight, while rare classes will
# get a higher weight. Remember, that the weights do not need to sum to one,
# so we don't have to normalize them.
#
# Since the `WeightedRandomSampler` will sample the elements from a list of
# ``[0, ..., len(weights)-1]`` we have to make sure to set the weight for each
# training sample in our dataset.

weights = 1. / torch.tensor(train_class_counts, dtype=torch.float)
samples_weights = weights[train_targets]
for name, count, weight in zip(class_names, train_class_counts, weights):
    print('Class {}: {} samples, {:.5} weight'.format(name, count, weight))

###############################################################################
# ``samples_weights`` now stores the individual weight for every sample.
# Using this ``tensor`` we will now create our sampler and pass it to the
# ``DataLoader``. Note that we cannot use ``shuffle=True`` anymore, since we
# have defined a sampler.

from torch.utils.data.sampler import WeightedRandomSampler

sampler = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=True)

train_loader_weighted = DataLoader(
    train_dataset_imbalanced,
    batch_size=batch_size,
    num_workers=2,
    sampler=sampler,
    pin_memory=torch.cuda.is_available())

###############################################################################
# Before starting the training, let's have another look at some batches and
# the class distribution in them.

train_iter_weighted = iter(train_loader_weighted)
for _ in range(5):
    _, target = train_iter_weighted.next()
    print('Classes {}, counts: {}'.format(*np.unique(
        target.numpy(), return_counts=True)))

###############################################################################
# Now we can see that the classes seem to be more balanced than before.
# Since the weights can be seen as probabilities, we do not force the
# ``DataLoader`` to sample an equal amount of instances for each class.
#
# As a side node: Since we are oversampling the minority classes, the model
# might overfit, since it sees the same duplicated samples a few times.
# Data augmentation might help with this issue, but keep in mind to validate
# your experiments properly and stop the training if the evaluation error is
# rising.
#
# Training with the WeightedRandomSampler
# ---------------------------------------
#
# Let's start the training again and have a look at the performance.

model = Net()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
for epoch in range(epochs):
    train(epoch, model, train_loader_weighted, optimizer, criterion)
evaluate(model, test_loader_imbalanced, plot_confusion_mat=True)

###############################################################################
# We got a mean per class accuracy of 51% using the ``WeightedRandomSampler``,
# which is better than our plain training on the imbalanced dataset!
#
# Another valid approach for training an imbalanced dataset would be to use a
# weighted criterion. Let's try this approach next.
#
# Training using a weighted criterion
# ===================================
#
# First, let's create a new DataLoader without the sampler and also a new model
# to restart the training from scratch.

train_loader_imbalanced = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=2,
    shuffle=True,
    pin_memory=torch.cuda.is_available())

model = Net()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

###############################################################################
# Now let's create a weighted criterion. We will still use ``nn.NLLLoss``, but
# will provide a weight tensor this time. The weight will be multiplied with
# the loss for the current class adding a penalty to the minority classes.
# It has to be a tensor specifying a weight for each class. Since the minority
# classes are undersampled by a factor of 10, let's use a weight of 10 and 1
# for the minority and majority classes, respectively.

weight = torch.cat((torch.tensor([10.] * 5), torch.tensor([1.] * 5)))
weight = weight.to(device)
criterion = nn.NLLLoss(weight=weight)

###############################################################################
# Let's train and evaluate the model again.

for epoch in range(epochs):
    train(epoch, model, train_loader_imbalanced, optimizer, criterion)
evaluate(model, test_loader_imbalanced, plot_confusion_mat=True)

###############################################################################
# The mean per class accuracy is comparable to the one we got using the
# ``WeightedRandomSampler`` and is also better than the plain training.
#
# In this tutorial we have seen, how to use different techniques to handle an
# imbalanced dataset. Depending on your use case, model, class distribution,
# and probably many more factors, one cannot say a specific approach is always
# better than the others.
# Like so many times, you have to try different approaches and see, which ones
# yields the desired performance.
#
# As an exercise you could combine the ``WeightedRandomSampler`` with a
# weigthed criterion and see, if this method performs any better than the
# others on its own. Also, you should play around with the ``weights`` for
# both approaches.
#
# A special thanks to `Josiane Rodrigues
# <https://discuss.pytorch.org/u/josiane_rodrigues>`_ for the idea to create
# this tutorial!
