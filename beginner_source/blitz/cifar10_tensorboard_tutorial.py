# -*- coding: utf-8 -*-
"""
Visualizing model training with Tensorboard
=====================

In the previous tutorial, we saw how to load in data, feed it through a model we defined as a subclass of ``nn.Module``, trained this model on training data, and tested it on testing data.

We were able to print out some statistics as the model was training to get a sense for whether training is progressing. However, we can do much better than that: PyTorch integrates with Tensorboard, a tool designed for visualizing the results of neural network training runs. Here we'll illustrate that functionality.  

----------------

Here are the steps we'll follow:

1. Read in data and apply appropriate transforms (same as prior tutorial).
2. Inspect the data by using `make_grid` to create a grid of images and log it to Tensorboard.
3. Log the model and inspect it using Tensorboard.
4. Log both the model's loss and the model's predictions on training batches as training proceeds 
"""
# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# constant for classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# helper function to show an image
def matplotlib_imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    # Note: need to transpose because matplotlib requires the channels
    # for color images to appear as the last dimension if they are 
    # present. See:
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
matplotlib_imshow(torchvision.utils.make_grid(images))

########################################################################
# We'll define the same model from before.

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
        x = self.fc3(x)
        return x


net = Net()

########################################################################
# We'll also define our `optimizer` and `criterion`

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 1. Tensorboard setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we'll set up Tensorboard, importing `tensorboard` from `torch.utils` and defining a 
# `SummaryWriter`, our key object for writing information to Tensorboard.
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs"
writer = SummaryWriter(log_dir='runs/cifar_experiment_1')

########################################################################
# This line alone creates a `runs/cifar_experiment_1` folder.
# 
# 2. Writing to Tensorboard
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now let's write an image to our Tensorboard.

# create grid of images
grid = torchvision.utils.make_grid(images)
    
# show images
matplotlib_imshow(grid)

# write to tensorboard
writer.add_image('four_cifar_images', grid)

########################################################################
# Now running
# ```
# tensorboard --logdir=runs
# ```
# from the command line and then navigating to `https://localhost:6006`
# should show the following.
# 
# .. figure:: /_static/img/tensorboard_first_view.png
#   :width: 600
# 
# Now you know how to use Tensorboard! This example doesn't give a good
# sense of what you might want to use Tensorboard *for*, however. One
# of Tensorboard's strengths is its ability to visualize complex model
# structures. Let's visualize the model we built.

writer.add_graph(net, images)
writer.close() # necessary to see Tensorboard refresh

########################################################################
# Now upon refreshing Tensorboard you should see a "Graphs" tab that
# looks like this:
#
# .. figure:: /_static/img/tensorboard_model_viz.png
#    :width: 600
#
# Go ahead and double click on "Net" to see it expand, seeing a more 
# detailed view of the structure of the get a deep dive  
# look into the structure of the model.
# 
# Next, we'll see how to use Tensorboard to track the model as it trains. 
# 
# 3. Tracking model training with Tensorboard
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the previous example, we simply *printed* the model's running loss 
# every 2000 iterations. Now, we'll instead log the running loss to 
# Tensorboard, along with a view into the predictions the model is 
# making via the `plot_classes_preds` function.

# Helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained 
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) 
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images 
    and labels from a batch. Uses the "images_to_probs" function above
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0}, {1:.1f}% (label: {2})".format(
            classes[preds[idx]], 
            probs[idx] * 100.0, 
            classes[labels[idx]]),
                     color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


########################################################################
# Finally, let's train the model using the same model training code from
# the prior tutorial, but writing results to Tensorboard instead of 
# printing to console

running_loss = 0.0
for epoch in range(2):  # loop over the dataset multiple times
    
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        if i % 2000 == 1999:    # every 2000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                              running_loss / 2000,
                              epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a 
            # random mini-batch             
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(net, inputs, labels),
                              global_step=epoch * len(trainloader) + i)
            running_loss = 0.0            
print('Finished Training')

########################################################################
# You can now look at the scalars tab to see the running loss plotted 
# over time, and the graphs tab to see the figure with the images, their
# probabilities, and whether the image was classified correctly, plotted
# as a function of the number of iterations (you may have to scroll
# down to see the figure in the images tab)
#  
# .. figure:: /_static/img/tensorboard_scalar_runs.png
#   :width: 600

########################################################################
# .. figure:: /_static/img/tensorboard_figure.png
#   :width: 600
