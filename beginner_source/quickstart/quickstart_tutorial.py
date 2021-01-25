"""
**Quickstart** >
`Tensors <tensor_tutorial.html>`_ > 
`Datasets & DataLoaders <dataquickstart_tutorial.html>`_ >
`Transforms <transforms_tutorial.html>`_ >
`Build Model <buildmodel_tutorial.html>`_ >
`Autograd <autograd_tutorial.html>`_ >
`Optimization <optimization_tutorial.html>`_ >
`Save & Load Model <saveloadrun_tutorial.html>`_

PyTorch Quickstart
===================

The basic machine learning concepts in any framework should include: Working with data, 
Creating models, Optimizing Parameters, Saving and Loading Models. In this PyTorch Quickstart we will
go through these concepts and how to apply them with PyTorch. The dataset we will be using is the 
FashionMNIST clothing images dataset that demonstrates these core steps applied to create ML Models. 

You will be introduced to the complete ML workflow using PyTorch with links to learn more at each step. 
Using this dataset we will be able to predict if the image is one of the following classes: T-shirt/top, 
Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot. Lets get started!


How to Use this Guide
-----------------
This guide is setup to cover machine learning concepts and how to apply them with PyTorch. This main page is a
highlevel intro to each step with the code examples to build the model. You have the option to jump 
into the concepts introduced in each section to get more details and explanations to better understand each concept 
and how to apply them with PyTorch. The topics are introduced in a sequenced order as listed below:

.. include:: /beginner_source/quickstart/qs_toc.txt

.. toctree::
   :hidden:

   /beginner/quickstart/tensor_tutorial
   /beginner/quickstart/dataquickstart_tutorial
   /beginner/quickstart/transforms_tutorial
   /beginner/quickstart/buildmodel_tutorial
   /beginner/quickstart/autograd_tutorial
   /beginner/quickstart/optimization_tutorial
   /beginner/quickstart/saveloadrun_tutorial


Running the Tutorial Code
------------------
The navigation above allows you to run the Jupyter Notebook on the cloud, download the Jupyter Notebook 
or download the python file to run locally.
If you want to run the code locally on your machine you will need some tools you 
may or may not have installed already.
Below are some good tool options for configuring local development or for more detailed instructions check out `get started locally <https://pytorch.org/get-started/locally/>`_.

- `Visual Studio Code <https://code.visualstudio.com/Download>`_ : You can open run python code in Visual Studio Code or open a Jupyter Notebook in VS Code.

- `Anaconda for Package Management <https://www.anaconda.com/products/individual>`_ : You will need to install the packages using either ``pip`` or ``conda`` to run the code locally.

Working with data
-----------------
"""

######################################################################
#
# PyTorch has two basic data primitives: ``DataSet`` and ``DataLoader``.
# The `torchvision.datasets` ``DataSet`` object includes a ``transforms`` mechanism to
# modify data in-place. Below is an example of how to load that data from the PyTorch open datasets and transform the data to a normalized tensor. 
# This example is using the `torchvision.datasets` which is a subclass from the primitive `torch.utils.data.Dataset`. Note that the primitive dataset doesnt have the built in transforms param like the built in dataset in `torchvision.datasets.`
# For more details on the concepts introduced here check out `Tensors <tensor_tutorial.html>`_,
# `DataSets & DataLoaders <dataquickstart_tutorial.html>`_,
# and `Transforms <transforms_tutorial.html>`_. 
#

import torch
import torch.nn as nn
import torch.onnx as onnx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Download training data from open datasets.
training_data = datasets.FashionMNIST('data', train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    target_transform=transforms.Compose([
        transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    ])
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST('data', train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    target_transform=transforms.Compose([
        transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    ])

)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0, pin_memory=True)

################################
# Creating Models
# ---------------
# 
# There are two ways of creating models: in-line or as a class.
# The most common way to define a neural network is to use a class inherited 
# from `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html)>`_.
# It provides great parameter management across all nested submodules, which gives us more 
# flexibility, because we can construct layers of any complexity, including the ones with shared weights. 
# For more details checkout `building the model <buildmodel_tutorial.html>`_.

# Get cpu or gpu device for training.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 512)
        self.layer2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return F.softmax(x, dim=1)
model = NeuralNetwork().to(device)
    
print(model)

######################################################################
# Optimizing Parameters and Training
# ---------------------
# 
# Optimizing model parameters requires a loss function, optimizer,
# and the optimization loop. 
# Training a model is essentially an optimization process similar to the one we described in the
# `Autograd <autograd_tutorial.html>`_ section. We run the optimization process on the whole dataset 
# several times, and each run is refered to as an **epoch**. During each run, we present data 
# in **minibatches**, and for each minibatch compute gradients and correct parameters of the model 
# according to back propagation algorithm. Read more about the `Optimization Loop <optimization_tutorial.html>`_.
#

# Cost function used to determine best parameters.
cost = torch.nn.BCELoss()

# This is used to create optimal parameters.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create the training function.
def train(dataloader, model, loss, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()
    
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


# Create the validation/test function

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y.argmax(1)).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')


# Call the train and test function in a training loop with the number of epochs indicated.

epochs = 5

for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_dataloader, model, cost, optimizer)
    test(test_dataloader, model)
print('Done!')

######################################################################
# Saving Models
# -------------
# 
# PyTorch has different ways you can save your model. One way is to serialize the internal model state to a file. Another would be to use the built-in `ONNX <https://github.com/onnx/tutorials>`_ support.
# 

torch.save(model.state_dict(), 'model.pth')
print('Saved PyTorch Model to model.pth')

# Save to ONNX, create dummy variable to traverse graph

x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
onnx.export(model, x, 'model.onnx')
print('Saved onnx model to model.onnx')

######################################################################
# Loading Models
# ----------------------------
# 
# Once a model has been serialized the process for loading the
# parameters includes re-creating the model shape and then loading
# the state dictionary. Once loaded the model can be used for either
# retraining or inference purposes (in this example it is used for
# inference). Check out more details on `saving, loading and running models with PyTorch <saveloadrun_tutorial.html>`_
#

loaded_model = NeuralNetwork()

loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()

# inference
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = loaded_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y.argmax(0)]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


#############################################################
#
# Looking for more resources? Check out the other tutorials on the `tutorials home page <https://pytorch.org/tutorials/>`_.
#

##################################################################
#
# Authors: `Seth Juarez <https://github.com/sethjuarez/>`_, `Cassie Breviu <https://github.com/cassieview/>`_, `Dmitry Soshnikov <https://soshnikov.com/>`_, `Ari Bornstein <https://github.com/aribornstein/>`_, `Suraj Subramanian <https://github.com/suraj813>`_
