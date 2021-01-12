"""
PyTorch Quickstart
===================


The basic machine learning concepts in any framework should include: Working with data, Creating models, Optimizing Parameters, Saving and Loading Models. In this PyTorch Quickstart we will go through these concepts and how to apply them with PyTorch. That dataset we will be using is the FashionMNIST clothing images dataset that demonstrates these core steps applied to create ML Models. You will be introduced to the complete ML workflow using PyTorch with links to learn more at each step. Using this dataset we will be able to predict if the image is one of the following classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot. Lets get started!

Working with data
-----------------
"""

######################################################################
#
# PyTorch has two basic data primitives: ``DataSet`` and ``DataLoader``.
# The `torchvision.datasets` ``DataSet`` object includes a ``transforms`` mechanism to
# modify data in-place. Below is an example of how to load that data from the PyTorch open datasets and transform the data to a normalized tensor. 
# This example is using the `torchvision.datasets` which is a subclass from the primitive `torch.utils.data.Dataset`. Note that the primitive dataset doesnt have the built in transforms param like the built in dataset in `torchvision.datasets.`
# 
# To see more examples and details of how to work with Tensors, Datasets, DataLoaders and Transforms in PyTorch with this example checkout these resources:
#  
#  - `Tensors <quickstart/tensor_tutorial.html>`_
#  - `DataSet and DataLoader <quickstart/dataquickstart_tutorial.html>`_
#  - `Transforms <quickstart/transforms_tutorial.html>`_

import torch
import torch.nn as nn
import torch.onnx as onnx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

training_data = datasets.FashionMNIST('data', train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    target_transform=transforms.Compose([
        transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    ])
)

test_data = datasets.FashionMNIST('data', train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    target_transform=transforms.Compose([
        transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    ])
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0, pin_memory=True)

################################
# Creating Models
# ---------------
# 
# There are two ways of creating models: in-line or as a class. This
# quickstart will consider a class definition. For more examples checkout `building the model <quickstart/buildmodel_tutorial.html>`_.

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
# Optimizing Parameters
# ---------------------
# 
# Optimizing model parameters requires a loss function, optimizer,
# and the optimization loop.
#
# To see more examples and details of how to work with Optimization and Training loops in Pytoch with this example checkout these resources:
#  - `Optimization and training loops <quickstart/optimization_tutorial.html>`_
#  - `Automatic differentiation and AutoGrad <quickstart/autograd_tutorial.html>`_
#

# cost function used to determine best parameters
cost = torch.nn.BCELoss()

# used to create optimal parameters
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create the training function

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

######################################################################
# Training Models
# -------------
# 
# Call the train and test function in a training loop with the number of epochs indicated
#

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
# Saving PyTorch Model Dictionary

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
# inference). Check out more details on `saving, loading and running models with Pytorch <quickstart/saveloadrun_tutorial.html>`_
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

##################################################################
# PyTorch Quickstart Topics
# ----------------------------------------
# | `Tensors <quickstart/tensor_tutorial.html>`_
# | `DataSets and DataLoaders <quickstart/dataquickstart_tutorial.html>`_
# | `Transforms <quickstart/transforms_tutorial.html>`_
# | `Build Model <quickstart/buildmodel_tutorial.html>`_
# | `Optimization Loop <quickstart/optimization_tutorial.html>`_
# | `AutoGrad <quickstart/autograd_tutorial.html>`_
# | `Save, Load and Run Model <saveloadrun_tutorial.html>`_
#
# *Authors: Seth Juarez, Ari Bornstein, Cassie Breviu, Dmitry Soshnikov*

