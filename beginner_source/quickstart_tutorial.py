"""
PyTorch Quickstart
===================

The basic machine learning concepts in any framework should include: Working with data, Creating models, Optimizing Parameters, Saving and Loading Models

"""

import torch
import torch.nn as nn
import torch.onnx as onnx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

######################################################################
# Working with data
# -----------------
# 
# PyTorch has two basic data primitives: ``DataSet`` and ``DataLoader``.
# These ``DataSet`` objects include a ``transforms`` mechanism to
# modify data in-place. 

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

######################################################################
# DataLoader

# batch size
batch_size = 64

# loader
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0, pin_memory=True)


######################################################################
# More details `DataSet and DataLoader <quickstart/data_quickstart_tutorial.html>`_
# More details `Tensors <quickstart/tensor_quickstart_tutorial.html>`_
# More details  `Transformations <transforms_tutorial.html>`_
#
#
# Creating Models
# ---------------
# 
# There are two ways of creating models: in-line or as a class. This
# quickstart will consider an in-line definition.

# where to run
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# model
model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(classes)),
        nn.Softmax(dim=1)
    ).to(device)
    
print(model)

######################################################################
# More details `on building the model <quickstart/build_model_tutorial.html>`_
#
# Optimizing Parameters
# ---------------------
# 
# Optimizing model parameters requires a loss function, and optimizer,
# and the optimization loop.

# cost function used to determine best parameters
cost = torch.nn.BCELoss()

# used to create optimal parameters
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

######################################################################
# training function
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

######################################################################
# validation/test function
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
# training loop
epochs = 5

for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_dataloader, model, cost, optimizer)
    test(test_dataloader, model)
print('Done!')

######################################################################
# More details `optimization and training loops <quickstart/optimization_tutorial.html>`_
# More deatils `AutoGrad <autograd_quickstart_tutorial.html>`_
#
# Saving Models
# -------------
# 
# PyTorch has can serialize the internal model state to a file. It also
# has built-in ONNX support.

# saving PyTorch Model Dictionary
torch.save(model.state_dict(), 'model.pth')
print('Saved PyTorch Model to model.pth')

# create dummy variable to traverse graph
x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
onnx.export(model, x, 'model.onnx')
print('Saved onnx model to model.onnx')

######################################################################
# More details `Saving loading and running <quickstart/save_load_run_tutorial.html>`_
#
# Loading Models
# ----------------------------
# 
# Once a model has been serialized the process for loading the
# parameters includes re-creating the model shape and then loading
# the state dictionary. Once loaded the model can be used for either
# retraining or inference purposes (in this example it is used for
# inference)

loaded_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(classes)),
        nn.Softmax(dim=1)
    )

loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()

# inference
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = loaded_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y.argmax(0)]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

