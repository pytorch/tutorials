"""
Save, Load and Use the Model
===================

We have trained the model! Now lets take a look at how to save, load and use the model created.

Full Section Example:
"""

import os
import torch
import torch.nn as nn
import torch.onnx as onnx

# create dummy variable to traverse graph
x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
onnx.export(model, x, 'model.onnx')
print('Saved onnx model to model.onnx')

# saving PyTorch Model Dictionary
torch.save(model.state_dict(), 'model.pth')
print('Saved PyTorch Model to model.pth')

draw_clothes(test_data)

#rehydrate model
loaded_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, len(classes)),
        nn.Softmax(dim=1)
    )

#load graph
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()

x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = loaded_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y.argmax(0)]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


######################################################
# Save the Model
# -----------------------
# Example:

# create dummy variable to traverse graph
x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
onnx.export(model, x, 'model.onnx')
print('Saved onnx model to model.onnx')

# saving PyTorch Model Dictionary
torch.save(model.state_dict(), 'model.pth')
print('Saved PyTorch Model to model.pth')


#######################################################################
# Load the Model
# ---------------------------
# Example:

draw_clothes(test_data)

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

######################################################################
# Test the Model
# ----------------------------------
# Example:

x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = loaded_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y.argmax(0)]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

##################################################################
# Pytorch Quickstart Topics
# ----------------------------------------
# | `Tensors <tensor_tutorial.html>`_
# | `DataSets and DataLoaders <data_quickstart_tutorial.html>`_
# | `Transforms <transforms_tutorial.html>`_
# | `Build Model <build_model_tutorial.html>`_
# | `Optimization Loop <optimization_tutorial.html>`_
# | `AutoGrad <autograd_quickstart_tutorial.html>`_
# | `Save, Load and Run Model <save_load_run_tutorial.html>`_

