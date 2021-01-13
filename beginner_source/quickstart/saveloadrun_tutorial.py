"""
7. Save, Load and Use the Model
============================

.. include:: /beginner_source/quickstart/qs_toc.txt

In this section we will look at how to save, load and use persisted model state
to run predictions.
"""

import torch
import torch.nn as nn
import torch.onnx as onnx

#######################################################################
# Save the Model
# --------------
# PyTorch stores the learned parameters in the model's internal
# state dictionary. These are persisted via the `torch.save`
# method:

# saving PyTorch Model Dictionary
torch.save(model.state_dict(), 'model.pth')

#######################################################################
# PyTorch also has native ONNX export support. Given the dynamic nature of the
# PyTorch execution graph however, the export process must
# traverse the execution graph to produce a persisted onnx model. As such, a
# test variable of the appropriate size should be passed in to the
# export routine:

# create test variable to traverse graph
x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
onnx.export(model, x, 'model.onnx')


#######################################################################
# Load the Model
# --------------
# Loading a persisted PyTorch model consists of two primary steps:
#
# 1. Recreating the appropriate model shape, and
# 2. Rehydrating the parameters into the newly created model's state dictionary
#
# These two steps are illustrated here:

# recreate model
loaded_model = NeuralNetwork()
# hydrate state dictionary
loaded_model.load_state_dict(torch.load('model.pth'))


######################################################################
# Use the Model
# -------------
# Once the model is loaded it can be used for both training as well
# as inference. The model's ``eval()`` method is called in this case
# to indicate the model will be used for inference. This method
# only affects internal modules like Dropout and BatchNorm which
# are not necessary for inference. Using ``torch.no_grad()`` turns off
# `automatic differentiation  <autograd_tutorial.html>`_ since it
# is also unnecessary:

loaded_model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = loaded_model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y.argmax(0)]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


