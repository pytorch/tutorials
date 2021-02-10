"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ || 
`Tensors <tensorqs_tutorial.html>`_ || 
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
**Save & Load Model**

Save and Load the Model
============================

In this section we will look at how to persist model state with saving, loading and running model predictions.
"""

import torch
import torch.onnx as onnx
import torchvision.models as models


#######################################################################
# Saving and Loading Model Weights
# --------------------------------
# PyTorch models store the learned parameters in an internal
# state dictionary, called ``state_dict``. These can be persisted via the ``torch.save``
# method:

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

##########################
# To load model weights, you need to create an instance of the same model first, and then load the parameters 
# using ``load_state_dict()`` method.

model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

###########################
# .. note:: be sure to call ``model.eval()`` method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.

#######################################################################
# Saving and Loading Models with Shapes
# -------------------------------------
# When loading model weights, we needed to instantiate the model class first, because the class 
# defines the structure of a network. We might want to save the structure of this class together with 
# the model, in which case we can pass ``model`` (and not ``model.state_dict()``) to the saving function:

torch.save(model, 'model.pth')

########################
# We can then load the model like this:

model = torch.load('model.pth')

########################
# .. note:: This approach uses Python `pickle <https://docs.python.org/3/library/pickle.html>`_ module when serializing the model, thus it relies on the actual class definition to be available when loading the model.

#######################################################################
# Exporting Model to ONNX
# -----------------------
# PyTorch also has native ONNX export support. Given the dynamic nature of the
# PyTorch execution graph, however, the export process must
# traverse the execution graph to produce a persisted ONNX model. For this reason, a
# test variable of the appropriate size should be passed in to the
# export routine (in our case, we will create a dummy zero tensor of the correct size):

input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'model.onnx')

###########################
# There are a lot of things you can do with ONNX model, including running inference on different platforms 
# and in different programming languages. For more details, we recommend 
# visiting `ONNX tutorial <https://github.com/onnx/tutorials>`_.
#
# Congratulations! You have completed the PyTorch beginner tutorial! Try 
# `revisting the first page <quickstart_tutorial.html>`_ to see the tutorial in its entirety
# again. We hope this tutorial has helped you get started with deep learning on PyTorch. 
# Good luck!
#

