"""
`Quickstart <quickstart_tutorial.html>`_ >
`Tensors <tensor_tutorial.html>`_ > 
`DataSets & DataLoaders <dataquickstart_tutorial.html>`_ >
`Transforms <transforms_tutorial.html>`_ >
`Build Model <buildmodel_tutorial.html>`_ >
`Autograd <autograd_tutorial.html>`_ >
`Optimization <optimization_tutorial.html>`_ >
**Save & Load Model**

Save and Load the Model
============================

In this section we will look at how to presist model state with saving, loading and running model predictions.
"""

import torch
import torch.onnx as onnx

#######################################################################
# Pre-trained Models
# ------------------
# 
# Many tasks, such as object classification in computer vision, rely on some pre-trained models. 
# While you can find some pre-trained models for common tasks online, PyTorch already includes the most 
# common model architectures. For example, to initialize a VGG-16 model for image classification, 
# we can use the following code:

import torchvision.models as models
model = models.vgg16(pretrained=True)

#############################
# To use this network on the input batch of images ``imgs``, we can just call it as an ordinary function:
#
# .. code-block:: Python
#
#    res = model(imgs)
#
# Let us see how a picture of a cat can be classified using this VGG-16 model. Once we load a 
# picture from the internet, we need to apply a series of transformations to it, to turn it into a 
# tensor of the appropriate size:
#
# * Resize image to 224x224 pixels
# * Convert it to tensor
# * Apply normalization with a given mean and standard deviation  
#
# Also, we need to turn a single tensor into a batch by adding one more dimension with ``unsqueeze()`` call.
# After doing the inference, we obtain a tensor of probabilities for each of the classes, and we get the index of the 
# most probable class by calling ``.argmax()``. 

import matplotlib.pyplot as plt
from PIL import Image
import requests
import torchvision.transforms as T

url = "https://upload.wikimedia.org/wikipedia/commons/6/66/An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg"
im = Image.open(requests.get(url, stream=True).raw)
plt.imshow(im)

transform = T.Compose([T.Resize(224), T.ToTensor(), 
                       T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
input_image = transform(im).unsqueeze(0)
with torch.no_grad():
    res = model(input_image).argmax().item()
    print(res)

######################
# The result obtained is a number of imagenet predicted class, in this case, *tiger cat*.
#
# .. note:: When running model inference, it is recommended to wrap the code into ``torch.no_grad()``, because  `automatic differentiation  <autograd_tutorial.html>`_ is unnecessary.

#######################################################################
# Saving and Loading Model Weights
# --------------------------------
# PyTorch stores the learned parameters in the model's internal
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
# export routine:

onnx.export(model, input_image, 'model.onnx')

###########################
# There are a lot of things you can do with ONNX model, including running inference on different platforms 
# and in different programming languages. For more details, we recommend 
# visiting `ONNX tutorial <https://github.com/onnx/tutorials>`_.
#
# Congratulations! You have completed the PyTorch beginner tutorial! You can 
# now `return to the first page <quickstart_tutorial.html>`_ and go over the sample code 
# again and we hope you have a better understanding of how to do deep learning with PyTorch. 
# Good luck on your deep learning journey!
