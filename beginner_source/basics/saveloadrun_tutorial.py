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
import torchvision.models as models


#######################################################################
# Saving and Loading Model Weights
# --------------------------------
# PyTorch models store the learned parameters in an internal
# state dictionary, called ``state_dict``. These can be persisted via the ``torch.save``
# method:

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

##########################
# To load model weights, you need to create an instance of the same model first, and then load the parameters
# using ``load_state_dict()`` method.
#
# In the code below, we set ``weights_only=True`` to limit the
# functions executed during unpickling to only those necessary for
# loading weights. Using ``weights_only=True`` is considered
# a best practice when loading weights.

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
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
# We can then load the model as demonstrated below.
#
# As described in `Saving and loading torch.nn.Modules <pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules>`__,
# saving ``state_dict``s is considered the best practice. However,
# below we use ``weights_only=False`` because this involves loading the
# model, which is a legacy use case for ``torch.save``.

model = torch.load('model.pth', weights_only=False),

########################
# .. note:: This approach uses Python `pickle <https://docs.python.org/3/library/pickle.html>`_ module when serializing the model, thus it relies on the actual class definition to be available when loading the model.

#######################
# Related Tutorials
# -----------------
# - `Saving and Loading a General Checkpoint in PyTorch <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>`_
# - `Tips for loading an nn.Module from a checkpoint <https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html?highlight=loading%20nn%20module%20from%20checkpoint>`_
