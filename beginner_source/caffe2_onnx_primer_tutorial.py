# -*- coding: utf-8 -*-
"""
Caffe2 ONNX Primer
==================
**Author**: `Nathan Inkawhich <https://github.com/inkawhich>`_

This tutorial is a brief look at how to use Caffe2 and
`ONNX <http://onnx.ai/about>`_ together. More specifically, we will
show how to export a model from Caffe2 to ONNX and how to import a model
from ONNX into Caffe2. Hopefully, the motivation is clear but this
tutorial shows how to use the very fast and efficient Caffe2 framework
with the flexibility enabling ONNX framework. One important fact to keep
in mind is that ONNX is designed to enable deployment and *inference* in
frameworks other than where the model was trained. Currently, there is
no streamlined way to finetune ONNX models. The workflow for this
document is as follows:

-  Run prediction with a Caffe2 model and collect initial prediction
-  Export the Caffe2 model to ONNX format
-  Import the saved ONNX model back into Caffe2
-  Run prediction on imported model and verify results

Let's get started with some imports.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import operator
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, models
import onnx
import caffe2.python.onnx.frontend # Required for Caffe2->ONNX export
import caffe2.python.onnx.backend # Required for ONNX->Caffe2 import


######################################################################
# Inputs
# ------
#
# Now we will specify the inputs. The *MODELS\_DIR* is where the
# downloaded Caffe2 models are saved, the *MODEL\_NAME* is the name of the
# model we want to use, and *SIZE* is the size of image the model expects.
# For more information about downloading a pretrained Caffe2 model, see
# the `Loading Pretrained Models
# Tutorial <https://github.com/caffe2/tutorials/blob/master/Loading_Pretrained_Models.ipynb>`__.
#

# User Inputs
MODELS_DIR = "../models"
MODEL_NAME = "squeezenet" # e.g. [squeezenet, bvlc_alexnet, bvlc_googlenet, bvlc_reference_caffenet]
SIZE = 224

# Construct path strings from inputs
INIT_NET = "{}/{}/init_net.pb".format(MODELS_DIR, MODEL_NAME)
PREDICT_NET = "{}/{}/predict_net.pb".format(MODELS_DIR, MODEL_NAME)
ONNX_MODEL = "{}/{}/my_model.onnx".format(MODELS_DIR, MODEL_NAME) # we will create this


######################################################################
# Load Caffe2 Model
# -----------------
#
# Before we perform the export we will first load the pretrained init and
# predict nets, then create a *Predictor*. Next, we will create a random
# input to get a baseline result for comparision later. Take note of the
# predicted label and confidence.
#

# Generate random NCHW input to run model
#   This is a placeholder for any real image that is processed and
#   put in NCHW order.
image = np.random.rand(1,3,SIZE,SIZE).astype(np.float32)
print("Input Shape: ",image.shape)

# Prepare the nets
predict_net = caffe2_pb2.NetDef()
with open(PREDICT_NET, 'rb') as f:
    predict_net.ParseFromString(f.read())
init_net = caffe2_pb2.NetDef()
with open(INIT_NET, 'rb') as f:
    init_net.ParseFromString(f.read())

# Initialize the predictor from the nets
p = workspace.Predictor(init_net, predict_net)

#### Run the sample data

# Run the net and return prediction
results = p.run({'data': image})
results = np.asarray(results)
print("Results Shape: ", results.shape)

# Quick way to get the top-1 prediction result
curr_pred, curr_conf = max(enumerate(np.squeeze(results)), key=operator.itemgetter(1))
print("Top-1 Prediction: {} @ {}".format(curr_pred, curr_conf))



######################################################################
# Caffe2 :math:`\rightarrow` ONNX Export
# --------------------------------------
#
# Finally, we have reached the interesting stuff. It is not hard to
# imagine why one may want to export a Caffe2 model to ONNX. Maybe you
# have a cool idea for an iPhone app and want to use a model trained in
# Caffe2 with CoreML as part of the app. Or, maybe you have a system built
# in Tensorflow but want to test out a model from the Caffe2 Model Zoo.
# ONNX enables this interoperability by allowing models to be imported and
# exported into different frameworks (for inference!).
#
# The code below shows how to **export** a model trained in Caffe2 to ONNX
# format. Once in ONNX format, the model can be imported into any other
# compatible framework to be used for *inference*. From the Caffe2 side,
# we only need the previously loaded *init\_net* and *predict\_net*
# *caffe2\_pb2.NetDef* objects.
#
# There are only a few steps to export once the nets are loaded. First, we
# must declare (via Python dictionary) the type and shape of inputs and
# outputs of the model. This information is not explicitly specified in
# the Caffe2 model architecture but is required by ONNX. Next, we must
# make sure the model has a name, otherwise the internal model checks in
# the ONNX converter will fail. Then, all thats left to do is create the
# ONNX model, check it, and save it.
#

# We need to provide type and shape of the model inputs
data_type = onnx.TensorProto.FLOAT
data_shape = (1, 3, 224, 224)
value_info = {
    'data': (data_type, data_shape)
}

# Make sure the net has a name. Otherwise, the checker will fail.
if predict_net.name == "":
    predict_net.name = "ModelNameHere"

# Create the ONNX model
onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
    predict_net,
    init_net,
    value_info,
)

# Check the ONNX model. Exception will be thrown if there is a problem here.
onnx.checker.check_model(onnx_model)

# Save the ONNX model
onnx.save(onnx_model, ONNX_MODEL)


######################################################################
# ONNX :math:`\rightarrow` Caffe2 Import
# --------------------------------------
#
# Now suppose someone has trained Alexnet2.0 which gets 99.9% top-1 test
# accuracy on ImageNet ... *gasp* ... in Tensorflow. As a Caffe2 user, all
# we have to do is convince them to convert the model to ONNX format, then
# we can import it and use it. Since we are running out of time in this
# 5-minute primer, here we will only show how to import the model we just
# exported back into Caffe2. The import happens in a single load command
# (``onnx.load``), then we can start feeding the model data in just one
# more command (``run_model``). Also, note that the predictions from this
# imported model and the original model are the exact same, indicating
# nothing was lost in the export/import process.
#

# Load the ONNX model
model = onnx.load(ONNX_MODEL)

# Run the ONNX model with Caffe2
outputs = caffe2.python.onnx.backend.run_model(model, [image])
print("Output Shape: ", np.array(outputs).shape)

# Get model prediction
curr_pred, curr_conf = max(enumerate(np.squeeze(results)), key=operator.itemgetter(1))
print("Top-1 Prediction: {} @ {}".format(curr_pred, curr_conf))



######################################################################
# Hopefully it is clear that the caffe2-onnx interface for both importing
# and exporting is relatively simple. For more information about ONNX and
# to see more tutorials on using ONNX with different frameworks see the
# `ONNX Tutorials <https://github.com/onnx/tutorials>`__. Also, although
# importing and exporting with Caffe2 is supported, and exporting a model
# from PyTorch to ONNX is supported, *importing* an ONNX model into
# PyTorch is *NOT*, but is coming soon!
#
# Here are some more cool ONNX resources for the curious reader:
#
# -  `ONNX Python API
#    Overview <https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md>`__
# -  `ONNX Model Zoo <https://github.com/onnx/models>`__
# -  `ONNX
#    Operators <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`__
# -  `ONNX Tutorials <https://github.com/onnx/tutorials>`__
#
