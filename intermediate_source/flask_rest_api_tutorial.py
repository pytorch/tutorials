# -*- coding: utf-8 -*-
"""
Deploying PyTorch and Building a REST API using Flask
=====================================================
**Author**: `Avinash Sajjanshetty <https://avi.im>`_


In this tutorial, we will deploy a PyTorch model using Flask and expose a
REST API for model inference. In particular, we will deploy a pretrained
DenseNet 121 model which detects the image.

.. tip:: All the code used here is released under MIT license and is available on `Github <https://github.com/avinassh/pytorch-flask-api>`_.

"""


######################################################################
# API Definition
# --------------
#
# We will first define our API endpoints, the request and response types. Our
# API endpoint will be at ``/predict`` which takes HTTP POST requests with a
# ``file`` parameter which contains the image. The response will be of JSON
# response containing the prediction:
#
# ::
#
#     {"class_id": "n02124075", "class_name": "Egyptian_cat"}
#
#

######################################################################
# Dependencies
# ------------
#
# Install the required dependenices by running the following command:
#
# ::
#
#     $ pip install Flask==1.0.3 torchvision-0.3.0


######################################################################
# Simple Web Server
# -----------------
#
# Following is a simple webserver, taken from Flask's documentaion


from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'

###############################################################################
# Save the above snippet in a file called ``app.py`` and you can now run a
# Flask development server by typing:
#
# ::
#
#     $ FLASK_ENV=development FLASK_APP=app.py flask run

###############################################################################
# When you visit ``http://localhost:5000/`` in your web browser, you will be
# greeted with ``Hello World!`` text

###############################################################################
# We will make slight changes to the above snippet, so that it suits our API
# definition. First, we will rename the method to ``predict``. We will update
# the endpoint path to ``/predict`` and make it so that it also accepts POST
# requests:


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return 'Hello World!'

###############################################################################
# We will also change the response type, so that it returns a JSON response
# containing ImageNet class id and name. The updated ``app.py`` file will
# be now:

from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})


######################################################################
# Inference
# -----------------
#
# In the next sections we will focus on writing the inference code. This will
# involve two parts, one where we prepare the image so that it can be fed
# to DenseNet and next, we will write the code to get the actual prediction
# from the model.
#
# Preparing the image
# ~~~~~~~~~~~~~~~~~~~
#
# DenseNet model requires the image to be of 3 channel RGB image of size
# 224 x 224. We will also normalise the image tensor with the required mean
# and standard deviation values. You can read more about it
# `here <https://pytorch.org/docs/stable/torchvision/models.html>`_.
#
# We will use ``transforms`` from ``torchvision`` library and build a
# transform pipeline, which transforms our images as required. You
# can read more about transforms `here <https://pytorch.org/docs/stable/torchvision/transforms.html>`_.

import io

import torchvision.transforms as transforms
from PIL import Image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


######################################################################
# Above method takes image data in bytes, applies the series of transforms
# and returns a tensor. To test the above method, read an image file in
# bytes mode and see if you get a tensor back:

with open('sample_file.jpeg', 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)

######################################################################
# Prediction
# ~~~~~~~~~~~~~~~~~~~
#
# Now will use a pretrained DenseNet 121 model to predict the image class. We
# will use one from ``torchvision`` library, load the model and get an
# inference. Instead of using a pretrained model, you can also load your own
# model, to learn more about it check this
# :doc:`tutorial </beginner/saving_loading_models>`.

from torchvision import models

# make sure to pass `pretrained` as `True` to use the pretrained weights
model = models.densenet121(pretrained=True)
# since we are using our model only for inference, switch to `eval` mode
model.eval()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat


######################################################################
# The tensor ``y_hat`` will contain the index of the predicted class id.
# However we need a human readable class name. For that we need a class id
# to name mapping. Download
# `this file <https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json>`_ 
# and place it in current directory as file ``imagenet_class_index.json``. We
# will load this JSON file and get the class name of the predicted index.

import json

imagenet_class_index = json.load(open('imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


######################################################################
# Before using ``imagenet_class_index`` dictionary, first we will convert
# tensor value to a string value, since the keys in the
# ``imagenet_class_index`` dictionary are strings.
# We will test our above method:


with open('sample_file.jpeg', 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))

######################################################################
# and you should get a response like this

['n02124075', 'Egyptian_cat']

######################################################################
# The first item in array is ImageNet class id and second item is the human
# readable name.
#
# .. Note ::
#    Did you notice why ``model`` variable is not part of ``get_prediction``
#    method? Or why it is a global variable? Loading a model can be an
#    expensive operation, memory, CPU wise. If we loaded the model in the
#    ``get_prediction`` method, then it will get unnecessarily loaded every
#    time the method is called. Since, we are building a web server, there
#    could be thousands of requests per second and we cannot keep loading the
#    model everytime. So, we keep the model loaded in memory once. In
#    production systems it is a good idea to load the model first and then
#    start serving the requests.

######################################################################
# Integrating the model in our API Server
# ---------------------------------------
#
# In this final part we will add our model to our Flask API server. Since
# our API server supposed to take an image file, we will update our ``predict``
# method to read files from the requests. We will also restrict this method
# to POST requests for the sake of simplicity

from flask import request


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

######################################################################
# The ``app.py`` file is now complete. Following is the full version:
#

import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()

######################################################################
# Let's test our web server! Run:
#
# ::
#
#     $ FLASK_ENV=development FLASK_APP=app.py flask run

#######################################################################
# We can use a command line tool like curl or Postman to send requests to
# this webserver
#
# ::
#
#     $ curl -X POST -F file=@cat_pic.jpeg http://localhost:5000/predict
#
# And you will get a response in the form
#
# ::
#
#     {"class_id": "n02124075", "class_name": "Egyptian_cat"}
#
#

######################################################################
# Next steps
# --------------
#
# The server we wrote is quite triavial and is not ready for production. So,
# here are some things you can do to make it better
#
# - The endpoint ``/predict`` assumes that always there will be a image file
#   in the request. This may not hold true for all requests. Our user may
#   send image with a different parameter or send no images at all.
#
# - User may send non-image type files too. Since we are not handling errors,
#   this will break our server. Loading the image within an exception block
#   is a good idea.
#
# - There is a possibility that the image sent by user may not get recoginzed
#   by our model. Above code should be updated to handle this case.
#
# - We run the Flask server in a development mode, which not suitable for
#   deploying in production. You may check `this tutorial <http://flask.pocoo.org/docs/1.0/tutorial/deploy/>`_ 
#   for deploying a Flask server in production.
#
# - You can also add a welcome page with a form which takes the image and
#   displays the prediction. Check the `demo <https://pytorch-imagenet.herokuapp.com/>`_ 
#   of the same or its `source code <https://github.com/avinassh/pytorch-flask-api-heroku>`_.
