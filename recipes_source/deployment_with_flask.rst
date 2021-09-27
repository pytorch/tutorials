Deploying with Flask
====================

In this recipe, you will learn:

-  How to wrap your trained PyTorch model in a Flask container to expose
   it via a web API
-  How to translate incoming web requests into PyTorch tensors for your
   model
-  How to package your model’s output for an HTTP response

Requirements
------------

You will need a Python 3 environment with the following packages (and
their dependencies) installed:

-  PyTorch 1.5
-  TorchVision 0.6.0
-  Flask 1.1

Optionally, to get some of the supporting files, you'll need git.

The instructions for installing PyTorch and TorchVision are available at
`pytorch.org`_. Instructions for installing Flask are available on `the
Flask site`_.

What is Flask?
--------------

Flask is a lightweight web server written in Python. It provides a
convenient way for you to quickly set up a web API for predictions from
your trained PyTorch model, either for direct use, or as a web service
within a larger system.

Setup and Supporting Files
--------------------------

We're going to create a web service that takes in images, and maps them
to one of the 1000 classes of the ImageNet dataset. To do this, you'll
need an image file for testing. Optionally, you can also get a file that
will map the class index output by the model to a human-readable class
name.

Option 1: To Get Both Files Quickly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can pull both of the supporting files quickly by checking out the
TorchServe repository and copying them to your working folder. *(NB:
There is no dependency on TorchServe for this tutorial - it's just a
quick way to get the files.)* Issue the following commands from your
shell prompt:

::

   git clone https://github.com/pytorch/serve
   cp serve/examples/image_classifier/kitten.jpg .
   cp serve/examples/image_classifier/index_to_name.json .

And you've got them!

Option 2: Bring Your Own Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``index_to_name.json`` file is optional in the Flask service below.
You can test your service with your own image - just make sure it's a
3-color JPEG.

Building Your Flask Service
---------------------------

The full Python script for the Flask service is shown at the end of this
recipe; you can copy and paste that into your own ``app.py`` file. Below
we'll look at individual sections to make their functions clear.

Imports
~~~~~~~

::

   import torchvision.models as models
   import torchvision.transforms as transforms
   from PIL import Image
   from flask import Flask, jsonify, request

In order:

-  We'll be using a pre-trained DenseNet model from
   ``torchvision.models``
-  ``torchvision.transforms`` contains tools for manipulating your image
   data
-  Pillow (``PIL``) is what we'll use to load the image file initially
-  And of course we'll need classes from ``flask``

Pre-Processing
~~~~~~~~~~~~~~

::

   def transform_image(infile):
       input_transforms = [transforms.Resize(255),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406],
               [0.229, 0.224, 0.225])]
       my_transforms = transforms.Compose(input_transforms)
       image = Image.open(infile)
       timg = my_transforms(image)
       timg.unsqueeze_(0)
       return timg

The web request gave us an image file, but our model expects a PyTorch
tensor of shape (N, 3, 224, 224) where *N* is the number of items in the
input batch. (We will just have a batch size of 1.) The first thing we
do is compose a set of TorchVision transforms that resize and crop the
image, convert it to a tensor, then normalize the values in the tensor.
(For more information on this normalization, see the documentation for
``torchvision.models_``.)

After that, we open the file and apply the transforms. The transforms
return a tensor of shape (3, 224, 224) - the 3 color channels of a
224x224 image. Because we need to make this single image a batch, we use
the ``unsqueeze_(0)`` call to modify the tensor in place by adding a new
first dimension. The tensor contains the same data, but now has shape
(1, 3, 224, 224).

In general, even if you're not working with image data, you will need to
transform the input from your HTTP request into a tensor that PyTorch
can consume.

Inference
~~~~~~~~~

::

   def get_prediction(input_tensor):
       outputs = model.forward(input_tensor)
       _, y_hat = outputs.max(1)
       prediction = y_hat.item()
       return prediction

The inference itself is the simplest part: When we pass the input tensor
to them model, we get back a tensor of values that represent the model's
estimated likelihood that the image belongs to a particular class. The
``max()`` call finds the class with the maximum likelihood value, and
returns that value with the ImageNet class index. Finally, we extract
that class index from the tensor containing it with the ``item()`` call, and
return it.

Post-Processing
~~~~~~~~~~~~~~~

::

   def render_prediction(prediction_idx):
       stridx = str(prediction_idx)
       class_name = 'Unknown'
       if img_class_map is not None:
           if stridx in img_class_map is not None:
               class_name = img_class_map[stridx][1]

       return prediction_idx, class_name

The ``render_prediction()`` method maps the predicted class index to a
human-readable class label. It's typical, after getting the prediction
from your model, to perform post-processing to make the prediction ready
for either human consumption, or for another piece of software.

Running The Full Flask App
--------------------------

Paste the following into a file called ``app.py``:

::

   import io
   import json
   import os

   import torchvision.models as models
   import torchvision.transforms as transforms
   from PIL import Image
   from flask import Flask, jsonify, request


   app = Flask(__name__)
   model = models.densenet121(pretrained=True)               # Trained on 1000 classes from ImageNet
   model.eval()                                              # Turns off autograd 



   img_class_map = None
   mapping_file_path = 'index_to_name.json'                  # Human-readable names for Imagenet classes
   if os.path.isfile(mapping_file_path):
       with open (mapping_file_path) as f:
           img_class_map = json.load(f)



   # Transform input into the form our model expects
   def transform_image(infile):
       input_transforms = [transforms.Resize(255),           # We use multiple TorchVision transforms to ready the image
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
               [0.229, 0.224, 0.225])]
       my_transforms = transforms.Compose(input_transforms)
       image = Image.open(infile)                            # Open the image file
       timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
       timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
       return timg


   # Get a prediction
   def get_prediction(input_tensor):
       outputs = model.forward(input_tensor)                 # Get likelihoods for all ImageNet classes
       _, y_hat = outputs.max(1)                             # Extract the most likely class
       prediction = y_hat.item()                             # Extract the int value from the PyTorch tensor
       return prediction

   # Make the prediction human-readable
   def render_prediction(prediction_idx):
       stridx = str(prediction_idx)
       class_name = 'Unknown'
       if img_class_map is not None:
           if stridx in img_class_map is not None:
               class_name = img_class_map[stridx][1]

       return prediction_idx, class_name


   @app.route('/', methods=['GET'])
   def root():
       return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


   @app.route('/predict', methods=['POST'])
   def predict():
       if request.method == 'POST':
           file = request.files['file']
           if file is not None:
               input_tensor = transform_image(file)
               prediction_idx = get_prediction(input_tensor)
               class_id, class_name = render_prediction(prediction_idx)
               return jsonify({'class_id': class_id, 'class_name': class_name})


   if __name__ == '__main__':
       app.run()

To start the server from your shell prompt, issue the following command:

::

   FLASK_APP=app.py flask run

By default, your Flask server is listening on port 5000. Once the server
is running, open another terminal window, and test your new inference
server:

::

   curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/predict -F "file=@kitten.jpg"

If everything is set up correctly, you should recevie a response similar
to the following:

::

   {"class_id":285,"class_name":"Egyptian_cat"}

Important Resources
-------------------

-  `pytorch.org`_ for installation instructions, and more documentation
   and tutorials
-  The `Flask site`_ has a `Quick Start guide`_ that goes into more
   detail on setting up a simple Flask service

.. _pytorch.org: https://pytorch.org
.. _Flask site: https://flask.palletsprojects.com/en/1.1.x/
.. _Quick Start guide: https://flask.palletsprojects.com/en/1.1.x/quickstart/
.. _torchvision.models: https://pytorch.org/vision/stable/models.html
.. _the Flask site: https://flask.palletsprojects.com/en/1.1.x/installation/
