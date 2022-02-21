
ONNX Live Tutorial
==================

This tutorial will show you to convert a neural style transfer model that has been exported from PyTorch into the Apple CoreML format using ONNX. This will allow you to easily run deep learning models on Apple devices and, in this case, live stream from the camera. 

What is ONNX?
-------------

ONNX (Open Neural Network Exchange) is an open format to represent deep learning models. With ONNX, AI developers can more easily move models between state-of-the-art tools and choose the combination that is best for them. ONNX is developed and supported by a community of partners. You can learn more about ONNX and what tools are supported by going to `onnx.ai <https://onnx.ai/>`_.

Tutorial Overview
-----------------

This tutorial will walk you through 4 main steps:


#. `Download (or train) PyTorch style transfer models`_
#. `Convert the PyTorch models to ONNX models`_
#. `Convert the ONNX models to CoreML models`_
#. `Run the CoreML models in a style transfer iOS App`_

Preparing the Environment
-------------------------

We will be working in a virtualenv in order to avoid conflicts with your local packages.
We are also using Python 3.6 for this tutorial, but other versions should work as well.

.. code-block:: python

   python3.6 -m venv venv
   source ./venv/bin/activate


You need to install pytorch and the onnx->coreml converter:

.. code-block:: bash

   pip install torchvision onnx-coreml


You will also need to install XCode if you want to run the iOS style transfer app on your iPhone.
You can also convert models in Linux, however to run the iOS app itself, you will need a Mac.

Download (or train) PyTorch style transfer models
-------------------------------------------------

For this tutorial, we will use the style transfer models that are published with pytorch in https://github.com/pytorch/examples/tree/master/fast_neural_style .
If you would like to use a different PyTorch or ONNX model, feel free to skip this step.

These models are meant for applying style transfer on still images and really not optimized to be fast enough for video. However if we reduce the resolution low enough, they can also work well on videos.

Let's download the models:

.. code-block:: bash

   git clone https://github.com/pytorch/examples
   cd examples/fast_neural_style


If you would like to train the models yourself, the pytorch/examples repository you just cloned has more information on how to do this.
For now, we'll just download pre-trained models with the script provided by the repository:

.. code-block:: bash

   python download_saved_models.py


This script downloads the pre-trained PyTorch models and puts them into the ``saved_models`` folder.
There should now be 4 files, ``candy.pth``\ , ``mosaic.pth``\ , ``rain_princess.pth`` and ``udnie.pth`` in your directory.

Convert the PyTorch models to ONNX models
-----------------------------------------

Now that we have the pre-trained PyTorch models as ``.pth`` files in the ``saved_models`` folder, we will need to convert them to ONNX format.
The model definition is in the pytorch/examples repository we cloned previously, and with a few lines of python we can export it to ONNX.
In this case, instead of actually running the neural net, we will call ``torch.onnx._export``\ , which is provided with PyTorch as an api to directly export ONNX formatted models from PyTorch.
However, in this case we don't even need to do that, because a script already exists ``neural_style/neural_style.py`` that will do this for us.
You can also take a look at that script if you would like to apply it to other models.

Exporting the ONNX format from PyTorch is essentially tracing your neural network so this api call will internally run the network on 'dummy data' in order to generate the graph.
For this, it needs an input image to apply the style transfer to which can simply be a blank image.
However, the pixel size of this image is important, as this will be the size for the exported style transfer model.
To get good performance, we'll use a resolution of 250x540. Feel free to take a larger resolution if you care less about
FPS and more about style transfer quality.

Let's use `ImageMagick <https://www.imagemagick.org/>`_ to create a blank image of the resolution we want:

.. code-block:: bash

   convert -size 250x540 xc:white png24:dummy.jpg


and use that to export the PyTorch models:

.. code-block:: bash

   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/candy.pth --cuda 0 --export_onnx ./saved_models/candy.onnx
   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/udnie.pth --cuda 0 --export_onnx ./saved_models/udnie.onnx
   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/rain_princess.pth --cuda 0 --export_onnx ./saved_models/rain_princess.onnx
   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/mosaic.pth --cuda 0 --export_onnx ./saved_models/mosaic.onnx


You should end up with 4 files, ``candy.onnx``\ , ``mosaic.onnx``\ , ``rain_princess.onnx`` and ``udnie.onnx``\ ,
created from the corresponding ``.pth`` files.

Convert the ONNX models to CoreML models
----------------------------------------

Now that we have ONNX models, we can convert them to CoreML models in order to run them on Apple devices.
For this, we use the onnx-coreml converter we installed previously.
The converter comes with a ``convert-onnx-to-coreml`` script, which the installation steps above added to our path. Unfortunately that won't work for us as we need to mark the input and output of the network as an image
and, while this is supported by the converter, it is only supported when calling the converter from python.

Looking at the style transfer model (for example opening the .onnx file in an application like `Netron <https://github.com/lutzroeder/Netron>`_\ ),
we see that the input is named '0' and the output is named '186'. These are just numeric ids assigned by PyTorch.
We will need to mark these as images.

So let's create a small python file and call it ``onnx_to_coreml.py``. This can be created by using the touch command and edited with your favorite editor to add the following lines of code.

.. code-block:: python

   import sys
   from onnx import onnx_pb
   from onnx_coreml import convert

   model_in = sys.argv[1]
   model_out = sys.argv[2]

   model_file = open(model_in, 'rb')
   model_proto = onnx_pb.ModelProto()
   model_proto.ParseFromString(model_file.read())
   coreml_model = convert(model_proto, image_input_names=['0'], image_output_names=['186'])
   coreml_model.save(model_out)


we now run it:

.. code-block:: bash

   python onnx_to_coreml.py ./saved_models/candy.onnx ./saved_models/candy.mlmodel
   python onnx_to_coreml.py ./saved_models/udnie.onnx ./saved_models/udnie.mlmodel
   python onnx_to_coreml.py ./saved_models/rain_princess.onnx ./saved_models/rain_princess.mlmodel
   python onnx_to_coreml.py ./saved_models/mosaic.onnx ./saved_models/mosaic.mlmodel


Now, there should be 4 CoreML models in your ``saved_models`` directory: ``candy.mlmodel``\ , ``mosaic.mlmodel``\ , ``rain_princess.mlmodel`` and ``udnie.mlmodel``.

Run the CoreML models in a style transfer iOS App
-------------------------------------------------

This repository (i.e. the one you're currently reading the README.md of) contains an iOS app able to run CoreML style transfer models on a live camera stream from your phone camera. Let's clone the repository:

.. code-block:: bash

   git clone https://github.com/onnx/tutorials


and open the ``tutorials/examples/CoreML/ONNXLive/ONNXLive.xcodeproj`` project in XCode.
We recommend using XCode 9.3 and an iPhone X. There might be issues running on older devices or XCode versions.

In the ``Models/`` folder, the project contains some .mlmodel files. We're going to replace them with the models we just created.

You then run the app on your iPhone and you are all set. Tapping on the screen switches through the models.

Conclusion
----------

We hope this tutorial gave you an overview of what ONNX is about and how you can use it to convert neural networks
between frameworks, in this case neural style transfer models moving from PyTorch to CoreML.

Feel free to experiment with these steps and test them on your own models.
Please let us know if you hit any issues or want to give feedback. We'd like to hear what you think.
