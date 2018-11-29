Welcome to PyTorch Tutorials
============================

To learn how to use PyTorch, begin with our Getting Started Tutorials.
The :doc:`60-minute blitz </beginner/deep_learning_60min_blitz>` is the most common
starting point, and provides a broad view into how to use PyTorch from the basics all the way into constructing deep neural networks.

Some considerations:

* If you would like to do the tutorials interactively via IPython / Jupyter,
  each tutorial has a download link for a Jupyter Notebook and Python source code.
* Additional high-quality examples are available, including image classification,
  unsupervised learning, reinforcement learning, machine translation, and
  many other applications, in `PyTorch Examples
  <https://github.com/pytorch/examples/>`_.
* You can find reference documentation for the PyTorch API and layers in `PyTorch Docs
  <https://pytorch.org/docs>`_ or via inline help.
* If you would like the tutorials section improved, please open a github issue
  `here <https://github.com/pytorch/tutorials>`_ with your feedback.

Lastly, some of the tutorials are marked as requiring the *Preview release*. These are tutorials that use the new functionality from the PyTorch 1.0 Preview. Please visit the `Get Started <https://pytorch.org/get-started>`_ section of the PyTorch website for instructions on how to install the latest Preview build before trying these tutorials.

Getting Started
------------------

.. customgalleryitem::
   :figure: /_static/img/thumbnails/pytorch-logo-flat.png
   :tooltip: Understand PyTorch’s Tensor library and neural networks at a high level
   :description: :doc:`/beginner/deep_learning_60min_blitz`

.. customgalleryitem::
   :figure: /_static/img/thumbnails/landmarked_face2.png
   :tooltip: Learn how to load and preprocess/augment data from a non trivial dataset
   :description: :doc:`/beginner/data_loading_tutorial`

.. customgalleryitem::
   :tooltip: This tutorial introduces the fundamental concepts of PyTorch through self-contained examples
   :figure: /_static/img/thumbnails/examples.png
   :description: :doc:`/beginner/pytorch_with_examples`

.. customgalleryitem::
   :figure: /_static/img/thumbnails/sphx_glr_transfer_learning_tutorial_001.png
   :tooltip: In transfer learning, a model created from one task is used in another
   :description: :doc:`beginner/transfer_learning_tutorial`

.. customgalleryitem::
   :figure: /_static/img/hybrid.png
   :tooltip: Experiment with some of the key features of the PyTorch hybrid frontend
   :description: :doc:`beginner/deploy_seq2seq_hybrid_frontend_tutorial`

.. customgalleryitem::
   :figure: /_static/img/thumbnails/floppy.png
   :tooltip: Explore use cases for the saving and loading of PyTorch models
   :description: :doc:`beginner/saving_loading_models`



.. .. galleryitem:: beginner/saving_loading_models.py

.. raw:: html

    <div style='clear:both'></div>


Image
----------------------

.. customgalleryitem::
   :figure: /_static/img/thumbnails/eye.png
   :tooltip: Finetune and feature extract the torchvision models
   :description: :doc:`beginner/finetuning_torchvision_models_tutorial`

.. customgalleryitem::
   :figure: /_static/img/stn/five.gif
   :tooltip: Learn how to augment your network using a visual attention mechanism called spatial transformer networks
   :description: :doc:`intermediate/spatial_transformer_tutorial`

.. customgalleryitem::
   :figure: /_static/img/neural-style/sphx_glr_neural_style_tutorial_004.png
   :tooltip: How to implement the Neural-Style algorithm developed by Gatys, Ecker, and Bethge
   :description: :doc:`advanced/neural_style_tutorial`

.. customgalleryitem::
   :figure: /_static/img/panda.png
   :tooltip: Raise your awareness to the security vulnerabilities of ML models, and get insight into the hot topic of adversarial machine learning
   :description: :doc:`beginner/fgsm_tutorial`

.. customgalleryitem::
   :figure: /_static/img/cat.jpg
   :tooltip: Use ONNX to convert a model defined in PyTorch into the ONNX format and then load it into Caffe2
   :description: :doc:`advanced/super_resolution_with_caffe2`

.. raw:: html

    <div style='clear:both'></div>


.. Audio
.. ----------------------

.. Uncomment below when adding content
.. .. raw:: html

    <div style='clear:both'></div>


Text
----------------------

.. customgalleryitem::
   :figure: /_static/img/chat.png
   :tooltip: Train a simple chatbot using movie scripts
   :description: :doc:`beginner/chatbot_tutorial`

.. customgalleryitem::
   :figure: /_static/img/char_rnn_generation.png
   :tooltip: Generate names from languages
   :description: :doc:`intermediate/char_rnn_generation_tutorial`

.. customgalleryitem::
   :figure: /_static/img/rnnclass.png
   :tooltip: Build and train a basic character-level RNN to classify words
   :description: :doc:`intermediate/char_rnn_classification_tutorial`

.. customgalleryitem::
    :tooltip: Explore the key concepts of deep learning programming using Pytorch
    :figure: /_static/img/thumbnails/babel.jpg
    :description: :doc:`/beginner/deep_learning_nlp_tutorial`

.. galleryitem:: intermediate/seq2seq_translation_tutorial.py
  :figure: _static/img/seq2seq_flat.png

.. raw:: html

    <div style='clear:both'></div>

Generative
----------------------

.. customgalleryitem::
    :tooltip: Train a generative adversarial network (GAN) to generate new celebrities
    :figure: /_static/img/dcgan_generator.png
    :description: :doc:`beginner/dcgan_faces_tutorial`

.. raw:: html

    <div style='clear:both'></div>


Reinforcement Learning
----------------------

.. customgalleryitem::
    :tooltip: Use PyTorch to train a Deep Q Learning (DQN) agent
    :figure: /_static/img/cartpole.gif
    :description: :doc:`intermediate/reinforcement_q_learning`

.. raw:: html

    <div style='clear:both'></div>

Extending PyTorch
----------------------

.. customgalleryitem::
    :tooltip: Create extensions using numpy and scipy
    :figure: /_static/img/scipynumpy.png
    :description: :doc:`advanced/numpy_extensions_tutorial`

.. customgalleryitem::
   :tooltip: Implement custom extensions in C++ or CUDA for eager PyTorch
   :description: :doc:`/advanced/cpp_extension`
   :figure: _static/img/cpp_logo.png

.. customgalleryitem::
   :tooltip: Implement custom operators in C++ or CUDA for TorchScript
   :description: :doc:`/advanced/torch_script_custom_ops`
   :figure: _static/img/cpp_logo.png


.. raw:: html

    <div style='clear:both'></div>


Production Usage
----------------------

.. customgalleryitem::
   :tooltip: Loading a PyTorch model in C++
   :description: :doc:`advanced/cpp_export`
   :figure: _static/img/cpp_logo.png

.. customgalleryitem::
   :tooltip: Convert a neural style transfer model that has been exported from PyTorch into the Apple CoreML format using ONNX
   :description: :doc:`advanced/ONNXLive`
   :figure: _static/img/ONNXLive.png


.. customgalleryitem::
   :tooltip: Parallelize computations across processes and clusters of machines
   :description: :doc:`/intermediate/dist_tuto`
   :figure: _static/img/distributed/DistPyTorch.jpg


.. raw:: html

    <div style='clear:both'></div>

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: Getting Started

   beginner/deep_learning_60min_blitz
   beginner/data_loading_tutorial
   beginner/pytorch_with_examples
   beginner/transfer_learning_tutorial
   beginner/deploy_seq2seq_hybrid_frontend_tutorial
   beginner/saving_loading_models

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Image

   beginner/finetuning_torchvision_models_tutorial
   intermediate/spatial_transformer_tutorial
   advanced/neural_style_tutorial
   beginner/fgsm_tutorial
   advanced/super_resolution_with_caffe2

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Audio

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Text

   beginner/chatbot_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/char_rnn_classification_tutorial
   beginner/deep_learning_nlp_tutorial
   intermediate/seq2seq_translation_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Generative

   beginner/dcgan_faces_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Reinforcement Learning

   intermediate/reinforcement_q_learning

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Extending PyTorch

   advanced/numpy_extensions_tutorial
   advanced/cpp_extension
   advanced/torch_script_custom_ops

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Production Usage

   intermediate/dist_tuto
   advanced/ONNXLive
   advanced/cpp_export
