Welcome to PyTorch Tutorials
============================

To learn how to use PyTorch, begin with our Getting Started Tutorials.
The :doc:`60-minute blitz </beginner/deep_learning_60min_blitz>` is the most common
starting point, and provides a broad view into how to use PyTorch from the basics all the way into constructing deep neural networks.

Some considerations:

* We’ve added a new feature to tutorials that allows users to open the notebook associated with a tutorial in Google Colab.
  Visit `this page <https://pytorch.org/tutorials/beginner/colab.html>`_ for more information.
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
* Check out our
  `PyTorch Cheat Sheet <https://pytorch.org/tutorials/beginner/ptcheat.html>`_
  for additional useful information.
* Finally, here's a link to the
  `PyTorch Release Notes <https://github.com/pytorch/pytorch/releases>`_

Learning PyTorch
------------------

.. customgalleryitem::
   :figure: /_static/img/thumbnails/pytorch-logo-flat.png
   :tooltip: Understand PyTorch’s Tensor library and neural networks at a high level
   :description: :doc:`/beginner/deep_learning_60min_blitz`

.. customgalleryitem::
   :tooltip: This tutorial introduces the fundamental concepts of PyTorch through self-contained examples
   :figure: /_static/img/thumbnails/examples.png
   :description: :doc:`/beginner/pytorch_with_examples`

.. customgalleryitem::
   :figure: /_static/img/torch.nn.png
   :tooltip: Use torch.nn to create and train a neural network
   :description: :doc:`beginner/nn_tutorial`

.. customgalleryitem::
   :figure: /_static/img/thumbnails/pytorch_tensorboard.png
   :tooltip: Learn to use TensorBoard to visualize data and model training
   :description: :doc:`intermediate/tensorboard_tutorial`

.. raw:: html

    <div style='clear:both'></div>


Image/Video
----------------------

.. customgalleryitem::
   :figure: /_static/img/thumbnails/tv-img.png
   :tooltip: Finetuning a pre-trained Mask R-CNN model
   :description: :doc:`intermediate/torchvision_tutorial`

.. customgalleryitem::
   :figure: /_static/img/thumbnails/sphx_glr_transfer_learning_tutorial_001.png
   :tooltip: In transfer learning, a model created from one task is used in another
   :description: :doc:`beginner/transfer_learning_tutorial`

.. customgalleryitem::
   :figure: /_static/img/panda.png
   :tooltip: Raise your awareness to the security vulnerabilities of ML models, and get insight into the hot topic of adversarial machine learning
   :description: :doc:`beginner/fgsm_tutorial`

.. customgalleryitem::
    :tooltip: Train a generative adversarial network (GAN) to generate new celebrities
    :figure: /_static/img/dcgan_generator.png
    :description: :doc:`beginner/dcgan_faces_tutorial`

.. customgalleryitem::
    :tooltip: (experimental) Static Quantization with Eager Mode in PyTorch
    :figure: /_static/img/qat.png
    :description: :doc:`advanced/static_quantization_tutorial`

.. customgalleryitem::
    :tooltip: Perform quantized transfer learning with feature extractor
    :description: :doc:`/intermediate/quantized_transfer_learning_tutorial`
    :figure: /_static/img/quantized_transfer_learning.png


.. raw:: html

    <div style='clear:both'></div>

Audio
----------------------

.. customgalleryitem::
   :figure: /_static/img/audio_preprocessing_tutorial_waveform.png
   :tooltip: Preprocessing with torchaudio Tutorial
   :description: :doc:`beginner/audio_preprocessing_tutorial`

.. raw:: html

    <div style='clear:both'></div>


Text
----------------------

.. customgalleryitem::
    :tooltip: Transformer Tutorial
    :figure: /_static/img/transformer_architecture.jpg
    :description: :doc:`/beginner/transformer_tutorial`

.. customgalleryitem::
   :figure: /_static/img/rnnclass.png
   :tooltip: Build and train a basic character-level RNN to classify words
   :description: :doc:`intermediate/char_rnn_classification_tutorial`

.. customgalleryitem::
   :figure: /_static/img/char_rnn_generation.png
   :tooltip: Generate names from languages
   :description: :doc:`intermediate/char_rnn_generation_tutorial`

.. galleryitem:: intermediate/seq2seq_translation_tutorial.py
  :figure: _static/img/seq2seq_flat.png

.. customgalleryitem::
    :tooltip: Sentiment Ngrams with Torchtext
    :figure: /_static/img/text_sentiment_ngrams_model.png
    :description: :doc:`/beginner/text_sentiment_ngrams_tutorial`

.. customgalleryitem::
    :tooltip: Language Translation with Torchtext
    :figure: /_static/img/thumbnails/german_to_english_translation.png
    :description: :doc:`/beginner/torchtext_translation_tutorial`

.. customgalleryitem::
   :tooltip: Perform dynamic quantization on a pre-trained PyTorch model
   :description: :doc:`/advanced/dynamic_quantization_tutorial`
   :figure: _static/img/quant_asym.png

.. customgalleryitem::
  :tooltip: Convert a well-known state-of-the-art model like BERT into dynamic quantized model
  :description: :doc:`/intermediate/dynamic_quantization_bert_tutorial`
  :figure: /_static/img/bert.png

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

Additional APIs
----------------------

.. customgalleryitem::
    :tooltip: Using the PyTorch C++ Frontend
    :figure: /_static/img/cpp-pytorch.png
    :description: :doc:`advanced/cpp_frontend`

.. customgalleryitem::
    :tooltip: Autograd in C++ Frontend
    :figure: /_static/img/cpp-pytorch.png
    :description: :doc:`advanced/cpp_autograd`

.. customgalleryitem::
   :figure: /_static/img/named_tensor.png
   :tooltip: Named Tensor
   :description: :doc:`intermediate/named_tensor_tutorial`

.. customgalleryitem::
   :tooltip: Use pruning to sparsify your neural networks
   :description: :doc:`/intermediate/pruning_tutorial`
   :figure: _static/img/pruning.png


.. raw:: html

    <div style='clear:both'></div>


.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: Learning PyTorch

   beginner/deep_learning_60min_blitz
   beginner/pytorch_with_examples
   beginner/nn_tutorial
   intermediate/tensorboard_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Image/Video

   intermediate/torchvision_tutorial
   beginner/transfer_learning_tutorial
   beginner/fgsm_tutorial
   beginner/dcgan_faces_tutorial
   advanced/static_quantization_tutorial
   intermediate/quantized_transfer_learning_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Audio

   beginner/audio_preprocessing_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Text

   beginner/transformer_tutorial
   intermediate/char_rnn_classification_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/seq2seq_translation_tutorial
   beginner/text_sentiment_ngrams_tutorial
   beginner/torchtext_translation_tutorial
   advanced/dynamic_quantization_tutorial
   intermediate/dynamic_quantization_bert_tutorial

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
   :caption: Additional APIs

   advanced/cpp_frontend
   advanced/cpp_autograd
   intermediate/named_tensor_tutorial
   intermediate/pruning_tutorial


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Recipes

   recipes/recipes_index
