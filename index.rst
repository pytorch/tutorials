Welcome to PyTorch Tutorials
============================

To get started with learning PyTorch, start with our Beginner Tutorials. 
The :doc:`60-minute blitz </beginner/deep_learning_60min_blitz>` is the most common 
starting point, and gives you a quick introduction to PyTorch.
If you like learning by examples, you will like the tutorial 
:doc:`/beginner/pytorch_with_examples`

If you would like to do the tutorials interactively via IPython / Jupyter, 
each tutorial has a download link for a Jupyter Notebook and Python source code.

We also provide a lot of high-quality examples covering image classification, 
unsupervised learning, reinforcement learning, machine translation and 
many other applications at https://github.com/pytorch/examples/

You can find reference documentation for PyTorch's API and layers at 
http://docs.pytorch.org or via inline help.

If you would like the tutorials section improved, please 
send feedback to `sasankchilamkurthy@gmail.com <mailto: sasankchilamkurthy@gmail.com>`_
or open a github issue here with your feedback: https://github.com/pytorch/tutorials

Beginner Tutorials
------------------
   
.. customgalleryitem::
   :figure: /_static/img/thumbnails/pytorch-logo-flat.png
   :tooltip: Understand PyTorch’s Tensor library and neural networks at a high level.
   :description: :doc:`/beginner/deep_learning_60min_blitz`

.. customgalleryitem::
   :tooltip: Understand similarities and differences between torch and pytorch.
   :figure: /_static/img/thumbnails/torch-logo.png
   :description: :doc:`/beginner/former_torchies_tutorial`

.. customgalleryitem::
   :tooltip: This tutorial introduces the fundamental concepts of PyTorch through self-contained examples.
   :figure: /_static/img/thumbnails/examples.png
   :description: :doc:`/beginner/pytorch_with_examples`

.. galleryitem:: beginner/transfer_learning_tutorial.py

.. raw:: html

    <div style='clear:both'></div>


.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: Beginner Tutorials

   beginner/deep_learning_60min_blitz
   beginner/former_torchies_tutorial
   beginner/pytorch_with_examples
   beginner/transfer_learning_tutorial
   beginner/deep_learning_nlp

Intermediate Tutorials
----------------------

RNNs for NLP
^^^^^^^^^^^^

Applying recurrent neural networks to natural language tasks, from
classification to generation.


.. galleryitem:: intermediate/char_rnn_classification_tutorial.py
.. galleryitem:: intermediate/char_rnn_generation_tutorial.py
  :figure: _static/img/char_rnn_generation.png
.. galleryitem:: intermediate/seq2seq_translation_tutorial.py
  :figure: _static/img/seq2seq_flat.png

.. raw:: html

    <div style='clear:both'></div>

Reinforcement Learning
^^^^^^^^^^^^^^^^^^^^^^

.. galleryitem:: intermediate/reinforcement_q_learning.py
    :figure: _static/img/cartpole.gif

.. raw:: html

    <div style='clear:both'></div>


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Intermediate Tutorials
   
   intermediate/char_rnn_classification_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/seq2seq_translation_tutorial
   intermediate/reinforcement_q_learning

Advanced Tutorials
------------------

.. galleryitem:: advanced/numpy_extensions_tutorial.py

.. customgalleryitem::
   :tooltip: Implement custom extensions in C.
   :description: :doc:`/advanced/c_extension`

.. raw:: html

    <div style='clear:both'></div>


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Advanced Tutorials
   
   advanced/numpy_extensions_tutorial
   advanced/c_extension

