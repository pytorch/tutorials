Deep Learning with PyTorch: A 60 Minute Blitz
---------------------------------------------
**Author**: `Soumith Chintala <http://soumith.ch>`_

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/u7x8RXwLKcA" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

What is PyTorch?
~~~~~~~~~~~~~~~~~~~~~
PyTorch is a Python-based scientific computing package serving two broad purposes:

-  A replacement for NumPy to use the power of GPUs and other accelerators.
-  An automatic differentiation library that is useful to implement neural networks.

Goal of this tutorial:
~~~~~~~~~~~~~~~~~~~~~~~~
- Understand PyTorch’s Tensor library and neural networks at a high level.
- Train a small neural network to classify images

To run the tutorials below, make sure you have the `torch`_, `torchvision`_,
and `matplotlib`_ packages installed.

.. _torch: https://github.com/pytorch/pytorch
.. _torchvision: https://github.com/pytorch/vision
.. _matplotlib: https://github.com/matplotlib/matplotlib

.. toctree::
   :hidden:

   /beginner/blitz/tensor_tutorial
   /beginner/blitz/autograd_tutorial
   /beginner/blitz/neural_networks_tutorial
   /beginner/blitz/cifar10_tutorial

.. grid:: 4

   .. grid-item-card::  :octicon:`file-code;1em` Tensors
      :link: blitz/tensor_tutorial.html

      In this tutorial, you will learn the basics of PyTorch tensors.
      +++
      :octicon:`code;1em` Code

   .. grid-item-card::  :octicon:`file-code;1em` A Gentle Introduction to torch.autograd
      :link: blitz/autograd_tutorial.html

      Learn about autograd.
      +++
      :octicon:`code;1em` Code

   .. grid-item-card::  :octicon:`file-code;1em` Neural Networks
      :link: blitz/neural_networks_tutorial.html

      This tutorial demonstrates how you can train neural networks in PyTorch.
      +++
      :octicon:`code;1em` Code

   .. grid-item-card::  :octicon:`file-code;1em` Training a Classifier
      :link: blitz/cifar10_tutorial.html

      Learn how to train an image classifier in PyTorch by using the
      CIFAR10 dataset.
      +++
      :octicon:`code;1em` Code
