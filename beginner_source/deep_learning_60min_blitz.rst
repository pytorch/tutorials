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


.. Note::
    Make sure you have the `torch`_ and `torchvision`_ packages installed.

.. _torch: https://github.com/pytorch/pytorch
.. _torchvision: https://github.com/pytorch/vision


.. toctree::
   :hidden:

   /beginner/blitz/tensor_tutorial
   /beginner/blitz/autograd_tutorial
   /beginner/blitz/neural_networks_tutorial
   /beginner/blitz/cifar10_tutorial

.. galleryitem:: /beginner/blitz/tensor_tutorial.py
    :figure: /_static/img/tensor_illustration_flat.png

.. galleryitem:: /beginner/blitz/autograd_tutorial.py
    :figure: /_static/img/autodiff.png

.. galleryitem:: /beginner/blitz/neural_networks_tutorial.py
    :figure: /_static/img/mnist.png

.. galleryitem:: /beginner/blitz/cifar10_tutorial.py
    :figure: /_static/img/cifar10.png

.. raw:: html

    <div style='clear:both'></div>
