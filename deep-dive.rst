:orphan:

Deep Dive
=========

Focused on enhancing model performance, this section includes
tutorials on profiling, hyperparameter tuning, quantization,
and other techniques to optimize PyTorch models for better efficiency
and speed.

.. raw:: html

    <div id="tutorial-cards-container">

    <nav class="navbar navbar-expand-lg navbar-light tutorials-nav col-12">
        <div class="tutorial-tags-container">
            <div id="dropdown-filter-tags">
                <div class="tutorial-filter-menu">
                    <div class="tutorial-filter filter-btn all-tag-selected" data-tag="all">All</div>
                </div>
            </div>
        </div>
    </nav>

    <hr class="tutorials-hr">

    <div class="row">

    <div id="tutorial-cards">
    <div class="list">

.. Add tutorial cards below this line
.. customcarditem::
   :header: Profiling PyTorch
   :card_description: Learn how to profile a PyTorch application
   :link: beginner/profiler.html
   :tags: Profiling

.. customcarditem::
   :header: Parametrizations Tutorial
   :card_description: Learn how to use torch.nn.utils.parametrize to put constraints on your parameters (e.g. make them orthogonal, symmetric positive definite, low-rank...)
   :image: _static/img/thumbnails/cropped/parametrizations.png
   :link: intermediate/parametrizations.html
   :tags: Model-Optimization,Best-Practice

.. customcarditem::
   :header: Pruning Tutorial
   :card_description: Learn how to use torch.nn.utils.prune to sparsify your neural networks, and how to extend it to implement your own custom pruning technique.
   :image: _static/img/thumbnails/cropped/Pruning-Tutorial.png
   :link: intermediate/pruning_tutorial.html
   :tags: Model-Optimization,Best-Practice


.. customcarditem::
   :header: Inductor CPU Backend Debugging and Profiling
   :card_description: Learn the usage, debugging and performance profiling for ``torch.compile`` with Inductor CPU backend.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/inductor_debug_cpu.html
   :tags: Model-Optimization,inductor

.. customcarditem::
   :header: (beta) Implementing High-Performance Transformers with SCALED DOT PRODUCT ATTENTION
   :card_description: This tutorial explores the new torch.nn.functional.scaled_dot_product_attention and how it can be used to construct Transformer components.
   :image: _static/img/thumbnails/cropped/pytorch-logo.png
   :link: intermediate/scaled_dot_product_attention_tutorial.html
   :tags: Model-Optimization,Attention,Transformer

.. customcarditem::
   :header: Knowledge Distillation in Convolutional Neural Networks
   :card_description:  Learn how to improve the accuracy of lightweight models using more powerful models as teachers.
   :image: _static/img/thumbnails/cropped/knowledge_distillation_pytorch_logo.png
   :link: beginner/knowledge_distillation_tutorial.html
   :tags: Model-Optimization,Image/Video

.. Frontend APIs
.. customcarditem::
   :header: (beta) Channels Last Memory Format in PyTorch
   :card_description: Get an overview of Channels Last memory format and understand how it is used to order NCHW tensors in memory preserving dimensions.
   :image: _static/img/thumbnails/cropped/experimental-Channels-Last-Memory-Format-in-PyTorch.png
   :link: intermediate/memory_format_tutorial.html
   :tags: Memory-Format,Best-Practice,Frontend-APIs

.. customcarditem::
   :header: Forward-mode Automatic Differentiation
   :card_description: Learn how to use forward-mode automatic differentiation.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/forward_ad_usage.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Jacobians, Hessians, hvp, vhp, and more
   :card_description: Learn how to compute advanced autodiff quantities using torch.func
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/jacobians_hessians.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Model Ensembling
   :card_description: Learn how to ensemble models using torch.vmap
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/ensembling.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Per-Sample-Gradients
   :card_description: Learn how to compute per-sample-gradients using torch.func
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/per_sample_grads.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Neural Tangent Kernels
   :card_description: Learn how to compute neural tangent kernels using torch.func
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/neural_tangent_kernels.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Using the PyTorch C++ Frontend
   :card_description: Walk through an end-to-end example of training a model with the C++ frontend by training a DCGAN – a kind of generative model – to generate images of MNIST digits.
   :image: _static/img/thumbnails/cropped/Using-the-PyTorch-Cpp-Frontend.png
   :link: advanced/cpp_frontend.html
   :tags: Frontend-APIs,C++

.. customcarditem::
   :header: Autograd in C++ Frontend
   :card_description: The autograd package helps build flexible and dynamic nerural netorks. In this tutorial, exploreseveral examples of doing autograd in PyTorch C++ frontend
   :image: _static/img/thumbnails/cropped/Autograd-in-Cpp-Frontend.png
   :link: advanced/cpp_autograd.html
   :tags: Frontend-APIs,C++

.. End of tutorial card section
.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:

   beginner/profiler
   beginner/vt_tutorial
   intermediate/parametrizations
   intermediate/pruning_tutorial
   intermediate/inductor_debug_cpu
   intermediate/scaled_dot_product_attention_tutorial
   beginner/knowledge_distillation_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Frontend APIs

   intermediate/memory_format_tutorial
   intermediate/forward_ad_usage
   intermediate/jacobians_hessians
   intermediate/ensembling
   intermediate/per_sample_grads
   intermediate/neural_tangent_kernels.py
   advanced/cpp_frontend
   advanced/cpp_autograd
