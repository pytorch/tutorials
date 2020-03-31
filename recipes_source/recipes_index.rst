PyTorch Recipes
---------------------------------------------
Recipes are bite-sized bite-sized, actionable examples of how to use specific PyTorch features, different from our full-length tutorials.

.. raw:: html

        </div>
    </div>

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

.. Add recipe cards below this line

.. Getting Started

.. customcarditem::
   :header: Writing Custom Datasets, DataLoaders and Transforms
   :card_description: Learn how to load and preprocess/augment data from a non trivial dataset.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/data_loading_tutorial.html
   :tags: Getting-Started


.. Production

.. customcarditem::
   :header: Deploying PyTorch in Python via a REST API with Flask
   :card_description: Deploy a PyTorch model using Flask and expose a REST API for model inference using the example of a pretrained DenseNet 121 model which detects the image.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/flask_rest_api_tutorial.html
   :tags: Production

.. customcarditem::
   :header: Introduction to TorchScript
   :card_description: Introduction to TorchScript, an intermediate representation of a PyTorch model (subclass of nn.Module) that can then be run in a high-performance environment such as C++.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/Intro_to_TorchScript_tutorial.html
   :tags: Production

.. customcarditem::
   :header: Loading a TorchScript Model in C++
   :card_description:  Learn how PyTorch provides to go from an existing Python model to a serialized representation that can be loaded and executed purely from C++, with no dependency on Python.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/cpp_export.html
   :tags: Production

.. customcarditem::
   :header: (optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime
   :card_description:  Convert a model defined in PyTorch into the ONNX format and then run it with ONNX Runtime.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/super_resolution_with_onnxruntime.html
   :tags: Production

.. Parallel-and-Distributed-Training

.. customcarditem::
   :header: Model Parallel Best Practices
   :card_description:  Learn how to implement model parallel, a distributed training technique which splits a single model onto different GPUs, rather than replicating the entire model on each GPU 
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/model_parallel_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Getting Started with Distributed Data Parallel
   :card_description: Learn the basics of when to use distributed data paralle versus data parallel and work through an example to set it up. 
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/ddp_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Writing Distributed Applications with PyTorch
   :card_description: Set up the distributed package of PyTorch, use the different communication strategies, and go over some the internals of the package.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/dist_tuto.html
   :tags: Parallel-and-Distributed-Training
   
.. customcarditem::
   :header: Getting Started with Distributed RPC Framework
   :card_description: Learn how to build distributed training using the torch.distributed.rpc package.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/rpc_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: (advanced) PyTorch 1.0 Distributed Trainer with Amazon AWS
   :card_description: Set up the distributed package of PyTorch, use the different communication strategies, and go over some the internals of the package.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/aws_distributed_training_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. Extending PyTorch

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Operators
   :card_description:  Implement a custom TorchScript operator in C++, how to build it into a shared library, how to use it in Python to define TorchScript models and lastly how to load it into a C++ application for inference workloads.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/torch_script_custom_ops.html
   :tags: Extending-PyTorch, TorchScript
   
.. customcarditem::
   :header: Extending TorchScript with Custom C++ Classes
   :card_description: This is a contiuation of the custom operator tutorial, and introduces the API we’ve built for binding C++ classes into TorchScript and Python simultaneously.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/torch_script_custom_classes.html
   :tags: Extending-PyTorch, TorchScript

.. customcarditem::
   :header: Creating Extensions Using numpy and scipy
   :card_description:  Create a neural network layer with no parameters using numpy. Then use scipy to create a neural network layer that has learnable weights. 
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/numpy_extensions_tutorial.html
   :tags: Extending-PyTorch, numpy, scipy

.. customcarditem::
   :header: Custom C++ and CUDA Extensions
   :card_description:  Create a neural network layer with no parameters using numpy. Then use scipy to create a neural network layer that has learnable weights. 
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/cpp_extension.html
   :tags: Extending-PyTorch, C++, CUDA


.. End of recipe card section

.. raw:: html

    </div>

    </div>

    </div>

    </div>

.. .. galleryitem:: beginner/saving_loading_models.py
