Compilers
=========

Explore PyTorch compilers to optimize and deploy models efficiently.
Learn about APIs like ``torch.compile`` and ``torch.export``
that let you enhance model performance and streamline deployment
processes.
Explore advanced topics such as compiled autograd, dynamic compilation
control, as well as third-party backend solutions.

.. warning::

   TorchScript is no longer in active development.

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

.. customcarditem::
   :header: torch.compile Tutorial
   :card_description: Speed up your models with minimal code changes using torch.compile, the latest PyTorch compiler solution.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/torch_compile_tutorial.html
   :tags: Model-Optimization,torch.compile

.. customcarditem::
   :header: Compiled Autograd: Capturing a larger backward graph for ``torch.compile``
   :card_description: Learn how to use compiled autograd to capture a larger backward graph.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/compiled_autograd_tutorial
   :tags: Model-Optimization,CUDA,torch.compile

.. customcarditem::
   :header: Inductor CPU Backend Debugging and Profiling
   :card_description: Learn the usage, debugging and performance profiling for ``torch.compile`` with Inductor CPU backend.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/inductor_debug_cpu.html
   :tags: Model-Optimization,torch.compile

.. customcarditem::
   :header: Dynamic Compilation Control with ``torch.compiler.set_stance``
   :card_description: Learn how to use torch.compiler.set_stance
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_compiler_set_stance_tutorial.html
   :tags: Model-Optimization,torch.compile

.. customcarditem::
   :header: Demonstration of torch.export flow, common challenges and the solutions to address them
   :card_description: Learn how to export models for popular usecases
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_export_challenges_solutions.html
   :tags: Model-Optimization,torch.compile

.. customcarditem::
   :header: (beta) Compiling the Optimizer with torch.compile
   :card_description: Speed up the optimizer using torch.compile
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/compiling_optimizer.html
   :tags: Model-Optimization,torch.compile

.. customcarditem::
   :header: (beta) Running the compiled optimizer with an LR Scheduler
   :card_description: Speed up training with LRScheduler and torch.compiled optimizer
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/compiling_optimizer_lr_scheduler.html
   :tags: Model-Optimization,torch.compile

.. customcarditem::
   :header: Using User-Defined Triton Kernels with ``torch.compile``
   :card_description: Learn how to use user-defined kernels with ``torch.compile``
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_compile_user_defined_triton_kernel_tutorial.html
   :tags: Model-Optimization,torch.compile

.. customcarditem::
   :header: Compile Time Caching in ``torch.compile``
   :card_description: Learn how to use compile time caching in ``torch.compile``
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_compile_caching_tutorial.html
   :tags: Model-Optimization,torch.compile

.. customcarditem::
   :header: Compile Time Caching Configurations
   :card_description: Learn how to configure compile time caching in ``torch.compile``
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_compile_caching_configuration_tutorial.html
   :tags: Model-Optimization,torch.compile

.. customcarditem::
   :header: Reducing torch.compile cold start compilation time with regional compilation
   :card_description: Learn how to use regional compilation to control cold start compile time
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/regional_compilation.html
   :tags: Model-Optimization,torch.compile

.. Export

.. customcarditem::
   :header: torch.export AOTInductor Tutorial for Python runtime
   :card_description: Learn an end-to-end example of how to use AOTInductor for python runtime.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_export_aoti_python.html
   :tags: Basics,torch.export

.. customcarditem::
   :header: Deep dive into torch.export
   :card_description: Learn how to use torch.export to export PyTorch models into standardized model representations.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: torch_export_tutorial.html
   :tags: Basics,torch.export

.. ONNX

.. customcarditem::
   :header: (optional) Exporting a PyTorch model to ONNX using TorchDynamo backend and Running it using ONNX Runtime
   :card_description: Build a image classifier model in PyTorch and convert it to ONNX before deploying it with ONNX Runtime.
   :image: _static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png
   :link: beginner/onnx/export_simple_model_to_onnx_tutorial.html
   :tags: Production,ONNX,Backends

.. customcarditem::
   :header: Extending the ONNX exporter operator support
   :card_description: Demonstrate end-to-end how to address unsupported operators in ONNX.
   :image: _static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png
   :link: beginner/onnx/onnx_registry_tutorial.html
   :tags: Production,ONNX,Backends

.. customcarditem::
   :header: Exporting a model with control flow to ONNX
   :card_description: Demonstrate how to handle control flow logic while exporting a PyTorch model to ONNX.
   :image: _static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png
   :link: beginner/onnx/export_control_flow_model_to_onnx_tutorial.html
   :tags: Production,ONNX,Backends

.. Code Transformations with FX

.. customcarditem::
   :header: Building a Convolution/Batch Norm fuser in FX
   :card_description: Build a simple FX pass that fuses batch norm into convolution to improve performance during inference.
   :image: _static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png
   :link: intermediate/fx_conv_bn_fuser.html
   :tags: FX

.. customcarditem::
   :header: Building a Simple Performance Profiler with FX
   :card_description: Build a simple FX interpreter to record the runtime of op, module, and function calls and report statistics
   :image: _static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png
   :link: intermediate/fx_profiling_tutorial.html
   :tags: FX

.. TorchScript

.. customcarditem::
   :header: Introduction to TorchScript
   :card_description: Introduction to TorchScript, an intermediate representation of a PyTorch model (subclass of nn.Module) that can then be run in a high-performance environment such as C++.
   :image: _static/img/thumbnails/cropped/Introduction-to-TorchScript.png
   :link: beginner/Intro_to_TorchScript_tutorial.html
   :tags: Production,TorchScript

.. customcarditem::
   :header: Loading a TorchScript Model in C++
   :card_description:  Learn how PyTorch provides to go from an existing Python model to a serialized representation that can be loaded and executed purely from C++, with no dependency on Python.
   :image: _static/img/thumbnails/cropped/Loading-a-TorchScript-Model-in-Cpp.png
   :link: advanced/cpp_export.html
   :tags: Production,TorchScript

.. customcarditem::
   :header: Distributed Optimizer with TorchScript support
   :card_description: How to enable TorchScript support for Distributed Optimizer.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/distributed_optim_torchscript.html
   :tags: Distributed-Training,TorchScript

.. customcarditem::
   :header: TorchScript for Deployment
   :card_description: Learn how to export your trained model in TorchScript format and how to load your TorchScript model in C++ and do inference.
   :image: ../_static/img/thumbnails/cropped/torchscript_overview.png
   :link: ../recipes/torchscript_inference.html
   :tags: TorchScript

.. customcarditem::
   :header: Deploying with Flask
   :card_description: Learn how to use Flask, a lightweight web server, to quickly setup a web API from your trained PyTorch model.
   :image: ../_static/img/thumbnails/cropped/using-flask-create-restful-api.png
   :link: ../recipes/deployment_with_flask.html
   :tags: Production,TorchScript

.. raw:: html

    </div>
    </div>

.. End of tutorial cards section

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: torch.compile

   intermediate/torch_compile_tutorial
   intermediate/compiled_autograd_tutorial
   intermediate/inductor_debug_cpu
   recipes/torch_compiler_set_stance_tutorial
   recipes/torch_export_challenges_solutions
   recipes/compiling_optimizer
   recipes/compiling_optimizer_lr_scheduler
   recipes/torch_compile_user_defined_triton_kernel_tutorial
   recipes/torch_compile_caching_tutorial
   recipes/regional_compilation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: torch.export

   intermediate/torch_export_tutorial
   recipes/torch_export_aoti_python
   recipes/torch_export_challenges_solutions

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: ONNX

   beginner/onnx/intro_onnx
   beginner/onnx/export_simple_model_to_onnx_tutorial
   beginner/onnx/onnx_registry_tutorial
   beginner/onnx/export_control_flow_model_to_onnx_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Code Transforms with FX

   intermediate/fx_conv_bn_fuser
   intermediate/fx_profiling_tutorial

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: TorchScript

   beginner/Intro_to_TorchScript_tutorial
   recipes/torchscript_inference
   recipes/distributed_optim_torchscript
   advanced/cpp_export
