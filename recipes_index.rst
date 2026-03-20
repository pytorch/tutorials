Recipes
========

Recipes are bite-sized, actionable examples of
how to use specific PyTorch features, different
from our full-length tutorials.

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

.. Add recipe cards below this line

.. Basics

.. customcarditem::
   :header: Defining a Neural Network
   :card_description: Learn how to use PyTorch's torch.nn package to create and define a neural network for the MNIST dataset.
   :image: _static/img/thumbnails/cropped/defining-a-network.PNG
   :link: recipes/recipes/defining_a_neural_network.html
   :tags: Basics

.. customcarditem::
   :header: What is a state_dict in PyTorch
   :card_description: Learn how state_dict objects and Python dictionaries are used in saving or loading models from PyTorch.
   :image: _static/img/thumbnails/cropped/what-is-a-state-dict.PNG
   :link: recipes/recipes/what_is_state_dict.html
   :tags: Basics


.. customcarditem::
   :header: Warmstarting model using parameters from a different model in PyTorch
   :card_description: Learn how warmstarting the training process by partially loading a model or loading a partial model can help your model converge much faster than training from scratch.
   :image: _static/img/thumbnails/cropped/warmstarting-models.PNG
   :link: recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html
   :tags: Basics

.. customcarditem::
   :header: Zeroing out gradients in PyTorch
   :card_description: Learn when you should zero out gradients and how doing so can help increase the accuracy of your model.
   :image: _static/img/thumbnails/cropped/zeroing-out-gradients.PNG
   :link: recipes/recipes/zeroing_out_gradients.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Benchmark
   :card_description: Learn how to use PyTorch's benchmark module to measure and compare the performance of your code
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/recipes/benchmark.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Benchmark (quick start)
   :card_description: Learn how to measure snippet run times and collect instructions.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/recipes/timer_quick_start.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Profiler
   :card_description: Learn how to use PyTorch's profiler to measure operators time and memory consumption
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/recipes/profiler_recipe.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Profiler with Instrumentation and Tracing Technology API (ITT API) support
   :card_description: Learn how to use PyTorch's profiler with Instrumentation and Tracing Technology API (ITT API) to visualize operators labeling in Intel® VTune™ Profiler GUI
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/profile_with_itt.html
   :tags: Basics

.. customcarditem::
   :header: Dynamic Compilation Control with ``torch.compiler.set_stance``
   :card_description: Learn how to use torch.compiler.set_stance
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/torch_compiler_set_stance_tutorial.html
   :tags: Compiler

.. customcarditem::
   :header: Reasoning about Shapes in PyTorch
   :card_description: Learn how to use the meta device to reason about shapes in your model.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/recipes/reasoning_about_shapes.html
   :tags: Basics

.. customcarditem::
   :header: Tips for Loading an nn.Module from a Checkpoint
   :card_description: Learn tips for loading an nn.Module from a checkpoint.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/recipes/module_load_state_dict_tips.html
   :tags: Basics

.. customcarditem::
   :header: (beta) Using TORCH_LOGS to observe torch.compile
   :card_description: Learn how to use the torch logging APIs to observe the compilation process.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/torch_logs.html
   :tags: Basics

.. customcarditem::
   :header: Extension points in nn.Module for loading state_dict and tensor subclasses
   :card_description: New extension points in nn.Module.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/recipes/swap_tensors.html
   :tags: Basics

.. customcarditem::
   :header: torch.export AOTInductor Tutorial for Python runtime
   :card_description: Learn an end-to-end example of how to use AOTInductor for python runtime.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/torch_export_aoti_python.html
   :tags: Basics

.. customcarditem::
   :header: Demonstration of torch.export flow, common challenges and the solutions to address them
   :card_description: Learn how to export models for popular usecases
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/torch_export_challenges_solutions.html
   :tags: Compiler,TorchCompile

.. Interpretability

.. customcarditem::
   :header: Model Interpretability using Captum
   :card_description: Learn how to use Captum attribute the predictions of an image classifier to their corresponding image features and visualize the attribution results.
   :image: _static/img/thumbnails/cropped/model-interpretability-using-captum.png
   :link: recipes/recipes/Captum_Recipe.html
   :tags: Interpretability,Captum

.. customcarditem::
   :header: How to use TensorBoard with PyTorch
   :card_description: Learn basic usage of TensorBoard with PyTorch, and how to visualize data in TensorBoard UI
   :image: _static/img/thumbnails/tensorboard_scalars.png
   :link: recipes/recipes/tensorboard_with_pytorch.html
   :tags: Visualization,TensorBoard

.. customcarditem::
   :header: DebugMode: Recording Dispatched Operations and Numerical Debugging
   :card_description: Inspect dispatched ops, tensor hashes, and module boundaries to debug eager and ``torch.compile`` runs.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/debug_mode_tutorial.html
   :tags: Interpretability,Compiler

.. Automatic Mixed Precision

.. customcarditem::
   :header: Automatic Mixed Precision
   :card_description: Use torch.cuda.amp to reduce runtime and save memory on NVIDIA GPUs.
   :image: _static/img/thumbnails/cropped/amp.png
   :link: recipes/recipes/amp_recipe.html
   :tags: Model-Optimization

.. Performance

.. customcarditem::
   :header: Performance Tuning Guide
   :card_description: Tips for achieving optimal performance.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/recipes/tuning_guide.html
   :tags: Model-Optimization

.. customcarditem::
   :header: Optimizing CPU Performance on Intel® Xeon® with run_cpu Script
   :card_description: How to use run_cpu script for optimal runtime configurations on Intel® Xeon CPUs.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/xeon_run_cpu.html
   :tags: Model-Optimization


.. (beta) Utilizing Torch Function modes with torch.compile

.. customcarditem::
   :header: (beta) Utilizing Torch Function modes with torch.compile
   :card_description: Override torch operators with Torch Function modes and torch.compile
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/torch_compile_torch_function_modes.html
   :tags: Model-Optimization

.. (beta) Compiling the Optimizer with torch.compile

.. customcarditem::
   :header: (beta) Compiling the Optimizer with torch.compile
   :card_description: Speed up the optimizer using torch.compile
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/compiling_optimizer.html
   :tags: Model-Optimization

.. (beta) Running the compiled optimizer with an LR Scheduler

.. customcarditem::
   :header: (beta) Running the compiled optimizer with an LR Scheduler
   :card_description: Speed up training with LRScheduler and torch.compiled optimizer
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/compiling_optimizer_lr_scheduler.html
   :tags: Model-Optimization

.. (beta) Explicit horizontal fusion with foreach_map and torch.compile
.. customcarditem::
   :header: (beta) Explicit horizontal fusion with foreach_map and torch.compile
   :card_description: Horizontally fuse pointwise ops with torch.compile
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/foreach_map.py
   :tags: Model-Optimization

.. Using User-Defined Triton Kernels with ``torch.compile``

.. customcarditem::
   :header: Using User-Defined Triton Kernels with ``torch.compile``
   :card_description: Learn how to use user-defined kernels with ``torch.compile``
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/torch_compile_user_defined_triton_kernel_tutorial.html
   :tags: Model-Optimization

.. Compile Time Caching in ``torch.compile``

.. customcarditem::
   :header: Compile Time Caching in ``torch.compile``
   :card_description: Learn how to use compile time caching in ``torch.compile``
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/torch_compile_caching_tutorial.html
   :tags: Model-Optimization

.. Compile Time Caching Configurations

.. customcarditem::
   :header: Compile Time Caching Configurations
   :card_description: Learn how to configure compile time caching in ``torch.compile``
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/torch_compile_caching_configuration_tutorial.html
   :tags: Model-Optimization

.. Reducing Cold Start Compilation Time with Regional Compilation

.. customcarditem::
   :header: Reducing torch.compile cold start compilation time with regional compilation
   :card_description: Learn how to use regional compilation to control cold start compile time
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/regional_compilation.html
   :tags: Model-Optimization

.. Intel(R) Neural Compressor for PyTorch*

.. customcarditem::
   :header: Intel® Neural Compressor for PyTorch
   :card_description: Ease-of-use quantization for PyTorch with Intel® Neural Compressor.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/intel_neural_compressor_for_pytorch.html
   :tags: Quantization,Model-Optimization

.. Distributed Training

.. customcarditem::
   :header: Getting Started with DeviceMesh
   :card_description: Learn how to use DeviceMesh
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/distributed_device_mesh.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Shard Optimizer States with ZeroRedundancyOptimizer
   :card_description: How to use ZeroRedundancyOptimizer to reduce memory consumption.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/zero_redundancy_optimizer.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Direct Device-to-Device Communication with TensorPipe RPC
   :card_description: How to use RPC with direct GPU-to-GPU communication.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: recipes/cuda_rpc.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Getting Started with Distributed Checkpoint (DCP)
   :card_description: Learn how to checkpoint distributed models with Distributed Checkpoint package.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-DCP.png
   :link: recipes/distributed_checkpoint_recipe.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Asynchronous Checkpointing (DCP)
   :card_description: Learn how to checkpoint distributed models with Distributed Checkpoint package.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-DCP.png
   :link: recipes/distributed_async_checkpoint_recipe.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Getting Started with CommDebugMode
   :card_description: Learn how to use CommDebugMode for DTensors
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/distributed_comm_debug_mode.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Reducing AoT cold start compilation time with regional compilation
   :card_description: Learn how to use regional compilation to control AoT cold start compile time
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: recipes/regional_aot.html
   :tags: Model-Optimization

.. End of tutorial card section

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :maxdepth: 2
   :hidden:

   recipes/recipes/defining_a_neural_network
   recipes/torch_logs
   recipes/recipes/what_is_state_dict
   recipes/recipes/warmstarting_model_using_parameters_from_a_different_model
   recipes/recipes/zeroing_out_gradients
   recipes/recipes/profiler_recipe
   recipes/recipes/profile_with_itt
   recipes/recipes/Captum_Recipe
   recipes/recipes/tensorboard_with_pytorch
   recipes/recipes/dynamic_quantization
   recipes/recipes/amp_recipe
   recipes/recipes/tuning_guide
   recipes/recipes/xeon_run_cpu
   recipes/compiling_optimizer
   recipes/recipes/timer_quick_start
   recipes/zero_redundancy_optimizer
   recipes/distributed_comm_debug_mode
   recipes/torch_export_challenges_solutions
   recipes/recipes/benchmark
   recipes/recipes/module_load_state_dict_tips
   recipes/recipes/reasoning_about_shapes
   recipes/recipes/swap_tensors
   recipes/torch_export_aoti_python
   recipes/recipes/tensorboard_with_pytorch
   recipes/torch_compile_torch_function_modes
   recipes/compiling_optimizer_lr_scheduler
   recipes/foreach_map
   recipes/torch_compile_user_defined_triton_kernel_tutorial
   recipes/torch_compile_caching_tutorial
   recipes/torch_compile_caching_configuration_tutorial
   recipes/regional_compilation
   recipes/regional_aot
   recipes/intel_neural_compressor_for_pytorch
   recipes/distributed_device_mesh
   recipes/distributed_checkpoint_recipe
   recipes/distributed_async_checkpoint_recipe
   recipes/debug_mode_tutorial
