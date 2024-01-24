PyTorch Recipes
---------------------------------------------
Recipes are bite-sized, actionable examples of how to use specific PyTorch features, different from our full-length tutorials.

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

.. Basics

.. customcarditem::
   :header: Loading data in PyTorch
   :card_description: Learn how to use PyTorch packages to prepare and load common datasets for your model.
   :image: ../_static/img/thumbnails/cropped/loading-data.PNG
   :link: ../recipes/recipes/loading_data_recipe.html
   :tags: Basics


.. customcarditem::
   :header: Defining a Neural Network
   :card_description: Learn how to use PyTorch's torch.nn package to create and define a neural network for the MNIST dataset.
   :image: ../_static/img/thumbnails/cropped/defining-a-network.PNG
   :link: ../recipes/recipes/defining_a_neural_network.html
   :tags: Basics

.. customcarditem::
   :header: What is a state_dict in PyTorch
   :card_description: Learn how state_dict objects and Python dictionaries are used in saving or loading models from PyTorch.
   :image: ../_static/img/thumbnails/cropped/what-is-a-state-dict.PNG
   :link: ../recipes/recipes/what_is_state_dict.html
   :tags: Basics

.. customcarditem::
   :header: Saving and loading models for inference in PyTorch
   :card_description: Learn about the two approaches for saving and loading models for inference in PyTorch - via the state_dict and via the entire model.
   :image: ../_static/img/thumbnails/cropped/saving-and-loading-models-for-inference.PNG
   :link: ../recipes/recipes/saving_and_loading_models_for_inference.html
   :tags: Basics


.. customcarditem::
   :header: Saving and loading a general checkpoint in PyTorch
   :card_description: Saving and loading a general checkpoint model for inference or resuming training can be helpful for picking up where you last left off. In this recipe, explore how to save and load multiple checkpoints.
   :image: ../_static/img/thumbnails/cropped/saving-and-loading-general-checkpoint.PNG
   :link: ../recipes/recipes/saving_and_loading_a_general_checkpoint.html
   :tags: Basics

.. customcarditem::
   :header: Saving and loading multiple models in one file using PyTorch
   :card_description: In this recipe, learn how saving and loading multiple models can be helpful for reusing models that you have previously trained.
   :image: ../_static/img/thumbnails/cropped/saving-multiple-models.PNG
   :link: ../recipes/recipes/saving_multiple_models_in_one_file.html
   :tags: Basics

.. customcarditem::
   :header: Warmstarting model using parameters from a different model in PyTorch
   :card_description: Learn how warmstarting the training process by partially loading a model or loading a partial model can help your model converge much faster than training from scratch.
   :image: ../_static/img/thumbnails/cropped/warmstarting-models.PNG
   :link: ../recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html
   :tags: Basics

.. customcarditem::
   :header: Saving and loading models across devices in PyTorch
   :card_description: Learn how saving and loading models across devices (CPUs and GPUs) is relatively straightforward using PyTorch.
   :image: ../_static/img/thumbnails/cropped/saving-and-loading-models-across-devices.PNG
   :link: ../recipes/recipes/save_load_across_devices.html
   :tags: Basics

.. customcarditem::
   :header: Zeroing out gradients in PyTorch
   :card_description: Learn when you should zero out gradients and how doing so can help increase the accuracy of your model.
   :image: ../_static/img/thumbnails/cropped/zeroing-out-gradients.PNG
   :link: ../recipes/recipes/zeroing_out_gradients.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Benchmark
   :card_description: Learn how to use PyTorch's benchmark module to measure and compare the performance of your code
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/benchmark.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Benchmark (quick start)
   :card_description: Learn how to measure snippet run times and collect instructions.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/timer_quick_start.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Profiler
   :card_description: Learn how to use PyTorch's profiler to measure operators time and memory consumption
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/profiler_recipe.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Profiler with Instrumentation and Tracing Technology API (ITT API) support
   :card_description: Learn how to use PyTorch's profiler with Instrumentation and Tracing Technology API (ITT API) to visualize operators labeling in Intel® VTune™ Profiler GUI
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/profile_with_itt.html
   :tags: Basics

.. customcarditem::
   :header: Torch Compile IPEX Backend
   :card_description: Learn how to use torch.compile IPEX backend
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_compile_backend_ipex.html
   :tags: Basics

.. customcarditem::
   :header: Reasoning about Shapes in PyTorch
   :card_description: Learn how to use the meta device to reason about shapes in your model.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/recipes/reasoning_about_shapes.html
   :tags: Basics

.. customcarditem::
   :header: Tips for Loading an nn.Module from a Checkpoint
   :card_description: Learn tips for loading an nn.Module from a checkpoint.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/recipes/module_load_state_dict_tips.html
   :tags: Basics

.. customcarditem::
   :header: (beta) Using TORCH_LOGS to observe torch.compile
   :card_description: Learn how to use the torch logging APIs to observe the compilation process.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_logs.html
   :tags: Basics


.. Interpretability

.. customcarditem::
   :header: Model Interpretability using Captum
   :card_description: Learn how to use Captum attribute the predictions of an image classifier to their corresponding image features and visualize the attribution results.
   :image: ../_static/img/thumbnails/cropped/model-interpretability-using-captum.png
   :link: ../recipes/recipes/Captum_Recipe.html
   :tags: Interpretability,Captum

.. customcarditem::
   :header: How to use TensorBoard with PyTorch
   :card_description: Learn basic usage of TensorBoard with PyTorch, and how to visualize data in TensorBoard UI
   :image: ../_static/img/thumbnails/tensorboard_scalars.png
   :link: ../recipes/recipes/tensorboard_with_pytorch.html
   :tags: Visualization,TensorBoard

.. Quantization

.. customcarditem::
   :header: Dynamic Quantization
   :card_description:  Apply dynamic quantization to a simple LSTM model.
   :image: ../_static/img/thumbnails/cropped/using-dynamic-post-training-quantization.png
   :link: ../recipes/recipes/dynamic_quantization.html
   :tags: Quantization,Text,Model-Optimization


.. Production Development

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

.. customcarditem::
   :header: PyTorch Mobile Performance Recipes
   :card_description: List of recipes for performance optimizations for using PyTorch on Mobile (Android and iOS).
   :image: ../_static/img/thumbnails/cropped/mobile.png
   :link: ../recipes/mobile_perf.html
   :tags: Mobile,Model-Optimization

.. customcarditem::
   :header: Making Android Native Application That Uses PyTorch Android Prebuilt Libraries
   :card_description: Learn how to make Android application from the scratch that uses LibTorch C++ API and uses TorchScript model with custom C++ operator.
   :image: ../_static/img/thumbnails/cropped/android.png
   :link: ../recipes/android_native_app_with_custom_op.html
   :tags: Mobile

.. customcarditem::
  :header: Fuse Modules recipe
  :card_description: Learn how to fuse a list of PyTorch modules into a single module to reduce the model size before quantization.
  :image: ../_static/img/thumbnails/cropped/mobile.png
  :link: ../recipes/fuse.html
  :tags: Mobile

.. customcarditem::
  :header: Quantization for Mobile Recipe
  :card_description: Learn how to reduce the model size and make it run faster without losing much on accuracy.
  :image: ../_static/img/thumbnails/cropped/mobile.png
  :link: ../recipes/quantization.html
  :tags: Mobile,Quantization

.. customcarditem::
  :header: Script and Optimize for Mobile
  :card_description: Learn how to convert the model to TorchScipt and (optional) optimize it for mobile apps.
  :image: ../_static/img/thumbnails/cropped/mobile.png
  :link: ../recipes/script_optimized.html
  :tags: Mobile

.. customcarditem::
  :header: Model Preparation for iOS Recipe
  :card_description: Learn how to add the model in an iOS project and use PyTorch pod for iOS.
  :image: ../_static/img/thumbnails/cropped/ios.png
  :link: ../recipes/model_preparation_ios.html
  :tags: Mobile

.. customcarditem::
  :header: Model Preparation for Android Recipe
  :card_description: Learn how to add the model in an Android project and use the PyTorch library for Android.
  :image: ../_static/img/thumbnails/cropped/android.png
  :link: ../recipes/model_preparation_android.html
  :tags: Mobile

.. customcarditem::
   :header: Mobile Interpreter Workflow in Android and iOS
   :card_description: Learn how to use the mobile interpreter on iOS and Andriod devices.
   :image: ../_static/img/thumbnails/cropped/mobile.png
   :link: ../recipes/mobile_interpreter.html
   :tags: Mobile

.. customcarditem::
   :header: Profiling PyTorch RPC-Based Workloads
   :card_description: How to use the PyTorch profiler to profile RPC-based workloads.
   :image: ../_static/img/thumbnails/cropped/profile.png
   :link: ../recipes/distributed_rpc_profiling.html
   :tags: Production

.. Automatic Mixed Precision

.. customcarditem::
   :header: Automatic Mixed Precision
   :card_description: Use torch.cuda.amp to reduce runtime and save memory on NVIDIA GPUs.
   :image: ../_static/img/thumbnails/cropped/amp.png
   :link: ../recipes/recipes/amp_recipe.html
   :tags: Model-Optimization

.. Performance

.. customcarditem::
   :header: Performance Tuning Guide
   :card_description: Tips for achieving optimal performance.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/tuning_guide.html
   :tags: Model-Optimization

.. customcarditem::
   :header: PyTorch Inference Performance Tuning on AWS Graviton Processors
   :card_description: Tips for achieving the best inference performance on AWS Graviton CPUs
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/inference_tuning_on_aws_graviton.html
   :tags: Model-Optimization

.. Leverage Advanced Matrix Extensions

.. customcarditem::
   :header: Leverage Intel® Advanced Matrix Extensions
   :card_description: Learn to leverage Intel® Advanced Matrix Extensions.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/amx.html
   :tags: Model-Optimization

.. (beta) Compiling the Optimizer with torch.compile

.. customcarditem::
   :header: (beta) Compiling the Optimizer with torch.compile
   :card_description: Speed up the optimizer using torch.compile
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/compiling_optimizer.html
   :tags: Model-Optimization

.. Intel(R) Extension for PyTorch*

.. customcarditem::
   :header: Intel® Extension for PyTorch*
   :card_description: Introduction of Intel® Extension for PyTorch*
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/intel_extension_for_pytorch.html
   :tags: Model-Optimization

.. Intel(R) Neural Compressor for PyTorch*

.. customcarditem::
   :header: Intel® Neural Compressor for PyTorch
   :card_description: Ease-of-use quantization for PyTorch with Intel® Neural Compressor.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/intel_neural_compressor_for_pytorch.html
   :tags: Quantization,Model-Optimization

.. Distributed Training

.. customcarditem::
   :header: Getting Started with DeviceMesh
   :card_description: Learn how to use DeviceMesh
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/distributed_device_mesh.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Shard Optimizer States with ZeroRedundancyOptimizer
   :card_description: How to use ZeroRedundancyOptimizer to reduce memory consumption.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/zero_redundancy_optimizer.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Direct Device-to-Device Communication with TensorPipe RPC
   :card_description: How to use RPC with direct GPU-to-GPU communication.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/cuda_rpc.html
   :tags: Distributed-Training

.. customcarditem::
   :header: Distributed Optimizer with TorchScript support
   :card_description: How to enable TorchScript support for Distributed Optimizer.
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/distributed_optim_torchscript.html
   :tags: Distributed-Training,TorchScript

.. customcarditem::
   :header: Getting Started with Distributed Checkpoint (DCP)
   :card_description: Learn how to checkpoint distributed models with Distributed Checkpoint package.
   :image: ../_static/img/thumbnails/cropped/Getting-Started-with-DCP.png
   :link: ../recipes/distributed_checkpoint_recipe.html
   :tags: Distributed-Training

.. TorchServe

.. customcarditem::
   :header: Deploying a PyTorch Stable Diffusion model as a Vertex AI Endpoint
   :card_description: Learn how to deploy model in Vertex AI with TorchServe
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torchserve_vertexai_tutorial.html
   :tags: Production

.. End of tutorial card section

.. raw:: html

    </div>

    <div class="pagination d-flex justify-content-center"></div>

    </div>

    </div>

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :hidden:

   /recipes/recipes/loading_data_recipe
   /recipes/recipes/defining_a_neural_network
   /recipes/torch_logs
   /recipes/recipes/what_is_state_dict
   /recipes/recipes/saving_and_loading_models_for_inference
   /recipes/recipes/saving_and_loading_a_general_checkpoint
   /recipes/recipes/saving_multiple_models_in_one_file
   /recipes/recipes/warmstarting_model_using_parameters_from_a_different_model
   /recipes/recipes/save_load_across_devices
   /recipes/recipes/zeroing_out_gradients
   /recipes/recipes/profiler_recipe
   /recipes/recipes/profile_with_itt
   /recipes/recipes/Captum_Recipe
   /recipes/recipes/tensorboard_with_pytorch
   /recipes/recipes/dynamic_quantization
   /recipes/recipes/amp_recipe
   /recipes/recipes/tuning_guide
   /recipes/recipes/intel_extension_for_pytorch
   /recipes/compiling_optimizer
   /recipes/torch_compile_backend_ipex
   /recipes/torchscript_inference
   /recipes/deployment_with_flask
   /recipes/distributed_rpc_profiling
   /recipes/zero_redundancy_optimizer
   /recipes/cuda_rpc
   /recipes/distributed_optim_torchscript
   /recipes/mobile_interpreter
