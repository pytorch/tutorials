Welcome to PyTorch Tutorials
============================

What's new in PyTorch tutorials?

* `Fast Transformer Inference with Better Transformer <https://pytorch.org/tutorials/beginner/bettertransformer_tutorial.html?utm_source=whats_new_tutorials&utm_medium=bettertransformer>`__
* `Introduction to TorchRec <https://pytorch.org/tutorials/intermediate/torchrec_tutorial.html?utm_source=whats_new_tutorials&utm_medium=torchrec>`__
* `Getting Started with Fully Sharded Data Parallel (FSDP) <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html?utm_source=whats_new_tutorials&utm_medium=FSDP>`__
* `Advanced model training with Fully Sharded Data Parallel (FSDP) <https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html?utm_source=whats_new_tutorials&utm_medium=FSDP_advanced>`__
* `Grokking PyTorch Intel CPU Performance from First Principles <https://pytorch.org/tutorials/intermediate/torchserve_with_ipex?utm_source=whats_new_tutorials&utm_medium=torchserve_ipex>`__
* `Customize Process Group Backends Using Cpp Extensions <https://pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html?utm_source=whats_new_tutorials&utm_medium=cpp_ext>`__
* `Forward-mode Automatic Differentiation <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html?utm_source=whats_new_tutorials&utm_medium=forward_ad>`__ (added functorch API capabilities)
* `Real Time Inference on Raspberry Pi 4 (30 fps!) <https://pytorch.org/tutorials/intermediate/realtime_rpi.html?utm_source=whats_new_tutorials&utm_medium=rpi>`__

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: Familiarize yourself with PyTorch concepts and modules. Learn how to load data, build deep neural networks, train and save your models in this quickstart guide.
   :header: Learn the Basics
   :button_link:  beginner/basics/intro.html
   :button_text: Get started with PyTorch

.. customcalloutitem::
   :description: Bite-size, ready-to-deploy PyTorch code examples.
   :header: PyTorch Recipes
   :button_link: recipes/recipes_index.html
   :button_text: Explore Recipes

.. End of callout item section

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

.. Add tutorial cards below this line

.. Learning PyTorch

.. customcarditem::
   :header: Learn the Basics
   :card_description: A step-by-step guide to building a complete ML workflow with PyTorch.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: beginner/basics/intro.html
   :tags: Getting-Started

.. customcarditem::
   :header: Introduction to PyTorch on YouTube
   :card_description: An introduction to building a complete ML workflow with PyTorch. Follows the PyTorch Beginner Series on YouTube.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: beginner/introyt.html
   :tags: Getting-Started

.. customcarditem::
   :header: Learning PyTorch with Examples
   :card_description: This tutorial introduces the fundamental concepts of PyTorch through self-contained examples.
   :image: _static/img/thumbnails/cropped/learning-pytorch-with-examples.png
   :link: beginner/pytorch_with_examples.html
   :tags: Getting-Started

.. customcarditem::
   :header: What is torch.nn really?
   :card_description: Use torch.nn to create and train a neural network.
   :image: _static/img/thumbnails/cropped/torch-nn.png
   :link: beginner/nn_tutorial.html
   :tags: Getting-Started

.. customcarditem::
   :header: Visualizing Models, Data, and Training with TensorBoard
   :card_description: Learn to use TensorBoard to visualize data and model training.
   :image: _static/img/thumbnails/cropped/visualizing-with-tensorboard.png
   :link: intermediate/tensorboard_tutorial.html
   :tags: Interpretability,Getting-Started,TensorBoard

.. Image/Video

.. customcarditem::
   :header: TorchVision Object Detection Finetuning Tutorial
   :card_description: Finetune a pre-trained Mask R-CNN model.
   :image: _static/img/thumbnails/cropped/TorchVision-Object-Detection-Finetuning-Tutorial.png
   :link: intermediate/torchvision_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Transfer Learning for Computer Vision Tutorial
   :card_description: Train a convolutional neural network for image classification using transfer learning.
   :image: _static/img/thumbnails/cropped/Transfer-Learning-for-Computer-Vision-Tutorial.png
   :link: beginner/transfer_learning_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Optimizing Vision Transformer Model
   :card_description: Apply cutting-edge, attention-based transformer models to computer vision tasks.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: beginner/vt_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Adversarial Example Generation
   :card_description: Train a convolutional neural network for image classification using transfer learning.
   :image: _static/img/thumbnails/cropped/Adversarial-Example-Generation.png
   :link: beginner/fgsm_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: DCGAN Tutorial
   :card_description: Train a generative adversarial network (GAN) to generate new celebrities.
   :image: _static/img/thumbnails/cropped/DCGAN-Tutorial.png
   :link: beginner/dcgan_faces_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Spatial Transformer Networks Tutorial
   :card_description: Learn how to augment your network using a visual attention mechanism.
   :image: _static/img/stn/Five.gif
   :link: intermediate/spatial_transformer_tutorial.html
   :tags: Image/Video

.. Audio

.. customcarditem::
   :header: Audio IO
   :card_description: Learn to load data with torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_io_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Resampling
   :card_description: Learn to resample audio waveforms using torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_resampling_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Data Augmentation
   :card_description: Learn to apply data augmentations using torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_data_augmentation_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Feature Extractions
   :card_description: Learn to extract features using torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_feature_extractions_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Feature Augmentation
   :card_description: Learn to augment features using torchaudio.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_feature_augmentation_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio Datasets
   :card_description: Learn to use torchaudio datasets.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_datasets_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Automatic Speech Recognition with Wav2Vec2 in torchaudio
   :card_description: Learn how to use torchaudio's pretrained models for building a speech recognition application.
   :image: _static/img/thumbnails/cropped/torchaudio-asr.png
   :link: intermediate/speech_recognition_pipeline_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Speech Command Classification
   :card_description: Learn how to correctly format an audio dataset and then train/test an audio classifier network on the dataset.
   :image: _static/img/thumbnails/cropped/torchaudio-speech.png
   :link: intermediate/speech_command_classification_with_torchaudio_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Text-to-Speech with torchaudio
   :card_description: Learn how to use torchaudio's pretrained models for building a text-to-speech application.
   :image: _static/img/thumbnails/cropped/torchaudio-speech.png
   :link: intermediate/text_to_speech_with_torchaudio.html
   :tags: Audio

.. customcarditem::
   :header: Forced Alignment with Wav2Vec2 in torchaudio
   :card_description: Learn how to use torchaudio's Wav2Vec2 pretrained models for aligning text to speech
   :image: _static/img/thumbnails/cropped/torchaudio-alignment.png
   :link: intermediate/forced_alignment_with_torchaudio_tutorial.html
   :tags: Audio

.. Text

.. customcarditem::
   :header: Fast Transformer Inference with Better Transformer
   :card_description: Deploy a PyTorch Transformer model using Better Transformer with high performance for inference
   :image: _static/img/thumbnails/cropped/pytorch-logo.png
   :link: beginner/bettertransformer_tutorial.html
   :tags: Production,Text

.. customcarditem::
   :header: Sequence-to-Sequence Modeling with nn.Transformer and torchtext
   :card_description: Learn how to train a sequence-to-sequence model that uses the nn.Transformer module.
   :image: _static/img/thumbnails/cropped/Sequence-to-Sequence-Modeling-with-nnTransformer-andTorchText.png
   :link: beginner/transformer_tutorial.html
   :tags: Text

.. customcarditem::
   :header: NLP from Scratch: Classifying Names with a Character-level RNN
   :card_description: Build and train a basic character-level RNN to classify word from scratch without the use of torchtext. First in a series of three tutorials.
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Classifying-Names-with-a-Character-Level-RNN.png
   :link: intermediate/char_rnn_classification_tutorial
   :tags: Text

.. customcarditem::
   :header: NLP from Scratch: Generating Names with a Character-level RNN
   :card_description: After using character-level RNN to classify names, leanr how to generate names from languages. Second in a series of three tutorials.
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Generating-Names-with-a-Character-Level-RNN.png
   :link: intermediate/char_rnn_generation_tutorial.html
   :tags: Text

.. customcarditem::
   :header: NLP from Scratch: Translation with a Sequence-to-sequence Network and Attention
   :card_description: This is the third and final tutorial on doing “NLP From Scratch”, where we write our own classes and functions to preprocess the data to do our NLP modeling tasks.
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Translation-with-a-Sequence-to-Sequence-Network-and-Attention.png
   :link: intermediate/seq2seq_translation_tutorial.html
   :tags: Text

.. customcarditem::
   :header: Text Classification with Torchtext
   :card_description: Learn how to build the dataset and classify text using torchtext library.
   :image: _static/img/thumbnails/cropped/Text-Classification-with-TorchText.png
   :link: beginner/text_sentiment_ngrams_tutorial.html
   :tags: Text

.. customcarditem::
   :header: Language Translation with Transformer
   :card_description: Train a language translation model from scratch using Transformer.
   :image: _static/img/thumbnails/cropped/Language-Translation-with-TorchText.png
   :link: beginner/translation_transformer.html
   :tags: Text


.. Reinforcement Learning

.. customcarditem::
   :header: Reinforcement Learning (DQN)
   :card_description: Learn how to use PyTorch to train a Deep Q Learning (DQN) agent on the CartPole-v0 task from the OpenAI Gym.
   :image: _static/img/cartpole.gif
   :link: intermediate/reinforcement_q_learning.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Train a Mario-playing RL Agent
   :card_description: Use PyTorch to train a Double Q-learning agent to play Mario.
   :image: _static/img/mario.gif
   :link: intermediate/mario_rl_tutorial.html
   :tags: Reinforcement-Learning


.. Deploying PyTorch Models in Production


.. customcarditem::
   :header: Deploying PyTorch in Python via a REST API with Flask
   :card_description: Deploy a PyTorch model using Flask and expose a REST API for model inference using the example of a pretrained DenseNet 121 model which detects the image.
   :image: _static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png
   :link: intermediate/flask_rest_api_tutorial.html
   :tags: Production

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
   :header: (optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime
   :card_description:  Convert a model defined in PyTorch into the ONNX format and then run it with ONNX Runtime.
   :image: _static/img/thumbnails/cropped/optional-Exporting-a-Model-from-PyTorch-to-ONNX-and-Running-it-using-ONNX-Runtime.png
   :link: advanced/super_resolution_with_onnxruntime.html
   :tags: Production

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

.. Frontend APIs

.. customcarditem::
   :header: (beta) Channels Last Memory Format in PyTorch
   :card_description: Get an overview of Channels Last memory format and understand how it is used to order NCHW tensors in memory preserving dimensions.
   :image: _static/img/thumbnails/cropped/experimental-Channels-Last-Memory-Format-in-PyTorch.png
   :link: intermediate/memory_format_tutorial.html
   :tags: Memory-Format,Best-Practice,Frontend-APIs

.. customcarditem::
   :header: Using the PyTorch C++ Frontend
   :card_description: Walk through an end-to-end example of training a model with the C++ frontend by training a DCGAN – a kind of generative model – to generate images of MNIST digits.
   :image: _static/img/thumbnails/cropped/Using-the-PyTorch-Cpp-Frontend.png
   :link: advanced/cpp_frontend.html
   :tags: Frontend-APIs,C++

.. customcarditem::
   :header: Custom C++ and CUDA Extensions
   :card_description:  Create a neural network layer with no parameters using numpy. Then use scipy to create a neural network layer that has learnable weights.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/cpp_extension.html
   :tags: Extending-PyTorch,Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Operators
   :card_description:  Implement a custom TorchScript operator in C++, how to build it into a shared library, how to use it in Python to define TorchScript models and lastly how to load it into a C++ application for inference workloads.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Operators.png
   :link: advanced/torch_script_custom_ops.html
   :tags: Extending-PyTorch,Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Classes
   :card_description: This is a continuation of the custom operator tutorial, and introduces the API we’ve built for binding C++ classes into TorchScript and Python simultaneously.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Classes.png
   :link: advanced/torch_script_custom_classes.html
   :tags: Extending-PyTorch,Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Dynamic Parallelism in TorchScript
   :card_description: This tutorial introduces the syntax for doing *dynamic inter-op parallelism* in TorchScript.
   :image: _static/img/thumbnails/cropped/TorchScript-Parallelism.jpg
   :link: advanced/torch-script-parallelism.html
   :tags: Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Real Time Inference on Raspberry Pi 4
   :card_description: This tutorial covers how to run quantized and fused models on a Raspberry Pi 4 at 30 fps.
   :image: _static/img/thumbnails/cropped/realtime_rpi.png
   :link: intermediate/realtime_rpi.html
   :tags: TorchScript,Model-Optimization,Image/Video,Quantization

.. customcarditem::
   :header: Autograd in C++ Frontend
   :card_description: The autograd package helps build flexible and dynamic nerural netorks. In this tutorial, exploreseveral examples of doing autograd in PyTorch C++ frontend
   :image: _static/img/thumbnails/cropped/Autograd-in-Cpp-Frontend.png
   :link: advanced/cpp_autograd.html
   :tags: Frontend-APIs,C++

.. customcarditem::
   :header: Registering a Dispatched Operator in C++
   :card_description: The dispatcher is an internal component of PyTorch which is responsible for figuring out what code should actually get run when you call a function like torch::add.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/dispatcher.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Extending Dispatcher For a New Backend in C++
   :card_description: Learn how to extend the dispatcher to add a new device living outside of the pytorch/pytorch repo and maintain it to keep in sync with native PyTorch devices.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/extend_dispatcher.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Custom Function Tutorial: Double Backward
   :card_description: Learn how to write a custom autograd Function that supports double backward.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/custom_function_double_backward_tutorial.html
   :tags: Extending-PyTorch,Frontend-APIs

.. customcarditem::
   :header: Custom Function Tutorial: Fusing Convolution and Batch Norm
   :card_description: Learn how to create a custom autograd Function that fuses batch norm into a convolution to improve memory usage.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/custom_function_conv_bn_tutorial.html
   :tags: Extending-PyTorch,Frontend-APIs

.. customcarditem::
   :header: Forward-mode Automatic Differentiation
   :card_description: Learn how to use forward-mode automatic differentiation.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/forward_ad_usage.html
   :tags: Frontend-APIs

.. Model Optimization

.. customcarditem::
   :header: Performance Profiling in PyTorch
   :card_description: Learn how to use the PyTorch Profiler to benchmark your module's performance.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: beginner/profiler.html
   :tags: Model-Optimization,Best-Practice,Profiling

.. customcarditem::
   :header: Performance Profiling in TensorBoard
   :card_description: Learn how to use the TensorBoard plugin to profile and analyze your model's performance.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: intermediate/tensorboard_profiler_tutorial.html
   :tags: Model-Optimization,Best-Practice,Profiling,TensorBoard

.. customcarditem::
   :header: Hyperparameter Tuning Tutorial
   :card_description: Learn how to use Ray Tune to find the best performing set of hyperparameters for your model.
   :image: _static/img/ray-tune.png
   :link: beginner/hyperparameter_tuning_tutorial.html
   :tags: Model-Optimization,Best-Practice

 .. customcarditem::
    :header: Optimizing Vision Transformer Model
    :card_description: Learn how to use Facebook Data-efficient Image Transformers DeiT and script and optimize it for mobile.
    :image: _static/img/thumbnails/cropped/mobile.png
    :link: beginner/vt_tutorial.html
    :tags: Model-Optimization,Best-Practice,Mobile

.. customcarditem::
   :header: Parametrizations Tutorial
   :card_description: Learn how to use torch.nn.utils.parametrize to put constriants on your parameters (e.g. make them orthogonal, symmetric positive definite, low-rank...)
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
   :header: (beta) Dynamic Quantization on an LSTM Word Language Model
   :card_description: Apply dynamic quantization, the easiest form of quantization, to a LSTM-based next word prediction model.
   :image: _static/img/thumbnails/cropped/experimental-Dynamic-Quantization-on-an-LSTM-Word-Language-Model.png
   :link: advanced/dynamic_quantization_tutorial.html
   :tags: Text,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Dynamic Quantization on BERT
   :card_description: Apply the dynamic quantization on a BERT (Bidirectional Embedding Representations from Transformers) model.
   :image: _static/img/thumbnails/cropped/experimental-Dynamic-Quantization-on-BERT.png
   :link: intermediate/dynamic_quantization_bert_tutorial.html
   :tags: Text,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Quantized Transfer Learning for Computer Vision Tutorial
   :card_description: Extends the Transfer Learning for Computer Vision Tutorial using a quantized model.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: intermediate/quantized_transfer_learning_tutorial.html
   :tags: Image/Video,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Static Quantization with Eager Mode in PyTorch
   :card_description: This tutorial shows how to do post-training static quantization.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: advanced/static_quantization_tutorial.html
   :tags: Quantization

.. customcarditem::
   :header: Grokking PyTorch Intel CPU Performance from First Principles
   :card_description: A case study on the TorchServe inference framework optimized with Intel® Extension for PyTorch.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/torchserve_with_ipex
   :tags: Model-Optimization,Production

.. customcarditem::
   :header: Introduction to nvFuser
   :card_description: An introduction to nvFuser
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/nvfuser_intro_tutorial.html
   :tags: Model-Optimization

.. customcarditem::
   :header: Multi-Objective Neural Architecture Search with Ax
   :card_description: Learn how to use Ax to search over architectures find optimal tradeoffs between accuracy and latency.
   :image: _static/img/ax_logo.png
   :link: intermediate/ax_multiobjective_nas_tutorial.html
   :tags: Model-Optimization,Best-Practice,Ax,TorchX

.. Parallel-and-Distributed-Training

.. customcarditem::
   :header: PyTorch Distributed Overview
   :card_description: Briefly go over all concepts and features in the distributed package. Use this document to find the distributed training technology that can best serve your application.
   :image: _static/img/thumbnails/cropped/PyTorch-Distributed-Overview.png
   :link: beginner/dist_overview.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Single-Machine Model Parallel Best Practices
   :card_description:  Learn how to implement model parallel, a distributed training technique which splits a single model onto different GPUs, rather than replicating the entire model on each GPU
   :image: _static/img/thumbnails/cropped/Model-Parallel-Best-Practices.png
   :link: intermediate/model_parallel_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Getting Started with Distributed Data Parallel
   :card_description: Learn the basics of when to use distributed data paralle versus data parallel and work through an example to set it up.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-Distributed-Data-Parallel.png
   :link: intermediate/ddp_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Writing Distributed Applications with PyTorch
   :card_description: Set up the distributed package of PyTorch, use the different communication strategies, and go over some the internals of the package.
   :image: _static/img/thumbnails/cropped/Writing-Distributed-Applications-with-PyTorch.png
   :link: intermediate/dist_tuto.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Customize Process Group Backends Using Cpp Extensions
   :card_description: Extend ProcessGroup with custom collective communication implementations.
   :image: _static/img/thumbnails/cropped/Customize-Process-Group-Backends-Using-Cpp-Extensions.png
   :link: intermediate/process_group_cpp_extension_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Getting Started with Distributed RPC Framework
   :card_description: Learn how to build distributed training using the torch.distributed.rpc package.
   :image: _static/img/thumbnails/cropped/Getting Started with Distributed-RPC-Framework.png
   :link: intermediate/rpc_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Implementing a Parameter Server Using Distributed RPC Framework
   :card_description: Walk through a through a simple example of implementing a parameter server using PyTorch’s Distributed RPC framework.
   :image: _static/img/thumbnails/cropped/Implementing-a-Parameter-Server-Using-Distributed-RPC-Framework.png
   :link: intermediate/rpc_param_server_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Distributed Pipeline Parallelism Using RPC
   :card_description: Demonstrate how to implement distributed pipeline parallelism using RPC
   :image: _static/img/thumbnails/cropped/Distributed-Pipeline-Parallelism-Using-RPC.png
   :link: intermediate/dist_pipeline_parallel_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Implementing Batch RPC Processing Using Asynchronous Executions
   :card_description: Learn how to use rpc.functions.async_execution to implement batch RPC
   :image: _static/img/thumbnails/cropped/Implementing-Batch-RPC-Processing-Using-Asynchronous-Executions.png
   :link: intermediate/rpc_async_execution.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Combining Distributed DataParallel with Distributed RPC Framework
   :card_description: Walk through a through a simple example of how to combine distributed data parallelism with distributed model parallelism.
   :image: _static/img/thumbnails/cropped/Combining-Distributed-DataParallel-with-Distributed-RPC-Framework.png
   :link: advanced/rpc_ddp_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Training Transformer models using Pipeline Parallelism
   :card_description: Walk through a through a simple example of how to train a transformer model using pipeline parallelism.
   :image: _static/img/thumbnails/cropped/Training-Transformer-models-using-Pipeline-Parallelism.png
   :link: intermediate/pipeline_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Training Transformer models using Distributed Data Parallel and Pipeline Parallelism
   :card_description: Walk through a through a simple example of how to train a transformer model using Distributed Data Parallel and Pipeline Parallelism
   :image: _static/img/thumbnails/cropped/Training-Transformer-Models-using-Distributed-Data-Parallel-and-Pipeline-Parallelism.png
   :link: advanced/ddp_pipeline.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Getting Started with Fully Sharded Data Parallel(FSDP)
   :card_description: Learn how to train models with Fully Sharded Data Parallel package.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-FSDP.png
   :link: intermediate/FSDP_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Advanced Model Training with Fully Sharded Data Parallel (FSDP)
   :card_description: Explore advanced model training with Fully Sharded Data Parallel package.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-FSDP.png
   :link: intermediate/FSDP_adavnced_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. Mobile

.. customcarditem::
   :header: Image Segmentation DeepLabV3 on iOS
   :card_description: A comprehensive step-by-step tutorial on how to prepare and run the PyTorch DeepLabV3 image segmentation model on iOS.
   :image: _static/img/thumbnails/cropped/ios.png
   :link: beginner/deeplabv3_on_ios.html
   :tags: Mobile

.. customcarditem::
   :header: Image Segmentation DeepLabV3 on Android
   :card_description: A comprehensive step-by-step tutorial on how to prepare and run the PyTorch DeepLabV3 image segmentation model on Android.
   :image: _static/img/thumbnails/cropped/android.png
   :link: beginner/deeplabv3_on_android.html
   :tags: Mobile

.. Recommendation Systems

.. customcarditem::
   :header: Introduction to TorchRec
   :card_description: TorchRec is a PyTorch domain library built to provide common sparsity & parallelism primitives needed for large-scale recommender systems.
   :image: _static/img/thumbnails/torchrec.png
   :link: intermediate/torchrec_tutorial.html
   :tags: TorchRec,Recommender

.. customcarditem::
   :header: Exploring TorchRec sharding
   :card_description: This tutorial covers the sharding schemes of embedding tables by using <code>EmbeddingPlanner</code> and <code>DistributedModelParallel</code> API.
   :image: _static/img/thumbnails/torchrec.png
   :link: advanced/sharding.html
   :tags: TorchRec,Recommender


.. End of tutorial card section

.. raw:: html

    </div>

    <div class="pagination d-flex justify-content-center"></div>

    </div>

    </div>
    <br>
    <br>


Additional Resources
============================

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :header: Examples of PyTorch
   :description: A set of examples around PyTorch in Vision, Text, Reinforcement Learning that you can incorporate in your existing work.
   :button_link: https://pytorch.org/examples?utm_source=examples&utm_medium=examples-landing
   :button_text: Check Out Examples

.. customcalloutitem::
   :header: PyTorch Cheat Sheet
   :description: Quick overview to essential PyTorch elements.
   :button_link: beginner/ptcheat.html
   :button_text: Open

.. customcalloutitem::
   :header: Tutorials on GitHub
   :description: Access PyTorch Tutorials from GitHub.
   :button_link: https://github.com/pytorch/tutorials
   :button_text: Go To GitHub

.. customcalloutitem::
   :header: Run Tutorials on Google Colab
   :description: Learn how to copy tutorial data into Google Drive so that you can run tutorials on Google Colab.
   :button_link: beginner/colab.html
   :button_text: Open

.. End of callout section

.. raw:: html

        </div>
    </div>

    <div style='clear:both'></div>

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: PyTorch Recipes

   See All Recipes <recipes/recipes_index>
   See All Prototype Recipes <prototype/prototype_index>

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: Introduction to PyTorch

   beginner/basics/intro
   beginner/basics/quickstart_tutorial
   beginner/basics/tensorqs_tutorial
   beginner/basics/data_tutorial
   beginner/basics/transforms_tutorial
   beginner/basics/buildmodel_tutorial
   beginner/basics/autogradqs_tutorial
   beginner/basics/optimization_tutorial
   beginner/basics/saveloadrun_tutorial

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: Introduction to PyTorch on YouTube

   beginner/introyt
   beginner/introyt/introyt1_tutorial
   beginner/introyt/tensors_deeper_tutorial
   beginner/introyt/autogradyt_tutorial
   beginner/introyt/modelsyt_tutorial
   beginner/introyt/tensorboardyt_tutorial
   beginner/introyt/trainingyt
   beginner/introyt/captumyt

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
   :caption: Image and Video

   intermediate/torchvision_tutorial
   beginner/transfer_learning_tutorial
   beginner/fgsm_tutorial
   beginner/dcgan_faces_tutorial
   intermediate/spatial_transformer_tutorial
   beginner/vt_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Audio

   beginner/audio_io_tutorial
   beginner/audio_resampling_tutorial
   beginner/audio_data_augmentation_tutorial
   beginner/audio_feature_extractions_tutorial
   beginner/audio_feature_augmentation_tutorial
   beginner/audio_datasets_tutorial
   intermediate/speech_recognition_pipeline_tutorial
   intermediate/speech_command_classification_with_torchaudio_tutorial
   intermediate/text_to_speech_with_torchaudio
   intermediate/forced_alignment_with_torchaudio_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Text

   beginner/transformer_tutorial
   beginner/bettertransformer_tutorial
   intermediate/char_rnn_classification_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/seq2seq_translation_tutorial
   beginner/text_sentiment_ngrams_tutorial
   beginner/translation_transformer


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: MaskedTensor

   beginner/maskedtensor_overview
   beginner/maskedtensor_sparsity
   beginner/maskedtensor_distinguish_gradient


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Reinforcement Learning

   intermediate/reinforcement_q_learning
   intermediate/mario_rl_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Deploying PyTorch Models in Production

   intermediate/flask_rest_api_tutorial
   beginner/Intro_to_TorchScript_tutorial
   advanced/cpp_export
   advanced/super_resolution_with_onnxruntime
   intermediate/realtime_rpi

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Code Transforms with FX

   intermediate/fx_conv_bn_fuser
   intermediate/fx_profiling_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Frontend APIs

   intermediate/memory_format_tutorial
   intermediate/forward_ad_usage
   advanced/cpp_frontend
   advanced/torch-script-parallelism
   advanced/cpp_autograd

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Extending PyTorch

   intermediate/custom_function_double_backward_tutorial
   intermediate/custom_function_conv_bn_tutorial
   advanced/cpp_extension
   advanced/torch_script_custom_ops
   advanced/torch_script_custom_classes
   advanced/dispatcher
   advanced/extend_dispatcher

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Model Optimization

   beginner/profiler
   intermediate/tensorboard_profiler_tutorial
   beginner/hyperparameter_tuning_tutorial
   beginner/vt_tutorial
   intermediate/parametrizations
   intermediate/pruning_tutorial
   advanced/dynamic_quantization_tutorial
   intermediate/dynamic_quantization_bert_tutorial
   intermediate/quantized_transfer_learning_tutorial
   advanced/static_quantization_tutorial
   intermediate/torchserve_with_ipex
   intermediate/nvfuser_intro_tutorial
   intermediate/ax_multiobjective_nas_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Parallel and Distributed Training

   beginner/dist_overview
   intermediate/model_parallel_tutorial
   intermediate/ddp_tutorial
   intermediate/dist_tuto
   intermediate/FSDP_tutorial
   intermediate/FSDP_adavnced_tutorial
   intermediate/process_group_cpp_extension_tutorial
   intermediate/rpc_tutorial
   intermediate/rpc_param_server_tutorial
   intermediate/dist_pipeline_parallel_tutorial
   intermediate/rpc_async_execution
   advanced/rpc_ddp_tutorial
   intermediate/pipeline_tutorial
   advanced/ddp_pipeline
   advanced/generic_join

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Mobile

   beginner/deeplabv3_on_ios
   beginner/deeplabv3_on_android

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Recommendation Systems

   intermediate/torchrec_tutorial
   advanced/sharding
