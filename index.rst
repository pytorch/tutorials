Welcome to PyTorch Tutorials
============================

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: The 60 min blitz is the most common starting point and provides a broad view on how to use PyTorch. It covers the basics all the way to constructing deep neural networks.
   :header: New to PyTorch?
   :button_link: beginner/deep_learning_60min_blitz.html
   :button_text: Start 60-min blitz

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
   :header: Deep Learning with PyTorch: A 60 Minute Blitz
   :card_description: Understand PyTorch’s Tensor library and neural networks at a high level.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: beginner/deep_learning_60min_blitz.html
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
   :header: Visualizing Models, Data, and Training with Tensorboard
   :card_description: Learn to use TensorBoard to visualize data and model training.
   :image: _static/img/thumbnails/cropped/visualizing-with-tensorboard.png
   :link: intermediate/tensorboard_tutorial.html
   :tags: Interpretability,Getting-Started,Tensorboard

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

.. Audio

.. customcarditem::
   :header: torchaudio Tutorial
   :card_description: Learn to load and preprocess data from a simple dataset with PyTorch's torchaudio library.
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_preprocessing_tutorial.html
   :tags: Audio
   
.. customcarditem::
   :header: Speech Command Recognition
   :card_description: Learn how to correctly format an audio dataset and then train/test an audio classifier network on the dataset.
   :image: _static/img/thumbnails/cropped/torchaudio-speech.png
   :link: intermediate/speech_command_recognition_with_torchaudio.html
   :tags: Audio

.. Text

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
   :card_description: This is the third and final tutorial on doing “NLP From Scratch”, where we write our own classes and functions to preprocess the data to do our NLP modeling tasks.
   :image: _static/img/thumbnails/cropped/Text-Classification-with-TorchText.png
   :link: beginner/text_sentiment_ngrams_tutorial.html
   :tags: Text

.. customcarditem::
   :header: Language Translation with Torchtext
   :card_description: Use torchtext to reprocess data from a well-known datasets containing both English and German. Then use it to train a sequence-to-sequence model.
   :image: _static/img/thumbnails/cropped/Language-Translation-with-TorchText.png
   :link: beginner/torchtext_translation_tutorial.html
   :tags: Text

.. Reinforcement Learning

.. customcarditem::
   :header: Reinforcement Learning (DQN)
   :card_description: Learn how to use PyTorch to train a Deep Q Learning (DQN) agent on the CartPole-v0 task from the OpenAI Gym.
   :image: _static/img/cartpole.gif
   :link: intermediate/reinforcement_q_learning.html
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
   :tags: Production

.. customcarditem::
   :header: Loading a TorchScript Model in C++
   :card_description:  Learn how PyTorch provides to go from an existing Python model to a serialized representation that can be loaded and executed purely from C++, with no dependency on Python.
   :image: _static/img/thumbnails/cropped/Loading-a-TorchScript-Model-in-Cpp.png
   :link: advanced/cpp_export.html
   :tags: Production

.. customcarditem::
   :header: (optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime
   :card_description:  Convert a model defined in PyTorch into the ONNX format and then run it with ONNX Runtime.
   :image: _static/img/thumbnails/cropped/optional-Exporting-a-Model-from-PyTorch-to-ONNX-and-Running-it-using-ONNX-Runtime.png
   :link: advanced/super_resolution_with_onnxruntime.html
   :tags: Production

.. Frontend APIs

.. customcarditem::
   :header: (prototype) Introduction to Named Tensors in PyTorch
   :card_description: Learn how to use PyTorch to train a Deep Q Learning (DQN) agent on the CartPole-v0 task from the OpenAI Gym.
   :image: _static/img/thumbnails/cropped/experimental-Introduction-to-Named-Tensors-in-PyTorch.png
   :link: intermediate/named_tensor_tutorial.html
   :tags: Frontend-APIs,Named-Tensor,Best-Practice

.. customcarditem::
   :header: (beta) Channels Last Memory Format in PyTorch
   :card_description: Get an overview of Channels Last memory format and understand how it is used to order NCHW tensors in memory preserving dimensions.
   :image: _static/img/thumbnails/cropped/experimental-Channels-Last-Memory-Format-in-PyTorch.png
   :link: intermediate/memory_format_tutorial.html
   :tags: Memory-Format,Best-Practice

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
   :tags: Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Operators
   :card_description:  Implement a custom TorchScript operator in C++, how to build it into a shared library, how to use it in Python to define TorchScript models and lastly how to load it into a C++ application for inference workloads.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Operators.png
   :link: advanced/torch_script_custom_ops.html
   :tags: Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Classes
   :card_description: This is a continuation of the custom operator tutorial, and introduces the API we’ve built for binding C++ classes into TorchScript and Python simultaneously.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Classes.png
   :link: advanced/torch_script_custom_classes.html
   :tags: Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Dynamic Parallelism in TorchScript
   :card_description: This tutorial introduces the syntax for doing *dynamic inter-op parallelism* in TorchScript.
   :image: _static/img/thumbnails/cropped/TorchScript-Parallelism.jpg
   :link: advanced/torch-script-parallelism.html
   :tags: Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Autograd in C++ Frontend
   :card_description: The autograd package helps build flexible and dynamic nerural netorks. In this tutorial, exploreseveral examples of doing autograd in PyTorch C++ frontend
   :image: _static/img/thumbnails/cropped/Autograd-in-Cpp-Frontend.png
   :link: advanced/cpp_autograd.html
   :tags: Frontend-APIs,C++

.. Model Optimization

.. customcarditem::
   :header: Hyperparameter Tuning Tutorial
   :card_description: Learn how to use Ray Tune to find the best performing set of hyperparameters for your model.
   :image: _static/img/ray-tune.png
   :link: beginner/hyperparameter_tuning_tutorial.html
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
   :header: (beta) Static Quantization with Eager Mode in PyTorch
   :card_description: Learn techniques to impove a model's accuracy =  post-training static quantization, per-channel quantization, and quantization-aware training.
   :image: _static/img/thumbnails/cropped/experimental-Static-Quantization-with-Eager-Mode-in-PyTorch.png
   :link: advanced/static_quantization_tutorial.html
   :tags: Image/Video,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Quantized Transfer Learning for Computer Vision Tutorial
   :card_description: Learn techniques to impove a model's accuracy -  post-training static quantization, per-channel quantization, and quantization-aware training.
   :image: _static/img/thumbnails/cropped/experimental-Quantized-Transfer-Learning-for-Computer-Vision-Tutorial.png
   :link: advanced/static_quantization_tutorial.html
   :tags: Image/Video,Quantization,Model-Optimization

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
   :description: A set of examples around pytorch in Vision, Text, Reinforcement Learning, etc.
   :button_link: https://github.com/pytorch/examples
   :button_text: Checkout Examples

.. customcalloutitem::
   :header: PyTorch Cheat Sheet
   :description: Quick overview to essential PyTorch elements.
   :button_link: beginner/ptcheat.html
   :button_text: Download

.. customcalloutitem::
   :header: Tutorials on GitHub
   :description: Access PyTorch Tutorials from GitHub.
   :button_link: https://github.com/pytorch/tutorials
   :button_text: Go To GitHub


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

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Audio

   beginner/audio_preprocessing_tutorial
   intermediate/speech_command_recognition_with_torchaudio
   

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
   :caption: Deploying PyTorch Models in Production

   intermediate/flask_rest_api_tutorial
   beginner/Intro_to_TorchScript_tutorial
   advanced/cpp_export
   advanced/super_resolution_with_onnxruntime

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Frontend APIs

   intermediate/named_tensor_tutorial
   intermediate/memory_format_tutorial
   advanced/cpp_frontend
   advanced/cpp_extension
   advanced/torch_script_custom_ops
   advanced/torch_script_custom_classes
   advanced/torch-script-parallelism
   advanced/cpp_autograd
   advanced/dispatcher

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Model Optimization

   beginner/hyperparameter_tuning_tutorial
   intermediate/pruning_tutorial
   advanced/dynamic_quantization_tutorial
   intermediate/dynamic_quantization_bert_tutorial
   advanced/static_quantization_tutorial
   intermediate/quantized_transfer_learning_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Parallel and Distributed Training

   beginner/dist_overview
   intermediate/model_parallel_tutorial
   intermediate/ddp_tutorial
   intermediate/dist_tuto
   intermediate/rpc_tutorial
   intermediate/rpc_param_server_tutorial
   intermediate/dist_pipeline_parallel_tutorial
   intermediate/rpc_async_execution
   advanced/rpc_ddp_tutorial
