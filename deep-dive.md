# Deep Dive

Focused on enhancing model performance, this section includes
tutorials on profiling, hyperparameter tuning, quantization,
and other techniques to optimize PyTorch models for better efficiency
and speed.

---

[#### Profiling PyTorch

Learn how to profile a PyTorch application

Profiling

![](_static/img/thumbnails/cropped/pytorch-logo.png)](beginner/profiler.html)

[#### Parametrizations Tutorial

Learn how to use torch.nn.utils.parametrize to put constraints on your parameters (e.g. make them orthogonal, symmetric positive definite, low-rank...)

Model-Optimization,Best-Practice

![](_static/img/thumbnails/cropped/parametrizations.png)](intermediate/parametrizations.html)

[#### Pruning Tutorial

Learn how to use torch.nn.utils.prune to sparsify your neural networks, and how to extend it to implement your own custom pruning technique.

Model-Optimization,Best-Practice

![](_static/img/thumbnails/cropped/Pruning-Tutorial.png)](intermediate/pruning_tutorial.html)

[#### Inductor CPU Backend Debugging and Profiling

Learn the usage, debugging and performance profiling for ``torch.compile`` with Inductor CPU backend.

Model-Optimization,inductor

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/inductor_debug_cpu.html)

[#### (beta) Implementing High-Performance Transformers with SCALED DOT PRODUCT ATTENTION

This tutorial explores the new torch.nn.functional.scaled_dot_product_attention and how it can be used to construct Transformer components.

Model-Optimization,Attention,Transformer

![](_static/img/thumbnails/cropped/pytorch-logo.png)](intermediate/scaled_dot_product_attention_tutorial.html)

[#### Knowledge Distillation in Convolutional Neural Networks

Learn how to improve the accuracy of lightweight models using more powerful models as teachers.

Model-Optimization,Image/Video

![](_static/img/thumbnails/cropped/knowledge_distillation_pytorch_logo.png)](beginner/knowledge_distillation_tutorial.html)

[#### (beta) Channels Last Memory Format in PyTorch

Get an overview of Channels Last memory format and understand how it is used to order NCHW tensors in memory preserving dimensions.

Memory-Format,Best-Practice,Frontend-APIs

![](_static/img/thumbnails/cropped/experimental-Channels-Last-Memory-Format-in-PyTorch.png)](intermediate/memory_format_tutorial.html)

[#### Forward-mode Automatic Differentiation

Learn how to use forward-mode automatic differentiation.

Frontend-APIs

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/forward_ad_usage.html)

[#### Jacobians, Hessians, hvp, vhp, and more

Learn how to compute advanced autodiff quantities using torch.func

Frontend-APIs

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/jacobians_hessians.html)

[#### Model Ensembling

Learn how to ensemble models using torch.vmap

Frontend-APIs

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/ensembling.html)

[#### Per-Sample-Gradients

Learn how to compute per-sample-gradients using torch.func

Frontend-APIs

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/per_sample_grads.html)

[#### Neural Tangent Kernels

Learn how to compute neural tangent kernels using torch.func

Frontend-APIs

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/neural_tangent_kernels.html)

[#### Using the PyTorch C++ Frontend

Walk through an end-to-end example of training a model with the C++ frontend by training a DCGAN - a kind of generative model - to generate images of MNIST digits.

Frontend-APIs,C++

![](_static/img/thumbnails/cropped/Using-the-PyTorch-Cpp-Frontend.png)](advanced/cpp_frontend.html)

[#### Autograd in C++ Frontend

The autograd package helps build flexible and dynamic nerural netorks. In this tutorial, exploreseveral examples of doing autograd in PyTorch C++ frontend

Frontend-APIs,C++

![](_static/img/thumbnails/cropped/Autograd-in-Cpp-Frontend.png)](advanced/cpp_autograd.html)