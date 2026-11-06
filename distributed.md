# Distributed

Distributed training is a model training paradigm that involves
spreading training workload across multiple worker nodes, therefore
significantly improving the speed of training and model accuracy. While
distributed training can be used for any type of ML model training, it
is most beneficial to use it for large models and compute demanding
tasks as deep learning.

There are a few ways you can perform distributed training in
PyTorch with each method having their advantages in certain use cases:

- DistributedDataParallel (DDP)
- Fully Sharded Data Parallel (FSDP2)
- Tensor Parallel (TP)
- Device Mesh
- Monarch Framework
- Custom Extensions

Read more about these options in [Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html?utm_source=distr_landing).

## Learn DDP

DDP Intro Video Tutorials

A step-by-step video series on how to get started with
DistributedDataParallel and advance to more complex topics

Code Video

[https://pytorch.org/tutorials/beginner/ddp_series_intro.html?utm_source=distr_landing&utm_medium=ddp_series_intro](https://pytorch.org/tutorials/beginner/ddp_series_intro.html?utm_source=distr_landing&utm_medium=ddp_series_intro)

Getting Started with Distributed Data Parallel

This tutorial provides a short and gentle intro to the PyTorch
DistributedData Parallel.

Code

[https://pytorch.org/tutorials/intermediate/ddp_tutorial.html?utm_source=distr_landing&utm_medium=intermediate_ddp_tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html?utm_source=distr_landing&utm_medium=intermediate_ddp_tutorial)

Distributed Training with Uneven Inputs Using
the Join Context Manager

This tutorial describes the Join context manager and
demonstrates it's use with DistributedData Parallel.

Code

[https://pytorch.org/tutorials/advanced/generic_join.html?utm_source=distr_landing&utm_medium=generic_join](https://pytorch.org/tutorials/advanced/generic_join.html?utm_source=distr_landing&utm_medium=generic_join)

## Learn FSDP2

Getting Started with FSDP2

This tutorial demonstrates how you can perform distributed training
with FSDP2 on a transformer model

Code

[https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html?utm_source=distr_landing&utm_medium=FSDP_getting_started](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html?utm_source=distr_landing&utm_medium=FSDP_getting_started)

## Learn Tensor Parallel (TP)

Large Scale Transformer model training with Tensor Parallel (TP)

This tutorial demonstrates how to train a large Transformer-like model across hundreds to thousands of GPUs using Tensor Parallel and Fully Sharded Data Parallel.

Code

[https://pytorch.org/tutorials/intermediate/TP_tutorial.html](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)

## Learn DeviceMesh

Getting Started with DeviceMesh

In this tutorial you will learn about DeviceMesh
and how it can help with distributed training.

Code

[https://pytorch.org/tutorials/recipes/distributed_device_mesh.html?highlight=devicemesh](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html?highlight=devicemesh)

## Learn Monarch

Interactive Distributed Applications with Monarch

Learn how to use Monarch's actor framework

Code

[https://docs.pytorch.org/tutorials/intermediate/monarch_distributed_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/monarch_distributed_tutorial.html)

## Custom Extensions

Customize Process Group Backends Using Cpp Extensions

In this tutorial you will learn to implement a custom ProcessGroup
backend and plug that into PyTorch distributed package using
cpp extensions.

Code

[https://pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html?utm_source=distr_landing&utm_medium=custom_extensions_cpp](https://pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html?utm_source=distr_landing&utm_medium=custom_extensions_cpp)