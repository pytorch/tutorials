# PyTorch Distributed Overview

**Author**: [Will Constable](https://github.com/wconstab/), [Wei Feng](https://github.com/weifengpy)

Note

[![edit](../_images/pencil-16.png)](../_images/pencil-16.png) View and edit this tutorial in [github](https://github.com/pytorch/tutorials/blob/main/beginner_source/dist_overview.rst).

This is the overview page for the `torch.distributed` package. The goal of
this page is to categorize documents into different topics and briefly
describe each of them. If this is your first time building distributed training
applications using PyTorch, it is recommended to use this document to navigate
to the technology that can best serve your use case.

## Introduction

The PyTorch Distributed library includes a collective of parallelism modules,
a communications layer, and infrastructure for launching and
debugging large training jobs.

### Parallelism APIs

These Parallelism Modules offer high-level functionality and compose with existing models:

- [Distributed Data-Parallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [Fully Sharded Data-Parallel Training (FSDP2)](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [Tensor Parallel (TP)](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [Pipeline Parallel (PP)](https://pytorch.org/docs/main/distributed.pipelining.html)

### Sharding primitives

`DTensor` and `DeviceMesh` are primitives used to build parallelism in terms of sharded or replicated tensors on N-dimensional process groups.

- [DTensor](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md) represents a tensor that is sharded and/or replicated, and communicates automatically to reshard tensors as needed by operations.
- [DeviceMesh](https://pytorch.org/docs/stable/distributed.html#devicemesh) abstracts the accelerator device communicators into a multi-dimensional array, which manages the underlying `ProcessGroup` instances for collective communications in multi-dimensional parallelisms. Try out our [Device Mesh Recipe](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html) to learn more.

### Communications APIs

The [PyTorch distributed communication layer (C10D)](https://pytorch.org/docs/stable/distributed.html) offers both collective communication APIs (e.g., [all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)

and [all_gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather))
and P2P communication APIs (e.g.,
[send](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send)
and [isend](https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend)),
which are used under the hood in all of the parallelism implementations.
[Writing Distributed Applications with PyTorch](../intermediate/dist_tuto.html)
shows examples of using c10d communication APIs.

### Launcher

[torchrun](https://pytorch.org/docs/stable/elastic/run.html) is a widely-used launcher script, which spawns processes on the local and remote machines for running distributed PyTorch programs.

## Applying Parallelism To Scale Your Model

Data Parallelism is a widely adopted single-program multiple-data training paradigm
where the model is replicated on every process, every model replica computes local gradients for
a different set of input data samples, gradients are averaged within the data-parallel communicator group before each optimizer step.

Model Parallelism techniques (or Sharded Data Parallelism) are required when a model doesn't fit in GPU, and can be combined together to form multi-dimensional (N-D) parallelism techniques.

When deciding what parallelism techniques to choose for your model, use these common guidelines:

1. Use [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/notes/ddp.html),
if your model fits in a single GPU but you want to easily scale up training using multiple GPUs.

- Use [torchrun](https://pytorch.org/docs/stable/elastic/run.html), to launch multiple pytorch processes if you are using more than one node.
- See also: [Getting Started with Distributed Data Parallel](../intermediate/ddp_tutorial.html)
2. Use [FullyShardedDataParallel (FSDP2)](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) when your model cannot fit on one GPU.

- See also: [Getting Started with FSDP2](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
3. Use [Tensor Parallel (TP)](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) and/or [Pipeline Parallel (PP)](https://pytorch.org/docs/main/distributed.pipelining.html) if you reach scaling limitations with FSDP2.

- Try our [Tensor Parallelism Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
- See also: [TorchTitan end to end example of 3D parallelism](https://github.com/pytorch/torchtitan)

Note

Data-parallel training also works with [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus).

## PyTorch Distributed Developers

If you'd like to contribute to PyTorch Distributed, refer to our
[Developer Guide](https://github.com/pytorch/pytorch/blob/master/torch/distributed/CONTRIBUTING.md).