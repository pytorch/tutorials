PyTorch Distributed Overview
============================
**Author**: `Shen Li <https://mrshenli.github.io/>`_


This is the overview page for the ``torch.distributed`` package. As there are
more and more documents, examples and tutorials added at different locations,
it becomes unclear which document or tutorial to consult for a specific problem
or what is the best order to read these contents. The goal of this page is to
address this problem by categorizing documents into different topics and briefly
describe each of them. If this is your first time building distributed training
applications using PyTorch, it is recommended to use this document to navigate
to the technology that can best serve your use case.


Introduction
------------

As of PyTorch v1.6.0, features in ``torch.distributed`` can be categorized into
three main components:

* `Distributed Data-Parallel Training <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__
  (DDP) is a widely adopted single-program multiple-data training paradigm. With
  DDP, the model is replicated on every process, and every model replica will be
  fed with a different set of input data samples. DDP takes care of gradient
  communications to keep model replicas synchronized and overlaps it with the
  gradient computations to speed up training.
* `RPC-Based Distributed Training <https://pytorch.org/docs/master/rpc.html>`__
  (RPC) is developed to support general training structures that cannot fit into
  data-parallel training, such as distributed pipeline parallelism, parameter
  server paradigm, and combination of DDP with other training paradigms. It
  helps manage remote object lifetime and extend autograd engine to beyond
  machine boundaries.
* `Collective Communication <https://pytorch.org/docs/stable/distributed.html>`__
  (c10d) library support sending tensors across processes within a group. It
  offers both collective communication APIs (e.g.,
  `all_reduce <https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce>`__
  and `all_gather <https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather>`__)
  and P2P communication APIs (e.g.,
  `send <https://pytorch.org/docs/stable/distributed.html#torch.distributed.send>`__
  and `isend <https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend>`__).
  DDP and RPC (`ProcessGroup Backend <https://pytorch.org/docs/master/rpc.html#process-group-backend>`__)
  are built on c10d as of v1.6.0, where the former uses collective communications
  and the latter uses P2P communications. Usually, developers do not need to
  directly use this raw communication API, as DDP and RPC features above can serve
  many distributed training scenarios. However, there are use cases where this API
  is still helpful. One example would be distributed parameter averaging, where
  applications would like to compute the average values of all model parameters
  after the backward pass instead of using DDP to communicate gradients. This can
  decouple communications from computations and allow finer-grain control over
  what to communicate, but on the other hand, it also gives up the performance
  optimizations offered by DDP. The
  `Writing Distributed Applications with PyTorch <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`__
  shows examples of using c10d communication APIs.


Most of the existing documents are written for either DDP or RPC, the remainder
of this page will elaborate materials for these two components.


Data Parallel Training
----------------------

PyTorch provides several options for data-parallel training. For applications
that gradually grow from simple to complex and from prototype to production, the
common development trajectory would be:

1. Use single-device training, if the data and model can fit in one GPU, and the
   training speed is not a concern.
2. Use single-machine multi-GPU
   `DataParallel <https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html>`__,
   if there are multiple GPUs on the server, and you would like to speed up
   training with the minimum code change.
3. Use single-machine multi-GPU
   `DistributedDataParallel <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__,
   if you would like to further speed up training and are willing to write a
   little more code to set it up.
4. Use multi-machine `DistributedDataParallel <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__
   and the `launching script <https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md>`__,
   if the application needs to scale across machine boundaries.
5. Use `torchelastic <https://pytorch.org/elastic>`__ to launch distributed
   training, if errors (e.g., OOM) are expected or if the resources can join and
   leave dynamically during the training.


.. note:: Data-parallel training also works with `Automatic Mixed Precision (AMP) <https://pytorch.org/docs/master/notes/amp_examples.html#working-with-multiple-gpus>`__.


``torch.nn.DataParallel``
~~~~~~~~~~~~~~~~~~~~~~~~~

The `DataParallel <https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html>`__
package enables single-machine multi-GPU parallelism with the lowest coding
hurdle. It only requires a one-line change to the application code. The tutorial
`Optional: Data Parallelism <https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html>`__
shows an example. The caveat is that, although ``DataParallel`` is very easy to
use, it usually does not offer the best performance. This is because the
implementation of ``DataParallel`` replicates the model in every forward pass,
and its single-process multi-thread parallelism naturally suffers from GIL
contentions. To get better performance, please consider using
`DistributedDataParallel <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__.


``torch.nn.parallel.DistributedDataParallel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compared to `DataParallel <https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html>`__,
`DistributedDataParallel <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__
requires one more step to set up, i.e., calling
`init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__.
DDP uses multi-process parallelism, and hence there is no GIL contention across
model replicas. Moreover, the model is broadcast at DDP construction time instead
of in every forward pass, which also helps to speed up training. DDP is shipped
with several performance optimization technologies. For a more in-depth
explanation, please refer to this
`DDP paper <https://arxiv.org/abs/2006.15704>`__ (VLDB'20).


DDP materials are listed below:

1. `DDP notes <https://pytorch.org/docs/stable/notes/ddp.html>`__
   offer a starter example and some brief descriptions of its design and
   implementation. If this is your first time using DDP, please start from this
   document.
2. `Getting Started with Distributed Data Parallel <../intermediate/ddp_tutorial.html>`__
   explains some common problems with DDP training, including unbalanced
   workload, checkpointing, and multi-device models. Note that, DDP can be
   easily combined with single-machine multi-device model parallelism which is
   described in the
   `Single-Machine Model Parallel Best Practices <../intermediate/model_parallel_tutorial.html>`__
   tutorial.
3. The `Launching and configuring distributed data parallel applications <https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md>`__
   document shows how to use the DDP launching script.
4. `PyTorch Distributed Trainer with Amazon AWS <aws_distributed_training_tutorial.html>`__
   demonstrates how to use DDP on AWS.

TorchElastic
~~~~~~~~~~~~

With the growth of the application complexity and scale, failure recovery
becomes an imperative requirement. Sometimes, it is inevitable to hit errors
like OOM when using DDP, but DDP itself cannot recover from those errors nor
does basic ``try-except`` block work. This is because DDP requires all processes
to operate in a closely synchronized manner and all ``AllReduce`` communications
launched in different processes must match. If one of the processes in the group
throws an OOM exception, it is likely to lead to desynchronization (mismatched
``AllReduce`` operations) which would then cause a crash or hang. If you expect
failures to occur during training or if resources might leave and join
dynamically, please launch distributed data-parallel training using
`torchelastic <https://pytorch.org/elastic>`__.


General Distributed Training
----------------------------

Many training paradigms do not fit into data parallelism, e.g.,
parameter server paradigm, distributed pipeline parallelism, reinforcement
learning applications with multiple observers or agents, etc. The
`torch.distributed.rpc <https://pytorch.org/docs/master/rpc.html>`__ aims at
supporting general distributed training scenarios.

The `torch.distributed.rpc <https://pytorch.org/docs/master/rpc.html>`__ package
has four main pillars:

* `RPC <https://pytorch.org/docs/master/rpc.html#rpc>`__ supports running
  a given function on a remote worker.
* `RRef <https://pytorch.org/docs/master/rpc.html#rref>`__ helps to manage the
  lifetime of a remote object. The reference counting protocol is presented in the
  `RRef notes <https://pytorch.org/docs/master/rpc/rref.html#remote-reference-protocol>`__.
* `Distributed Autograd <https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework>`__
  extends the autograd engine beyond machine boundaries. Please refer to
  `Distributed Autograd Design <https://pytorch.org/docs/master/rpc/distributed_autograd.html#distributed-autograd-design>`__
  for more details.
* `Distributed Optimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`__
  that automatically reaches out to all participating workers to update
  parameters using gradients computed by the distributed autograd engine.

RPC Tutorials are listed below:

1. The `Getting Started with Distributed RPC Framework <../intermediate/rpc_tutorial.html>`__
   tutorial first uses a simple Reinforcement Learning (RL) example to
   demonstrate RPC and RRef. Then, it applies a basic distributed model
   parallelism to an RNN example to show how to use distributed autograd and
   distributed optimizer.
2. The `Implementing a Parameter Server Using Distributed RPC Framework <../intermediate/rpc_param_server_tutorial.html>`__
   tutorial borrows the spirit of
   `HogWild! training <https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf>`__
   and applies it to an asynchronous parameter server (PS) training application.
3. The `Distributed Pipeline Parallelism Using RPC <../intermediate/dist_pipeline_parallel_tutorial.html>`__
   tutorial extends the single-machine pipeline parallel example (presented in
   `Single-Machine Model Parallel Best Practices <../intermediate/model_parallel_tutorial.html>`__)
   to a distributed environment and shows how to implement it using RPC.
4. The `Implementing Batch RPC Processing Using Asynchronous Executions <../intermediate/rpc_async_execution.html>`__
   tutorial demonstrates how to implement RPC batch processing using the
   `@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__
   decorator, which can help speed up inference and training. It uses similar
   RL and PS examples employed in the above tutorials 1 and 2.
5. The `Combining Distributed DataParallel with Distributed RPC Framework <../advanced/rpc_ddp_tutorial.html>`__
   tutorial demonstrates how to combine DDP with RPC to train a model using 
   distributed data parallelism combined with distributed model parallelism.


PyTorch Distributed Developers
------------------------------

If you'd like to contribute to PyTorch Distributed, please refer to our 
`Developer Guide <https://github.com/pytorch/pytorch/blob/master/torch/distributed/CONTRIBUTING.md>`_.
