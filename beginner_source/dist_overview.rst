PyTorch Distributed Overview
============================
**Author**: `Shen Li <https://mrshenli.github.io/>`_

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/beginner_source/dist_overview.rst>`__.

This is the overview page for the ``torch.distributed`` package. The goal of
this page is to categorize documents into different topics and briefly
describe each of them. If this is your first time building distributed training
applications using PyTorch, it is recommended to use this document to navigate
to the technology that can best serve your use case.


Introduction
------------

As of PyTorch v1.6.0, features in ``torch.distributed`` can be categorized into
three main components:

* `Distributed Data-Parallel Training <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
  (DDP) is a widely adopted single-program multiple-data training paradigm. With
  DDP, the model is replicated on every process, and every model replica will be
  fed with a different set of input data samples. DDP takes care of gradient
  communication to keep model replicas synchronized and overlaps it with the
  gradient computations to speed up training.
* `RPC-Based Distributed Training <https://pytorch.org/docs/stable/rpc.html>`__
  (RPC) supports general training structures that cannot fit into
  data-parallel training such as distributed pipeline parallelism, parameter
  server paradigm, and combinations of DDP with other training paradigms. It
  helps manage remote object lifetime and extends the
  `autograd engine <https://pytorch.org/docs/stable/autograd.html>`__ beyond
  machine boundaries.
* `Collective Communication <https://pytorch.org/docs/stable/distributed.html>`__
  (c10d) library supports sending tensors across processes within a group. It
  offers both collective communication APIs (e.g.,
  `all_reduce <https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce>`__
  and `all_gather <https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather>`__)
  and P2P communication APIs (e.g.,
  `send <https://pytorch.org/docs/stable/distributed.html#torch.distributed.send>`__
  and `isend <https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend>`__).
  DDP and RPC (`ProcessGroup Backend <https://pytorch.org/docs/stable/rpc.html#process-group-backend>`__)
  are built on c10d, where the former uses collective communications
  and the latter uses P2P communications. Usually, developers do not need to
  directly use this raw communication API, as the DDP and RPC APIs can serve
  many distributed training scenarios. However, there are use cases where this API
  is still helpful. One example would be distributed parameter averaging, where
  applications would like to compute the average values of all model parameters
  after the backward pass instead of using DDP to communicate gradients. This can
  decouple communications from computations and allow finer-grain control over
  what to communicate, but on the other hand, it also gives up the performance
  optimizations offered by DDP.
  `Writing Distributed Applications with PyTorch <../intermediate/dist_tuto.html>`__
  shows examples of using c10d communication APIs.


Data Parallel Training
----------------------

PyTorch provides several options for data-parallel training. For applications
that gradually grow from simple to complex and from prototype to production, the
common development trajectory would be:

1. Use single-device training if the data and model can fit in one GPU, and
   training speed is not a concern.
2. Use single-machine multi-GPU
   `DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`__
   to make use of multiple GPUs on a single machine to speed up training with
   minimal code changes.
3. Use single-machine multi-GPU
   `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__,
   if you would like to further speed up training and are willing to write a
   little more code to set it up.
4. Use multi-machine `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
   and the `launching script <https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md>`__,
   if the application needs to scale across machine boundaries.
5. Use multi-GPU `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`__
   training on a single-machine or multi-machine when the data and model cannot
   fit on one GPU.
6. Use `torch.distributed.elastic <https://pytorch.org/docs/stable/distributed.elastic.html>`__
   to launch distributed training if errors (e.g., out-of-memory) are expected or if
   resources can join and leave dynamically during training.


.. note:: Data-parallel training also works with `Automatic Mixed Precision (AMP) <https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus>`__.


``torch.nn.DataParallel``
~~~~~~~~~~~~~~~~~~~~~~~~~

The `DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`__
package enables single-machine multi-GPU parallelism with the lowest coding
hurdle. It only requires a one-line change to the application code. The tutorial
`Optional: Data Parallelism <../beginner/blitz/data_parallel_tutorial.html>`__
shows an example. Although ``DataParallel`` is very easy to
use, it usually does not offer the best performance because it replicates the
model in every forward pass, and its single-process multi-thread parallelism
naturally suffers from
`GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`__ contention. To get
better performance, consider using
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__.


``torch.nn.parallel.DistributedDataParallel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compared to `DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`__,
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
requires one more step to set up, i.e., calling
`init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__.
DDP uses multi-process parallelism, and hence there is no GIL contention across
model replicas. Moreover, the model is broadcast at DDP construction time instead
of in every forward pass, which also helps to speed up training. DDP is shipped
with several performance optimization technologies. For a more in-depth
explanation, refer to this
`paper <http://www.vldb.org/pvldb/vol13/p3005-li.pdf>`__ (VLDB'20).


DDP materials are listed below:

1. `DDP notes <https://pytorch.org/docs/stable/notes/ddp.html>`__
   offer a starter example and some brief descriptions of its design and
   implementation. If this is your first time using DDP, start from this
   document.
2. `Getting Started with Distributed Data Parallel <../intermediate/ddp_tutorial.html>`__
   explains some common problems with DDP training, including unbalanced
   workload, checkpointing, and multi-device models. Note that, DDP can be
   easily combined with single-machine multi-device model parallelism which is
   described in the
   `Single-Machine Model Parallel Best Practices <../intermediate/model_parallel_tutorial.html>`__
   tutorial.
3. The `Launching and configuring distributed data parallel applications <https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md>`__
   document shows how to use the DDP launching script.
4. The `Shard Optimizer States With ZeroRedundancyOptimizer <../recipes/zero_redundancy_optimizer.html>`__
   recipe demonstrates how `ZeroRedundancyOptimizer <https://pytorch.org/docs/stable/distributed.optim.html>`__
   helps to reduce optimizer memory footprint.
5. The `Distributed Training with Uneven Inputs Using the Join Context Manager <../advanced/generic_join.html>`__
   tutorial walks through using the generic join context for distributed training with uneven inputs.


``torch.distributed.FullyShardedDataParallel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`__
(FSDP) is a type of data parallelism paradigm which maintains a per-GPU copy of a model’s
parameters, gradients and optimizer states, it shards all of these states across
data-parallel workers. The support for FSDP was added starting PyTorch v1.11. The tutorial
`Getting Started with FSDP <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__
provides in depth explanation and example of how FSDP works.


torch.distributed.elastic
~~~~~~~~~~~~~~~~~~~~~~~~~

With the growth of the application complexity and scale, failure recovery
becomes a requirement. Sometimes it is inevitable to hit errors
like out-of-memory (OOM) when using DDP, but DDP itself cannot recover from those errors,
and it is not possible to handle them using a standard ``try-except`` construct.
This is because DDP requires all processes to operate in a closely synchronized manner
and all ``AllReduce`` communications launched in different processes must match.
If one of the processes in the group
throws an exception, it is likely to lead to desynchronization (mismatched
``AllReduce`` operations) which would then cause a crash or hang.
`torch.distributed.elastic <https://pytorch.org/docs/stable/distributed.elastic.html>`__
adds fault tolerance and the ability to make use of a dynamic pool of machines (elasticity).

RPC-Based Distributed Training
------------------------------

Many training paradigms do not fit into data parallelism, e.g.,
parameter server paradigm, distributed pipeline parallelism, reinforcement
learning applications with multiple observers or agents, etc.
`torch.distributed.rpc <https://pytorch.org/docs/stable/rpc.html>`__ aims at
supporting general distributed training scenarios.

`torch.distributed.rpc <https://pytorch.org/docs/stable/rpc.html>`__
has four main pillars:

* `RPC <https://pytorch.org/docs/stable/rpc.html#rpc>`__ supports running
  a given function on a remote worker.
* `RRef <https://pytorch.org/docs/stable/rpc.html#rref>`__ helps to manage the
  lifetime of a remote object. The reference counting protocol is presented in the
  `RRef notes <https://pytorch.org/docs/stable/rpc/rref.html#remote-reference-protocol>`__.
* `Distributed Autograd <https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework>`__
  extends the autograd engine beyond machine boundaries. Please refer to
  `Distributed Autograd Design <https://pytorch.org/docs/stable/rpc/distributed_autograd.html#distributed-autograd-design>`__
  for more details.
* `Distributed Optimizer <https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim>`__
  automatically reaches out to all participating workers to update
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
   `@rpc.functions.async_execution <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.functions.async_execution>`__
   decorator, which can help speed up inference and training. It uses
   RL and PS examples similar to those in the above tutorials 1 and 2.
5. The `Combining Distributed DataParallel with Distributed RPC Framework <../advanced/rpc_ddp_tutorial.html>`__
   tutorial demonstrates how to combine DDP with RPC to train a model using
   distributed data parallelism combined with distributed model parallelism.


PyTorch Distributed Developers
------------------------------

If you'd like to contribute to PyTorch Distributed, please refer to our
`Developer Guide <https://github.com/pytorch/pytorch/blob/master/torch/distributed/CONTRIBUTING.md>`_.
