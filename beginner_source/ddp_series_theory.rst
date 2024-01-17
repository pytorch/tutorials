`Introduction <ddp_series_intro.html>`__ \|\| **What is DDP** \|\|
`Single-Node Multi-GPU Training <ddp_series_multigpu.html>`__ \|\|
`Fault Tolerance <ddp_series_fault_tolerance.html>`__ \|\|
`Multi-Node training <../intermediate/ddp_series_multinode.html>`__ \|\|
`minGPT Training <../intermediate/ddp_series_minGPT.html>`__

What is Distributed Data Parallel (DDP)
=======================================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      *  How DDP works under the hood
      *  What is ``DistributedSampler``
      *  How gradients are synchronized across GPUs


   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Familiarity with `basic non-distributed training  <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__ in PyTorch

Follow along with the video below or on `youtube <https://www.youtube.com/watch/Cvdhwx-OBBo>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/Cvdhwx-OBBo" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

This tutorial is a gentle introduction to PyTorch `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__ (DDP)
which enables data parallel training in PyTorch. Data parallelism is a way to
process multiple data batches across multiple devices simultaneously
to achieve better performance. In PyTorch, the `DistributedSampler <https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>`__
ensures each device gets a non-overlapping input batch. The model is replicated on all the devices;
each replica calculates gradients and simultaneously synchronizes with the others using the `ring all-reduce
algorithm <https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/>`__.

This `illustrative tutorial <https://pytorch.org/tutorials/intermediate/dist_tuto.html#>`__ provides a more in-depth python view of the mechanics of DDP.

Why you should prefer DDP over ``DataParallel`` (DP)
----------------------------------------------------

`DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`__ 
is an older approach to data parallelism. DP is trivially simple (with just one extra line of code) but it is much less performant.
DDP improves upon the architecture in a few ways:

+---------------------------------------+------------------------------+
| ``DataParallel``                      | ``DistributedDataParallel``  |
+=======================================+==============================+
| More overhead; model is replicated    | Model is replicated only     |
| and destroyed at each forward pass    | once                         |
+---------------------------------------+------------------------------+
| Only supports single-node parallelism | Supports scaling to multiple |
|                                       | machines                     |
+---------------------------------------+------------------------------+
| Slower; uses multithreading on a      | Faster (no GIL contention)   |
| single process and runs into Global   | because it uses              |
| Interpreter Lock (GIL) contention     | multiprocessing              |
+---------------------------------------+------------------------------+

Further Reading
---------------

-  `Multi-GPU training with DDP <ddp_series_multigpu.html>`__ (next tutorial in this series)
-  `DDP
   API <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
-  `DDP Internal
   Design <https://pytorch.org/docs/master/notes/ddp.html#internal-design>`__
-  `DDP Mechanics Tutorial <https://pytorch.org/tutorials/intermediate/dist_tuto.html#>`__
