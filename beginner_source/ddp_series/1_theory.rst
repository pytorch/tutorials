`Introduction <0_intro.html>`__ \|\| **What is DDP** \|\| `Multi-GPU
training <2_multigpu.html>`__ \|\| `Fault
Tolerance <3_fault_tolerance.html>`__ \|\| `Multi-node
training <4_multinode.html>`__ \|\| `mingpt training <5_minGPT.html>`__

What is Distributed Data Parallel (DDP)
=======================================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

.. raw:: html

   <embed video>

-  Data parallelism refers to processing multiple data batches on
   multiple devices simultaneously.
-  PyTorch provides data parallel training across multiple GPUs via
   `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
   (DDP)
-  The
   `DistributedSampler <https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>`__
   ensures each replica gets a non-overlapping input batch.
-  All the replicas simultaneously calculate gradients and synchronize
   them with each other, ensuring maximum GPU utilization. See here for
   a detailed explanation of the `ring all-reduce
   algorithm <https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/>`__

Use DDP instead of DataParallel (DP)!
-------------------------------------

DP is an older approach to data parallelism. DDP improves upon the
architecture in a few ways:

+---------------------------------------+------------------------------+
| DP                                    | DDP                          |
+=======================================+==============================+
| More overhead; model is replicated    | Model is replicated only     |
| and destroyed at each forward pass    | once                         |
+---------------------------------------+------------------------------+
| Only supports single-node parallelism | Supports scaling to multiple |
|                                       | machines                     |
+---------------------------------------+------------------------------+
| Slower; uses multithreading on a      | Faster (no GIL contention)   |
| single process and runs into GIL      | because it uses              |
| contention                            | multiprocessing              |
+---------------------------------------+------------------------------+

Further Reading
---------------

-  `DDP
   API <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
-  `DDP Internal
   Design <https://pytorch.org/docs/master/notes/ddp.html#internal-design>`__
