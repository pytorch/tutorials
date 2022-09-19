`Introduction <beginner/ddp_series_intro.html>`__ \|\| `What is DDP <beginner/ddp_theory.html>`__ \|\| `Single-node
Multi-GPU training <beginner/ddp_multigpu.html>`__ \|\| `Fault
Tolerance <beginner/ddp_fault_tolerance.html>`__ \|\| **Multi-node
training** \|\| `mingpt training <ddp_minGPT.html>`__

Multinode Training
==================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

Multinode training involves deploying a training job across several
machines. There are two ways to do this:  
- running a torchrun command
on each machine with identical rendezvous arguments, or 
- deploying it on a
compute cluster using a workload manager (like SLURM)

In this video we will go over the (minimal) code changes required to move from single-node multigpu to 
multinode training, and run our training script in both of the above ways.


What you will learn
-------------------
-  Launching multinode training jobs with ``torchrun``
-  Code changes (and things to keep in mind) when moving from single-node to multinode training.


View the code used in this video: https://github.com/suraj813/distributed-pytorch/blob/main/multinode_torchrun.py


.. raw:: html

   <embed video>


.. note:: 
-  In a single-node setup, local ranks are sufficient to identify each
process uniquely. When running a multinode setup, use the global rank
(given by ``os.environ["RANK"]`` when using ``torchrun``) to uniquely
identify processes.

- ``RANK`` is NOT stable. On restarting a training job, the local workers
on a node can be assigned a different range of ranks than before. Do not
use ``RANK`` and ``LOCAL_RANK`` in any functionality that assumes their
stability.


.. note:: 
Torchrun supports *heteregenous scaling* i.e. each of your multinode
machines can have different number of workers participating in the
training job. In the video, I deployed the code on 2 machines with 4 and
2 GPUs each.



Troubleshooting
~~~~~~~~~~~~~~~

-  Ensure that your nodes are able to communicate with each other over
   TCP.
-  Set env variable ``NCCL_DEBUG`` to ``INFO`` (using
   ``export NCCL_DEBUG=INFO``) to print verbose logs that can help
   diagnose the issue.
-  Sometimes you might need to explicitly set the network interface for
   the distributed backend (``export NCCL_SOCKET_IFNAME=eth0``). Read
   more about this
   `here <https://pytorch.org/docs/stable/distributed.html#choosing-the-network-interface-to-use>`__.


Further Reading
---------------
-  `Training a GPT model with DDP <ddp_minGPT.html>`__  (next tutorial in this series)
-  `Fault Tolerant distributed training <beginner/ddp_fault_tolerance.html>`__ (previous tutorial in this series)
-  `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__
-  `Rendezvous
   arguments <https://pytorch.org/docs/stable/elastic/run.html#note-on-rendezvous-backend>`__
-  `Setting up a cluster on
   AWS <https://github.com/suraj813/minGPT-ddp/blob/master/mingpt/slurm/setup_pcluster_slurm.md>`__
-  `Slurm docs <https://slurm.schedmd.com/>`__
