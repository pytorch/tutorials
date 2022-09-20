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


.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
      :shadow: none

      -  Launching multinode training jobs with ``torchrun``
      -  Code changes (and things to keep in mind) when moving from single-node to multinode training.

   .. grid-item-card:: :octicon:list-unordered;1em;` Prerequisites
      :shadow: none

      * Familiarity with `multi-GPU training <beginner/ddp_multigpu.html>`__ and `torchrun <beginner/ddp_fault_tolerance.html>`__ 
      * 2 or more TCP-reachable GPU machines (this tutorial uses AWS p3.2xlarge instances)
      * PyTorch `installed <https://pytorch.org/get-started/locally/>`__ with CUDA on all machines


View the code used in this video: https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series/blob/main/multinode_torchrun.py


.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/KaAJtI1T2x4" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>


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
