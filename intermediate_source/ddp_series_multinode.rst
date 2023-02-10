`Introduction <../beginner/ddp_series_intro.html>`__ \|\| `What is DDP <../beginner/ddp_series_theory.html>`__ \|\| `Single-Node
Multi-GPU Training <../beginner/ddp_series_multigpu.html>`__ \|\| `Fault
Tolerance <../beginner/ddp_series_fault_tolerance.html>`__ \|\| **Multi-Node
training** \|\| `minGPT Training <ddp_series_minGPT.html>`__

Multinode Training
==================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      -  Launching multinode training jobs with ``torchrun``
      -  Code changes (and things to keep in mind) when moving from single-node to multinode training.

      .. grid:: 1

         .. grid-item::

            :octicon:`code-square;1.0em;` View the code used in this tutorial on `GitHub <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py>`__

   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      -  Familiarity with `multi-GPU training <../beginner/ddp_series_multigpu.html>`__ and `torchrun <../beginner/ddp_series_fault_tolerance.html>`__ 
      -  2 or more TCP-reachable GPU machines (this tutorial uses AWS p3.2xlarge instances)
      -  PyTorch `installed <https://pytorch.org/get-started/locally/>`__ with CUDA on all machines

Follow along with the video below or on `youtube <https://www.youtube.com/watch/KaAJtI1T2x4>`__. 

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/KaAJtI1T2x4" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

Multinode training involves deploying a training job across several
machines. There are two ways to do this:

-  running a ``torchrun`` command on each machine with identical rendezvous arguments, or
-  deploying it on a compute cluster using a workload manager (like SLURM)

In this video we will go over the (minimal) code changes required to move from single-node multigpu to
multinode training, and run our training script in both of the above ways.

Note that multinode training is bottlenecked by inter-node communication latencies. Running a training job
on 4 GPUs on a single node will be faster than running it on 4 nodes with 1 GPU each.

Local and Global ranks
~~~~~~~~~~~~~~~~~~~~~~~~
In single-node settings, we were tracking the 
``gpu_id`` of each device running our training process. ``torchrun`` tracks this value in an environment variable ``LOCAL_RANK``
which uniquely identifies each GPU-process on a node. For a unique identifier across all the nodes, ``torchrun`` provides another variable
``RANK`` which refers to the global rank of a process.

.. warning::
   Do not use ``RANK`` for critical logic in your training job. When ``torchrun`` restarts processes after a failure or membership changes, there is no guarantee
   that the processes will hold the same ``LOCAL_RANK`` and ``RANKS``. 
 

Heteregeneous Scaling
~~~~~~~~~~~~~~~~~~~~~~
Torchrun supports *heteregenous scaling* i.e. each of your multinode machines can have different number of 
GPUs participating in the training job. In the video, I deployed the code on 2 machines where one machine has 4 GPUs and the
other used only 2 GPUs.


Troubleshooting
~~~~~~~~~~~~~~~~~~

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
-  `Training a GPT model with DDP <ddp_series_minGPT.html>`__  (next tutorial in this series)
-  `Fault Tolerant distributed training <../beginner/ddp_series_fault_tolerance.html>`__ (previous tutorial in this series)
-  `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__
-  `Rendezvous
   arguments <https://pytorch.org/docs/stable/elastic/run.html#note-on-rendezvous-backend>`__
-  `Setting up a cluster on
   AWS <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/setup_pcluster_slurm.md>`__
-  `Slurm docs <https://slurm.schedmd.com/>`__
