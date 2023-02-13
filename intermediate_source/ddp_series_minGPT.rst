`Introduction <../beginner/ddp_series_intro.html>`__ \|\| `What is DDP <../beginner/ddp_series_theory.html>`__ \|\| `Single-Node
Multi-GPU Training <../beginner/ddp_series_multigpu.html>`__ \|\| `Fault
Tolerance <../beginner/ddp_series_fault_tolerance.html>`__ \|\| `Multi-Node
training <ddp_series_multinode.html>`__ \|\| **minGPT Training**

Training “real-world” models with DDP
=====================================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      -  Best practices when writing a distributed training script
      -  Increased flexibility with saving/loading artifacts in the cloud
      -  When DDP is NOT suitable

      .. grid:: 1

         .. grid-item::

            :octicon:`code-square;1.0em;` View the code used in this tutorial on `GitHub <https://github.com/pytorch/examples/tree/main/distributed/minGPT-ddp>`__

   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      - Familiarity with `multi-GPU training <../beginner/ddp_series_multigpu.html>`__ and `torchrun <../beginner/ddp_series_fault_tolerance.html>`__ 
      - [Optional] Familiarity with `multinode training <ddp_series_multinode.html>`__
      - 2 or more TCP-reachable GPU machines (this tutorial uses AWS p3.2xlarge instances)
      - PyTorch `installed <https://pytorch.org/get-started/locally/>`__ with CUDA on all machines

Follow along with the video below or on `youtube <https://www.youtube.com/watch/XFsFDGKZHh4>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/XFsFDGKZHh4" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

In this video, we will review the process of training a GPT model in multinode DDP.
We first clone the `minGPT repo <https://github.com/karpathy/minGPT>`__ and refactor the Trainer
to resemble the structure we have used in this series. Watch the video for details on these changes.

We use `hydra <https://hydra.cc/>`__ to centrally manage all the configurations for our training run.
Once the code has been refactored, we run it first on a single-node with 4 GPUs, and then on a slurm cluster.

Files used for training
~~~~~~~~~~~~~~~~~~~~~~~~
- `trainer.py <https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/trainer.py>`__ includes the Trainer class that runs the distributed training iterations on the model with the provided dataset.
- `model.py <https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/model.py>`__ defines the model architecture.
- `char_dataset.py <https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/char_dataset.py>`__ contains the ``Dataset`` class for a character-level dataset.
- `gpt2_train_cfg.yaml <https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/gpt2_train_cfg.yaml>`__ contains the configurations for data, model, optimizer, and training run.
- `main.py <https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/main.py>`__ is the entry point to the training job. It sets up the DDP process group, reads all the configurations and runs the training job.


Saving and Loading from the cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the video above, we save training snapshots directly to the cloud. This gives us the flexibility to continue training
from any node that has access to the cloud bucket.


Using Mixed Precision
~~~~~~~~~~~~~~~~~~~~~~~~
To speed things up, you might be able to use `Mixed Precision <https://pytorch.org/docs/stable/amp.html>`__ to train your models. 
In Mixed Precision, some parts of the training process are carried out in reduced precision, while other steps 
that are more sensitive to precision drops are maintained in FP32 precision. 


When is DDP not enough?
~~~~~~~~~~~~~~~~~~~~~~~~
A typical training run's memory footprint consists of model weights, activations, gradients, the input batch, and the optimizer state.
Since DDP replicates the model on each GPU, it only works when GPUs have sufficient capacity to accomodate the full footprint. 
When models grow larger, more aggressive techniques might be useful:

-  `activation checkpointing <https://pytorch.org/docs/stable/checkpoint.html>`__: Instead of saving intermediate activations during the forward pass, the activations are recomputed during the backward pass. In this approach, we run more compute but save on memory footprint.
-  `Fully-Sharded Data Parallel <https://pytorch.org/docs/stable/fsdp.html>`__: Here the model is not replicated but "sharded" across all the GPUs, and computation is overlapped with communication in the forward and backward passes. Read our `blog <https://medium.com/pytorch/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff>`__ to learn how we trained a 1 Trillion parameter model with FSDP.


Further Reading
---------------
-  `Multi-Node training with DDP <ddp_series_multinode.html>`__ (previous tutorial in this series)
-  `Mixed Precision training <https://pytorch.org/docs/stable/amp.html>`__
-  `Fully-Sharded Data Parallel <https://pytorch.org/docs/stable/fsdp.html>`__
-  `Training a 1T parameter model with FSDP <https://medium.com/pytorch/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff>`__
-  `FSDP Video Tutorial Series <https://www.youtube.com/playlist?list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT>`__ 
