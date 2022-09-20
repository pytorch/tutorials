`Introduction <beginner/ddp_series_intro.html>`__ \|\| `What is DDP <beginner/ddp_theory.html>`__ \|\| `Single-node
Multi-GPU training <beginner/ddp_multigpu.html>`__ \|\| `Fault
Tolerance <beginner/ddp_fault_tolerance.html>`__ \|\| `Multi-node
training <ddp_multinode.html>` \|\| **mingpt training**

Training “real-world” models with DDP
=====================================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

In this video we walk through the process of training a GPT model in multinode DDP.
We first clone the `minGPT repo <https://github.com/karpathy/minGPT>` and refactor the Trainer
to resemble the structure we have used in this series. Watch the video for details on these changes.

We use `hydra <https://hydra.cc/>` to centrally manage all the configurations for our training run.  
Once the code has been refactored, we run it first on a single-node with 4 GPUs, and then on a slurm cluster.


What you will learn
-------------------
-  Refactor a (nicely structured) project to use DDP training
-  Best practices when writing a distributed training script
-  Increased flexibility with saving/loading artifacts in the cloud
-  When is DDP NOT suitable


View the code used in this video: https://github.com/suraj813/minGPT-ddp


.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/XFsFDGKZHh4" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>



Files used for training
~~~~~~~~~~~~~~~~~~~~~~~~
- `trainer.py <https://github.com/suraj813/minGPT-ddp/blob/master/mingpt/trainer.py>` includes the Trainer class that runs the distributed training iterations on the model
with the provided dataset.
- `model.py <https://github.com/suraj813/minGPT-ddp/blob/master/mingpt/model.py>` defines the model architecture.
- `char_dataset.py <https://github.com/suraj813/minGPT-ddp/blob/master/mingpt/char_dataset.py>` contains the `Dataset`class for a character-level dataset.
- `gpt2_train_cfg.yaml <https://github.com/suraj813/minGPT-ddp/blob/master/mingpt/gpt2_train_cfg.yaml>` contains the configurations for data, model, optimizer and training run.
- `main.py <https://github.com/suraj813/minGPT-ddp/blob/master/mingpt/main.py>` is the entry point to the trainig job. 
It sets up the DDP process group, reads all the configurations and runs the training job.


Saving and Loading from the cloud
~~~~~~~~~~~~~~~~~~~~~~~~
In the video, we save training snapshots directly to the cloud. This gives us the flexibility to continue training
from any node that has access to the cloud bucket.


Using Mixed Precision
~~~~~~~~~~~~~~~~~~~~~~~~
To speed things up, you might be able to use `Mixed Precision <https://pytorch.org/docs/stable/amp.html>`__ to train your models. 
In Mixed Precision, some parts of the training process are carried out in FP16 half-precision, while other steps 
that are more sensitive to precision drops are maintained in FP32 precision. The `use_amp <https://github.com/suraj813/minGPT-ddp/tree/use_amp>`
branch contains the code for training with Mixed Precision.


When is DDP not enough?
~~~~~~~~~~~~~~~~~~~~~~~~
A typical training run's memory footprint consists of model weights, activations, gradients, the input batch, and the optimizer state.
Since DDP replicates the model on each GPU, it only works when GPUs have sufficient capacity to accomodate the full footprint. 
When models grow larger, more aggressive techniques like `FSDP <https://pytorch.org/docs/stable/fsdp.html>`__ are required; here the model is not replicated but "sharded" across all the GPUs,
and computation is overlapped with communication in the forward and backward passes. Read our `blog <https://medium.com/pytorch/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff>`__
to learn how we trained a 1 Trillion parameter model with FSDP.


Further Reading
---------------
-  `Multi-node training with DDP <ddp_multinode.html>`__ (previous tutorial in this series)
-  `Mixed Precision training <https://pytorch.org/docs/stable/amp.html>`__
-  `Fully-Sharded Data Parallel <https://pytorch.org/docs/stable/fsdp.html>`__
-  `Training a 1T parameter model with FSDP <https://medium.com/pytorch/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff>`__
-  Less' FSDP videos
