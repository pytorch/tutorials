**Introduction** \|\| `What is DDP <ddp_series_theory.html>`__ \|\|
`Single-Node Multi-GPU Training <ddp_series_multigpu.html>`__ \|\|
`Fault Tolerance <ddp_series_fault_tolerance.html>`__ \|\|
`Multi-Node training <../intermediate/ddp_series_multinode.html>`__ \|\|
`minGPT Training <../intermediate/ddp_series_minGPT.html>`__

Distributed Data Parallel in PyTorch - Video Tutorials
======================================================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

Follow along with the video below or on `youtube <https://www.youtube.com/watch/-K3bZYHYHEA>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/-K3bZYHYHEA" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

This series of video tutorials walks you through distributed training in
PyTorch via DDP.

The series starts with a simple non-distributed training job, and ends
with deploying a training job across several machines in a cluster.
Along the way, you will also learn about
`torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__ for
fault-tolerant distributed training.

The tutorial assumes a basic familiarity with model training in PyTorch.

Running the code
----------------

You will need multiple CUDA GPUs to run the tutorial code. Typically,
this can be done on a cloud instance with multiple GPUs (the tutorials
use an Amazon EC2 P3 instance with 4 GPUs).

The tutorial code is hosted in this
`github repo <https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series>`__.
Clone the repository and follow along!

Tutorial sections
-----------------

0. Introduction (this page)
1. `What is DDP? <ddp_series_theory.html>`__ Gently introduces what DDP is doing
   under the hood
2. `Single-Node Multi-GPU Training <ddp_series_multigpu.html>`__ Training models
   using multiple GPUs on a single machine
3. `Fault-tolerant distributed training <ddp_series_fault_tolerance.html>`__
   Making your distributed training job robust with torchrun
4. `Multi-Node training <../intermediate/ddp_series_multinode.html>`__ Training models using
   multiple GPUs on multiple machines
5. `Training a GPT model with DDP <../intermediate/ddp_series_minGPT.html>`__ “Real-world”
   example of training a `minGPT <https://github.com/karpathy/minGPT>`__
   model with DDP
