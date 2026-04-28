**Introduction** || [What is DDP](ddp_series_theory.html) ||
[Single-Node Multi-GPU Training](ddp_series_multigpu.html) ||
[Fault Tolerance](ddp_series_fault_tolerance.html) ||
[Multi-Node training](../intermediate/ddp_series_multinode.html) ||
[minGPT Training](../intermediate/ddp_series_minGPT.html)

# Distributed Data Parallel in PyTorch - Video Tutorials

Authors: [Suraj Subramanian](https://github.com/subramen)

Follow along with the video below or on [youtube](https://www.youtube.com/watch/-K3bZYHYHEA).

This series of video tutorials walks you through distributed training in
PyTorch via DDP.

The series starts with a simple non-distributed training job, and ends
with deploying a training job across several machines in a cluster.
Along the way, you will also learn about
[torchrun](https://pytorch.org/docs/stable/elastic/run.html) for
fault-tolerant distributed training.

The tutorial assumes a basic familiarity with model training in PyTorch.

## Running the code

You will need multiple CUDA GPUs to run the tutorial code. Typically,
this can be done on a cloud instance with multiple GPUs (the tutorials
use an Amazon EC2 P3 instance with 4 GPUs).

The tutorial code is hosted in this
[github repo](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series).
Clone the repository and follow along!

## Tutorial sections

1. Introduction (this page)
2. [What is DDP?](ddp_series_theory.html) Gently introduces what DDP is doing
under the hood
3. [Single-Node Multi-GPU Training](ddp_series_multigpu.html) Training models
using multiple GPUs on a single machine
4. [Fault-tolerant distributed training](ddp_series_fault_tolerance.html)
Making your distributed training job robust with torchrun
5. [Multi-Node training](../intermediate/ddp_series_multinode.html) Training models using
multiple GPUs on multiple machines
6. [Training a GPT model with DDP](../intermediate/ddp_series_minGPT.html) "Real-world"
example of training a [minGPT](https://github.com/karpathy/minGPT)
model with DDP