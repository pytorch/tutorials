[Introduction](ddp_series_intro.html) || **What is DDP** ||
[Single-Node Multi-GPU Training](ddp_series_multigpu.html) ||
[Fault Tolerance](ddp_series_fault_tolerance.html) ||
[Multi-Node training](../intermediate/ddp_series_multinode.html) ||
[minGPT Training](../intermediate/ddp_series_minGPT.html)

# What is Distributed Data Parallel (DDP)

Authors: [Suraj Subramanian](https://github.com/subramen)

 What you will learn

- How DDP works under the hood
- What is `DistributedSampler`
- How gradients are synchronized across GPUs

 Prerequisites

- Familiarity with [basic non-distributed training](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) in PyTorch

Follow along with the video below or on [youtube](https://www.youtube.com/watch/Cvdhwx-OBBo).

This tutorial is a gentle introduction to PyTorch [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP)
which enables data parallel training in PyTorch. Data parallelism is a way to
process multiple data batches across multiple devices simultaneously
to achieve better performance. In PyTorch, the [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
ensures each device gets a non-overlapping input batch. The model is replicated on all the devices;
each replica calculates gradients and simultaneously synchronizes with the others using the [ring all-reduce
algorithm](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/).

This [illustrative tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html#) provides a more in-depth python view of the mechanics of DDP.

## Why you should prefer DDP over `DataParallel` (DP)

[DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
is an older approach to data parallelism. DP is trivially simple (with just one extra line of code) but it is much less performant.
DDP improves upon the architecture in a few ways:

| `DataParallel` | `DistributedDataParallel` |
| --- | --- |
| More overhead; model is replicated and destroyed at each forward pass | Model is replicated only once |
| Only supports single-node parallelism | Supports scaling to multiple machines |
| Slower; uses multithreading on a single process and runs into Global Interpreter Lock (GIL) contention | Faster (no GIL contention) because it uses multiprocessing |

## Further Reading

- [Multi-GPU training with DDP](ddp_series_multigpu.html) (next tutorial in this series)
- [DDP
API](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [DDP Internal
Design](https://pytorch.org/docs/master/notes/ddp.html#internal-design)
- [DDP Mechanics Tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html#)