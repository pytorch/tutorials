[Introduction](0_intro.html) ||
[What is DDP](1_theory.html) ||
[Multi-GPU training](2_multigpu.html) ||
[Fault Tolerance](3_fault_tolerance.html) ||
**Multi-node training** ||
[mingpt training](5_minGPT.html)

# Multinode Training

Authors: [Suraj Subramanian](https://github.com/suraj813)

<embed video>

Multinode training involves deploying a training job across several machines. There are two ways to do this:
* running a torchrun command on each machine with identical rendezvous arguments
* deploying it on a compute cluster using a workload manager (like SLURM)

In both cases, it is essential that the machines can communicate with each other over TCP.

In a single-node setup, local ranks are sufficient to identify each process uniquely. When running a multinode setup, use the global rank (given by `os.environ["RANK"]` when using `torchrun`) to uniquely identify processes.

Torchrun supports _heteregenous scaling_ i.e. each of your multinode machines can have different number of workers participating in the training job. In the video, I deployed the code on 2 machines with 4 and 2 GPUs each.

### Note
`RANK` is NOT stable. On restarting a training job, the local workers on a node can be assigned a different range of ranks than before. Do not use `RANK` and `LOCAL_RANK` in any functionality that assumes their stability.

### Troubleshooting
* Ensure that your nodes are able to communicate with each other over TCP.
* Set env variable `NCCL_DEBUG` to `INFO` (using `export NCCL_DEBUG=INFO`) to print verbose logs that can help diagnose the issue.
* Sometimes you might need to explicitly set the network interface for the distributed backend (`export NCCL_SOCKET_IFNAME=eth0`). Read more about this [here](https://pytorch.org/docs/stable/distributed.html#choosing-the-network-interface-to-use).



## Further Reading
* [torchrun](https://pytorch.org/docs/stable/elastic/run.html)
* [Rendezvous arguments](https://pytorch.org/docs/stable/elastic/run.html#note-on-rendezvous-backend)
* [Setting up a cluster on AWS](https://github.com/suraj813/minGPT-ddp/blob/master/mingpt/slurm/setup_pcluster_slurm.md)
* [Slurm docs](https://slurm.schedmd.com/)

