Shard Optimizer States with ZeroRedundancyOptimizer
===================================================

In this recipe, you will learn:

- The high-level idea of `ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__.
- How to use `ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__
  in distributed training and its impact.


Requirements
------------

- PyTorch 1.8+
- `Getting Started With Distributed Data Parallel <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_


What is ``ZeroRedundancyOptimizer``?
------------------------------------

The idea of `ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__
comes from `DeepSpeed/ZeRO project <https://github.com/microsoft/DeepSpeed>`_ and
`Marian <https://github.com/marian-nmt/marian-dev>`_ that shard
optimizer states across distributed data-parallel processes to
reduce per-process memory footprint. In the
`Getting Started With Distributed Data Parallel <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_
tutorial, we have shown how to use
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_
(DDP) to train models. In that tutorial, each process keeps a dedicated replica
of the optimizer. Since DDP has already synchronized gradients in the
backward pass, all optimizer replicas will operate on the same parameter and
gradient values in every iteration, and this is how DDP keeps model replicas in
the same state. Oftentimes, optimizers also maintain local states. For example,
the ``Adam`` optimizer uses per-parameter ``exp_avg`` and ``exp_avg_sq`` states. As a
result, the ``Adam`` optimizer's memory consumption is at least twice the model
size. Given this observation, we can reduce the optimizer memory footprint by
sharding optimizer states across DDP processes. More specifically, instead of
creating per-param states for all parameters, each optimizer instance in
different DDP processes only keeps optimizer states for a shard of all model
parameters. The optimizer ``step()`` function only updates the parameters in its
shard and then broadcasts its updated parameters to all other peer DDP
processes, so that all model replicas still land in the same state.

How to use ``ZeroRedundancyOptimizer``?
---------------------------------------

The code below demonstrates how to use
`ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__.
The majority of the code is similar to the simple DDP example presented in
`Distributed Data Parallel notes <https://pytorch.org/docs/stable/notes/ddp.html>`_.
The main difference is the ``if-else`` clause in the ``example`` function which
wraps optimizer constructions, toggling between
`ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__
and ``Adam`` optimizer.


::

    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from torch.nn.parallel import DistributedDataParallel as DDP

    def print_peak_memory(prefix, device):
        if device == 0:
            print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

    def example(rank, world_size, use_zero):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        # create default process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # create local model
        model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        print_peak_memory("Max memory allocated after creating local model", rank)

        # construct DDP model
        ddp_model = DDP(model, device_ids=[rank])
        print_peak_memory("Max memory allocated after creating DDP", rank)

        # define loss function and optimizer
        loss_fn = nn.MSELoss()
        if use_zero:
            optimizer = ZeroRedundancyOptimizer(
                ddp_model.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=0.01
            )
        else:
            optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)

        # forward pass
        outputs = ddp_model(torch.randn(20, 2000).to(rank))
        labels = torch.randn(20, 2000).to(rank)
        # backward pass
        loss_fn(outputs, labels).backward()

        # update parameters
        print_peak_memory("Max memory allocated before optimizer step()", rank)
        optimizer.step()
        print_peak_memory("Max memory allocated after optimizer step()", rank)

        print(f"params sum is: {sum(model.parameters()).sum()}")



    def main():
        world_size = 2
        print("=== Using ZeroRedundancyOptimizer ===")
        mp.spawn(example,
            args=(world_size, True),
            nprocs=world_size,
            join=True)

        print("=== Not Using ZeroRedundancyOptimizer ===")
        mp.spawn(example,
            args=(world_size, False),
            nprocs=world_size,
            join=True)

    if __name__=="__main__":
        main()

The output is shown below. When enabling ``ZeroRedundancyOptimizer`` with ``Adam``,
the optimizer ``step()`` peak memory consumption is half of vanilla ``Adam``'s
memory consumption. This agrees with our expectation, as we are sharding
``Adam`` optimizer states across two processes. The output also shows that, with
``ZeroRedundancyOptimizer``, the model parameters still end up with the same
values after one iterations (the parameters sum is the same with and without
``ZeroRedundancyOptimizer``).

::

    === Using ZeroRedundancyOptimizer ===
    Max memory allocated after creating local model: 335.0MB
    Max memory allocated after creating DDP: 656.0MB
    Max memory allocated before optimizer step(): 992.0MB
    Max memory allocated after optimizer step(): 1361.0MB
    params sum is: -3453.6123046875
    params sum is: -3453.6123046875
    === Not Using ZeroRedundancyOptimizer ===
    Max memory allocated after creating local model: 335.0MB
    Max memory allocated after creating DDP: 656.0MB
    Max memory allocated before optimizer step(): 992.0MB
    Max memory allocated after optimizer step(): 1697.0MB
    params sum is: -3453.6123046875
    params sum is: -3453.6123046875
