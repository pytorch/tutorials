Getting Started with Distributed Checkpoint (DCP)
=====================================================

**Author**: `Iris Zhang <https://github.com/wz337>`__, `Rodrigo Kumpera <https://github.com/kumpera>`__, `Chien-Chin Huang <https://github.com/fegin>`__

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/DCP_tutorial.rst>`__.

Checkpointing AI models during distributed training could be challenging, as parameters and gradients are partitioned across GPUs and the number of GPUs available could change when resume training.
Pytorch Distributed Checkpointing (DCP) can help make this easier.

In this tutorial, we show how to use DCP APIs with a simple FSDP wrapped model.


How DCP works
--------------

:func:`torch.distributed.checkpoint` enables saving and loading models from multiple ranks in parallel.
In addition, checkpointing automatically handles fully-qualified-name (FQN) mappings across models and optimizers, enabling load-time resharding across differing cluster topologies.

DCP is different than torch.save and torch.load in a few significant ways:
* It produces multiple files per checkpoint, with at least one per rank.
* It operates in place, meaning that the model should allocate its data first and DCP uses that storage instead.


How to use DCP
--------------

Here we use a toy model wrapped with FSDP for demonstration purposes. Similarly the APIs and logic can be applied to larger models for checkpointing.

*Save*

Now, let’s create a toy module, wrap it with FSDP, feed it with some dummy input data, and save it.

.. code-block:: python

    import os

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as DCP
    import torch.multiprocessing as mp
    import torch.nn as nn

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    CHECKPOINT_DIR = "checkpoint"


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(16, 16)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355 "

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


    def cleanup():
        dist.destroy_process_group()


    def run_fsdp_checkpoint_save_example(rank, world_size):
        print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
        setup(rank, world_size)

        # create a model and move it to GPU with id rank
        model = ToyModel().to(rank)
        model = FSDP(model)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        optimizer.zero_grad()
        model(torch.rand(8, 16, device="cuda")).sum().backward()
        optimizer.step()

        # set FSDP StateDictType to SHARDED_STATE_DICT so we can use DCP to checkpoint sharded model state dict
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        state_dict = {
            "model": model.state_dict(),
        }

        DCP.save_state_dict(
            state_dict=state_dict,
            storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
        )

        cleanup()


    if __name__ == "__main__":
        world_size = torch.cuda.device_count()
        print(f"Running fsdp checkpoint example on {world_size} devices.")
        mp.spawn(
            run_fsdp_checkpoint_save_example,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


*Load*

After saving, let’s create the same FSDP wrapped model, and load the saved state dict from storage into the model. You can load in the same world size or different world size.

.. code-block:: python

    import os

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as DCP
    import torch.multiprocessing as mp
    import torch.nn as nn

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    CHECKPOINT_DIR = "checkpoint"


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(16, 16)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355 "

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


    def cleanup():
        dist.destroy_process_group()


    def run_fsdp_checkpoint_load_example(rank, world_size):
        print(f"Running basic FSDP checkpoint loading example on rank {rank}.")
        setup(rank, world_size)

        # create a model and move it to GPU with id rank
        model = ToyModel().to(rank)
        model = FSDP(model)

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        state_dict = {
            "model": model.state_dict(),
        }

        DCP.load_state_dict(
            state_dict=state_dict,
            storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
        )
        model.load_state_dict(state_dict["model"])

        cleanup()


    if __name__ == "__main__":
        world_size = torch.cuda.device_count()
        print(f"Running fsdp checkpoint example on {world_size} devices.")
        mp.spawn(
            run_fsdp_checkpoint_load_example,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

If you would like to load the saved checkpoint into a non-FSDP wrapped model in a non distributed setup, perhaps for inference, you can also do that with DCP.
By default, DCP saves and loads a distributed state_dict in Single Program Multiple Data(SPMD) style. To load without a distributed setup, please set ``no_dist`` to ``True`` when loading with DCP.

.. code-block:: python
    import os

    import torch
    import torch.distributed.checkpoint as DCP
    import torch.nn as nn


    CHECKPOINT_DIR = "checkpoint"


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(16, 16)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def run_checkpoint_load_example():
        # create the non FSDP-wrapped toy model
        model = ToyModel()
        state_dict = {
            "model": model.state_dict(),
        }

        # turn no_dist to be true to load in non-distributed setting
        DCP.load_state_dict(
            state_dict=state_dict,
            storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
            no_dist=True,
        )
        model.load_state_dict(state_dict["model"])

    if __name__ == "__main__":
        print(f"Running basic DCP checkpoint loading example.")
        run_checkpoint_load_example()
