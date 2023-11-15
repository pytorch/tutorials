Getting Started with Distributed Checkpoint (DCP)
=====================================================

**Author**: `Iris Zhang <https://github.com/wz337>`__, `Rodrigo Kumpera <https://github.com/kumpera>`__, `Chien-Chin Huang <https://github.com/fegin>`__

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_checkpoint_recipe.rst>`__.


Prerequisites:

-  `FullyShardedDataParallel API documents <https://pytorch.org/docs/master/fsdp.html>`__
-  `torch.load API documents <https://pytorch.org/docs/stable/generated/torch.load.html>`__


Checkpointing AI models during distributed training could be challenging, as parameters and gradients are partitioned across trainers and the number of trainers available could change when you resume training.
Pytorch Distributed Checkpointing (DCP) can help make this process easier.

In this tutorial, we show how to use DCP APIs with a simple FSDP wrapped model.


How DCP works
--------------

:func:`torch.distributed.checkpoint` enables saving and loading models from multiple ranks in parallel.
In addition, checkpointing automatically handles fully-qualified-name (FQN) mappings across models and optimizers, enabling load-time resharding across differing cluster topologies.

DCP is different from :func:`torch.save` and :func:`torch.load` in a few significant ways:

* It produces multiple files per checkpoint, with at least one per rank.
* It operates in place, meaning that the model should allocate its data first and DCP uses that storage instead.

.. note::
  The code in this tutorial runs on an 8-GPU server, but it can be easily
  generalized to other environments.

How to use DCP
--------------

Here we use a toy model wrapped with FSDP for demonstration purposes. Similarly, the APIs and logic can be applied to larger models for checkpointing.

Saving
~~~~~~

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
        # note that we do not support FSDP StateDictType.LOCAL_STATE_DICT
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

Please go ahead and check the `checkpoint` directory. You should see 8 checkpoint files as shown below.

.. figure:: /_static/img/distributed/distributed_checkpoint_generated_files.png
   :width: 100%
   :align: center
   :alt: Distributed Checkpoint

Loading
~~~~~~~

After saving, let’s create the same FSDP-wrapped model, and load the saved state dict from storage into the model. You can load in the same world size or different world size.

Please note that you will have to call :func:`model.state_dict` prior to loading and pass it to DCP's :func:`load_state_dict` API.
This is fundamentally different from :func:`torch.load`, as :func:`torch.load` simply requires the path to the checkpoint prior for loading.
The reason that we need the ``state_dict`` prior to loading is:

* DCP uses the pre-allocated storage from model state_dict to load from the checkpoint directory. During loading, the state_dict passed in will be updated in place.
* DCP requires the sharding information from the model prior to loading to support resharding.

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
        # different from ``torch.load()``, DCP requires model state_dict prior to loading to get
        # the allocated storage and sharding information.
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

If you would like to load the saved checkpoint into a non-FSDP wrapped model in a non-distributed setup, perhaps for inference, you can also do that with DCP.
By default, DCP saves and loads a distributed ``state_dict`` in Single Program Multiple Data(SPMD) style. To load without a distributed setup, please set ``no_dist`` to ``True`` when loading with DCP.

.. note::
  Distributed checkpoint support for Multi-Program Multi-Data is still under development.

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


Conclusion
----------
In conclusion, we have learned how to use DCP's :func:`save_state_dict` and :func:`load_state_dict` APIs, as well as how they are different form :func:`torch.save` and :func:`torch.load`.

For more information, please see the following:

-  `Saving and loading models tutorial <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`__
-  `Getting started with FullyShardedDataParallel tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__
