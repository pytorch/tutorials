Asynchronous Saving with Distributed Checkpoint (DCP)
=====================================================

Checkpointing is often a bottle-neck in the critical path for distributed training workloads, incurring larger and larger costs as both model and world sizes grow.
One excellent strategy for offsetting this cost is to checkpoint in parallel, asynchronously. Below, we expand the save example
from the `Getting Started with Distributed Checkpoint Tutorial <https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_checkpoint_recipe.rst>`__
to show how this can be integrated quite easily with ``torch.distributed.checkpoint.async_save``.

**Author**: , `Lucas Pasqualin <https://github.com/lucasllc>`__, `Iris Zhang <https://github.com/wz337>`__, `Rodrigo Kumpera <https://github.com/kumpera>`__, `Chien-Chin Huang <https://github.com/fegin>`__

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to use DCP to generate checkpoints in parallel
       * Effective strategies to optimize performance

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.4.0 or later
       * `Getting Started with Distributed Checkpoint Tutorial <https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_checkpoint_recipe.rst>`__


Asynchronous Checkpointing Overview
------------------------------------
Before getting started with Asynchronous Checkpointing, it's important to understand it's differences and limitations as compared to synchronous checkpointing.
Specifically:

* Memory requirements - Asynchronous checkpointing works by first copying models into internal CPU-buffers.
    This is helpful since it ensures model and optimizer weights are not changing while the model is still checkpointing,
    but does raise CPU memory by a factor of ``checkpoint_size_per_rank X number_of_ranks``. Additionally, users should take care to understand
    the memory constraints of their systems. Specifically, pinned memory implies the usage of ``page-lock`` memory, which can be scarce as compared to
    ``pageable`` memory.

* Checkpoint Management - Since checkpointing is asynchronous, it is up to the user to manage concurrently run checkpoints. In general, users can
    employ their own management strategies by handling the future object returned form ``async_save``. For most users, we recommend limiting
    checkpoints to one asynchronous request at a time, avoiding additional memory pressure per request.



.. code-block:: python

    import os

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.multiprocessing as mp
    import torch.nn as nn

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.checkpoint.stateful import Stateful
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    CHECKPOINT_DIR = "checkpoint"


    class AppState(Stateful):
        """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
        with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
        dcp.save/load APIs.

        Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
        and optimizer.
        """

        def __init__(self, model, optimizer=None):
            self.model = model
            self.optimizer = optimizer

        def state_dict(self):
            # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
            model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
            return {
                "model": model_state_dict,
                "optim": optimizer_state_dict
            }

        def load_state_dict(self, state_dict):
            # sets our state dicts on the model and optimizer, now that we've loaded
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"]
            )

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

        checkpoint_future = None
        for step in range(10):
            optimizer.zero_grad()
            model(torch.rand(8, 16, device="cuda")).sum().backward()
            optimizer.step()

            # waits for checkpointing to finish if one exists, avoiding queuing more then one checkpoint request at a time
            if checkpoint_future is not None:
                checkpoint_future.result()

            state_dict = { "app": AppState(model, optimizer) }
            checkpoint_future = dcp.async_save(state_dict, checkpoint_id=f"{CHECKPOINT_DIR}_step{step}")

        cleanup()


    if __name__ == "__main__":
        world_size = torch.cuda.device_count()
        print(f"Running async checkpoint example on {world_size} devices.")
        mp.spawn(
            run_fsdp_checkpoint_save_example,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


Even more performance with Pinned Memory
-----------------------------------------
If the above optimization is still not performant enough, you can take advantage of an additional optimization for GPU models which utilizes a pinned memory buffer for checkpoint staging.
Specifically, this optimization attacks the main overhead of asynchronous checkpointing, which is the in-memory copying to checkpointing buffers. By maintaining a pinned memory buffer between
checkpoint requests users can take advantage of direct memory access to speed up this copy.

.. note:: The main drawback of this optimization is the persistence of the buffer in between checkpointing steps. Without the pinned memory optimization (as demonstrated above),
any checkpointing buffers are released as soon as checkpointing is finished. With the pinned memory implementation, this buffer is maintained between steps, leading to the same
peak memory pressure being sustained through the application life.


.. code-block:: python

    import os

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.multiprocessing as mp
    import torch.nn as nn

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.checkpoint.stateful import Stateful
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
    from torch.distributed.checkpoint import StorageWriter

    CHECKPOINT_DIR = "checkpoint"


    class AppState(Stateful):
        """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
        with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
        dcp.save/load APIs.

        Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
        and optimizer.
        """

        def __init__(self, model, optimizer=None):
            self.model = model
            self.optimizer = optimizer

        def state_dict(self):
            # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
            model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
            return {
                "model": model_state_dict,
                "optim": optimizer_state_dict
            }

        def load_state_dict(self, state_dict):
            # sets our state dicts on the model and optimizer, now that we've loaded
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"]
            )

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

        # The storage writer defines our 'staging' strategy, where staging is considered the process of copying
        # checkpoints to in-memory buffers. By setting `cached_state_dict=True`, we enable efficient memory copying
        # into a persistent buffer with pinned memory enabled.
        # Note: It's important that the writer persists in between checkpointing requests, since it maintains the
        # pinned memory buffer.
        writer = StorageWriter(cached_state_dict=True)
        checkpoint_future = None
        for step in range(10):
            optimizer.zero_grad()
            model(torch.rand(8, 16, device="cuda")).sum().backward()
            optimizer.step()

            state_dict = { "app": AppState(model, optimizer) }
            if checkpoint_future is not None:
                # waits for checkpointing to finish, avoiding queuing more then one checkpoint request at a time
                checkpoint_future.result()
            dcp.async_save(state_dict, storage_writer=writer, checkpoint_id=f"{CHECKPOINT_DIR}_step{step}")

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


Conclusion
----------
In conclusion, we have learned how to use DCP's :func:`async_save` API to generate checkpoints off the critical training path. We've also learned about the
additional memory and concurrency overhead introduced by using this API, as well as additional optimizations which utilize pinned memory to speed things up
even further.

-  `Saving and loading models tutorial <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`__
-  `Getting started with FullyShardedDataParallel tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__
