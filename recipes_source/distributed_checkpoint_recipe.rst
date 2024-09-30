Getting Started with Distributed Checkpoint (DCP)
=====================================================

**Author**: `Iris Zhang <https://github.com/wz337>`__, `Rodrigo Kumpera <https://github.com/kumpera>`__, `Chien-Chin Huang <https://github.com/fegin>`__, `Lucas Pasqualin <https://github.com/lucasllc>`__

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

:func:`torch.distributed.checkpoint` enables saving and loading models from multiple ranks in parallel. You can use this module to save on any number of ranks in parallel,
and then re-shard across differing cluster topologies at load time.

Addditionally, through the use of modules in :func:`torch.distributed.checkpoint.state_dict`,
DCP offers support for gracefully handling ``state_dict`` generation and loading in distributed settings.
This includes managing fully-qualified-name (FQN) mappings across models and optimizers, and setting default parameters for PyTorch provided parallelisms.

DCP is different from :func:`torch.save` and :func:`torch.load` in a few significant ways:

* It produces multiple files per checkpoint, with at least one per rank.
* It operates in place, meaning that the model should allocate its data first and DCP uses that storage instead.
* DCP offers special handling of Stateful objects (formally defined in `torch.distributed.checkpoint.stateful`), automatically calling both `state_dict` and `load_state_dict` methods if they are defined.

.. note::
  The code in this tutorial runs on an 8-GPU server, but it can be easily
  generalized to other environments.

How to use DCP
--------------

Here we use a toy model wrapped with FSDP for demonstration purposes. Similarly, the APIs and logic can be applied to larger models for checkpointing.

Saving
~~~~~~

Now, let's create a toy module, wrap it with FSDP, feed it with some dummy input data, and save it.

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
            model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
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

        optimizer.zero_grad()
        model(torch.rand(8, 16, device="cuda")).sum().backward()
        optimizer.step()

        state_dict = { "app": AppState(model, optimizer) }
        dcp.save(state_dict, checkpoint_id=CHECKPOINT_DIR)

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
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.stateful import Stateful
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    import torch.multiprocessing as mp
    import torch.nn as nn

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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
            model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
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


    def run_fsdp_checkpoint_load_example(rank, world_size):
        print(f"Running basic FSDP checkpoint loading example on rank {rank}.")
        setup(rank, world_size)

        # create a model and move it to GPU with id rank
        model = ToyModel().to(rank)
        model = FSDP(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        state_dict = { "app": AppState(model, optimizer)}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        # generates the state dict we will load into
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict
        }
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )

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
By default, DCP saves and loads a distributed ``state_dict`` in Single Program Multiple Data(SPMD) style. However if no process group is initialized, DCP infers
the intent is to save or load in "non-distributed" style, meaning entirely in the current process.

.. note::
  Distributed checkpoint support for Multi-Program Multi-Data is still under development.

.. code-block:: python

    import os

    import torch
    import torch.distributed.checkpoint as dcp
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

        # since no progress group is initialized, DCP will disable any collectives.
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )
        model.load_state_dict(state_dict["model"])

    if __name__ == "__main__":
        print(f"Running basic DCP checkpoint loading example.")
        run_checkpoint_load_example()


Formats
----------
One drawback not yet mentioned is that DCP saves checkpoints in a format which is inherently different then those generated using torch.save.
Since this can be an issue when users wish to share models with users used to the torch.save format, or in general just want to add format flexibility
to their applications. For this case, we provide the ``format_utils`` module in ``torch.distributed.checkpoint.format_utils``.

A command line utility is provided for the users convenience, which follows the following format:

.. code-block:: bash

    python -m torch.distributed.checkpoint.format_utils -m <checkpoint location> <location to write formats to> <mode>

In the command above, ``mode`` is one of ``torch_to_dcp``` or ``dcp_to_torch``.


Alternatively, methods are also provided for users who may wish to convert checkpoints directly.

.. code-block:: python

    import os

    import torch
    import torch.distributed.checkpoint as DCP
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

    CHECKPOINT_DIR = "checkpoint"
    TORCH_SAVE_CHECKPOINT_DIR = "torch_save_checkpoint.pth"

    # convert dcp model to torch.save (assumes checkpoint was generated as above)
    dcp_to_torch_save(CHECKPOINT_DIR, TORCH_SAVE_CHECKPOINT_DIR)

    # converts the torch.save model back to DCP
    dcp_to_torch_save(TORCH_SAVE_CHECKPOINT_DIR, f"{CHECKPOINT_DIR}_new")



Conclusion
----------
In conclusion, we have learned how to use DCP's :func:`save` and :func:`load` APIs, as well as how they are different form :func:`torch.save` and :func:`torch.load`.
Additionally, we've learned how to use :func:`get_state_dict` and :func:`set_state_dict` to automatically manage parallelism-specific FQN's and defaults during state dict
generation and loading.

For more information, please see the following:

-  `Saving and loading models tutorial <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`__
-  `Getting started with FullyShardedDataParallel tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__
