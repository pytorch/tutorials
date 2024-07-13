Getting Started with Distributed Data Parallel
=================================================
**Author**: `Shen Li <https://mrshenli.github.io/>`_

**Edited by**: `Joe Zhu <https://github.com/gunandrose4u>`_

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/ddp_tutorial.rst>`__.

Prerequisites:

-  `PyTorch Distributed Overview <../beginner/dist_overview.html>`__
-  `DistributedDataParallel API documents <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__
-  `DistributedDataParallel notes <https://pytorch.org/docs/master/notes/ddp.html>`__


`DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#module-torch.nn.parallel>`__
(DDP) implements data parallelism at the module level which can run across
multiple machines. Applications using DDP should spawn multiple processes and
create a single DDP instance per process. DDP uses collective communications in the
`torch.distributed <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`__
package to synchronize gradients and buffers. More specifically, DDP registers
an autograd hook for each parameter given by ``model.parameters()`` and the
hook will fire when the corresponding gradient is computed in the backward
pass. Then DDP uses that signal to trigger gradient synchronization across
processes. Please refer to
`DDP design note <https://pytorch.org/docs/master/notes/ddp.html>`__ for more details.


The recommended way to use DDP is to spawn one process for each model replica,
where a model replica can span multiple devices. DDP processes can be
placed on the same machine or across machines, but GPU devices cannot be
shared across processes. This tutorial starts from a basic DDP use case and
then demonstrates more advanced use cases including checkpointing models and
combining DDP with model parallel.


.. note::
  The code in this tutorial runs on an 8-GPU server, but it can be easily
  generalized to other environments.


Comparison between ``DataParallel`` and ``DistributedDataParallel``
-------------------------------------------------------------------

Before we dive in, let's clarify why, despite the added complexity, you would
consider using ``DistributedDataParallel`` over ``DataParallel``:

- First, ``DataParallel`` is single-process, multi-thread, and only works on a
  single machine, while ``DistributedDataParallel`` is multi-process and works
  for both single- and multi- machine training. ``DataParallel`` is usually
  slower than ``DistributedDataParallel`` even on a single machine due to GIL
  contention across threads, per-iteration replicated model, and additional
  overhead introduced by scattering inputs and gathering outputs.
- Recall from the
  `prior tutorial <https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html>`__
  that if your model is too large to fit on a single GPU, you must use **model parallel**
  to split it across multiple GPUs. ``DistributedDataParallel`` works with
  **model parallel**; ``DataParallel`` does not at this time. When DDP is combined
  with model parallel, each DDP process would use model parallel, and all processes
  collectively would use data parallel.
- If your model needs to span multiple machines or if your use case does not fit
  into data parallelism paradigm, please see `the RPC API <https://pytorch.org/docs/stable/rpc.html>`__
  for more generic distributed training support.

Basic Use Case
--------------

To create a DDP module, you must first set up process groups properly. More details can
be found in
`Writing Distributed Applications with PyTorch <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`__.

.. code:: python

    import os
    import sys
    import tempfile
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    import torch.multiprocessing as mp

    from torch.nn.parallel import DistributedDataParallel as DDP

    # On Windows platform, the torch.distributed package only
    # supports Gloo backend, FileStore and TcpStore.
    # For FileStore, set init_method parameter in init_process_group
    # to a local file. Example as follow:
    # init_method="file:///f:/libtmp/some_file"
    # dist.init_process_group(
    #    "gloo",
    #    rank=rank,
    #    init_method=init_method,
    #    world_size=world_size)
    # For TcpStore, same way as on Linux.

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

Now, let's create a toy module, wrap it with DDP, and feed it some dummy
input data. Please note, as DDP broadcasts model states from rank 0 process to
all other processes in the DDP constructor, you do not need to worry about
different DDP processes starting from different initial model parameter values.

.. code:: python

    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def demo_basic(rank, world_size):
        print(f"Running basic DDP example on rank {rank}.")
        setup(rank, world_size)

        # create model and move it to GPU with id rank
        model = ToyModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()

        cleanup()


    def run_demo(demo_fn, world_size):
        mp.spawn(demo_fn,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)

As you can see, DDP wraps lower-level distributed communication details and
provides a clean API as if it were a local model. Gradient synchronization
communications take place during the backward pass and overlap with the
backward computation. When the ``backward()`` returns, ``param.grad`` already
contains the synchronized gradient tensor. For basic use cases, DDP only
requires a few more LoCs to set up the process group. When applying DDP to more
advanced use cases, some caveats require caution.

Skewed Processing Speeds
------------------------

In DDP, the constructor, the forward pass, and the backward pass are
distributed synchronization points. Different processes are expected to launch
the same number of synchronizations and reach these synchronization points in
the same order and enter each synchronization point at roughly the same time.
Otherwise, fast processes might arrive early and timeout while waiting for
stragglers. Hence, users are responsible for balancing workload distributions
across processes. Sometimes, skewed processing speeds are inevitable due to,
e.g., network delays, resource contentions, or unpredictable workload spikes. To
avoid timeouts in these situations, make sure that you pass a sufficiently
large ``timeout`` value when calling
`init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__.

Save and Load Checkpoints
-------------------------

It's common to use ``torch.save`` and ``torch.load`` to checkpoint modules
during training and recover from checkpoints. See
`SAVING AND LOADING MODELS <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`__
for more details. When using DDP, one optimization is to save the model in
only one process and then load it to all processes, reducing write overhead.
This is correct because all processes start from the same parameters and
gradients are synchronized in backward passes, and hence optimizers should keep
setting parameters to the same values. If you use this optimization, make sure no process starts 
loading before the saving is finished. Additionally, when
loading the module, you need to provide an appropriate ``map_location``
argument to prevent a process from stepping into others' devices. If ``map_location``
is missing, ``torch.load`` will first load the module to CPU and then copy each
parameter to where it was saved, which would result in all processes on the
same machine using the same set of devices. For more advanced failure recovery
and elasticity support, please refer to `TorchElastic <https://pytorch.org/elastic>`__.

.. code:: python

    def demo_checkpoint(rank, world_size):
        print(f"Running DDP checkpoint example on rank {rank}.")
        setup(rank, world_size)

        model = ToyModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])


        CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location))

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(rank)

        loss_fn(outputs, labels).backward()
        optimizer.step()

        # Not necessary to use a dist.barrier() to guard the file deletion below
        # as the AllReduce ops in the backward pass of DDP already served as
        # a synchronization.

        if rank == 0:
            os.remove(CHECKPOINT_PATH)

        cleanup()

Combining DDP with Model Parallelism
------------------------------------

DDP also works with multi-GPU models. DDP wrapping multi-GPU models is especially
helpful when training large models with a huge amount of data.

.. code:: python

    class ToyMpModel(nn.Module):
        def __init__(self, dev0, dev1):
            super(ToyMpModel, self).__init__()
            self.dev0 = dev0
            self.dev1 = dev1
            self.net1 = torch.nn.Linear(10, 10).to(dev0)
            self.relu = torch.nn.ReLU()
            self.net2 = torch.nn.Linear(10, 5).to(dev1)

        def forward(self, x):
            x = x.to(self.dev0)
            x = self.relu(self.net1(x))
            x = x.to(self.dev1)
            return self.net2(x)

When passing a multi-GPU model to DDP, ``device_ids`` and ``output_device``
must NOT be set. Input and output data will be placed in proper devices by
either the application or the model ``forward()`` method.

.. code:: python

    def demo_model_parallel(rank, world_size):
        print(f"Running DDP with model parallel example on rank {rank}.")
        setup(rank, world_size)

        # setup mp_model and devices for this process
        dev0 = rank * 2
        dev1 = rank * 2 + 1
        mp_model = ToyMpModel(dev0, dev1)
        ddp_mp_model = DDP(mp_model)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        # outputs will be on dev1
        outputs = ddp_mp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(dev1)
        loss_fn(outputs, labels).backward()
        optimizer.step()

        cleanup()


    if __name__ == "__main__":
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(demo_basic, world_size)
        run_demo(demo_checkpoint, world_size)
        world_size = n_gpus//2
        run_demo(demo_model_parallel, world_size)

Initialize DDP with torch.distributed.run/torchrun
---------------------------------------------------

We can leverage PyTorch Elastic to simplify the DDP code and initialize the job more easily.
Let's still use the Toymodel example and create a file named ``elastic_ddp.py``.

.. code:: python

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim

    from torch.nn.parallel import DistributedDataParallel as DDP

    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def demo_basic():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")
   
        # create model and move it to GPU with id rank
        device_id = rank % torch.cuda.device_count()
        model = ToyModel().to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(device_id)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        dist.destroy_process_group()
        
    if __name__ == "__main__":
        demo_basic()

One can then run a `torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ command 
on all nodes to initialize the DDP job created above:

.. code:: bash

    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py

We are running the DDP script on two hosts, and each host we run with 8 processes, aka, we 
are running it on 16 GPUs. Note that ``$MASTER_ADDR`` must be the same across all nodes.

Here torchrun will launch 8 process and invoke ``elastic_ddp.py`` 
on each process on the node it is launched on, but user also needs to apply cluster 
management tools like slurm to actually run this command on 2 nodes.

For example, on a SLURM enabled cluster, we can write a script to run the command above
and set ``MASTER_ADDR`` as:

.. code:: bash

    export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)


Then we can just run this script using the SLURM command: ``srun --nodes=2 ./torchrun_script.sh``.
Of course, this is just an example; you can choose your own cluster scheduling tools
to initiate the torchrun job.

For more information about Elastic run, one can check this 
`quick start document <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ to learn more.
