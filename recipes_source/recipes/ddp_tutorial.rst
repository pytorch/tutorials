Getting Started with Distributed Data Parallel
=================================================
**Author**: `Shen Li <https://mrshenli.github.io/>`_

`DistributedDataParallel <https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html>`__
(DDP) implements data parallelism at the module level. It uses communication
collectives in the `torch.distributed <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`__
package to synchronize gradients, parameters, and buffers. Parallelism is
available both within a process and across processes. Within a process, DDP
replicates the input module to devices specified in ``device_ids``, scatters
inputs along the batch dimension accordingly, and gathers outputs to the
``output_device``, which is similar to
`DataParallel <https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html>`__.
Across processes, DDP inserts necessary parameter synchronizations in forward
passes and gradient synchronizations in backward passes. It is up to users to
map processes to available resources, as long as processes do not share GPU
devices. The recommended (usually fastest) approach is to create a process for
every module replica, i.e., no module replication within a process. The code in
this tutorial runs on an 8-GPU server, but it can be easily generalized to
other environments.

Comparison between ``DataParallel`` and ``DistributedDataParallel``
-------------------------------------------------------------------

Before we dive in, let's clarify why, despite the added complexity, you would
consider using ``DistributedDataParallel`` over ``DataParallel``:

- First, recall from the
  `prior tutorial <https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html>`__
  that if your model is too large to fit on a single GPU, you must use **model parallel**
  to split it across multiple GPUs. ``DistributedDataParallel`` works with
  **model parallel**; ``DataParallel`` does not at this time.
- ``DataParallel`` is single-process, multi-thread, and only works on a single
  machine, while ``DistributedDataParallel`` is multi-process and works for both
  single- and multi- machine training. Thus, even for single machine training,
  where your **data** is small enough to fit on a single machine, ``DistributedDataParallel``
  is expected to be faster than ``DataParallel``. ``DistributedDataParallel``
  also replicates models upfront instead of on each iteration and gets Global
  Interpreter Lock out of the way.
- If both your data is too large to fit on one machine **and** your
  model is too large to fit on a single GPU, you can combine model parallel
  (splitting a single model across multiple GPUs) with ``DistributedDataParallel``.
  Under this regime, each ``DistributedDataParallel`` process could use model parallel,
  and all processes collectively would use data parallel.

Basic Use Case
--------------

To create DDP modules, first set up process groups properly. More details can
be found in
`Writing Distributed Applications with PyTorch <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`__.

.. code:: python

    import os
    import tempfile
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    import torch.multiprocessing as mp

    from torch.nn.parallel import DistributedDataParallel as DDP


    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(42)


    def cleanup():
        dist.destroy_process_group()

Now, let's create a toy module, wrap it with DDP, and feed it with some dummy
input data. Please note, if training starts from random parameters, you might
want to make sure that all DDP processes use the same initial values.
Otherwise, global gradient synchronizes will not make sense.

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
        setup(rank, world_size)

        # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
        # rank 2 uses GPUs [4, 5, 6, 7].
        n = torch.cuda.device_count() // world_size
        device_ids = list(range(rank * n, (rank + 1) * n))

        # create model and move it to device_ids[0]
        model = ToyModel().to(device_ids[0])
        # output_device defaults to device_ids[0]
        ddp_model = DDP(model, device_ids=device_ids)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(device_ids[0])
        loss_fn(outputs, labels).backward()
        optimizer.step()

        cleanup()


    def run_demo(demo_fn, world_size):
        mp.spawn(demo_fn,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)

As you can see, DDP wraps lower level distributed communication details, and
provides a clean API as if it is a local model. For basic use cases, DDP only
requires a few more LoCs to set up the process group. When applying DDP to more
advanced use cases, there are some caveats that require cautions.

Skewed Processing Speeds
------------------------

In DDP, constructor, forward method, and differentiation of the outputs are
distributed synchronization points. Different processes are expected to reach
synchronization points in the same order and enter each synchronization point
at roughly the same time. Otherwise, fast processes might arrive early and
timeout on waiting for stragglers. Hence, users are responsible for balancing
workloads distributions across processes. Sometimes, skewed processing speeds
are inevitable due to, e.g., network delays, resource contentions,
unpredictable workload spikes. To avoid timeouts in these situations, make
sure that you pass a sufficiently large ``timeout`` value when calling
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
setting parameters to same values. If you use this optimization, make sure all
processes do not start loading before the saving is finished. Besides, when
loading the module, you need to provide an appropriate ``map_location``
argument to prevent a process to step into others' devices. If ``map_location``
is missing, ``torch.load`` will first load the module to CPU and then copy each
parameter to where it was saved, which would result in all processes on the
same machine using the same set of devices.

.. code:: python

    def demo_checkpoint(rank, world_size):
        setup(rank, world_size)

        # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
        # rank 2 uses GPUs [4, 5, 6, 7].
        n = torch.cuda.device_count() // world_size
        device_ids = list(range(rank * n, (rank + 1) * n))

        model = ToyModel().to(device_ids[0])
        # output_device defaults to device_ids[0]
        ddp_model = DDP(model, device_ids=device_ids)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

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
        rank0_devices = [x - rank * len(device_ids) for x in device_ids]
        device_pairs = zip(rank0_devices, device_ids)
        map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
        ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location))

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(device_ids[0])
        loss_fn = nn.MSELoss()
        loss_fn(outputs, labels).backward()
        optimizer.step()

        # Use a barrier() to make sure that all processes have finished reading the
        # checkpoint
        dist.barrier()

        if rank == 0:
            os.remove(CHECKPOINT_PATH)

        cleanup()

Combine DDP with Model Parallelism
----------------------------------

DDP also works with multi-GPU models, but replications within a process are not
supported. You need to create one process per module replica, which usually
leads to better performance compared to multiple replicas per process. DDP
wrapping multi-GPU models is especially helpful when training large models with
a huge amount of data. When using this feature, the multi-GPU model needs to be
carefully implemented to avoid hard-coded devices, because different model
replicas will be placed to different devices.

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
        run_demo(demo_basic, 2)
        run_demo(demo_checkpoint, 2)

        if torch.cuda.device_count() >= 8:
            run_demo(demo_model_parallel, 4)
