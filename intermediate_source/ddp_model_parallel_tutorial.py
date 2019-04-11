# -*- coding: utf-8 -*-
"""
Getting Started with Distributed Data Parallel
*************************************************************
**Author**: `Shen Li <https://mrshenli.github.io/>`_

`DistributedDataParallel <https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html>`_
(DDP) implements data parallelism at the module level. It uses communication
collectives in the `torch.distributed <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`_
package to synchronize gradients, parameters, and buffers. Parallelism is
available both within a process and across processes. Within a process, DDP
replicates the input module to devices specified in `device_ids`, scatters
inputs along the batch dimension accordingly, and gathers outputs to the
`output_device`, which is similar to
`DataParallel <https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html>`_.
Across processes, DDP inserts necessary synchronizations in forward and
backward passes. It is up to users to map processes to available resources, as
long as processes do not share GPU devices. The
recommended (usually fastest) approach is to create a process for every module
replica, i.e., no module replication within a process. For demonstration
purpose, this tutorial will create two processes on a 8-GPU machine, with each
exclusively occupying 4 GPUs.

**Basic Usage**

To create DDP modules, first set up process groups properly. More details can be
found in `WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`_.
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as multiprocessing

from torch.multiprocessing import Process
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

######################################################################
# Now, let's create a toy module, wrap it with DDP, and feed it with
# some dummy input data. Please note, if training starts from random
# parameters, you might want to make sure that all DDP processes use the same
# initial values. Otherwise, global gradient synchronizes will not make sense.


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size, device_ids):
    setup(rank, world_size)

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
    per_process = torch.cuda.device_count() // world_size
    processes = []
    for rank in range(world_size):
        # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
        # rank 2 uses GPUs [4, 5, 6, 7].
        device_ids = list(range(rank * per_process, (rank + 1) * per_process))
        p = Process(target=demo_fn, args=(rank, world_size, device_ids))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    run_demo(demo_basic, 2)

######################################################################
# As you can see, DDP wraps lower level distributed communication details, and
# provides a clean API as if it is a local model. For basic use cases, DDP only
# requires a few more LoCs to setup the process group. When applying DDP to
# more advanced use cases, there are some caveats that require cautions.

######################################################################
# Skewed Processing Speeds
# =======================
#
# In DDP, constructor, forward method, and differentiation of the outputs are
# distributed synchronization points. Different processes are expected to reach
# synchronization points in the same order and enter each synchronization point
# at roughly the same time. Otherwise, fast processes might arrive early and
# timeout on waiting for stragglers. Hence, users are responsible for balancing
# workloads distributions across processes. Sometimes, skewed processing speeds
# are inevitable due to, e.g., network delays, resource contentions,
# unpredictable workload spikes. To avoid timeouts in these situations, make
# sure that you pass a sufficiently large `timeout` value when calling
# `init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`_.

######################################################################
# Save and Load Checkpoints
# =======================
#
# It's common to use `torch.save` and `torch.load` to checkpoint modules during
# training and recover from checkpoints. See
# `SAVING AND LOADING MODELS <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_
# for more details. When using DDP, one optimization is to save the model in
# only one process and then load it to all processes, reducing write overhead.
# This is correct
# because all processes start from same parameters and gradients are
# synchronized in backward passes, and hence optimizers should keep setting
# parameters to same values. If use this optimization, make sure all processes
# do not start loading before the saving is finished. Besdies, when loading the
# module, you need to provide an appropriate `map_location` argument to prevent
# a process
# to step into others' devices. If `map_location` is missing, `torch.load` will
# first load the module to CPU and then copy each parameter to where it was
# saved, which would result in all processes on the same machine using the same
# set of devices.


def demo_checkpoint(rank, world_size, device_ids):
    setup(rank, world_size)

    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = "./model.checkpoint"
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
    outputs = ddp_model(torch.randn(20, 10).to(device_ids[0]))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


######################################################################
# Combine DDP with Model Parallelism
# =======================
#
# DDP also works with multi-GPU models, i.e., you may use model parallelism in
# each replica and data parallelism across replicas. This is especially helpful
# when training large models with a huge amount of data. When using this
# feature, the multi-GPU model needs to be carefully implemented to avoid
# hard-coded devices, because different model replicas will be placed to
# different devices. In the example below, the `forward` pass explicitly uses
# the device on layer weights instead of `dev0` and `dev1`.


class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU().to(dev0)
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        # avoid hard-coding devices
        x = x.to(self.net1.weight.device)
        x = self.relu(self.net1(x))
        x = x.to(self.net2.weight.device)
        return self.net2(x)

######################################################################
# To pass a multi-GPU model to DDP, you need to provide a 2D list for
# `device_ids`, where devices in a row are exclusively used by one model
# replica. The devices in `device_ids[0]` must match devices of the input
# model, and the order does not matter. DDP will find model parameters and
# buffers on `devices_ids[0][i]` and replicate them to `devices_ids[j][i]` for
# all replica `j`. Let us walk through an example. Say you construct a
# `ToyMpModel` object using `dev0=0` and `dev1=1`. So `net1` and `relu` resides
# on device 0 and `net2` resides on device 1. If you do not need multiploe
# model replicas within a DDP process, pass `[[0, 1]]` to `device_ids`. If you
# need two replicas, you may set `device_ids` to `[[0, 1], [2, 3]]` instead.
# DDP will create a model replica on devices 2 and 3, with `net1` and `relu`
# replicated to 2, and `net2` replicated to 3.


def demo_model_parallel(rank, world_size):
    setup(rank, world_size)

    # setup mp_model and devices for this process, rank 1 uses GPUs
    # [[0, 1], [2, 3]] and rank 2 uses GPUs [[4, 5], [6, 7]].
    if rank == 0:
        mp_model = ToyMpModel(0, 1)
        device_ids = [[0, 1], [2, 3]]
    else:
        mp_model = ToyMpModel(4, 5)
        device_ids = [[4, 5], [6, 7]]

    ddp_mp_model = DDP(mp_model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # output_device defaults to device_ids[0][0]
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0][0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
