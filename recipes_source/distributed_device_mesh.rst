Getting Started with DeviceMesh
=====================================================

**Author**: `Iris Zhang <https://github.com/wz337>`__, `Wanchao Liang <https://github.com/wanchaol>`__

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_device_mesh.rst>`__.

Prerequisites:

- `Distributed Communication Package - torch.distributed <https://pytorch.org/docs/stable/distributed.html>`__
- Python
- PyTorch 2.2


Setting up the NVIDIA Collective Communication Library (NCCL) communicators for distributed communication during distributed training can pose a significant challenge. For workloads where users need to compose different parallelisms,
users would need to manually set up and manage NCCL communicators (for example, :class:`ProcessGroup`) for each parallelism solutions. This process could be complicated and susceptible to errors.
:class:`DeviceMesh` can simplify this process, making it more manageable and less prone to errors.

What is DeviceMesh
------------------
:class:`DeviceMesh` is a higher level abstraction that manages :class:`ProcessGroup`. It allows users to effortlessly
create inter-node and intra-node process groups without worrying about how to set up ranks correctly for different sub process groups.
Users can also easily manage the underlying process_groups/devices for multi-dimensional parallelism via :class:`DeviceMesh`.

.. figure:: /_static/img/distributed/device_mesh.png
   :width: 100%
   :align: center
   :alt: PyTorch DeviceMesh

Why DeviceMesh is Useful
------------------------

The following code snippet illustrates a 2D setup without :class:`DeviceMesh`. First, we need to manually calculate the shard group and replicate group. Then, we need to assign the correct shard and
replicate group to each rank.

.. code-block:: python
    import os

    import torch
    import torch.distributed as dist

    # Understand world topology
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Running example on {rank=} in a world with {world_size=}")

    # Create process groups to manage 2-D like parallel pattern
    dist.init_process_group("nccl")

    # Create shard groups (e.g. (0, 1, 2, 3), (4, 5, 6, 7))
    # and assign the correct shard group to each rank
    num_node_devices = torch.cuda.device_count()
    shard_rank_lists = list(range(0, num_node_devices // 2)), list(range(num_node_devices // 2, num_node_devices))
    shard_groups = (
        dist.new_group(shard_rank_lists[0]),
        dist.new_group(shard_rank_lists[1]),
    )
    current_shard_group = (
        shard_groups[0] if rank in shard_rank_lists[0] else shard_groups[1]
    )

    # Create replicate groups (for example, (0, 4), (1, 5), (2, 6), (3, 7))
    # and assign the correct replicate group to each rank
    current_replicate_group = None
    shard_factor = len(shard_rank_lists[0])
    for i in range(num_node_devices // 2):
        replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
        replicate_group = dist.new_group(replicate_group_ranks)
        if rank in replicate_group_ranks:
            current_replicate_group = replicate_group

To run the above code snippet, we can leverage PyTorch Elastic. Let's create a file named ``2d_setup.py``.
Then, run the following `torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ command.

.. code-block:: python
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint=localhost:29400 2d_setup.py


With the help of :func:`init_device_mesh`, we can accomplish the above 2D setup in just two lines.


.. code-block:: python
    from torch.distributed.device_mesh import init_device_mesh
    device_mesh = init_device_mesh("cuda", (2, 4))

Let's create a file named ``2d_setup_with_device_mesh.py``.
Then, run the following `torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ command.

.. code-block:: python
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint=localhost:29400 2d_setup_with_device_mesh.py


How to use DeviceMesh with HSDP
-------------------------------

Hybrid Sharding Data Parallel(HSDP) is 2D strategy to perform FSDP within a host and DDP across hosts.

Let's see an example of how DeviceMesh can assist with applying HSDP to your model. With DeviceMesh,
users would not need to manually create and manage shard group and replicate group.

.. code-block:: python
    import torch
    import torch.nn as nn

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    # HSDP: MeshShape(2, 4)
    mesh_2d = init_device_mesh("cuda", (2, 4))
    model = FSDP(
        ToyModel(), device_mesh=mesh_2d, sharding_strategy=ShardingStrategy.HYBRID_SHARD
    )

Let's create a file named ``hsdp.py``.
Then, run the following `torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ command.

.. code-block:: python
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint=localhost:29400  hsdp.py

Conclusion
----------
In conclusion, we have learned about :class:`DeviceMesh` and :func:`init_device_mesh`, as well as how
they can be used to describe the layout of devices across the cluster.

For more information, please see the following:

- `2D parallel combining Tensor/Sequance Parallel with FSDP <https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py>`__
- `Composable PyTorch Distributed with PT2 <chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://static.sched.com/hosted_files/pytorch2023/d1/%5BPTC%2023%5D%20Composable%20PyTorch%20Distributed%20with%20PT2.pdf>`__
