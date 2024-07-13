Exploring TorchRec sharding
===========================

This tutorial will mainly cover the sharding schemes of embedding tables
via ``EmbeddingPlanner`` and ``DistributedModelParallel`` API and
explore the benefits of different sharding schemes for the embedding
tables by explicitly configuring them.

Installation
------------

Requirements: - python >= 3.7

We highly recommend CUDA when using torchRec. If using CUDA: - cuda >=
11.0

.. code:: python

    # install conda to make installying pytorch with cudatoolkit 11.3 easier. 
    !sudo rm Miniconda3-py37_4.9.2-Linux-x86_64.sh Miniconda3-py37_4.9.2-Linux-x86_64.sh.*
    !sudo wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh
    !sudo chmod +x Miniconda3-py37_4.9.2-Linux-x86_64.sh
    !sudo bash ./Miniconda3-py37_4.9.2-Linux-x86_64.sh -b -f -p /usr/local

.. code:: python

    # install pytorch with cudatoolkit 11.3
    !sudo conda install pytorch cudatoolkit=11.3 -c pytorch-nightly -y

Installing torchRec will also install
`FBGEMM <https://github.com/pytorch/fbgemm>`__, a collection of CUDA
kernels and GPU enabled operations to run

.. code:: python

    # install torchrec
    !pip3 install torchrec-nightly

Install multiprocess which works with ipython to for multi-processing
programming within colab

.. code:: python

    !pip3 install multiprocess

The following steps are needed for the Colab runtime to detect the added
shared libraries. The runtime searches for shared libraries in /usr/lib,
so we copy over the libraries which were installed in /usr/local/lib/.
**This is a very necessary step, only in the colab runtime**.

.. code:: python

    !sudo cp /usr/local/lib/lib* /usr/lib/

**Restart your runtime at this point for the newly installed packages
to be seen.** Run the step below immediately after restarting so that
python knows where to look for packages. **Always run this step after
restarting the runtime.**

.. code:: python

    import sys
    sys.path = ['', '/env/python', '/usr/local/lib/python37.zip', '/usr/local/lib/python3.7', '/usr/local/lib/python3.7/lib-dynload', '/usr/local/lib/python3.7/site-packages', './.local/lib/python3.7/site-packages']


Distributed Setup
-----------------

Due to the notebook enviroment, we cannot run
`SPMD <https://en.wikipedia.org/wiki/SPMD>`_ program here but we
can do multiprocessing inside the notebook to mimic the setup. Users
should be responsible for setting up their own
`SPMD <https://en.wikipedia.org/wiki/SPMD>`_ launcher when using
Torchrec. We setup our environment so that torch distributed based
communication backend can work.

.. code:: python

    import os
    import torch
    import torchrec

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

Constructing our embedding model
--------------------------------

Here we use TorchRec offering of
`EmbeddingBagCollection <https://github.com/facebookresearch/torchrec/blob/main/torchrec/modules/embedding_modules.py#L59>`_
to construct our embedding bag model with embedding tables.

Here, we create an EmbeddingBagCollection (EBC) with four embedding
bags. We have two types of tables: large tables and small tables
differentiated by their row size difference: 4096 vs 1024. Each table is
still represented by 64 dimension embedding.

We configure the ``ParameterConstraints`` data structure for the tables,
which provides hints for the model parallel API to help decide the
sharding and placement strategy for the tables. In TorchRec, we support
\* ``table-wise``: place the entire table on one device; \*
``row-wise``: shard the table evenly by row dimension and place one
shard on each device of the communication world; \* ``column-wise``:
shard the table evenly by embedding dimension, and place one shard on
each device of the communication world; \* ``table-row-wise``: special
sharding optimized for intra-host communication for available fast
intra-machine device interconnect, e.g. NVLink; \* ``data_parallel``:
replicate the tables for every device;

Note how we initially allocate the EBC on device "meta". This will tell
EBC to not allocate memory yet.

.. code:: python

    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.embedding_types import EmbeddingComputeKernel
    from torchrec.distributed.types import ShardingType
    from typing import Dict

    large_table_cnt = 2
    small_table_cnt = 2
    large_tables=[
      torchrec.EmbeddingBagConfig(
        name="large_table_" + str(i),
        embedding_dim=64,
        num_embeddings=4096,
        feature_names=["large_table_feature_" + str(i)],
        pooling=torchrec.PoolingType.SUM,
      ) for i in range(large_table_cnt)
    ]
    small_tables=[
      torchrec.EmbeddingBagConfig(
        name="small_table_" + str(i),
        embedding_dim=64,
        num_embeddings=1024,
        feature_names=["small_table_feature_" + str(i)],
        pooling=torchrec.PoolingType.SUM,
      ) for i in range(small_table_cnt)
    ]

    def gen_constraints(sharding_type: ShardingType = ShardingType.TABLE_WISE) -> Dict[str, ParameterConstraints]:
      large_table_constraints = {
        "large_table_" + str(i): ParameterConstraints(
          sharding_types=[sharding_type.value],
        ) for i in range(large_table_cnt)
      }
      small_table_constraints = {
        "small_table_" + str(i): ParameterConstraints(
          sharding_types=[sharding_type.value],
        ) for i in range(small_table_cnt)
      }
      constraints = {**large_table_constraints, **small_table_constraints}
      return constraints

.. code:: python

    ebc = torchrec.EmbeddingBagCollection(
        device="cuda",
        tables=large_tables + small_tables
    )

DistributedModelParallel in multiprocessing
-------------------------------------------

Now, we have a single process execution function for mimicking one
rank's work during `SPMD <https://en.wikipedia.org/wiki/SPMD>`_
execution.

This code will shard the model collectively with other processes and
allocate memories accordingly. It first sets up process groups and do
embedding table placement using planner and generate sharded model using
``DistributedModelParallel``.

.. code:: python

    def single_rank_execution(
        rank: int,
        world_size: int,
        constraints: Dict[str, ParameterConstraints],
        module: torch.nn.Module,
        backend: str,
    ) -> None:
        import os
        import torch
        import torch.distributed as dist
        from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
        from torchrec.distributed.model_parallel import DistributedModelParallel
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.types import ModuleSharder, ShardingEnv
        from typing import cast

        def init_distributed_single_host(
            rank: int,
            world_size: int,
            backend: str,
            # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
        ) -> dist.ProcessGroup:
            os.environ["RANK"] = f"{rank}"
            os.environ["WORLD_SIZE"] = f"{world_size}"
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            return dist.group.WORLD

        if backend == "nccl":
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        topology = Topology(world_size=world_size, compute_device="cuda")
        pg = init_distributed_single_host(rank, world_size, backend)
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
        )
        sharders = [cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())]
        plan: ShardingPlan = planner.collective_plan(module, sharders, pg)
    
        sharded_model = DistributedModelParallel(
            module,
            env=ShardingEnv.from_process_group(pg),
            plan=plan,
            sharders=sharders,
            device=device,
        )
        print(f"rank:{rank},sharding plan: {plan}")
        return sharded_model


Multiprocessing Execution
~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's execute the code in multi-processes representing multiple GPU
ranks.

.. code:: python

    import multiprocess
       
    def spmd_sharing_simulation(
        sharding_type: ShardingType = ShardingType.TABLE_WISE,
        world_size = 2,
    ):
      ctx = multiprocess.get_context("spawn")
      processes = []
      for rank in range(world_size):
          p = ctx.Process(
              target=single_rank_execution,
              args=(
                  rank,
                  world_size,
                  gen_constraints(sharding_type),
                  ebc,
                  "nccl"
              ),
          )
          p.start()
          processes.append(p)
    
      for p in processes:
          p.join()
          assert 0 == p.exitcode

Table Wise Sharding
~~~~~~~~~~~~~~~~~~~

Now let's execute the code in two processes for 2 GPUs. We can see in
the plan print that how our tables are sharded across GPUs. Each node
will have one large table and one small which shows our planner tries
for load balance for the embedding tables. Table-wise is the de-factor
go-to sharding schemes for many small-medium size tables for load
balancing over the devices.

.. code:: python

    spmd_sharing_simulation(ShardingType.TABLE_WISE)


.. parsed-literal::

    rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:0/cuda:0)])), 'large_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:0/cuda:0)])), 'small_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:1/cuda:1)]))}}
    rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:0/cuda:0)])), 'large_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:0/cuda:0)])), 'small_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:1/cuda:1)]))}}

Explore other sharding modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have initially explored what table-wise sharding would look like and
how it balances the tables placement. Now we explore sharding modes with
finer focus on load balance: row-wise. Row-wise is specifically
addressing large tables which a single device cannot hold due to the
memory size increase from large embedding row numbers. It can address
the placement of the super large tables in your models. Users can see
that in the ``shard_sizes`` section in the printed plan log, the tables
are halved by row dimension to be distributed onto two GPUs.

.. code:: python

    spmd_sharing_simulation(ShardingType.ROW_WISE)


.. parsed-literal::

    rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)]))}}
    rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)]))}}

Column-wise on the other hand, address the load imbalance problems for
tables with large embedding dimensions. We will split the table
vertically. Users can see that in the ``shard_sizes`` section in the
printed plan log, the tables are halved by embedding dimension to be
distributed onto two GPUs.

.. code:: python

    spmd_sharing_simulation(ShardingType.COLUMN_WISE)


.. parsed-literal::

    rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)]))}}
    rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)]))}}

For ``table-row-wise``, unfortuately we cannot simulate it due to its
nature of operating under multi-host setup. We will present a python
`SPMD <https://en.wikipedia.org/wiki/SPMD>`_ example in the future
to train models with ``table-row-wise``.

With data parallel, we will repeat the tables for all devices.

.. code:: python

    spmd_sharing_simulation(ShardingType.DATA_PARALLEL)


.. parsed-literal::

    rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'large_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None)}}
    rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'large_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None)}}

