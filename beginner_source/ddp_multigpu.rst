`Introduction <ddp_series_intro.html>`__ \|\| `What is DDP <ddp_theory.html>`__ \|\| **Single-node Multi-GPU training** \|\| `Fault
Tolerance <ddp_fault_tolerance.html>`__ \|\| `Multi-node
training <intermediate/ddp_multinode.html>`__ \|\| `mingpt training <intermediate/ddp_minGPT.html>`__


Multi GPU training with DDP
===========================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

In the `previous tutorial <ddp_theory.html>`__, we got a high-level overview of how DDP works; now we see how to use DDP in code.
In this tutorial we start with a single-GPU training script, and migrate that to running it on 4 GPUs on a single node. 
Along the way, we will talk through important concepts in distributed training while implementing them in our code.


What you will learn
-------------------
-  How to migrate a single-GPU training script to multi-GPU via DDP
-  Setting up the distributed process group
-  Replicating your model with the DDP constructor
-  Distributing the input batch via DistributedSampler
-  Saving and loading models in a distributed setup


Code used in this video
-----------------------------
https://github.com/suraj813/distributed-pytorch/blob/main/multigpu.py


.. raw:: html

   <embed video>


.. note:: 
If your model contains any ``BatchNorm`` layer, it needs to be converted to ``SyncBatchNorm`` to sync the running stats of ``BatchNorm`` 
layers across replicas.

Use the helper function 
`torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm>`__ to convert all ``BatchNorm`` layers in the model to ``SyncBatchNorm``.


Diff for `single_gpu.py <https://github.com/suraj813/distributed-pytorch/blob/main/single_gpu.py>`__ v/s `multigpu.py <https://github.com/suraj813/distributed-pytorch/blob/main/multigpu.py>`__
----------------------------------------------------

These are the changes you typically make to a single-GPU training script to enable DDP.

Imports
~~~~~~~

.. code:: diff

   import torch
   import torch.nn.functional as F
   from utils import MyTrainDataset
    
   + import torch.multiprocessing as mp
   + from torch.utils.data.distributed import DistributedSampler
   + from torch.nn.parallel import DistributedDataParallel as DDP
   + from torch.distributed import init_process_group, destroy_process_group
   + import os

-  ``torch.multiprocessing`` is a PyTorch wrapper around python's native
   multiprocessing
-  The dsitributed process group contains all the processes that can
   communicate and synchronize with each other.

Constructing the process group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: diff

   + def ddp_setup(rank, world_size):
   +   """
   +   Args:
   +       rank: Unique identifier of each process
   +      world_size: Total number of processes
   +   """
   +   os.environ["MASTER_ADDR"] = "localhost"
   +   os.environ["MASTER_PORT"] = "12355"
   +   init_process_group(backend="nccl", rank=rank, world_size=world_size)

-  `Choosing a DDP
   backend <https://pytorch.org/docs/stable/distributed.html#which-backend-to-use>`__
-  The process group can be initialized by TCP (default) or from a
   shared file-system. `Read more on process group
   initialization <https://pytorch.org/docs/stable/distributed.html#tcp-initialization>`__
-  `init_process_group <https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group>`__
   initializes the distributed process group.

Constructing the DDP model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: diff

   - self.model = model.to(gpu_id)
   + self.model = DDP(model, device_ids=[gpu_id])

Distributing input data
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: diff

   train_data = torch.utils.data.DataLoader(
       dataset=train_dataset,
       batch_size=32,
   -   shuffle=True,
   +   shuffle=False,
   +   sampler=DistributedSampler(train_dataset),
   )

-  `DistributedSampler <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler>`__
   chunks the input data across all distributed processes.
-  Each process will receive an input batch of 32 samples; the effective
   batch size is ``32 * nprocs``, or 128 when using 4 GPUs.

Saving model checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: diff

   - ckp = self.model.state_dict()
   + ckp = self.model.module.state_dict()
   ...
   ...
   - if epoch % self.save_every == 0:
   + if self.gpu_id == 0 and epoch % self.save_every == 0:
      self._save_checkpoint(epoch)

We only need to save model checkpoints from one process. Without this
condition, each process would save its copy of the identical mode. Read
more on `saving and loading models with
DDP <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#save-and-load-checkpoints>`__

Running the distributed training job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: diff

   - def main(device, total_epochs, save_every):
   + def main(rank, world_size, total_epochs, save_every):
   +  ddp_setup(rank, world_size)
      dataset, model, optimizer = load_train_objs()
      train_data = prepare_dataloader(dataset, batch_size=32)
   -  trainer = Trainer(model, dataset, optimizer, device, save_every)
   +  trainer = Trainer(model, dataset, optimizer, rank, save_every)
      trainer.train(total_epochs)
   +  destroy_process_group()
    
   if __name__ == "__main__":
      import sys
      total_epochs = int(sys.argv[1])
      save_every = int(sys.argv[2])
   -  device = 0      # shorthand for cuda:0
   -  main(device, total_epochs, save_every)
   +  world_size = torch.cuda.device_count()
   +  mp.spawn(main, args=(world_size, total_epochs, save_every,), nprocs=world_size)

-  Include new arguments ``rank`` (replacing ``device``) and
   ``world_size``.
-  ``rank`` is auto-allocated by DDP when calling
   ```mp.spawn`` <https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses>`__.
-  ``world_size`` is the number of processes/GPUs we want to use
   (typically 1 process per GPU).


Further Reading
---------------

-  `Fault Tolerant distributed training <ddp_fault_tolerance.html>`__  (next tutorial in this series)
-  `Intro to DDP <ddp_theory.html>`__ (previous tutorial in this series)
-  `Getting Started with DDP <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__ 
-  `Process Group
   initialization <https://pytorch.org/docs/stable/distributed.html#tcp-initialization>`__
