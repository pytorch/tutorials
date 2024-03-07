`Introduction <ddp_series_intro.html>`__ \|\|
`What is DDP <ddp_series_theory.html>`__ \|\|
**Single-Node Multi-GPU Training** \|\|
`Fault Tolerance <ddp_series_fault_tolerance.html>`__ \|\|
`Multi-Node training <../intermediate/ddp_series_multinode.html>`__ \|\|
`minGPT Training <../intermediate/ddp_series_minGPT.html>`__


Multi GPU training with DDP
===========================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      -  How to migrate a single-GPU training script to multi-GPU via DDP
      -  Setting up the distributed process group
      -  Saving and loading models in a distributed setup
      
      .. grid:: 1

         .. grid-item::

            :octicon:`code-square;1.0em;` View the code used in this tutorial on `GitHub <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py>`__
      
   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * High-level overview of `how DDP works  <ddp_series_theory.html>`__
      * A machine with multiple GPUs (this tutorial uses an AWS p3.8xlarge instance)
      * PyTorch `installed <https://pytorch.org/get-started/locally/>`__ with CUDA

Follow along with the video below or on `youtube <https://www.youtube.com/watch/-LAtx9Q6DA8>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/-LAtx9Q6DA8" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

In the `previous tutorial <ddp_series_theory.html>`__, we got a high-level overview of how DDP works; now we see how to use DDP in code.
In this tutorial, we start with a single-GPU training script and migrate that to running it on 4 GPUs on a single node.
Along the way, we will talk through important concepts in distributed training while implementing them in our code.

.. note:: 
   If your model contains any ``BatchNorm`` layers, it needs to be converted to ``SyncBatchNorm`` to sync the running stats of ``BatchNorm``
   layers across replicas.

   Use the helper function 
   `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm>`__ to convert all ``BatchNorm`` layers in the model to ``SyncBatchNorm``.


Diff for `single_gpu.py <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/single_gpu.py>`__ v/s `multigpu.py <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py>`__

These are the changes you typically make to a single-GPU training script to enable DDP.

Imports
~~~~~~~
-  ``torch.multiprocessing`` is a PyTorch wrapper around Python's native
   multiprocessing
-  The distributed process group contains all the processes that can
   communicate and synchronize with each other.

.. code-block:: diff

    import torch
    import torch.nn.functional as F
    from utils import MyTrainDataset

    + import torch.multiprocessing as mp
    + from torch.utils.data.distributed import DistributedSampler
    + from torch.nn.parallel import DistributedDataParallel as DDP
    + from torch.distributed import init_process_group, destroy_process_group
    + import os


Constructing the process group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  First, before initializing the group process, call `set_device <https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html?highlight=set_device#torch.cuda.set_device>`__,
   which sets the default GPU for each process. This is important to prevent hangs or excessive memory utilization on `GPU:0`
-  The process group can be initialized by TCP (default) or from a
   shared file-system. Read more on `process group
   initialization <https://pytorch.org/docs/stable/distributed.html#tcp-initialization>`__
-  `init_process_group <https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group>`__
   initializes the distributed process group.
-  Read more about `choosing a DDP
   backend <https://pytorch.org/docs/stable/distributed.html#which-backend-to-use>`__

.. code-block:: diff

    + def ddp_setup(rank: int, world_size: int):
    +   """
    +   Args:
    +       rank: Unique identifier of each process
    +      world_size: Total number of processes
    +   """
    +   os.environ["MASTER_ADDR"] = "localhost"
    +   os.environ["MASTER_PORT"] = "12355"
    +   torch.cuda.set_device(rank)
    +   init_process_group(backend="nccl", rank=rank, world_size=world_size)



Constructing the DDP model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: diff

    - self.model = model.to(gpu_id)
    + self.model = DDP(model, device_ids=[gpu_id])

Distributing input data
~~~~~~~~~~~~~~~~~~~~~~~

-  `DistributedSampler <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler>`__
   chunks the input data across all distributed processes.
-  Each process will receive an input batch of 32 samples; the effective
   batch size is ``32 * nprocs``, or 128 when using 4 GPUs.

.. code-block:: diff

    train_data = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
    -   shuffle=True,
    +   shuffle=False,
    +   sampler=DistributedSampler(train_dataset),
    )

-  Calling the ``set_epoch()`` method on the ``DistributedSampler`` at the beginning of each epoch is necessary to make shuffling work 
   properly across multiple epochs. Otherwise, the same ordering will be used in each epoch.

.. code-block:: diff

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
    +   self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
          ...
          self._run_batch(source, targets)


Saving model checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~
-  We only need to save model checkpoints from one process. Without this 
   condition, each process would save its copy of the identical mode. Read
   more on saving and loading models with
   DDP `here <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#save-and-load-checkpoints>`__  

.. code-block:: diff

    - ckp = self.model.state_dict()
    + ckp = self.model.module.state_dict()
    ...
    ...
    - if epoch % self.save_every == 0:
    + if self.gpu_id == 0 and epoch % self.save_every == 0:
      self._save_checkpoint(epoch)

.. warning::
   `Collective calls <https://pytorch.org/docs/stable/distributed.html#collective-functions>`__ are functions that run on all the distributed processes,
   and they are used to gather certain states or values to a specific process. Collective calls require all ranks to run the collective code.
   In this example, `_save_checkpoint` should not have any collective calls because it is only run on the ``rank:0`` process. 
   If you need to make any collective calls, it should be before the ``if self.gpu_id == 0`` check.


Running the distributed training job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Include new arguments ``rank`` (replacing ``device``) and
   ``world_size``.
-  ``rank`` is auto-allocated by DDP when calling
   `mp.spawn <https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses>`__.
-  ``world_size`` is the number of processes across the training job. For GPU training, 
   this corresponds to the number of GPUs in use, and each process works on a dedicated GPU.

.. code-block:: diff

   - def main(device, total_epochs, save_every):
   + def main(rank, world_size, total_epochs, save_every):
   +  ddp_setup(rank, world_size)
      dataset, model, optimizer = load_train_objs()
      train_data = prepare_dataloader(dataset, batch_size=32)
   -  trainer = Trainer(model, train_data, optimizer, device, save_every)
   +  trainer = Trainer(model, train_data, optimizer, rank, save_every)
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



Further Reading
---------------

-  `Fault Tolerant distributed training <ddp_series_fault_tolerance.html>`__  (next tutorial in this series)
-  `Intro to DDP <ddp_series_theory.html>`__ (previous tutorial in this series)
-  `Getting Started with DDP <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__ 
-  `Process Group
   initialization <https://pytorch.org/docs/stable/distributed.html#tcp-initialization>`__
