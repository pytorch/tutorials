`Introduction <ddp_series_intro.html>`__ \|\|
`What is DDP <ddp_series_theory.html>`__ \|\|
`Single-Node Multi-GPU Training <ddp_series_multigpu.html>`__ \|\|
**Fault Tolerance** \|\|
`Multi-Node training <../intermediate/ddp_series_multinode.html>`__ \|\|
`minGPT Training <../intermediate/ddp_series_minGPT.html>`__


Fault-tolerant Distributed Training with ``torchrun``
=====================================================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
      :margin: 0
      
      -  Launching multi-GPU training jobs with ``torchrun``
      -  Saving and loading snapshots of your training job
      -  Structuring your training script for graceful restarts

      .. grid:: 1

         .. grid-item::

            :octicon:`code-square;1.0em;` View the code used in this tutorial on `GitHub <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py>`__

   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
      :margin: 0

      * High-level `overview <ddp_series_theory.html>`__ of DDP
      * Familiarity with `DDP code <ddp_series_multigpu.html>`__
      * A machine with multiple GPUs (this tutorial uses an AWS p3.8xlarge instance)
      * PyTorch `installed <https://pytorch.org/get-started/locally/>`__ with CUDA

Follow along with the video below or on `youtube <https://www.youtube.com/watch/9kIvQOiwYzg>`__.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/9kIvQOiwYzg" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

In distributed training, a single process failure can
disrupt the entire training job. Since the susceptibility for failure can be higher here, making your training
script robust is particularly important here. You might also prefer your training job to be *elastic*, for example,
compute resources can join and leave dynamically over the course of the job.

PyTorch offers a utility called ``torchrun`` that provides fault-tolerance and 
elastic training. When a failure occurs, ``torchrun`` logs the errors and
attempts to automatically restart all the processes from the last saved
“snapshot” of the training job. 

The snapshot saves more than just the model state; it can include
details about the number of epochs run, optimizer states or any other
stateful attribute of the training job necessary for its continuity.

Why use ``torchrun``
~~~~~~~~~~~~~~~~~~~~

``torchrun`` handles the minutiae of distributed training so that you
don't need to. For instance,

-  You don't need to set environment variables or explicitly pass the ``rank`` and ``world_size``; ``torchrun`` assigns this along with several other `environment variables <https://pytorch.org/docs/stable/elastic/run.html#environment-variables>`__.
-  No need to call ``mp.spawn`` in your script; you only need a generic ``main()`` entry point, and launch the script with ``torchrun``. This way the same script can be run in non-distributed as well as single-node and multinode setups.
-  Gracefully restarting training from the last saved training snapshot.


Graceful restarts
~~~~~~~~~~~~~~~~~~~~~
For graceful restarts, you should structure your train script like:

.. code:: python

   def main():
     load_snapshot(snapshot_path)
     initialize()
     train()

   def train():
     for batch in iter(dataset):
       train_step(batch)

       if should_checkpoint:
         save_snapshot(snapshot_path)

If a failure occurs, ``torchrun`` will terminate all the processes and restart them. 
Each process entry point first loads and initializes the last saved snapshot, and continues training from there.
So at any failure, you only lose the training progress from the last saved snapshot. 

In elastic training, whenever there are any membership changes (adding or removing nodes), ``torchrun`` will terminate and spawn processes
on available devices. Having this structure ensures your training job can continue without manual intervention.


Diff for `multigpu.py <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py>`__ v/s `multigpu_torchrun.py <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py>`__

Process group initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``torchrun`` assigns ``RANK`` and ``WORLD_SIZE`` automatically,
   among `other envvariables <https://pytorch.org/docs/stable/elastic/run.html#environment-variables>`__

.. code-block:: diff

    - def ddp_setup(rank, world_size):
    + def ddp_setup():
    -     """
    -     Args:
    -         rank: Unique identifier of each process
    -         world_size: Total number of processes
    -     """
    -     os.environ["MASTER_ADDR"] = "localhost"
    -     os.environ["MASTER_PORT"] = "12355"
    -     init_process_group(backend="nccl", rank=rank, world_size=world_size)
    +     init_process_group(backend="nccl")
         torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

Use torchrun-provided environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: diff

    - self.gpu_id = gpu_id
    + self.gpu_id = int(os.environ["LOCAL_RANK"])

Saving and loading snapshots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regularly storing all the relevant information in snapshots allows our
training job to seamlessly resume after an interruption.

.. code-block:: diff

    + def _save_snapshot(self, epoch):
    +     snapshot = {}
    +     snapshot["MODEL_STATE"] = self.model.module.state_dict()
    +     snapshot["EPOCHS_RUN"] = epoch
    +     torch.save(snapshot, "snapshot.pt")
    +     print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")

    + def _load_snapshot(self, snapshot_path):
    +     snapshot = torch.load(snapshot_path)
    +     self.model.load_state_dict(snapshot["MODEL_STATE"])
    +     self.epochs_run = snapshot["EPOCHS_RUN"]
    +     print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


Loading a snapshot in the Trainer constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When restarting an interrupted training job, your script will first try
to load a snapshot to resume training from.

.. code-block:: diff

    class Trainer:
       def __init__(self, snapshot_path, ...):
       ...
    +  if os.path.exists(snapshot_path):
    +     self._load_snapshot(snapshot_path)
       ...


Resuming training
~~~~~~~~~~~~~~~~~

Training can resume from the last epoch run, instead of starting all
over from scratch.

.. code-block:: diff

    def train(self, max_epochs: int):
    -  for epoch in range(max_epochs):
    +  for epoch in range(self.epochs_run, max_epochs):
          self._run_epoch(epoch)


Running the script
~~~~~~~~~~~~~~~~~~

Simply call your entry point function as you would for a non-multiprocessing script; ``torchrun`` automatically
spawns the processes.

.. code-block:: diff

    if __name__ == "__main__":
       import sys
       total_epochs = int(sys.argv[1])
       save_every = int(sys.argv[2])
    -  world_size = torch.cuda.device_count()
    -  mp.spawn(main, args=(world_size, total_epochs, save_every,), nprocs=world_size)
    +  main(save_every, total_epochs)


.. code-block:: diff

    - python multigpu.py 50 10
    + torchrun --standalone --nproc_per_node=4 multigpu_torchrun.py 50 10

Further Reading
---------------

-  `Multi-Node training with DDP <../intermediate/ddp_series_multinode.html>`__  (next tutorial in this series)
-  `Multi-GPU Training with DDP <ddp_series_multigpu.html>`__ (previous tutorial in this series)
-  `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__
-  `Torchrun launch
   options <https://github.com/pytorch/pytorch/blob/bbe803cb35948df77b46a2d38372910c96693dcd/torch/distributed/run.py#L401>`__
-  `Migrating from torch.distributed.launch to
   torchrun <https://pytorch.org/docs/stable/elastic/train_script.html#elastic-train-script>`__
