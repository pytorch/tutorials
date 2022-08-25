`Introduction <0_intro.html>`__ \|\| `What is DDP <1_theory.html>`__
\|\| `Multi-GPU training <2_multigpu.html>`__ \|\| **Fault Tolerance**
\|\| `Multi-node training <4_multinode.html>`__ \|\| `mingpt
training <5_minGPT.html>`__

Fault-tolerant Distributed Training with ``torchrun``
=====================================================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__

.. raw:: html

   <embed video>

When a training job fails, we relaunch it from the last saved
checkpoint. In distributed training, a single process failing can
disrupt the entire training job. Manually restarting a training job
after failure is tedious enough, more so for distributed jobs where the
susceptibility for failure can be higher.

Especially in multinode training (covered in the next section), you
might also prefer to be *elastic* i.e. increase or decrease the number
of processes in realtime while running the training job.

Fault tolerance
~~~~~~~~~~~~~~~

PyTorch has a utility ``torchrun`` that provides fault-tolerance as well
as elastic training. When a failure occurs, torchrun logs the errors and
attempts to automatically restart all the processes from the last saved
“snapshot” of the training job. It is recommended for your script to
have the following structure:

.. code:: python

   def main():
     load_checkpoint(checkpoint_path)
     initialize()
     train()

   def train():
     for batch in iter(dataset):
       train_step(batch)

       if should_checkpoint:
         save_checkpoint(checkpoint_path)

See the diff below for the functions that implement this. The
snapshot/checkpoint saves more than just the model state; it can include
details about the number of epochs run, optimizer states or any other
mutable attribute of the training job necessary for its continuity.

Automatic
~~~~~~~~~

``torchrun`` handles the minutiae of distributed training so that you
don’t need to, for instance: \* you don’t need to set environment
variables or explicitly pass the ``rank`` and ``world_size``; torchrun
assigns this along with several other `environment
variables <https://pytorch.org/docs/stable/elastic/run.html#environment-variables>`__.
\* No need to call ``mp.spawn`` in your script; you only need a generic
``main()`` entrypoint, and launch the script with ``torchrun``. This way
the same script can be run in non-distributed as well as single-node and
multinode setups. \* Gracefully restarting training from the last saved
checkpoint in case of runtime errors or elastic scaling.

Diff for `multigpu.py <>`__ v/s `multigpu_torchrun.py <>`__
-----------------------------------------------------------

Process group initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: diff

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

-  ``torchrun`` assigns ``RANK`` and ``WORLD_SIZE`` automatically,
   amongst `other env
   variables <https://pytorch.org/docs/stable/elastic/run.html#environment-variables>`__

Use Torchrun-provided env variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: diff

   - self.gpu_id = gpu_id
   + self.gpu_id = int(os.environ["LOCAL_RANK"])

Saving and loading snapshots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: diff

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

Regularly storing all the relevant information in snapshots allows our
training job to seamlessly resume after an interruption.

Loading a snapshot in the Trainer constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: diff

   class Trainer:
      def __init__(self, snapshot_path, ...):
      ...
   +  if os.path.exists(snapshot_path):
   +     self._load_snapshot(snapshot_path)
      ...

When restarting an interrupted training job, your script will first try
to load a snapshot to resume training from.

Resuming training
~~~~~~~~~~~~~~~~~

.. code:: diff

   def train(self, max_epochs: int):
   -  for epoch in range(max_epochs):
   +  for epoch in range(self.epochs_run, max_epochs):
         self._run_epoch(epoch)

Training can resume from the last epoch run, instead of starting all
over from scratch.

Running the script
~~~~~~~~~~~~~~~~~~

.. code:: diff

   if __name__ == "__main__":
      import sys
      total_epochs = int(sys.argv[1])
      save_every = int(sys.argv[2])
   -  world_size = torch.cuda.device_count()
   -  mp.spawn(main, args=(world_size, total_epochs, save_every,), nprocs=world_size)
   +  main(save_every, total_epochs)

Call your entrypoint function as usual; ``torchrun`` automatically
spawns the processes.

.. code:: diff

   - python multigpu.py 50 10
   + torchrun --standalone --nproc_per_node=4 multigpu_torchrun.py 50 10

Further Reading
---------------

-  `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__
-  `Torchrun
   options <https://github.com/pytorch/pytorch/blob/bbe803cb35948df77b46a2d38372910c96693dcd/torch/distributed/run.py#L401>`__
-  `Migrating from ``torch.distributed.launch`` to
   ``torchrun`` <https://pytorch.org/docs/stable/elastic/train_script.html#elastic-train-script>`__
