==========================================================
Interactive Distributed Applications with Monarch
==========================================================

**Author**: `Amir Afzali <https://github.com/amirafzali>`_

Introduction
------------

As deep learning models continue to grow in size and complexity, training them efficiently requires coordinating computation across multiple GPUs and nodes.
In this tutorial, you will learn how to easily set up and run large-scale distributed workflows using Monarch's actor framework together with TorchTitan, on a SLURM-managed cluster.

What is Monarch?
^^^^^^^^^^^^^^^^

Monarch is an actor framework designed to streamline the development of distributed applications. At its core, Monarch provides:

- **Actor-based programming model**: Encapsulate stateful computations in actors that can run on remote processes and machines
- **Process mesh abstractions**: Easily manage and coordinate distributed processes across your cluster, with scalable Actor messaging
- **Fault tolerance**: Actors and processes form a tree and failures propagate up the tree, providing good default error behavior and enabling fine-grained fault recovery.
- **Flexible resource management**: Support for multiple cluster schedulers including SLURM, Kubernetes, custom host management, and local processes
- **Integrated monitoring**: Stream logs from remote processes back to your client for easy debugging and aggregation

For more details, see the `Monarch documentation <https://meta-pytorch.org/monarch/generated/examples/getting_started.html>`_.

Why Use Monarch with TorchTitan?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TorchTitan is a PyTorch native library for pre-training at scale.
While TorchTitan provides excellent primitives for distributed training, launching and managing these jobs across clusters can be complex. Monarch addresses this with:

1. **Simplified cluster interaction**: Reserve and manage compute resources with simple async Python calls instead of writing bash scripts
2. **Interactive development**: Modify and re-run training code on existing allocations without waiting for new resources
3. **Unified workflow**: Seamlessly move between local testing and cluster execution with the same code
4. **Failure supervision**: Handle errors and failures gracefully, with fine-grained recovery options from the controller

Prerequisites
-------------

To run this tutorial, you must have:

1. **Monarch nightly installed:**
   `Install script <https://github.com/meta-pytorch/monarch/blob/main/scripts/install_nightly.py>`_
2. **TorchTitan nightly installed:**
   `TorchTitan install instructions <https://github.com/pytorch/torchtitan?tab=readme-ov-fileightly-builds>`_
3. **A valid Titan model config** and **tokenizer** in your working directory (e.g., ``debug_model.toml`` from `TorchTitan configs <https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/train_configs/debug_model.toml>`_).
4. **SLURM cluster access:**

   - Sufficient permissions to reserve nodes and launch jobs.
   - CUDA environment configured for distributed GPU training.


Now let's implement this step by step!

Step 1: Reserve Machine Resources
---------------------------------

First, we'll define a function to programmatically reserve a machine allocation.

**Monarch Highlight**: Instead of submitting an SBATCH script, you can reserve and manage resources interactively from Python.
The JobTrait design pattern allows for interfacing with custom schedulers, such as SLURM and Kubernetes, through a consistent API.

.. code-block:: python

    from monarch.job import SlurmJob, JobTrait


    def create_slurm_job(
        mesh_name: str,
        num_nodes: int,
        time_limit: str = "06:00:00"
    ) -> SlurmJob:
        """
        Args:
            mesh_name: Name assigned to the primary mesh for this example.
                       A JobTrait can consist of multiple meshes, and
                       Monarch allows for re-attaching to ongoing jobs.
            num_nodes: Number of nodes allocated per mesh

            Note: SlurmJob is just one instance of a Monarch scheduler interface.
                  Consult the JobTrait documentation to find one that's right for your usecase.
        """
        default_job_name = "monarch_titan"
        return SlurmJob(
            meshes={mesh_name: num_nodes},
            job_name=default_job_name,
            time_limit=time_limit,
            # ... additional args can be passed here
        )

Step 2: Define the Trainer Actor
--------------------------------

Now we create a Monarch Actor that wraps TorchTitan's Trainer. This is the
key abstraction that allows TorchTitan to run in Monarch's distributed
environment.

**Monarch Highlight**: The Actor pattern provides several benefits:

1. **Remote execution**: Methods marked with @endpoint can be called remotely
2. **Lifecycle management**: Monarch handles initialization, execution, and cleanup
3. **Error handling**: Exceptions are properly propagated back to the client, enabling progressive error handling

.. code-block:: python

    import torch
    from monarch.actor import Actor, current_rank, endpoint
    from monarch.utils import setup_env_for_distributed
    from torchtitan.tools.logging import init_logger, logger
    from torchtitan.train import Trainer


    class TrainerActor(Actor):
        """
        Monarch Actor wrapper for TorchTitan's Trainer.

        This actor encapsulates a complete TorchTitan training process, handling
        initialization, training loop execution, and cleanup. Each instance runs
        on a single GPU in the distributed training job.

        The actor's lifetime:
            1. __init__: Initialize with job configuration
            2. start_training:
               Execute the training loop
               Destroy process group and release resources

        Attributes:
            job_config: TorchTitan configuration for this trainer
            uid: Unique identifier for logging (includes rank)
        """

        def __init__(self, job_config: "JobConfig") -> None:
            """
            Initialize the trainer actor.

            Args:
                job_config: TorchTitan JobConfig with training parameters
            """
            self.job_config = job_config

            # current_rank() provides access to this actor's rank in the process mesh
            self.rank = current_rank().rank
            self.uid = f"[trainer_{rank}]"

        @endpoint
        async def ping_rank(self) -> None:
            """
                A dummy logging function we will use for demonstration purposes.
            """
            logger.info(f"{self.uid} Ping!")

        @endpoint
        async def start_training(self) -> None:
            """
            Execute the TorchTitan training loop.

            This remote endpoint:
            1. Initializes TorchTitan's logger
            2. Creates a Trainer instance with the job configuration
            3. Runs the training loop
            4. Handles cleanup and error conditions

            The @endpoint decorator makes this method callable from the Monarch
            client, even though it runs on a remote GPU node.

            Raises:
                Exception: Any exception from TorchTitan training is propagated
                          back to the client
            """
            init_logger()
            trainer: Trainer | None = None
            try:
                # Initialize TorchTitan trainer
                trainer = Trainer(self.job_config)
                logger.info(f"{self.uid} initialized successfully and starting training")

                # Run the training loop
                trainer.train()

            except Exception as e:
                logger.error(f"{self.uid} training failed: {e}")
                if trainer:
                    trainer.close()
                # Note: error is propagated back to the controller
                raise e

            else:
                # Training completed successfully
                trainer.close()
                logger.info(f"{self.uid} training completed successfully")

            finally:
                # Clean up distributed process group
                torch.distributed.destroy_process_group()
                logger.info(f"{self.uid} trainer cleaned up")

Actor endpoints can be invoked in a variety of patterns. We'll explore a concrete example in `Step 4: Execute the Training Workflow`_,
but here are some common usages:

.. code-block:: python

    try:
        # where mesh0 is 4 nodes * 8 GPUs
        proc_mesh = mesh0.spawn_procs({"gpus": 32})
        trainer_actor = proc_mesh.spawn(...)

        # Call on all ranks
        await trainer_actor.ping_rank.call()

        # Call-and-forget on all ranks
        trainer_actor.ping_rank.broadcast()

        # Call on ONE random rank
        await trainer_actor.ping_rank.choose()

    except Exception as e:
        # handle SupervisionEvents from remote actor failures
        pass

Remote actor endpoints can also utilize Python native breakpoints, enabling interactive debugging sessions.
For a complete deep-dive into Monarch debuggers, `refer to the documentation <https://meta-pytorch.org/monarch/generated/examples/debugging.html>`_.

.. code-block:: python

    @endpoint
        async def ping_debuggable_rank(self) -> None:
            logger.info(f"{self.uid} Ping!")
            if self.rank == 0:
                breakpoint()
            logger.info(f"{self.uid} Pong!")


Step 3: Define Training Parameters
-----------------------------------

Next, we define some common parameters for our training job and cluster resources.
This configuration determines both the scale of training (number of nodes and GPUs),
and some of the training hyperparameters.

.. code-block:: python

    from dataclasses import dataclass


    @dataclass
    class RunParams:
        """
        Configuration for cluster resources and training parameters.

        Attributes:
            training_steps: Number of training iterations to run
            model_config: Path to TorchTitan model configuration file
            tokenizer: Path to tokenizer directory
            dataset: Dataset to use for training (e.g., 'c4', 'c4_test')
            num_nodes: Number of compute nodes to request
            gpus_per_node: Number of GPUs per node

        Adjust these values based on your model size and available resources.
        """

        training_steps: int = 50
        model_config: str = "debug_model.toml"
        tokenizer: str = "tokenizer"
        dataset: str = "c4"
        num_nodes: int = 2
        gpus_per_node: int = 8

TorchTitan uses a JobConfig object to control all aspects of training.
Here we create a function that builds this configuration from our RunParams.

.. code-block:: python

    import os
    from torchtitan.config import ConfigManager, JobConfig


    def make_job_config() -> JobConfig:
        """
        Create a TorchTitan JobConfig from RunParams.

        This function constructs the complete training configuration, including
        parallelism settings, model architecture, and dataset paths
        """
        # Calculate total parallelism based on cluster size
        data_parallel_shard_degree = RunParams.num_nodes * RunParams.gpus_per_node
        output_path = "./outputs"
        # Construct paths relative to script directory
        script_dir = os.getcwd()

        # Build argument list for TorchTitan's ConfigManager
        # These override defaults from the model config file
        default_args = [
            "--job.config_file",
            os.path.join(script_dir, RunParams.model_config),
            "--model.tokenizer_path",
            os.path.join(script_dir, RunParams.tokenizer),
            "--parallelism.data_parallel_shard_degree",
            str(data_parallel_shard_degree),
            "--training.steps",
            str(RunParams.training_steps),
            "--training.dataset",
            RunParams.dataset,
            "--job.dump_folder",
            output_path,
            # continue to configure as needed
        ]
        config_manager = ConfigManager()
        job_config = config_manager.parse_args(default_args)
        return job_config

Step 4: Execute the Training Workflow
--------------------------------------

With all components defined, we now orchestrate the complete workflow.
This is where Monarch's power becomes most apparent.

**Monarch Highlights**:

1. **Interactive iteration**: After reserving the machine allocation, you can adjust your logic
   and re-spawn actors, without requesting new resources.
2. **Transparent logging**: All logs from remote workers stream back to your
   client in real-time, making debugging feel like local execution

Workflow:
    Reserve Machines → Create Proc Mesh → Configure Logging → Spawn Actors → Train → Cleanup

.. code-block:: python

    async def execute_training() -> None:
        """
        Execute the complete distributed training workflow.
        """
        job_config = make_job_config()
        slurm_job = None
        mesh_name = "mesh0"
        try:
            # 1. Create a SLURM job with N nodes
            #    This leverages Monarch to reserve a persistent machine allocation
            slurm_job = create_slurm_job(mesh_name, RunParams.num_nodes)
            job_state = slurm_job.state()

            # 2. Create a process mesh on the machine allocation
            #    This creates one process per GPU across all allocated nodes
            logger.info("Creating process mesh...")
            total_gpus = RunParams.gpus_per_node * RunParams.num_nodes
            proc_mesh = job_state.mesh0.spawn_procs({"gpus": total_gpus})

            # 3. Configure remote logging behavior
            #    - stream_to_client: Forward all remote logs to your local console
            #    - aggregate_window_sec: Batch logs for efficiency
            logger.info("Configuring logging...")
            await proc_mesh.logging_option(
                stream_to_client=True,
                # aggregate_window_sec=None  # Uncomment to disable log batching
            )

            # 4. Setup environment for torch.distributed
            #    This configures torch.distributed across all processes in the mesh
            logger.info("Setting up distributed environment...")
            await setup_env_for_distributed(proc_mesh)

            # 5. Spawn TrainerActor on each GPU
            #    Each process in the mesh creates its own TrainerActor instance
            logger.info("Spawning trainer actors...")
            trainer = proc_mesh.spawn(
                "trainer_actor",  # Name for the actor group
                TrainerActor,  # Actor class to instantiate
                job_config,  # Arguments to __init__
            )

            # 6. Execute the training job across all actors
            #    The .call() method invokes start_training() on all actors in parallel
            logger.info("Starting distributed training...")
            await trainer.start_training.call()

            logger.info("Training completed successfully!")

        except Exception as e:
            logger.error(f"Training workflow failed: {e}")

        finally:
            # Always clean up the machine allocation
            if slurm_job:
                await cleanup_job(slurm_job)

Step 5: Clean Up Resources
--------------------------

After training completes (or if you're done experimenting), it's important
to free up cluster resources by terminating the SLURM job.

**Monarch Highlight**: While you can keep allocations alive for multiple
training runs during development, always remember to release cluster resources.

.. code-block:: python

    async def cleanup_job(job: JobTrait) -> None:
        """
        This function cancels the SLURM job, releasing all reserved nodes back
        to the cluster for other users.

        Args:
            job: A JobTrait, like the one returned from create_slurm_job()

        Note:
            The job will also terminate automatically when the configured TTL
            is exceeded, but explicit cleanup is recommended for long-running
            notebooks or scripts.
        """
        job.kill()
        logger.info("Job terminated successfully")

Step 6: Run the Complete Pipeline
---------------------------------

Finally, we tie everything together in a main function that kicks off the workflow

.. code-block:: python

    import asyncio


    if __name__ == "__main__":
        """
        Run the complete workflow: reserve resources, train, and cleanup.
        """
        logger.info("Starting Monarch + TorchTitan Distributed Training")

        asyncio.run(execute_training())

        logger.info("Workflow completed!")

Summary
-------

Congrats! In this tutorial, you learned how to combine Monarch's actor framework with
TorchTitan for scalable distributed training.

**Further Reading**

- Monarch also integrates with TorchFT to provide per-step fault-tolerance across replicated workers.
You can find a comprehensive `proof of concept <https://github.com/meta-pytorch/torchft/tree/main/torchft/examples/slurm>`_ of this integration in the TorchFT repo.
- For an interactive notebook covering similar topics to this tutorial, please consult `this Monarch example <https://github.com/meta-pytorch/monarch/blob/main/examples/slurm_titan.ipynb>`_.