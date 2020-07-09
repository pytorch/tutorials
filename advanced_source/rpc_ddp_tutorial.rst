Combining Distributed DataParallel with Distributed RPC Framework
=================================================================
**Author**: `Pritam Damania <https://github.com/pritamdamania87>`_


This tutorial uses a simple example to demonstrate how you can combine 
`DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__ (DDP)
with the `Distributed RPC framework <https://pytorch.org/docs/master/rpc.html>`__ 
to combine distributed data parallelism with distributed model parallelism to 
train a simple model. Source code of the example can be found `here <https://github.com/pytorch/examples/tree/master/distributed/rpc/ddp_rpc>`__.

Previous tutorials,
`Getting Started With Distributed Data Parallel <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__
and `Getting Started with Distributed RPC Framework <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`__,
described how to perform distributed data parallel and distributed model 
parallel training respectively. Although, there are several training paradigms 
where you might want to combine these two techniques. For example:

1) If we have a model with a sparse part (large embedding table) and a dense 
   part (FC layers), we might want to put the embedding table on a parameter 
   server and replicate the FC layer across multiple trainers using `DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__.
   The `Distributed RPC framework <https://pytorch.org/docs/master/rpc.html>`__ 
   can be used to perform embedding lookups on the parameter server.
2) Enable hybrid parallelism as described in the `PipeDream <https://arxiv.org/abs/1806.03377>`__ paper.
   We can use the `Distributed RPC framework <https://pytorch.org/docs/master/rpc.html>`__ 
   to pipeline stages of the model across multiple workers and replicate each 
   stage (if needed) using `DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__.

|
In this tutorial we will cover case 1 mentioned above. We have a total of 4 
workers in our setup as follows:


1) 1 Master, which is responsible for creating an embedding table 
   (nn.EmbeddingBag) on the parameter server. The master also drives the 
   training loop on the two trainers.
2) 1 Parameter Server, which basically holds the embedding table in memory and 
   responds to RPCs from the Master and Trainers.
3) 2 Trainers, which store a FC layer (nn.Linear) which is replicated amongst 
   themselves using `DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__.
   The trainers are also responsible for executing the forward pass, backward 
   pass and optimizer step.

|
The entire training process is executed as follows:

1) The master creates an embedding table on the Parameter Server and holds a 
   `RRef <https://pytorch.org/docs/master/rpc.html#rref>`__ to it.
2) The master, then kicks of the training loop on the trainers and passes the 
   embedding table RRef to the trainers.
3) The trainers create a ``HybridModel`` which first performs an embedding lookup 
   using the embedding table RRef provided by the master and then executes the 
   FC layer which is wrapped inside DDP.
4) The trainer executes the forward pass of the model and uses the loss to 
   execute the backward pass using `Distributed Autograd <https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework>`__.
5) As part of the backward pass, the gradients for the FC layer are computed 
   first and synced to all trainers via allreduce in DDP.
6) Next, Distributed Autograd propagates the gradients to the parameter server, 
   where the gradients for the embedding table are updated.
7) Finally, the `Distributed Optimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`__ is used to update all the parameters.


|
**NOTE**: You should always use `Distributed Autograd <https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework>`__ for the backward pass if you're combining DDP and RPC.


Now, lets go through each part in detail. Firstly, we need to setup all of our 
workers before we can perform any training. We create 4 processes such that 
ranks 0 and 1 are our trainers, rank 2 is the master and rank 3 is the 
parameter server.

We initialize the RPC framework on all 4 workers using the TCP init_method. 
Once RPC initialization is done, the master creates an `EmbeddingBag <https://pytorch.org/docs/master/generated/torch.nn.EmbeddingBag.html>`__ 
on the Parameter Server using `rpc.remote <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.remote>`__.
The master then loops through each trainer and kicks of the training loop by 
calling ``_run_trainer`` on each trainer using `rpc_async <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.rpc_async>`__.
Finally, the master waits for all training to finish before exiting.

The trainers first initialize a ``ProcessGroup`` for DDP with world_size=2 
(for two trainers) using `init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__.
Next, they initialize the RPC framework using the TCP init_method. Note that 
the ports are different in RPC initialization and ProcessGroup intialization. 
This is to avoid port conflicts between initialization of both frameworks. 
Once the initialization is done, the trainers just wait for the ``_run_trainer`` 
RPC from the master.

The parameter server just initializes the RPC framework and waits for RPCs from 
the trainers and master.

.. code:: python

    def run_worker(rank, world_size):
        r"""
        A wrapper function that initializes RPC, calls the function, and shuts down
        RPC.
        """

        # We need to use different port numbers in TCP init_method for init_rpc and
        # init_process_group to avoid port conflicts.
        rpc_backend_options = ProcessGroupRpcBackendOptions()
        rpc_backend_options.init_method='tcp://localhost:29501'

        # Rank 2 is master, 3 is ps and 0 and 1 are trainers.
        if rank == 2:
            rpc.init_rpc(
                    "master",
                    rank=rank,
                    world_size=world_size,
                    rpc_backend_options=rpc_backend_options)

            # Build the embedding table on the ps.
            emb_rref = rpc.remote(
                    "ps",
                    torch.nn.EmbeddingBag,
                    args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
                    kwargs={"mode": "sum"})

            # Run the training loop on trainers.
            futs = []
            for trainer_rank in [0, 1]:
                trainer_name = "trainer{}".format(trainer_rank)
                fut = rpc.rpc_async(
                        trainer_name, _run_trainer, args=(emb_rref, rank))
                futs.append(fut)

            # Wait for all training to finish.
            for fut in futs:
                fut.wait()
        elif rank <= 1:
            # Initialize process group for Distributed DataParallel on trainers.
            dist.init_process_group(
                    backend="gloo", rank=rank, world_size=2,
                    init_method='tcp://localhost:29500')

            # Initialize RPC.
            trainer_name = "trainer{}".format(rank)
            rpc.init_rpc(
                    trainer_name,
                    rank=rank,
                    world_size=world_size,
                    rpc_backend_options=rpc_backend_options)

            # Trainer just waits for RPCs from master.
        else:
            rpc.init_rpc(
                    "ps",
                    rank=rank,
                    world_size=world_size,
                    rpc_backend_options=rpc_backend_options)
            # parameter server do nothing
            pass

        # block until all rpcs finish
        rpc.shutdown()


    if __name__=="__main__":
        # 2 trainers, 1 parameter server, 1 master.
        world_size = 4
        mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)

Before we discuss details of the Trainer, lets introduce the ``HybridModel`` that 
the trainer uses. As described below, the ``HybridModel`` is initialized using an 
RRef to the embedding table (emb_rref) on the parameter server and the ``device`` 
to use for DDP. The initialization of the model wraps a 
`nn.Linear <https://pytorch.org/docs/master/generated/torch.nn.Linear.html>`__ 
layer inside DDP to replicate and synchronize this layer across all trainers.

The forward method of the model is pretty straightforward. It performs an 
embedding lookup on the parameter server using an 
`RRef helper <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.rpc_sync>`__ 
and passes its output onto the FC layer.


.. code:: python

  class HybridModel(torch.nn.Module):
      r"""
      The model consists of a sparse part and a dense part. The dense part is an
      nn.Linear module that is replicated across all trainers using
      DistributedDataParallel. The sparse part is an nn.EmbeddingBag that is
      stored on the parameter server.

      The model holds a Remote Reference to the embedding table on the parameter
      server.
      """

      def __init__(self, emb_rref, device):
          super(HybridModel, self).__init__()
          self.emb_rref = emb_rref
          self.fc = DDP(torch.nn.Linear(16, 8).cuda(device), device_ids=[device])
          self.device = device

      def forward(self, indices, offsets):
          emb_lookup = self.emb_rref.rpc_sync().forward(indices, offsets)
          return self.fc(emb_lookup.cuda(self.device))

Next, lets look at the setup on the Trainer. The trainer first creates the 
``HybridModel`` described above using an RRef to the embedding table on the 
parameter server and its own rank.

Now, we need to retrieve a list of RRefs to all the parameters that we would 
like to optimize with `DistributedOptimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`__. 
To retrieve the parameters for the embedding table from the parameter server, 
we define a simple helper function ``_retrieve_embedding_parameters``, which 
basically walks through all the parameters for the embedding table and returns 
a list of RRefs. The trainer calls this method on the parameter server via RPC 
to receive a list of RRefs to the desired parameters. Since the 
DistributedOptimizer always takes a list of RRefs to parameters that need to 
be optimized, we need to create RRefs even for the local parameters for our 
FC layers. This is done by walking ``model.parameters()``, creating an RRef for 
each parameter and appending it to a list. Note that ``model.parameters()`` only 
returns local parameters and doesn't include ``emb_rref``.

Finally, we create our DistributedOptimizer using all the RRefs and define a 
CrossEntropyLoss function.

.. code:: python

  def _retrieve_embedding_parameters(emb_rref):
      return [RRef(p) for p in emb_rref.local_value().parameters()]


  def _run_trainer(emb_rref, rank):
      r"""
      Each trainer runs a forward pass which involves an embedding lookup on the
      parameter server and running nn.Linear locally. During the backward pass,
      DDP is responsible for aggregating the gradients for the dense part
      (nn.Linear) and distributed autograd ensures gradients updates are
      propagated to the parameter server.
      """

      # Setup the model.
      model = HybridModel(emb_rref, rank)

      # Retrieve all model parameters as rrefs for DistributedOptimizer.

      # Retrieve parameters for embedding table.
      model_parameter_rrefs = rpc.rpc_sync(
              "ps", _retrieve_embedding_parameters, args=(emb_rref,))

      # model.parameters() only includes local parameters.
      for param in model.parameters():
          model_parameter_rrefs.append(RRef(param))

      # Setup distributed optimizer
      opt = DistributedOptimizer(
          optim.SGD,
          model_parameter_rrefs,
          lr=0.05,
      )

      criterion = torch.nn.CrossEntropyLoss()

Now we're ready to introduce the main training loop that is run on each trainer. 
``get_next_batch`` is just a helper function to generate random inputs and 
targets for training. We run the training loop for multiple epochs and for each 
batch:

1) Setup a `Distributed Autograd Context <https://pytorch.org/docs/master/rpc.html#torch.distributed.autograd.context>`__ 
   for Distributed Autograd.
2) Run the forward pass of the model and retrieve its output.
3) Compute the loss based on our outputs and targets using the loss function.
4) Use Distributed Autograd to execute a distributed backward pass using the loss.
5) Finally, run a Distributed Optimizer step to optimize all the parameters.

.. code:: python

    # def _run_trainer(emb_rref, rank): continued...

    def get_next_batch(rank):
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # Generate offsets.
            offsets = []
            start = 0
            batch_size = 0
            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            target = torch.LongTensor(batch_size).random_(8).cuda(rank)
            yield indices, offsets_tensor, target

    # Train for 100 epochs
    for epoch in range(100):
        # create distributed autograd context
        for indices, offsets, target in get_next_batch(rank):
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])

                # Tun distributed optimizer
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training done for epoch {}".format(epoch))

|
Source code for the entire example can be found `here <https://github.com/pytorch/examples/tree/master/distributed/rpc/ddp_rpc>`__.
