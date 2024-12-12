Introduction to Distributed Pipeline Parallelism
================================================
**Authors**: `Howard Huang <https://github.com/H-Huang>`_

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/pipelining_tutorial.rst>`__.

This tutorial uses a gpt-style transformer model to demonstrate implementing distributed
pipeline parallelism with `torch.distributed.pipelining <https://pytorch.org/docs/main/distributed.pipelining.html>`__
APIs.

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
      :class-card: card-prerequisites

      *  How to use ``torch.distributed.pipelining`` APIs
      *  How to apply pipeline parallelism to a transformer model
      *  How to utilize different schedules on a set of microbatches


   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
      :class-card: card-prerequisites

      * Familiarity with `basic distributed training  <https://pytorch.org/tutorials/beginner/dist_overview.html>`__ in PyTorch

Setup
-----

With ``torch.distributed.pipelining`` we will be partitioning the execution of a model and scheduling computation on micro-batches. We will be using a simplified version
of a transformer decoder model. The model architecture is for educational purposes and has multiple transformer decoder layers as we want to demonstrate how to split the model into different
chunks. First, let us define the model:

.. code:: python

   import torch
   import torch.nn as nn
   from dataclasses import dataclass

   @dataclass
   class ModelArgs:
      dim: int = 512
      n_layers: int = 8
      n_heads: int = 8
      vocab_size: int = 10000

   class Transformer(nn.Module):
      def __init__(self, model_args: ModelArgs):
         super().__init__()

         self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

         # Using a ModuleDict lets us delete layers witout affecting names,
         # ensuring checkpoints will correctly save and load.
         self.layers = torch.nn.ModuleDict()
         for layer_id in range(model_args.n_layers):
               self.layers[str(layer_id)] = nn.TransformerDecoderLayer(model_args.dim, model_args.n_heads)

         self.norm = nn.LayerNorm(model_args.dim)
         self.output = nn.Linear(model_args.dim, model_args.vocab_size)

      def forward(self, tokens: torch.Tensor):
         # Handling layers being 'None' at runtime enables easy pipeline splitting
         h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

         for layer in self.layers.values():
               h = layer(h, h)

         h = self.norm(h) if self.norm else h
         output = self.output(h).clone() if self.output else h
         return output

Then, we need to import the necessary libraries in our script and initialize the distributed training process. In this case, we are defining some global variables to use
later in the script:

.. code:: python

   import os
   import torch.distributed as dist
   from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe

   global rank, device, pp_group, stage_index, num_stages
   def init_distributed():
      global rank, device, pp_group, stage_index, num_stages
      rank = int(os.environ["LOCAL_RANK"])
      world_size = int(os.environ["WORLD_SIZE"])
      device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
      dist.init_process_group()

      # This group can be a sub-group in the N-D parallel case
      pp_group = dist.new_group()
      stage_index = rank
      num_stages = world_size

The ``rank``, ``world_size``, and ``init_process_group()`` code should seem familiar to you as those are commonly used in
all distributed programs. The globals specific to pipeline parallelism include ``pp_group`` which is the process
group that will be used for send/recv communications, ``stage_index`` which, in this example, is a single rank
per stage so the index is equivalent to the rank, and ``num_stages`` which is equivalent to world_size.

The ``num_stages`` is used to set the number of stages that will be used in the pipeline parallelism schedule. For example,
for ``num_stages=4``, a microbatch will need to go through 4 forwards and 4 backwards before it is completed. The ``stage_index``
is necessary for the framework to know how to communicate between stages. For example, for the first stage (``stage_index=0``), it will
use data from the dataloader and does not need to receive data from any previous peers to perform its computation.


Step 1: Partition the Transformer Model
---------------------------------------

There are two different ways of partitioning the model:

First is the manual mode in which we can manually create two instances of the model by deleting portions of
attributes of the model. In this example for 2 stages (2 ranks) the model is cut in half.

.. code:: python

   def manual_model_split(model) -> PipelineStage:
      if stage_index == 0:
         # prepare the first stage model
         for i in range(4, 8):
               del model.layers[str(i)]
         model.norm = None
         model.output = None

      elif stage_index == 1:
         # prepare the second stage model
         for i in range(4):
               del model.layers[str(i)]
         model.tok_embeddings = None

      stage = PipelineStage(
         model,
         stage_index,
         num_stages,
         device,
      )
      return stage

As we can see the first stage does not have the layer norm or the output layer, and it only includes the first four transformer blocks.
The second stage does not have the input embedding layers, but includes the output layers and the final four transformer blocks. The function
then returns the ``PipelineStage`` for the current rank.

The second method is the tracer-based mode which automatically splits the model based on a ``split_spec`` argument. Using the pipeline specification, we can instruct
``torch.distributed.pipelining`` where to split the model. In the following code block,
we are splitting before the before 4th transformer decoder layer, mirroring the manual split described above. Similarly,
we can retrieve a ``PipelineStage`` by calling ``build_stage`` after this splitting is done.

.. code:: python
   def tracer_model_split(model, example_input_microbatch) -> PipelineStage:
      pipe = pipeline(
         module=model,
         mb_args=(example_input_microbatch,),
         split_spec={
            "layers.4": SplitPoint.BEGINNING,
         }
      )
      stage = pipe.build_stage(stage_index, device, pp_group)
      return stage


Step 2: Define The Main Execution
---------------------------------

In the main function we will create a particular pipeline schedule that the stages should follow. ``torch.distributed.pipelining``
supports multiple schedules including supports multiple schedules, including single-stage-per-rank schedules ``GPipe`` and ``1F1B``,
as well as multiple-stage-per-rank schedules such as ``Interleaved1F1B`` and ``LoopedBFS``.

.. code:: python

   if __name__ == "__main__":
      init_distributed()
      num_microbatches = 4
      model_args = ModelArgs()
      model = Transformer(model_args)

      # Dummy data
      x = torch.ones(32, 500, dtype=torch.long)
      y = torch.randint(0, model_args.vocab_size, (32, 500), dtype=torch.long)
      example_input_microbatch = x.chunk(num_microbatches)[0]

      # Option 1: Manual model splitting
      stage = manual_model_split(model)

      # Option 2: Tracer model splitting
      # stage = tracer_model_split(model, example_input_microbatch)

      model.to(device)
      x = x.to(device)
      y = y.to(device)

      def tokenwise_loss_fn(outputs, targets):
         loss_fn = nn.CrossEntropyLoss()
         outputs = outputs.reshape(-1, model_args.vocab_size)
         targets = targets.reshape(-1)
         return loss_fn(outputs, targets)

      schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)

      if rank == 0:
         schedule.step(x)
      elif rank == 1:
         losses = []
         output = schedule.step(target=y, losses=losses)
         print(f"losses: {losses}")
      dist.destroy_process_group()

In the example above, we are using the manual method to split the model, but the code can be uncommented to also try the
tracer-based model splitting function. In our schedule, we need to pass in the number of microbatches and
the loss function used to evaluate the targets.

The ``.step()`` function processes the entire minibatch and automatically splits it into microbatches based
on the ``n_microbatches`` passed previously. The microbatches are then operated on according to the schedule class.
In the example above, we are using GPipe, which follows a simple all-forwards and then all-backwards schedule. The output
returned from rank 1 will be the same as if the model was on a single GPU and run with the entire batch. Similarly,
we can pass in a ``losses`` container to store the corresponding losses for each microbatch.

Step 3: Launch the Distributed Processes
----------------------------------------

Finally, we are ready to run the script. We will use ``torchrun`` to create a single host, 2-process job.
Our script is already written in a way rank 0 that performs the required logic for pipeline stage 0, and rank 1
performs the logic for pipeline stage 1.

``torchrun --nnodes 1 --nproc_per_node 2 pipelining_tutorial.py``

Conclusion
----------

In this tutorial, we have learned how to implement distributed pipeline parallelism using PyTorch's ``torch.distributed.pipelining`` APIs.
We explored setting up the environment, defining a transformer model, and partitioning it for distributed training.
We discussed two methods of model partitioning, manual and tracer-based, and demonstrated how to schedule computations on
micro-batches across different stages. Finally, we covered the execution of the pipeline schedule and the launch of distributed
processes using ``torchrun``.

Additional Resources
--------------------

We have successfully integrated ``torch.distributed.pipelining`` into the `torchtitan repository <https://github.com/pytorch/torchtitan>`__. TorchTitan is a clean, minimal code base for
large-scale LLM training using native PyTorch. For a production ready usage of pipeline
parallelism as well as composition with other distributed techniques, see
`TorchTitan end to end example of 3D parallelism <https://github.com/pytorch/torchtitan>`__.
