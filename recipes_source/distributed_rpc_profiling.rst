Profiling PyTorch RPC-Based Workloads
======================================

In this recipe, you will learn:

-  An overview of the Distributed RPC Framework
-  An overview of the PyTorch Profiler
-  How to use the PyTorch Profiler to profile RPC-based workloads

Requirements
------------

-  PyTorch 1.6

The instructions for installing PyTorch are
available at `pytorch.org`_.

What is the Distributed RPC Framework?
---------------------------------------

The ** Distributed RPC Framework ** provides mechanisms for multi-machine model
training through a set of primitives to allow for remote communication, and a 
higher-level API to automatically differentiate models split across several machines.
For this recipe, it would be helpful to be familiar with the Distributed RPC Framework
as well as the tutorials. 

What is the PyTorch Profiler?
---------------------------------------
The profiler is a context manager based API that allows for on-demand profiling of
operators in a model's workload. The profiler can be used to analyze various aspects
of a model including execution time, operators invoked, and memory consumption. For a
detailed tutorial on using the profiler to profile a single-node model, please see the
Profiler Recipe.



How to use the Profiler for RPC-based workloads
-----------------------------------------------

The profiler supports profiling of calls made of RPC and allows the user to have a
detailed view into the operations that take place on different nodes. To demonstrate an
example of this, let's first set up the RPC framework:

.. code:: python
  import torch
  import torch.distributed.rpc as rpc
  import torch.autograd.profiler as profiler
  import torch.multiprocessing as mp
  import os
  import logging
  import sys

  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  logger = logging.getLogger()

  def random_tensor():
      return torch.rand((3, 3), requires_grad=True)


  def worker(rank, world_size):
      os.environ["MASTER_ADDR"] = "localhost"
      os.environ["MASTER_PORT"] = "29500"
      worker_name = f"worker{rank}"

      # Initialize RPC framework.
      rpc.init_rpc(
          name=worker_name,
          rank=rank,
          world_size=world_size
      )
      logger.debug(f"{worker_name} successfully initialized RPC.")

      pass # to be continued below


  if __name__ == '__main__':
      # Run 2 RPC workers.
      world_size = 2
      mp.spawn(worker, args=(world_size,), nprocs=world_size)

Running the above program should present you with the following output:

.. 
  DEBUG:root:worker0 successfully initialized RPC.
  DEBUG:root:worker1 successfully initialized RPC.

And you're done!







Important Resources
-------------------

-  `pytorch.org`_ for installation instructions, and more documentation
   and tutorials.
-  `Introduction to TorchScript tutorial`_ for a deeper initial
   exposition of TorchScript
-  `Full TorchScript documentation`_ for complete TorchScript language
   and API reference

.. _pytorch.org: https://pytorch.org/
.. _Introduction to TorchScript tutorial: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
.. _Full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
.. _Loading A TorchScript Model in C++ tutorial: https://pytorch.org/tutorials/advanced/cpp_export.html
.. _full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
