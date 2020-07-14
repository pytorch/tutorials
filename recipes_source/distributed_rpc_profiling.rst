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

As an example, let’s take a pretrained vision model. All of the
pretrained models in TorchVision are compatible with TorchScript.


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
