Getting Started with Fully Sharded Data Parallel(FSDP)
======================================================

**Author**: `Wei Feng <https://github.com/weifengpy>`__, `Will Constable <https://github.com/wconstab>`__, `Yifan Mao <https://github.com/mori360>`__

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/FSDP2_tutorial.rst>`__.

How FSDP2 works
--------------
In `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__ (DDP) training, each rank owns a model replica and processes a batch of data, finally it uses all-reduce to sync gradients across ranks.

Comparing with DDP, FSDP reduces GPU memory footprint by sharding model parameters, gradients, and optimizer states. It makes it feasible to train models that cannot fit on a single GPU. As shown below in the picture,
* Outside of forward and backward, parameters stay fully sharded.
* Before forward and backward, all-gather to unshard parameters for computation.
* Inside backward, reduce-scatter to get fully sharded gradients.
* Optimizer updates sharded parameters according to sharded gradients, resulting in sharded optimizer states.

.. figure:: /_static/img/distributed/fsdp_workflow.png
   :width: 100%
   :align: center
   :alt: FSDP workflow

   FSDP Workflow


FSDP can be considered as decomposing DDP all-reduce into reduce-scatter and all-gather.

.. figure:: /_static/img/distributed/fsdp_sharding.png
   :width: 100%
   :align: center
   :alt: FSDP allreduce

   FSDP Allreduce

Comparing with FSDP1, FSDP2 has following advantages:
* Representing sharded parameters as DTensors sharded on dim-i, allowing for easy manipulation of individual parameters, communication-free sharded state dicts, and a simpler meta-device initialization flow.
* Improving memory management system that achieves lower and deterministic GPU memory by avoiding recordStream and does so without any CPU synchronization.
* Offers an extension point to customize the all-gather, e.g. for fp8 all-gather for fp8 linears.
* Mixing frozen and non-frozen parameters can in the the same communication group without using extra memory.

How to use FSDP2
---------------
Model Initialization: nested wrapping, dim-0 sharding, AC

Loading State Dict

Forward and Backward

Gradient Clipping and Scaler, and Optimizer with DTensor

Saving State Dict

FSDP1-to-FSDP2 Migration Guide
---------------
