Distributed and Parallel Training Tutorials
===========================================

Distributed training is a model training paradigm that involves
spreading training workload across multiple worker nodes, therefore
significantly improving the speed of training and model accuracy. While
distributed training can be used for any type of ML model training, it
is most beneficial to use it for large models and compute demanding
tasks as deep learning.

There are a few ways you can perform distributed training in
PyTorch with each method having their advantages in certain use cases:

* `DistributedDataParallel (DDP) <#learn-ddp>`__
* `Fully Sharded Data Parallel (FSDP) <#learn-fsdp>`__
* `Remote Procedure Call (RPC) distributed training <#learn-rpc>`__
* `Pipeline Parallelism <#learn-pipeline-parallelism>`__

Read more about these options in [Distributed Overview](../beginner/dist_overview.rst).

.. _learn-ddp:

Learn DDP
---------

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        DDP Intro Video Tutorials
        :shadow: none
        :link: https://example.com
        :link-type: url

        A step-by-step video series on how to get started with
        `DistributedDataParallel` and advance to more complex topics
        +++
        :octicon:`code;1em` Code :octicon:`square-fill;1em` :octicon:`video;1em` Video

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting Started with PyTorch Distributed
        :shadow: none
        :link: https://example.com
        :link-type: url

        This tutorial provides a short and gentle intro to the PyTorch
        DistributedData Parallel.
        +++
        :octicon:`code;1em` Code

.. _learn-fsdp:

Learn FSDP
----------

Fully-Sharded Data Parallel (FSDP) is a tool that distributes model
parameters across multiple workers, therefore enabling you to train larger
models.


.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting Started with FSDP
        :shadow: none
        :link: https://example.com
        :link-type: url

        This tutorial demonstrates how you can perform distributed training
        with FSDP on a MNIST dataset.
        +++
        :octicon:`code;1em` Code

     .. grid-item-card:: :octicon:`file-code;1em`
        FSDP Advanced
        :shadow: none
        :link: https://example.com
        :link-type: url

        In this tutorial, you will learn how to fine-tune a HuggingFace (HF) T5
        model with FSDP for text summarization.
        +++
        :octicon:`code;1em` Code

.. _learn-rpc:

Learn RPC
---------

Distributed Remote Procedure Call (RPC) framework provides
mechanisms for multi-machine model training

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting Started with Distributed RPC Framework
        :shadow: none
        :link: https://example.com
        :link-type: url

        This tutorial demonstrates how to get started with RPC-based distributed
        training.
        +++
        :octicon:`code;1em` Code

     .. grid-item-card:: :octicon:`file-code;1em`
        Implementing a Parameter Server Using Distributed RPC Framework
        :shadow: none
        :link: https://example.com
        :link-type: url

        This tutorial walks you through a simple example of implementing a
        parameter server using PyTorch’s Distributed RPC framework.
        +++
        :octicon:`code;1em` Code

     .. grid-item-card:: :octicon:`file-code;1em`
        Distributed Pipeline Parallelism Using RPC
        :shadow: none
        :link: https://example.com
        :link-type: url

        Learn how to use a Resnet50 model for distributed pipeline parallelism
        with the Distributed RPC APIs.
        +++
        :octicon:`code;1em` Code

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Implementing Batch RPC Processing Using Asynchronous Executions
        :shadow: none
        :link: https://example.com
        :link-type: url

        In this tutorial you will build batch-processing RPC applications
        with the @rpc.functions.async_execution decorator.
        +++
        :octicon:`code;1em` Code

     .. grid-item-card:: :octicon:`file-code;1em`
        Combining Distributed DataParallel with Distributed RPC Framework
        :shadow: none
        :link: https://example.com
        :link-type: url

        In this tutorial you will learn how to combine distributed data
        parallelism with distributed model parallelism.
        +++
        :octicon:`code;1em` Code

.. _learn-pipeline-parallelism:
