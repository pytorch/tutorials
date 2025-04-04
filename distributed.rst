Distributed
===========

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
* `Tensor Parallel (TP) <#learn-tp>`__
* `Device Mesh <#device-mesh>`__
* `Remote Procedure Call (RPC) distributed training <#learn-rpc>`__
* `Custom Extensions <#custom-extensions>`__

Read more about these options in `Distributed Overview <../beginner/dist_overview.html>`__.

.. _learn-ddp:

Learn DDP
---------

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        DDP Intro Video Tutorials
        :link: https://pytorch.org/tutorials/beginner/ddp_series_intro.html?utm_source=distr_landing&utm_medium=ddp_series_intro
        :link-type: url

        A step-by-step video series on how to get started with
        `DistributedDataParallel` and advance to more complex topics
        +++
        :octicon:`code;1em` Code :octicon:`square-fill;1em` :octicon:`video;1em` Video

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting Started with Distributed Data Parallel
        :link: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html?utm_source=distr_landing&utm_medium=intermediate_ddp_tutorial
        :link-type: url

        This tutorial provides a short and gentle intro to the PyTorch
        DistributedData Parallel.
        +++
        :octicon:`code;1em` Code

     .. grid-item-card:: :octicon:`file-code;1em`
        Distributed Training with Uneven Inputs Using
        the Join Context Manager
        :link: https://pytorch.org/tutorials/advanced/generic_join.html?utm_source=distr_landing&utm_medium=generic_join
        :link-type: url

        This tutorial describes the Join context manager and
        demonstrates it's use with DistributedData Parallel.
        +++
        :octicon:`code;1em` Code

.. _learn-fsdp:

Learn FSDP
----------

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting Started with FSDP
        :link: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html?utm_source=distr_landing&utm_medium=FSDP_getting_started
        :link-type: url

        This tutorial demonstrates how you can perform distributed training
        with FSDP on a MNIST dataset.
        +++
        :octicon:`code;1em` Code

     .. grid-item-card:: :octicon:`file-code;1em`
        FSDP Advanced
        :link: https://pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html?utm_source=distr_landing&utm_medium=FSDP_advanced
        :link-type: url

        In this tutorial, you will learn how to fine-tune a HuggingFace (HF) T5
        model with FSDP for text summarization.
        +++
        :octicon:`code;1em` Code


.. _learn-tp:

Learn Tensor Parallel (TP)
---------------

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Large Scale Transformer model training with Tensor Parallel (TP)
        :link: https://pytorch.org/tutorials/intermediate/TP_tutorial.html
        :link-type: url

        This tutorial demonstrates how to train a large Transformer-like model across hundreds to thousands of GPUs using Tensor Parallel and Fully Sharded Data Parallel.
        +++
        :octicon:`code;1em` Code


.. _device-mesh:

Learn DeviceMesh
----------------

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting Started with DeviceMesh
        :link: https://pytorch.org/tutorials/recipes/distributed_device_mesh.html?highlight=devicemesh
        :link-type: url

        In this tutorial you will learn about `DeviceMesh`
        and how it can help with distributed training.
        +++
        :octicon:`code;1em` Code

.. _learn-rpc:

Learn RPC
---------

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting Started with Distributed RPC Framework
        :link: https://pytorch.org/tutorials/intermediate/rpc_tutorial.html?utm_source=distr_landing&utm_medium=rpc_getting_started
        :link-type: url

        This tutorial demonstrates how to get started with RPC-based distributed
        training.
        +++
        :octicon:`code;1em` Code

     .. grid-item-card:: :octicon:`file-code;1em`
        Implementing a Parameter Server Using Distributed RPC Framework
        :link: https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html?utm_source=distr_landing&utm_medium=rpc_param_server_tutorial
        :link-type: url

        This tutorial walks you through a simple example of implementing a
        parameter server using PyTorch’s Distributed RPC framework.
        +++
        :octicon:`code;1em` Code

     .. grid-item-card:: :octicon:`file-code;1em`
        Implementing Batch RPC Processing Using Asynchronous Executions
        :link: https://pytorch.org/tutorials/intermediate/rpc_async_execution.html?utm_source=distr_landing&utm_medium=rpc_async_execution
        :link-type: url

        In this tutorial you will build batch-processing RPC applications
        with the @rpc.functions.async_execution decorator.
        +++
        :octicon:`code;1em` Code

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Combining Distributed DataParallel with Distributed RPC Framework
        :link: https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html?utm_source=distr_landing&utm_medium=rpc_plus_ddp
        :link-type: url

        In this tutorial you will learn how to combine distributed data
        parallelism with distributed model parallelism.
        +++
        :octicon:`code;1em` Code

.. _custom-extensions:

Custom Extensions
-----------------

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Customize Process Group Backends Using Cpp Extensions
        :link: https://pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html?utm_source=distr_landing&utm_medium=custom_extensions_cpp
        :link-type: url

        In this tutorial you will learn to implement a custom `ProcessGroup`
        backend and plug that into PyTorch distributed package using
        cpp extensions.
        +++
        :octicon:`code;1em` Code
