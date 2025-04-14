Introduction to Context Parallel
======================================
**Authors**: `Xilun Wu <https://github.com/XilunWu>`_, `Chien-Chin Huang <https://github.com/fegin>`__

.. note::
    |edit| View and edit this tutorial in `GitHub <https://github.com/pytorch/tutorials/blob/main/prototype_source/context_parallel.rst>`__.

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
      :class-card: card-prerequisites

      * `Context Parallel APIs <https://pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.experimental.context_parallel>`__
      * `1M sequence training in TorchTitan with Context Parallel <https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082>`__


   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
      :class-card: card-prerequisites

      * PyTorch 2.7 or later


Introduction
------------

Context Parallel is an approach used in large language model training to reduce peak activation size by sharding the long input sequence across multiple devices.
It breaks the constraint on input sequence length resulting from peak memory usage on storing activations in Transformer blocks.

The core of Context Parallel is Ring Attention, a novel parallel implementation of the Attention layer.
Ring Attention shuffles the KV shards and calculates the partial attention scores, repeats until all KV shards have been used on each device.
Two Ring Attention variants have been implemented: `the all-gather based pass-KV <https://arxiv.org/abs/2407.21783>`__ and `the all-to-all based pass-KV <https://openreview.net/forum?id=WsRHpHH4s0>`__:
1.  The all-gather based pass-KV algorithm is used in Llama3 training, which initially performs an all-gather on the key and value tensors, followed by computing the attention output for the
    local query tensor chunk. Our modified all-gather based pass-KV algorithm concurrently all-gathers KV shards and computes attention output for the local query tensor chunk
    using local key and value tensor chunks, followed by a final computation of attention output for the local query tensor and remaining KV shards. This allows some degree of
    overlap between the attention computation and the all-gather collective.
2.  The all-to-all approach uses interleaved all-to-all collectives to ring shuffle KV shards to overlap the SDPA computation and the all-to-all communication
    necessary for the next SDPA.

The Context Parallel APIs consist of two parts:

1. ``context_parallel()`` allows users to create a Python context where the SDPA function (``torch.nn.functional.scaled_dot_product_attention``)
   will be automatically replaced with Ring Attention. To shard Tensors along a dimension, simply pass the Tensors and their sharding dimensions to
   argument ``buffers`` and ``buffer_seq_dims`` respectively.
2. ``set_rotate_method()`` allows users to choose between the all-gather based pass-KV approach and the all-to-all based pass-KV approach.


Setup
---------------------

With ``torch.distributed.tensor.experimental.context_parallel()``, users can easily shard the Tensor input and parallelize the execution of the SDPA function.
To better demonstrate the usage of this API, we start with a simple code snippet doing SDPA and then parallelize it using the API:

.. code:: python

    import torch
    import torch.nn.functional as F

    from torch.nn.attention import sdpa_kernel, SDPBackend


    def sdpa_example():
        assert torch.cuda.is_available()
        torch.cuda.set_device("cuda:0")
        torch.cuda.manual_seed(0)

        batch = 8
        nheads = 8
        qkv_len = 8192
        dim = 32
        backend = SDPBackend.FLASH_ATTENTION
        dtype = (
            torch.bfloat16
            if backend == SDPBackend.FLASH_ATTENTION
            or backend == SDPBackend.CUDNN_ATTENTION
            else torch.float32
        )

        qkv = [
            torch.rand(
                (batch, nheads, qkv_len, dim),
                dtype=dtype,
                requires_grad=True,
                device='cuda',
            )
            for _ in range(3)
        ]
        # specify the SDPBackend to use
        with sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(*qkv, is_causal=True)


    if __name__ == "__main__":
        sdpa_example()


Enable Context Parallel
-----------------------

Now, let's first adapt it to a distributed program where each rank has the same tensor input. Then we apply the context parallel API to
shard to input and distribute the computation across ranks:

.. code:: python

    # file: cp_sdpa_example.py
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import context_parallel_unshard
    from torch.nn.attention import sdpa_kernel, SDPBackend


    def context_parallel_sdpa_example(world_size: int, rank: int):
        assert torch.cuda.is_available()
        assert dist.is_nccl_available()
        torch.cuda.set_device(f"cuda:{rank}")
        torch.cuda.manual_seed(0)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        device_mesh = init_device_mesh(
            device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("cp",)
        )

        batch = 8
        nheads = 8
        qkv_len = 64
        dim = 32
        backend = SDPBackend.FLASH_ATTENTION
        dtype = (
            torch.bfloat16
            if backend == SDPBackend.FLASH_ATTENTION
            or backend == SDPBackend.CUDNN_ATTENTION
            else torch.float32
        )

        qkv = [
            torch.rand(
                (batch, nheads, qkv_len, dim),
                dtype=dtype,
                requires_grad=True,
                device='cuda',
            )
            for _ in range(3)
        ]
        # specify the SDPBackend to use
        with sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(*qkv, is_causal=True)

        # make a clean copy of QKV for output comparison
        cp_qkv = [t.detach().clone() for t in qkv]

        with sdpa_kernel(backend):
            # This `context_parallel()` performs two actions:
            # 1. Shard the tensor objects in `buffers` in-place along the dimension
            #    specified in `buffer_seq_dims`, the tensors in `buffers` and their
            #    sharding dims in `buffer_seq_dims` are organized in the same order.
            # 2. Replace the execution of `F.scaled_dot_product_attention` with a
            #    context-paralleled-enabled Ring Attention.
            with context_parallel(
                device_mesh, buffers=tuple(cp_qkv), buffer_seq_dims=(2, 2, 2)
            ):
                cp_out = F.scaled_dot_product_attention(*cp_qkv, is_causal=True)

            # The output `cp_out` is still sharded in the same way as QKV
            # the `context_parallel_unshard` API allows users to easily
            # unshard to gain the full tensor.
            (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [2])

        assert torch.allclose(
            cp_out,
            out,
            atol=(1e-08 if dtype == torch.float32 else 1e-03 * world_size),
        )


    if __name__ == "__main__":
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        try:
            context_parallel_sdpa_example(world_size, rank)
        finally:
            dist.barrier()
            dist.destroy_process_group()


You can use the command ``torchrun --standalone --nnodes=1 --nproc-per-node=4 cp_sdpa_example.py`` to launch the above context parallel
SDPA on 4 GPUs. We demonstrate the numeric correctness by comparing the output of Ring Attention to that of SDPA on a single GPU.


Select Rotation Approach
------------------------

You can choose the desired shards rotation approach in Ring Attention by using ``torch.distributed.tensor.experimental._attention.set_rotate_method()``:

.. code:: python

    # file: cp_sdpa_example.py
    from torch.distributed.tensor.experimental._attention import set_rotate_method

    set_rotate_method("alltoall")  # rotate shards using all-to-all

    with sdpa_kernel(backend):
        with context_parallel(
            device_mesh, buffers=tuple(cp_qkv), buffer_seq_dims=(2, 2, 2)
        ):
            cp_out = F.scaled_dot_product_attention(*cp_qkv, is_causal=True)


The default rotation approach is the all-gather based pass-KV.


Conclusion
----------

In this tutorial, we have learned how to parallelize the SDPA computation along the sequence dimension easily with our Context Parallel APIs. For
design and implementation details, performance analysis, and an end-to-end training example in `TorchTitan <https://github.com/pytorch/torchtitan>`__,
see our post on `PyTorch native long-context training <https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082>`__.
