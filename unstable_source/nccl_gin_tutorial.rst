GPU-Initiated Networking with NCCL and PyTorch Symmetric Memory
===============================================================

.. note::
    |edit| View and edit this tutorial in `GitHub <https://github.com/pytorch/tutorials/blob/main/unstable_source/nccl_gin_tutorial.rst>`__.

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
      :class-card: card-prerequisites

      * What GPU-Initiated Networking (GIN) is and how it relates to the
        NCCL device API
      * How to allocate and exchange symmetric memory tensors with the
        NCCL backend of ``torch.distributed._symmetric_memory``
      * How to use one-sided put and signal operations between ranks
      * How to run device-initiated collectives such as
        ``one_shot_all_reduce``
      * How to write a custom communication kernel in Python with the
        CuTe DSL

   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
      :class-card: card-prerequisites

      * PyTorch 2.11 or later (nightly recommended)
      * NCCL 2.28 or later (2.28.7 or later for GIN over the network)
      * A host with two or more CUDA GPUs
      * For multi-node GIN: RDMA-capable NICs (ConnectX-4 or newer) with
        GPUDirect RDMA
      * For the custom kernel section: ``nvidia-cutlass-dsl`` 4.5 or
        later
      * Familiarity with `PyTorch Distributed <https://docs.pytorch.org/tutorials/beginner/dist_overview.html>`__

Introduction
------------

In the traditional PyTorch distributed model, communication is
*host-initiated*: the CPU enqueues each collective (for example,
``dist.all_reduce``) onto a CUDA stream, and NCCL launches a kernel that
performs the communication. This model works well for large, structured
collectives, but it adds CPU launch latency to every operation and makes
it hard to fuse communication with computation inside a single kernel.

NCCL 2.28 introduced a *device API* that turns this model around:
communication can be initiated directly from GPU code, without a round
trip through the CPU. The device API has three building blocks:

* **LSA (Load/Store Accessible)**: peers reachable over NVLink or PCIe
  P2P are accessed with direct loads and stores.
* **Multimem**: uses NVLink SHARP multicast on supported hardware.
* **GIN (GPU-Initiated Networking)**: a CUDA kernel initiates RDMA
  transfers over the network to remote nodes. This is the piece that
  extends device-initiated communication beyond a single machine.

PyTorch exposes this functionality through `Symmetric Memory
<https://docs.pytorch.org/docs/main/symmetric_memory.html>`__.
A symmetric memory tensor is allocated with the same size on every rank
and registered with NCCL as a *window*, which makes it remotely
accessible by all peers. Once a window is established, PyTorch can run
device-initiated collectives and one-sided operations (put, get,
signal) on it. Within a node, transfers use LSA; across nodes, NCCL
uses GIN under the hood.

.. note::
   ``torch.distributed._symmetric_memory`` is an unstable API. Names
   and signatures may change between releases. The examples in this
   tutorial were written against PyTorch nightly builds.

Enabling the NCCL backend for symmetric memory
----------------------------------------------

Symmetric memory supports several allocation backends (``CUDA``,
``NVSHMEM``, and ``NCCL``). To use the NCCL device API, select the
``NCCL`` backend before allocating any tensors:

.. code:: python

    import torch.distributed._symmetric_memory as symm_mem

    symm_mem.set_backend("NCCL")

Alternatively, set the environment variable ``TORCH_SYMMMEM=NCCL``
before starting the process.

The NCCL backend requires an eagerly initialized NCCL communicator.
Pass ``device_id`` to ``init_process_group`` so that the communicator
is created up front, and issue one warm-up collective before the first
symmetric memory allocation:

.. code:: python

    dist.init_process_group(backend="nccl", device_id=device)
    dist.all_reduce(torch.ones(1, device=device))

A first example: device-initiated all-reduce
--------------------------------------------

The following script allocates a symmetric memory tensor on each rank,
establishes the NCCL windows through ``rendezvous``, and runs a
one-shot all-reduce. A one-shot all-reduce is a single fused kernel:
each rank reads its peers' buffers directly and reduces them locally,
with no separate communication kernel launched by the host. Save this
program as ``symm_mem_all_reduce.py``:

.. code:: python

    # file: symm_mem_all_reduce.py
    import os

    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem


    def main():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)

        dist.init_process_group(backend="nccl", device_id=device)
        symm_mem.set_backend("NCCL")

        # Warm up the NCCL communicator before the first allocation.
        dist.all_reduce(torch.ones(1, device=device))
        group_name = dist.group.WORLD.group_name

        # Allocate a symmetric tensor. Every rank must allocate the
        # same size, and the allocation must happen on all ranks.
        t = symm_mem.empty(4096, dtype=torch.float32, device=device)
        t.fill_(rank + 1)

        # Establish the symmetric memory windows. This is a collective
        # operation and returns a handle for one-sided operations.
        symm_mem.rendezvous(t, group=group_name)

        # Device-initiated all-reduce over the symmetric tensor.
        res = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)

        expected = sum(range(1, dist.get_world_size() + 1))
        assert res.eq(expected).all().item()
        if rank == 0:
            print(f"one_shot_all_reduce OK, every element == {expected}")

        dist.destroy_process_group()


    if __name__ == "__main__":
        main()

Run it with ``torchrun`` on a machine with at least two GPUs:

.. code:: shell

    torchrun --nnodes=1 --nproc_per_node=2 symm_mem_all_reduce.py

You should see:

.. code:: shell

    one_shot_all_reduce OK, every element == 3

Two things distinguish this from a regular ``dist.all_reduce``:

1. The input tensor must be a rendezvoused symmetric memory tensor —
   the kernel relies on every rank's buffer being remotely accessible.
2. The whole operation is one kernel that both communicates and
   reduces. There is no separate NCCL collective launch, which makes
   the operation cheap for small messages and easy to capture in a
   CUDA graph.

One-sided communication: put and signal
---------------------------------------

Collectives are symmetric by nature: every rank participates in the
same operation at the same time. The device API also enables
*one-sided* operations, where a rank writes into a peer's buffer and
notifies it, without the peer posting a matching receive. This is the
communication style used by MoE token dispatch, pipeline transfers, and
other irregular patterns.

The ``torch.ops.symm_mem`` namespace provides NCCL put/get operations
that run as device kernels. In the following example, ranks are paired:
each odd rank writes its buffer into the previous even rank's symmetric
tensor and raises a signal; the even rank blocks until the signal
arrives, then reads the delivered data. Save it as
``symm_mem_put_signal.py``:

.. code:: python

    # file: symm_mem_put_signal.py
    import os

    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem


    def main():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)

        dist.init_process_group(backend="nccl", device_id=device)
        symm_mem.set_backend("NCCL")
        dist.all_reduce(torch.ones(1, device=device))
        group_name = dist.group.WORLD.group_name

        t = symm_mem.empty(1024, dtype=torch.float32, device=device)
        t.fill_(rank)
        hdl = symm_mem.rendezvous(t, group=group_name)

        # Make sure all ranks finished writing their initial values
        # before any peer starts writing remotely.
        dist.barrier()

        signal_val = 1
        if rank % 2 == 1:
            # Write our buffer into the peer's symmetric tensor and
            # raise its signal so it knows the data arrived.
            peer = rank - 1
            torch.ops.symm_mem.nccl_put_with_signal(t, signal_val, peer)
        elif rank + 1 < hdl.world_size:
            # Block until the peer's put has been delivered.
            peer = rank + 1
            torch.ops.symm_mem.nccl_wait_for_signal(t, signal_val)
            torch.cuda.synchronize()
            assert t.eq(peer).all().item()
            print(f"rank {rank}: received data from rank {peer}")

        dist.barrier()
        dist.destroy_process_group()


    if __name__ == "__main__":
        main()

Run it the same way:

.. code:: shell

    torchrun --nnodes=1 --nproc_per_node=2 symm_mem_put_signal.py

Expected output:

.. code:: shell

    rank 0: received data from rank 1

A few notes on the one-sided primitives:

* ``torch.ops.symm_mem.nccl_put(tensor, peer)`` writes the local
  symmetric tensor into ``peer``'s corresponding buffer, and
  ``torch.ops.symm_mem.nccl_get(tensor, peer)`` reads a peer's buffer
  into the local one. ``nccl_put_with_signal`` additionally raises a
  signal on the destination rank after the data lands, and
  ``nccl_wait_for_signal`` blocks the stream until that signal arrives.
* With NCCL 2.29 or later, the handle also exposes host-initiated
  one-sided signaling: ``hdl.put_signal(dst_rank=peer)`` and
  ``hdl.wait_signal(src_rank=peer)``.
* Because the operations are one-sided, the target rank does not post
  a matching receive. You are responsible for ordering remote writes
  against local reads, using signals or a barrier.

Going multi-node: where GIN takes over
--------------------------------------

Nothing in the code above changes when you scale from one node to
several. When a peer is reachable over NVLink or PCIe, the NCCL backend
services these operations with direct loads and stores (LSA). When a
peer lives on another node, NCCL services the same window operations
with GIN: the GPU posts RDMA operations to the NIC directly from the
kernel, without waking up the CPU.

Launch across two nodes the usual way:

.. code:: shell

    # On node 0
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
        symm_mem_ring_put.py

    # On node 1
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
        symm_mem_ring_put.py

For GIN to be available, the cluster must satisfy NCCL's requirements:

* NCCL 2.28.7 or later
* CUDA 12.2 or later, driver 510.40.3 or later
* ConnectX-4 or newer NICs with ``rdma-core`` 44 or later
* GPUDirect RDMA, via DMA-BUF (Linux kernel 6.1 or later) or the
  ``nvidia-peermem`` module
* Full NIC connectivity between all rails; ``NCCL_CROSS_NIC=0`` is not
  supported with the device API

NCCL chooses between two GIN transports automatically: GDAKI
(GPUDirect Async Kernel-Initiated), where the GPU drives the NIC
doorbells directly, and a CPU proxy transport that works on a wider
range of NICs by relaying GPU requests through lock-free queues. Run
with ``NCCL_DEBUG=INFO`` to see which transport was selected. If
symmetric window registration is not available (for example, an old
NCCL or missing RDMA support), rendezvous or the device-initiated ops
raise an error; check the debug log for the reason.

Writing custom communication kernels in Python with CuTe DSL
------------------------------------------------------------

The operations used so far are prebuilt kernels that ship with
PyTorch. Symmetric memory also lets you write your *own* communication
kernels, and with the `CuTe DSL
<https://docs.nvidia.com/cutlass/latest/cute_dsl_general/cute_dsl.html>`__
(``nvidia-cutlass-dsl``) you can do it in Python. The key enabler is
``hdl.get_buffer(peer, shape, dtype)``: it returns a regular CUDA
tensor whose data pointer is the mapped address of a *peer's*
symmetric buffer. A kernel that loads from that tensor performs a
remote read over NVLink or PCIe, so you can pass the peer buffers into
a CuTe DSL kernel like any local tensor.

The following example implements a one-shot all-reduce as a custom
kernel: every rank reads all peers' buffers directly and accumulates
them into a local output. It is adapted from the `distributed CuTe DSL
examples
<https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell/kernel/distributed>`__
in the CUTLASS repository. Install the DSL with
``pip install nvidia-cutlass-dsl`` (Python 3.12 recommended) and save
the program as ``cute_all_reduce.py``:

.. code:: python

    # file: cute_all_reduce.py
    import os

    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack


    @cute.kernel
    def all_reduce_kernel(
        inputs: list[cute.Tensor],
        gOut: cute.Tensor,
        thr_layout: cute.Layout,
        val_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        blk_coord = ((None, None), bidx)
        local_tile_out = gOut[blk_coord]
        local_tile_list = [t[blk_coord] for t in inputs]

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), inputs[0].element_type
        )
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)

        thr_tensor_list = [thr_copy.partition_S(t) for t in local_tile_list]
        thr_out = thr_copy.partition_D(local_tile_out)
        frg_acc = cute.make_fragment_like(thr_out)
        frg_acc.fill(0.0)

        # Each iteration loads the same tile from a different rank's
        # buffer. Loads from peer tensors are remote reads.
        for thr in thr_tensor_list:
            frg = cute.make_fragment_like(thr)
            cute.copy(copy_atom, thr, frg)
            frg_acc.store(frg.load() + frg_acc.load())

        cute.copy(copy_atom, frg_acc, thr_out)


    @cute.jit
    def all_reduce(
        inputs: list[cute.Tensor],
        output: cute.Tensor,
        copy_bits: cutlass.Constexpr = 128,
    ):
        vector_size = copy_bits // inputs[0].element_type.width
        thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
        val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        divided_inputs = [cute.zipped_divide(t, tiler_mn) for t in inputs]
        gOut = cute.zipped_divide(output, tiler_mn)
        all_reduce_kernel(divided_inputs, gOut, thr_layout, val_layout).launch(
            grid=[cute.size(gOut, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
        )


    def main():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)

        dist.init_process_group(backend="nccl", device_id=device)
        symm_mem.set_backend("NCCL")
        dist.all_reduce(torch.ones(1, device=device))
        world_size = dist.get_world_size()

        M, N = 1024, 512
        t = symm_mem.empty((M, N), dtype=torch.float32, device=device)
        hdl = symm_mem.rendezvous(t, dist.group.WORLD)
        t.random_(0, 100)
        output = torch.zeros((M, N), device=device)

        # One tensor view per rank, each backed by that rank's
        # symmetric buffer mapped into this process.
        peer_tensors = [
            hdl.get_buffer(r, t.shape, t.dtype) for r in range(world_size)
        ]

        compiled = cute.compile(
            all_reduce,
            [from_dlpack(p) for p in peer_tensors],
            from_dlpack(output),
        )

        # Every rank must finish writing its input before any peer
        # reads it, and no rank may exit before all reads finish.
        dist.barrier()
        compiled([from_dlpack(p) for p in peer_tensors], from_dlpack(output))
        dist.barrier()

        expected = t.clone()
        dist.all_reduce(expected)
        torch.testing.assert_close(output, expected)
        if rank == 0:
            print("custom CuTe DSL all_reduce OK")

        dist.destroy_process_group()


    if __name__ == "__main__":
        main()

Run it with ``torchrun``:

.. code:: shell

    torchrun --nnodes=1 --nproc_per_node=2 cute_all_reduce.py

You should see:

.. code:: shell

    custom CuTe DSL all_reduce OK

Note the division of labor: PyTorch symmetric memory handles all the
setup (allocation, window registration, peer mapping), and the CuTe
DSL kernel is ordinary tile-based code — the only distributed aspect
is that some of its input tensors happen to live on other GPUs.
Custom Python kernels like this one can reach peers over NVLink or
PCIe (the LSA path) within a node. Cross-node RDMA through GIN is not
accessible from Python kernels today; that requires the C++ device
API described in the next section.

Writing custom device kernels with GIN
--------------------------------------

The operations above are host-visible entry points to device-initiated
communication. The full power of GIN — issuing puts, gets, and signals
from *inside your own CUDA kernel*, interleaved with computation — is
exposed at the NCCL level through ``nccl_device.h``. A kernel obtains a
``ncclGin`` object from a device communicator and calls primitives such
as:

.. code:: cpp

    // NCCL device API (C++/CUDA), sketch only
    ncclGin gin(devComm, /*context=*/0);
    gin.put(team, peer, remoteWindow, dstOffset,
            localWindow, srcOffset, bytes,
            ncclGin_SignalInc{signalIndex});
    gin.waitSignal(signalIndex, expectedValue);
    gin.flush();

PyTorch does not yet expose these device-side primitives in Python.
Today they are the domain of C++/CUDA extension authors: you can obtain
the NCCL communicator backing a process group, create a device
communicator with GIN resources, and launch your own kernels against
the symmetric memory windows that PyTorch registered. See the
`NCCL device API documentation
<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html>`__
for the complete device-side interface, and the
`kraken repository <https://github.com/meta-pytorch/kraken>`__ for
examples of device-initiated communication kernels written against
PyTorch symmetric memory.

Conclusion
----------

In this tutorial, we used the NCCL backend of PyTorch symmetric memory
to program GPU-initiated communication: we allocated symmetric tensors,
established NCCL windows with ``rendezvous``, ran a fused
device-initiated all-reduce, exchanged data with one-sided put and
signal primitives, and wrote a custom all-reduce kernel in Python with
the CuTe DSL. We also saw how the same code scales across nodes, where
NCCL's GPU-Initiated Networking (GIN) services window operations with
kernel-initiated RDMA.

For further reading:

* `PyTorch Symmetric Memory documentation <https://docs.pytorch.org/docs/main/symmetric_memory.html>`__
* `PyTorch SymmetricMemory deep dive on dev-discuss <https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798>`__
* `NCCL device API documentation <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html>`__
* `CuTe DSL documentation <https://docs.nvidia.com/cutlass/latest/cute_dsl_general/cute_dsl.html>`__ and `distributed examples <https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell/kernel/distributed>`__
* `NVIDIA blog: Fusing Communication and Compute with the NCCL 2.28 Device API <https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/>`__
* `GPU-Initiated Networking for NCCL (paper) <https://arxiv.org/abs/2511.15076>`__
