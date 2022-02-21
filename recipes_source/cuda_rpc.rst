Direct Device-to-Device Communication with TensorPipe CUDA RPC
==============================================================

.. note:: Direct device-to-device RPC (CUDA RPC) is introduced in PyTorch 1.8
    as a prototype feature. This API is subject to change.

In this recipe, you will learn:

- The high-level idea of CUDA RPC.
- How to use CUDA RPC.


Requirements
------------

- PyTorch 1.8+
- `Getting Started With Distributed RPC Framework <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`_


What is CUDA RPC?
------------------------------------

CUDA RPC supports directly sending Tensors from local CUDA memory to remote
CUDA memory. Prior to v1.8 release, PyTorch RPC only accepts CPU Tensors. As a
result, when an application needs to send a CUDA Tensor through RPC, it has
to first move the Tensor to CPU on the caller, send it via RPC, and then move
it to the destination device on the callee, which incurs both unnecessary
synchronizations and D2H and H2D copies. Since v1.8, RPC allows users to
configure a per-process global device map using the
`set_device_map <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map>`_
API, specifying how to map local devices to remote devices. More specifically,
if ``worker0``'s device map has an entry ``"worker1" : {"cuda:0" : "cuda:1"}``,
all RPC arguments on ``"cuda:0"`` from ``worker0`` will be directly sent to
``"cuda:1"`` on ``worker1``. The response of an RPC will use the inverse of
the caller device map, i.e., if ``worker1`` returns a Tensor on ``"cuda:1"``,
it will be directly sent to ``"cuda:0"`` on ``worker0``. All intended
device-to-device direct communication must be specified in the per-process
device map. Otherwise, only CPU tensors are allowed.

Under the hood, PyTorch RPC relies on `TensorPipe <https://github.com/pytorch/tensorpipe>`_
as the communication backend. PyTorch RPC extracts all Tensors from each
request or response into a list and packs everything else into a binary
payload. Then, TensorPipe will automatically choose a communication channel
for each Tensor based on Tensor device type and channel availability on both
the caller and the callee. Existing TensorPipe channels cover NVLink, InfiniBand,
SHM, CMA, TCP, etc.

How to use CUDA RPC?
---------------------------------------

The code below shows how to use CUDA RPC. The model contains two linear layers
and is split into two shards. The two shards are placed on ``worker0`` and
``worker1`` respectively, and ``worker0`` serves as the master that drives the
forward and backward passes. Note that we intentionally skipped
`DistributedOptimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`_
to highlight the performance improvements when using CUDA RPC. The experiment
repeats the forward and backward passes 10 times and measures the total
execution time. It compares using CUDA RPC against manually staging to CPU
memory and using CPU RPC.


::

    import torch
    import torch.distributed.autograd as autograd
    import torch.distributed.rpc as rpc
    import torch.multiprocessing as mp
    import torch.nn as nn

    import os
    import time


    class MyModule(nn.Module):
        def __init__(self, device, comm_mode):
            super().__init__()
            self.device = device
            self.linear = nn.Linear(1000, 1000).to(device)
            self.comm_mode = comm_mode

        def forward(self, x):
            # x.to() is a no-op if x is already on self.device
            y = self.linear(x.to(self.device))
            return y.cpu() if self.comm_mode == "cpu" else y

        def parameter_rrefs(self):
            return [rpc.RRef(p) for p in self.parameters()]


    def measure(comm_mode):
        # local module on "worker0/cuda:0"
        lm = MyModule("cuda:0", comm_mode)
        # remote module on "worker1/cuda:1"
        rm = rpc.remote("worker1", MyModule, args=("cuda:1", comm_mode))
        # prepare random inputs
        x = torch.randn(1000, 1000).cuda(0)

        tik = time.time()
        for _ in range(10):
            with autograd.context() as ctx:
                y = rm.rpc_sync().forward(lm(x))
                autograd.backward(ctx, [y.sum()])
        # synchronize on "cuda:0" to make sure that all pending CUDA ops are
        # included in the measurements
        torch.cuda.current_stream("cuda:0").synchronize()
        tok = time.time()
        print(f"{comm_mode} RPC total execution time: {tok - tik}")


    def run_worker(rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

        if rank == 0:
            options.set_device_map("worker1", {0: 1})
            rpc.init_rpc(
                f"worker{rank}",
                rank=rank,
                world_size=2,
                rpc_backend_options=options
            )
            measure(comm_mode="cpu")
            measure(comm_mode="cuda")
        else:
            rpc.init_rpc(
                f"worker{rank}",
                rank=rank,
                world_size=2,
                rpc_backend_options=options
            )

        # block until all rpcs finish
        rpc.shutdown()


    if __name__=="__main__":
        world_size = 2
        mp.spawn(run_worker, nprocs=world_size, join=True)

Outputs are displayed below, which shows that CUDA RPC can help to achieve
34X speed up compared to CPU RPC in this experiment.

::

    cpu RPC total execution time: 2.3145179748535156 Seconds
    cuda RPC total execution time: 0.06867480278015137 Seconds
