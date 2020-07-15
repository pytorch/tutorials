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
    if rank == 0:  
      dst_worker_rank = (rank + 1) % world_size
      dst_worker_name = f"worker{dst_worker_rank}"
      t1, t2 = random_tensor(), random_tensor() 
      # Send and wait RPC completion under profiling scope.
      with profiler.profile() as p:
        fut1 = rpc.rpc_async(dst_worker_name, torch.add, args=(t1, t2))
        fut2 = rpc.rpc_async(dst_worker_name, torch.mul, args=(t1, t2))
        # RPCs must be awaited within profiling scope.
        fut1.wait()
        fut2.wait()

      print(p.key_averages().table())



if __name__ == '__main__':
    # Run 2 RPC workers.
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size)