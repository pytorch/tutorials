Distributed Pipeline Parallelism Using RPC
==========================================
**Author**: `Shen Li <https://mrshenli.github.io/>`_

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/dist_pipeline_parallel_tutorial.rst>`__.

Prerequisites:

-  `PyTorch Distributed Overview <../beginner/dist_overview.html>`__
-  `Single-Machine Model Parallel Best Practices <https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html>`__
-  `Getting started with Distributed RPC Framework <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`__
-  RRef helper functions:
   `RRef.rpc_sync() <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.rpc_sync>`__,
   `RRef.rpc_async() <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.rpc_async>`__, and
   `RRef.remote() <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.remote>`__



This tutorial uses a Resnet50 model to demonstrate implementing distributed
pipeline parallelism with `torch.distributed.rpc <https://pytorch.org/docs/master/rpc.html>`__
APIs. This can be viewed as the distributed counterpart of the multi-GPU
pipeline parallelism discussed in
`Single-Machine Model Parallel Best Practices <model_parallel_tutorial.html>`_.

.. note:: This tutorial requires PyTorch v1.6.0 or above.

.. note:: Full source code of this tutorial can be found at
    `pytorch/examples <https://github.com/pytorch/examples/tree/master/distributed/rpc/pipeline>`__.

Basics
------


The previous tutorial, `Getting Started with Distributed RPC Framework <rpc_tutorial.html>`_
shows how to use `torch.distributed.rpc <https://pytorch.org/docs/master/rpc.html>`_
to implement distributed model parallelism for an RNN model. That tutorial uses
one GPU to host the ``EmbeddingTable``, and the provided code works fine.
However, if a model lives on multiple GPUs, it would require some extra steps to
increase the amortized utilization of all GPUs. Pipeline parallelism is one type
of paradigm that can help in this case.

In this tutorial, we use ``ResNet50`` as an example model which is also used by
the `Single-Machine Model Parallel Best Practices <model_parallel_tutorial.html>`_
tutorial. Similarly, the ``ResNet50`` model is divided into two shards and
the input batch is partitioned into multiple splits and fed into the two model
shards in a pipelined fashion. The difference is that, instead of parallelizing
the execution using CUDA streams, this tutorial invokes asynchronous RPCs. So,
the solution presented in this tutorial also works across machine boundaries.
The remainder of this tutorial presents the implementation in four steps.



Step 1: Partition ResNet50 Model
--------------------------------

This is the preparation step which implements ``ResNet50`` in two model shards.
The code below is borrowed from the
`ResNet implementation in torchvision <https://github.com/pytorch/vision/blob/7c077f6a986f05383bcb86b535aedb5a63dd5c4b/torchvision/models/resnet.py#L124>`_.
The ``ResNetBase`` module contains the common building blocks and attributes for
the two ResNet shards.


.. code:: python

    import threading

    import torch
    import torch.nn as nn

    from torchvision.models.resnet import Bottleneck

    num_classes = 1000


    def conv1x1(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


    class ResNetBase(nn.Module):
        def __init__(self, block, inplanes, num_classes=1000,
                    groups=1, width_per_group=64, norm_layer=None):
            super(ResNetBase, self).__init__()

            self._lock = threading.Lock()
            self._block = block
            self._norm_layer = nn.BatchNorm2d
            self.inplanes = inplanes
            self.dilation = 1
            self.groups = groups
            self.base_width = width_per_group

        def _make_layer(self, planes, blocks, stride=1):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if stride != 1 or self.inplanes != planes * self._block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * self._block.expansion, stride),
                    norm_layer(planes * self._block.expansion),
                )

            layers = []
            layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * self._block.expansion
            for _ in range(1, blocks):
                layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer))

            return nn.Sequential(*layers)

        def parameter_rrefs(self):
            return [RRef(p) for p in self.parameters()]


Now, we are ready to define the two model shards. For the constructor, we
simply split all ResNet50 layers into two parts and move each part into the
provided device. The ``forward`` functions of both shards take an ``RRef`` of
the input data, fetch the data locally, and then move it to the expected device.
After applying all layers to the input, it moves the output to CPU and returns.
It is because the RPC API requires tensors to reside on CPU to avoid invalid
device errors when the numbers of devices in the caller and the callee do not
match.


.. code:: python

    class ResNetShard1(ResNetBase):
        def __init__(self, device, *args, **kwargs):
            super(ResNetShard1, self).__init__(
                Bottleneck, 64, num_classes=num_classes, *args, **kwargs)

            self.device = device
            self.seq = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                self._norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_layer(64, 3),
                self._make_layer(128, 4, stride=2)
            ).to(self.device)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x_rref):
            x = x_rref.to_here().to(self.device)
            with self._lock:
                out =  self.seq(x)
            return out.cpu()


    class ResNetShard2(ResNetBase):
        def __init__(self, device, *args, **kwargs):
            super(ResNetShard2, self).__init__(
                Bottleneck, 512, num_classes=num_classes, *args, **kwargs)

            self.device = device
            self.seq = nn.Sequential(
                self._make_layer(256, 6, stride=2),
                self._make_layer(512, 3, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
            ).to(self.device)

            self.fc =  nn.Linear(512 * self._block.expansion, num_classes).to(self.device)

        def forward(self, x_rref):
            x = x_rref.to_here().to(self.device)
            with self._lock:
                out = self.fc(torch.flatten(self.seq(x), 1))
            return out.cpu()


Step 2: Stitch ResNet50 Model Shards Into One Module
----------------------------------------------------


Then, we create a ``DistResNet50`` module to assemble the two shards and
implement the pipeline parallel logic. In the constructor, we use two
``rpc.remote`` calls to put the two shards on two different RPC workers
respectively and hold on to the ``RRef`` to the two model parts so that they
can be referenced in the forward pass.  The ``forward`` function
splits the input batch into multiple micro-batches, and feeds these
micro-batches to the two model parts in a pipelined fashion. It first uses an
``rpc.remote`` call to apply the first shard to a micro-batch and then forwards
the returned intermediate output ``RRef`` to the second model shard. After that,
it collects the ``Future`` of all micro-outputs, and waits for all of them after
the loop. Note that both ``remote()`` and ``rpc_async()`` return immediately and
run asynchronously. Therefore, the entire loop is non-blocking, and will launch
multiple RPCs concurrently. The execution order of one micro-batch on two model
parts are preserved by intermediate output ``y_rref``. The execution order
across micro-batches does not matter. In the end, the forward function
concatenates outputs of all micro-batches into one single output tensor and
returns. The ``parameter_rrefs`` function is a helper to
simplify distributed optimizer construction, which will be used later.



.. code:: python

    class DistResNet50(nn.Module):
        def __init__(self, num_split, workers, *args, **kwargs):
            super(DistResNet50, self).__init__()

            self.num_split = num_split

            # Put the first part of the ResNet50 on workers[0]
            self.p1_rref = rpc.remote(
                workers[0],
                ResNetShard1,
                args = ("cuda:0",) + args,
                kwargs = kwargs
            )

            # Put the second part of the ResNet50 on workers[1]
            self.p2_rref = rpc.remote(
                workers[1],
                ResNetShard2,
                args = ("cuda:1",) + args,
                kwargs = kwargs
            )

        def forward(self, xs):
            out_futures = []
            for x in iter(xs.split(self.num_split, dim=0)):
                x_rref = RRef(x)
                y_rref = self.p1_rref.remote().forward(x_rref)
                z_fut = self.p2_rref.rpc_async().forward(y_rref)
                out_futures.append(z_fut)

            return torch.cat(torch.futures.wait_all(out_futures))

        def parameter_rrefs(self):
            remote_params = []
            remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
            remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
            return remote_params


Step 3: Define The Training Loop
--------------------------------


After defining the model, let us implement the training loop. We use a
dedicated "master" worker to prepare random inputs and labels, and control the
distributed backward pass and distributed optimizer step. It first creates an
instance of the ``DistResNet50`` module. It specifies the number of
micro-batches for each batch, and also provides the name of the two RPC workers
(i.e., "worker1", and "worker2"). Then it defines the loss function and creates
a ``DistributedOptimizer`` using the ``parameter_rrefs()`` helper to acquire a
list of parameter ``RRefs``. Then, the main training loop is very similar to
regular local training, except that it uses ``dist_autograd`` to launch
backward and provides the ``context_id`` for both backward and optimizer
``step()``.


.. code:: python

    import torch.distributed.autograd as dist_autograd
    import torch.optim as optim
    from torch.distributed.optim import DistributedOptimizer

    num_batches = 3
    batch_size = 120
    image_w = 128
    image_h = 128


    def run_master(num_split):
        # put the two model parts on worker1 and worker2 respectively
        model = DistResNet50(num_split, ["worker1", "worker2"])
        loss_fn = nn.MSELoss()
        opt = DistributedOptimizer(
            optim.SGD,
            model.parameter_rrefs(),
            lr=0.05,
        )

        one_hot_indices = torch.LongTensor(batch_size) \
                            .random_(0, num_classes) \
                            .view(batch_size, 1)

        for i in range(num_batches):
            print(f"Processing batch {i}")
            # generate random inputs and labels
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                        .scatter_(1, one_hot_indices, 1)

            with dist_autograd.context() as context_id:
                outputs = model(inputs)
                dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
                opt.step(context_id)


Step 4: Launch RPC Processes
----------------------------


Finally, the code below shows the target function for all processes. The main
logic is defined in ``run_master``. The workers passively waiting for
commands from the master, and hence simply runs ``init_rpc`` and ``shutdown``,
where the ``shutdown`` by default will block until all RPC participants finish.

.. code:: python

    import os
    import time

    import torch.multiprocessing as mp


    def run_worker(rank, world_size, num_split):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

        if rank == 0:
            rpc.init_rpc(
                "master",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=options
            )
            run_master(num_split)
        else:
            rpc.init_rpc(
                f"worker{rank}",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=options
            )
            pass

        # block until all rpcs finish
        rpc.shutdown()


    if __name__=="__main__":
        world_size = 3
        for num_split in [1, 2, 4, 8]:
            tik = time.time()
            mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
            tok = time.time()
            print(f"number of splits = {num_split}, execution time = {tok - tik}")


