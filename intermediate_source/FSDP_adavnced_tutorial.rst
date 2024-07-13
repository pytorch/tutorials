Advanced Model Training with Fully Sharded Data Parallel (FSDP)
===============================================================

**Author**: `Hamid Shojanazeri <https://github.com/HamidShojanazeri>`__, `Less
Wright <https://github.com/lessw2020>`__, `Rohan Varma
<https://github.com/rohan-varma/>`__, `Yanli Zhao
<https://github.com/zhaojuanmao>`__


This tutorial introduces more advanced features of Fully Sharded Data Parallel
(FSDP) as part of the PyTorch 1.12 release. To get familiar with FSDP, please
refer to the `FSDP getting started tutorial
<https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__.

In this tutorial, we fine-tune a HuggingFace (HF) T5 model with FSDP for text
summarization as a working example. 

The example uses Wikihow and for simplicity, we will showcase the training on a
single node, P4dn instance with 8 A100 GPUs. We will soon have a blog post on
large scale FSDP training on a multi-node cluster, please stay tuned for that on
the PyTorch medium channel.

FSDP is a production ready package with focus on ease of use, performance, and
long-term support.  One of the main benefits of FSDP is reducing the memory
footprint on each GPU. This enables training of larger models with lower total
memory vs DDP, and leverages the overlap of computation and communication to
train models efficiently. 
This reduced memory pressure can be leveraged to either train larger models or
increase batch size, potentially helping overall training throughput.  You can
read more about PyTorch FSDP `here
<https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`__.


FSDP Features in This Tutorial
------------------------------
* Transformer Auto Wrap Policy
* Mixed Precision
* Initializing FSDP Model on Device
* Sharding Strategy
* Backward Prefetch
* Model Checkpoint Saving via Streaming to CPU



Recap on How FSDP Works
-----------------------

At a high level FDSP works as follow:

*In constructor*

* Shard model parameters and each rank only keeps its own shard

*In forward pass*

* Run `all_gather` to collect all shards from all ranks to recover the full
  parameter for this FSDP unit Run forward computation
* Discard non-owned parameter shards it has just collected to free memory

*In backward pass*

* Run `all_gather` to collect all shards from all ranks to recover the full
  parameter in this FSDP unit Run backward computation
* Discard non-owned parameters to free memory. 
* Run reduce_scatter to sync gradients


Fine-tuning HF T5
-----------------
HF T5 pre-trained models are available in four different sizes, ranging from
small with 60 Million parameters to XXL with 11 Billion parameters. In this
tutorial, we demonstrate the fine-tuning of a T5 3B with FSDP for text
summarization using WikiHow dataset.  The main focus of this tutorial is to
highlight different available features in FSDP that are helpful for training
large scale model above 3B parameters. Also, we cover specific features for
Transformer based models. The code for this tutorial is available in  `Pytorch
examples
<https://github.com/pytorch/examples/tree/main/distributed/FSDP/>`__.


*Setup*

1.1 Install PyTorch Nightlies

We will install PyTorch nightlies, as some of the features such as activation
checkpointing is available in nightlies and will be added in next PyTorch
release after 1.12.

.. code-block:: bash 

    pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html

1.2 Dataset Setup

Please create a `data` folder, download the WikiHow dataset from `wikihowAll.csv
<https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358>`__  and
`wikihowSep.cs <https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag>`__,
and place them in the `data` folder.  We will use the wikihow dataset from
`summarization_dataset
<https://github.com/pytorch/examples/blob/main/distributed/FSDP/summarization_dataset.py>`__.

Next, we add the following code snippets to a Python script “T5_training.py”.

.. note::
   The full source code for this tutorial is available in `PyTorch examples
   <https://github.com/pytorch/examples/tree/main/distributed/FSDP/>`__.

1.3  Import necessary packages:

.. code-block:: python

    import os
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from transformers import AutoTokenizer, GPT2TokenizerFast
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import functools
    from torch.optim.lr_scheduler import StepLR
    import torch.nn.functional as F
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from transformers.models.t5.modeling_t5 import T5Block

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
     checkpoint_wrapper,
     CheckpointImpl,
     apply_activation_checkpointing_wrapper)

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        enable_wrap,
        wrap,
    )
    from functools import partial
    from torch.utils.data import DataLoader
    from pathlib import Path
    from summarization_dataset import *
    from transformers.models.t5.modeling_t5 import T5Block
    from typing import Type
    import time
    import tqdm
    from datetime import datetime

1.4 Distributed training setup. 
Here we use two helper functions to initialize the processes for distributed
training,  and then to clean up after training completion.  In this tutorial, we
are going to use torch elastic, using `torchrun
<https://pytorch.org/docs/stable/elastic/run.html>`__ , which will set the
worker `RANK` and `WORLD_SIZE` automatically.

.. code-block:: python

    def setup():
        # initialize the process group
        dist.init_process_group("nccl")

    def cleanup():
        dist.destroy_process_group()

2.1  Set up the HuggingFace T5 model:

.. code-block:: python

    def setup_model(model_name):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer =  T5Tokenizer.from_pretrained(model_name)
        return model, tokenizer

We also, add couple of helper functions here for date and formatting memory
metrics.

.. code-block:: python

    def get_date_of_run():
        """create date and time for file save uniqueness
        example: 2022-05-07-08:31:12_PM'
        """
        date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
        print(f"--> current date and time of run = {date_of_run}")
        return date_of_run
   
    def format_metrics_to_gb(item):
        """quick function to format numbers to gigabyte and round to 4 digit precision"""
        metric_num = item / g_gigabyte
        metric_num = round(metric_num, ndigits=4)
        return metric_num
    

2.2 Define a train function:

.. code-block:: python

    def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
        model.train()
        local_rank = int(os.environ['LOCAL_RANK'])
        fsdp_loss = torch.zeros(2).to(local_rank)

        if sampler:
            sampler.set_epoch(epoch)
        if rank==0:
            inner_pbar = tqdm.tqdm(
                range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
            )
        for batch in train_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            optimizer.zero_grad()
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(batch)
            if rank==0:
                inner_pbar.update(1)

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        train_accuracy = fsdp_loss[0] / fsdp_loss[1]


        if rank == 0:
            inner_pbar.close()
            print(
                    f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
                )
        return train_accuracy

2.3 Define a validation function:

.. code-block:: python

    def validation(model, rank, world_size, val_loader):
        model.eval()
        correct = 0
        local_rank = int(os.environ['LOCAL_RANK'])
        fsdp_loss = torch.zeros(3).to(local_rank)
        if rank == 0:
            inner_pbar = tqdm.tqdm(
                range(len(val_loader)), colour="green", desc="Validation Epoch"
            )
        with torch.no_grad():
            for batch in val_loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)
                output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
                fsdp_loss[0] += output["loss"].item()  # sum up batch loss
                fsdp_loss[1] += len(batch)

                if rank==0:
                    inner_pbar.update(1)

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        val_loss = fsdp_loss[0] / fsdp_loss[1]
        if rank == 0:
            inner_pbar.close()
            print(f"Validation Loss: {val_loss:.4f}")
        return val_loss


2.4 Define a distributed train function that wraps the model in FSDP:


.. code-block:: python

    
    def fsdp_main(args):

        model, tokenizer = setup_model("t5-base")

        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])


        dataset = load_dataset('wikihow', 'all', data_dir='data/')
        print(dataset.keys())
        print("Size of train dataset: ", dataset['train'].shape)
        print("Size of Validation dataset: ", dataset['validation'].shape)


        #wikihow(tokenizer, type_path, num_samples, input_length, output_length, print_text=False)
        train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False) 
        val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False)

        sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
        sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

        setup()


        train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
        test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
        cuda_kwargs = {'num_workers': 2,
                        'pin_memory': True,
                        'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

        t5_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                T5Block,
            },
        )
        sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
        torch.cuda.set_device(local_rank)


        #init_start_event = torch.cuda.Event(enable_timing=True)
        #init_end_event = torch.cuda.Event(enable_timing=True)

        #init_start_event.record()

        bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
        )

        if bf16_ready:
            mp_policy = bfSixteen
        else:
            mp_policy = None # defaults to fp32

        # model is on CPU before input to FSDP
        model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=mp_policy,
            #sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device())

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        best_val_loss = float("inf")
        curr_val_loss = float("inf")
        file_save_name = "T5-model-"

        if rank == 0:
            time_of_run = get_date_of_run()
            dur = []
            train_acc_tracking = []
            val_acc_tracking = []
            training_start_time = time.time()

        if rank == 0 and args.track_memory:
            mem_alloc_tracker = []
            mem_reserved_tracker = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
            if args.run_validation:
                curr_val_loss = validation(model, rank, world_size, val_loader)
            scheduler.step()

            if rank == 0:

                print(f"--> epoch {epoch} completed...entering save and stats zone")

                dur.append(time.time() - t0)
                train_acc_tracking.append(train_accuracy.item())

                if args.run_validation:
                    val_acc_tracking.append(curr_val_loss.item())

                if args.track_memory:
                    mem_alloc_tracker.append(
                        format_metrics_to_gb(torch.cuda.memory_allocated())
                    )
                    mem_reserved_tracker.append(
                        format_metrics_to_gb(torch.cuda.memory_reserved())
                    )
                print(f"completed save and stats zone...")

            if args.save_model and curr_val_loss < best_val_loss:

                # save
                if rank == 0:
                    print(f"--> entering save model state")

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    cpu_state = model.state_dict()
                #print(f"saving process: rank {rank}  done w state_dict")


                if rank == 0:
                    print(f"--> saving model ...")
                    currEpoch = (
                        "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                    )
                    print(f"--> attempting to save model prefix {currEpoch}")
                    save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                    print(f"--> saving as model name {save_name}")

                    torch.save(cpu_state, save_name)

            if curr_val_loss < best_val_loss:

                best_val_loss = curr_val_loss
                if rank==0:
                    print(f"-->>>> New Val Loss Record: {best_val_loss}")

        dist.barrier()
        cleanup()


2.5 Parse the arguments and set the main function:

.. code-block:: python

    
    if __name__ == '__main__':
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
        parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=2, metavar='N',
                            help='number of epochs to train (default: 3)')
        parser.add_argument('--lr', type=float, default=.002, metavar='LR',
                            help='learning rate (default: .002)')
        parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--track_memory', action='store_false', default=True,
                            help='track the gpu memory')
        parser.add_argument('--run_validation', action='store_false', default=True,
                            help='running the validation')
        parser.add_argument('--save-model', action='store_false', default=True,
                            help='For Saving the current Model')
        args = parser.parse_args()

        torch.manual_seed(args.seed)

        fsdp_main(args)


To run the the training using torchrun:

.. code-block:: bash 

    torchrun --nnodes 1 --nproc_per_node 4  T5_training.py

.. _transformer_wrapping_policy:

Transformer Wrapping Policy
---------------------------

As discussed in the `previous tutorial
<https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__,
auto_wrap_policy is one of the FSDP features that make it easy to automatically
shard a given model and put the model, optimizer and gradient shards into
distinct FSDP units.

For some architectures such as Transformer encoder-decoders, some parts of the
model such as embedding table is being shared with both encoder and decoder.  In
this case, we need to place the embedding table in the outer FSDP unit so that
it could be accessed from both encoder and decoder.  In addition, by registering
the layer class for a transformer, the sharding plan can be made much more
communication efficient.  In PyTorch 1.12, FSDP added this support and now we
have a wrapping policy for transfomers.

It can be created as follows, where the T5Block represents the T5 transformer
layer class (holding MHSA and FFN).  


.. code-block:: python

    t5_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                T5Block,
            },
        )
    torch.cuda.set_device(local_rank)
  

    model = FSDP(model,
        fsdp_auto_wrap_policy=t5_auto_wrap_policy)

To see the wrapped model, you can easily print the model and visually inspect
the sharding and FSDP units as well.


Mixed Precision
---------------
FSDP supports flexible mixed precision training allowing for arbitrary reduced
precision types (such as fp16 or bfloat16). Currently BFloat16 is only available
on Ampere GPUs, so you need to confirm native support before you use it. On
V100s for example, BFloat16 can still be run but due to it running non-natively,
it can result in significant slowdowns.

To check if BFloat16 is natively supported, you can use the following :

.. code-block:: python
    
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported() 
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

One of the advantages of mixed percision in FSDP is providing granular control
over different precision levels for parameters, gradients, and buffers as
follows:

.. code-block:: python

    fpSixteen = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )

    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    fp32_policy = MixedPrecision(
        param_dtype=torch.float32,
        # Gradient communication precision.
        reduce_dtype=torch.float32,
        # Buffer precision.
        buffer_dtype=torch.float32,
    )

Note that if a certain type (parameter, reduce, buffer) is not specified, they
will not be casted at all.

This flexibility allows users fine grained control, such as only setting
gradient communication to happen in reduced precision, and all parameters /
buffer computation to be done in full precision. This is potentially useful in
cases where intra-node communication is the main bottleneck and parameters /
buffers must be in full precision to avoid accuracy issues. This can be done
with the following policy:

.. code-block:: bash

    grad_bf16 = MixedPrecision(reduce_dtype=torch.bfloat16)
    

In 2.4 we just add the relevant mixed precision policy to the FSDP wrapper:


.. code-block:: python

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen)

In our experiments, we have observed up to 4x speed up by using BFloat16 for
training and memory reduction of approximately 30% in some experiments that can
be used for batch size increases.


Intializing FSDP Model on Device
--------------------------------
In 1.12, FSDP supports a `device_id` argument meant to initialize input CPU
module on the device given by `device_id`. This is useful when the entire model
does not fit on a single GPU, but fits in a host's CPU memory. When `device_id`
is specified, FSDP will move the model to the specified device on a per-FSDP
unit basis, avoiding GPU OOM issues while initializing several times faster than
CPU-based initialization:

.. code-block:: python

    torch.cuda.set_device(local_rank)

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device())
     

    
Sharding Strategy
-----------------
FSDP sharding strategy by default is set to fully shard the model parameters,
gradients and optimizer states get sharded across all ranks. (also termed Zero3
sharding). In case you are interested to have the Zero2 sharding strategy, where
only optimizer states and gradients are sharded, FSDP support this feature by
passing the Sharding strategy by using  "ShardingStrategy.SHARD_GRAD_OP",
instead of "ShardingStrategy.FULL_SHARD" to the FSDP initialization  as follows:

.. code-block:: python

    torch.cuda.set_device(local_rank)

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP # ZERO2)

This will reduce the communication overhead in FSDP, in this case, it holds full
parameters after forward and through the backwards pass. 

This saves an all_gather during backwards so there is less communication at the
cost of a higher memory footprint. Note that full model params are freed at the
end of backwards and all_gather will happen on the next forward pass.

Backward Prefetch
-----------------
The backward prefetch setting controls the timing of when the next FSDP unit's
parameters should be requested.  By setting it to `BACKWARD_PRE`, the next
FSDP's unit params can begin to be requested and arrive sooner before the
computation of the current unit starts. This overlaps the `all_gather`
communication and gradient computation which can increase the training speed in
exchange for slightly higher memory consumption. It can be utilized in the FSDP
wrapper in 2.4 as follows:

.. code-block:: python

    torch.cuda.set_device(local_rank)

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device(),
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE)
            
`backward_prefetch` has two modes, `BACKWARD_PRE` and `BACKWARD_POST`.
`BACKWARD_POST` means that the next FSDP unit's params will not be requested
until the current FSDP unit processing is complete, thus minimizing memory
overhead.  In some cases, using `BACKWARD_PRE` can increase model training speed
up to 2-10%, with even higher speed improvements noted for larger models. 

Model Checkpoint Saving, by streaming to the Rank0 CPU
------------------------------------------------------
To save model checkpoints using FULL_STATE_DICT saving which saves model in the
same fashion as a local model, PyTorch 1.12 offers a few utilities to support
the saving of larger models.

First, a FullStateDictConfig can be specified, allowing the state_dict to be
populated on rank 0 only and offloaded to the CPU.

When using this configuration, FSDP will allgather model parameters, offloading
them to the CPU one by one, only on rank 0. When the state_dict is finally
saved, it will only be populated on rank 0 and contain CPU tensors. This avoids
potential OOM for models that are larger than a single GPU memory and allows
users to checkpoint models whose size is roughly the available CPU RAM on the
user's machine.

This feature can be run as follows:

.. code-block:: python

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
    if rank == 0:
     save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
     torch.save(cpu_state, save_name)

Summary
-------

In this tutorial, we have introduced many new features for FSDP available in
Pytorch 1.12 and used HF T5 as the running example.  Using the proper wrapping
policy especially for transformer models, along with mixed precision and
backward prefetch should speed up your training runs. Also, features such as
initializing the model on device, and checkpoint saving via streaming to CPU
should help to avoid OOM error in dealing with large models. 

We are actively working to add new features to FSDP for the next release. If
you have feedback, feature requests, questions or are encountering issues
using FSDP, please feel free to contact us by opening an issue in the
`PyTorch Github repository <https://github.com/pytorch/pytorch>`__.
