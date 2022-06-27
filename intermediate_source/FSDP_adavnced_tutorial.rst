Advanced Fully Sharded Data Parallel(FSDP) Tutorial
=====================================================

**Author**: `Hamid Shojanazeri <https://github.com/HamidShojanazeri>`__, `Less Wright <https://github.com/lessw2020>`__,  `Yanli Zhao <https://github.com/zhaojuanmao>`__


This tutorial introduces more advanced features of Fully Sharded Data Parallel (FSDP) as part of the Pytorch 1.12 release. To get familiar with FSDP, please refer to the `FSDP getting started tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__.

In this tutorial, we fine-tune of a HuggingFace (HF) T5 model with FSDP for text summarization as the running example. 

The example uses Wikihow and for simplicty, we will showcase the training on a single node, P4dn instance with 8, A100 GPUs. We will soon have a blog post on large scale FSDP training on cluster, please stay tuned for that Pytorch medium channel.

FSDP is a production ready pakcage with focus on  ease of use, performance and long term support. One of the main values of FSDP is reducing the memory footprint on each GPU. This enable training larger models with less compute. This would also help to fit larger batch sizes during the training and ideally positvely impact the training speed and cost. Please read more Pytorch FSDP `here <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`__.


FSDP Features in This Tutorial
--------------
* :Transfromer Auto Wrap Policy:`transformer_wrapping_policy`
* Mixed Percision
* Intializing FSDP Model on Device
* Activation Checkpointing
* Sharding Starategy
* Backward Preftech
* Checkpoint Saving Streamed on CPU



Recap on How FSDP Works
--------------

At high level FDSP works as follow:

*In constructor*

* Shard model parameters and each rank only keeps its own shard

*In forward path*

* Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
* Run forward computation
* Discard parameter shards it has just collected

*In backward path*

* Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
* Run backward computation
* Run reduce_scatter to sync gradients
* Discard parameters. 

Fine-tuning HF T5
--------------
HF T5 pretrained models are availbe in 4 different sizes, ranging from small with 60 M parameters to 11 B parameters. In this tutorial, we demonstrate the finetuing of a T5 3B with FSDP for text summarization using WikiHow dataset.
The main focus of this tutorial is to highligh different available features in FSDP that would be helpful for training large scale model above 3B parameters. Also, we cover specific features for Transformer based models. The code for this tutorial is available in ,  `Pytorch Examples <https://github.com/HamidShojanazeri/examples/blob/FSDP_example>`__.


*Setup*

1.1 Install Pytorch 1.12 

.. code-block:: bash 

    pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html

1.2 Dataset Setup

Please create a "data" folder, download the WikiHow dataset from `wikihowAll.csv <https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358>`__  and `wikihowSep.cs <https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag>`__ and place them in the "data" folder. 
We will use the wikihow dataset from  `summarization_dataset <https://github.com/HamidShojanazeri/examples/blob/FSDP_example/FSDP/summarization_dataset.py>`__.

Next, we add the following code snippets to a python script “T5_training.py”, the source code for this tutorial is availbe in `Pytorch examples <https://github.com/HamidShojanazeri/examples/tree/FSDP_example/FSDP>`__ 

1.3  Import necessary packages

.. code-block:: python

    import os
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import functools
    from torch.optim.lr_scheduler import StepLR
    import torch.nn.functional as F
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel as FSDP,
        CPUOffload,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
    )
    from torch.utils.data import DataLoader
    from pathlib import Path
    from summerization_dataset import *
    from transformers.models.t5.modeling_t5 import T5Block
    from typing import Type

1.4 Distributed training setup. As we mentioned FSDP is a type of data parallelism which requires a distributed training environment, so here we use two helper functions to initialize the processes for distributed training and clean up.
In this tutrial, we are going to use torch elastic, using `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__ , it will set the worker RANK and WORLD_SIZE automatically for us.

.. code-block:: python

    def setup():
        # initialize the process group
        dist.init_process_group("nccl")

    def cleanup():
        dist.destroy_process_group()

2.1  Setup the HuggingFace T5 model. 

.. code-block:: python

    def setup_model(model_name):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer =  T5Tokenizer.from_pretrained(model_name)
        return model, tokenizer

    

2.2 define a train function 

.. code-block:: python

    def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
        model.train()
        local_rank = int(os.environ['LOCAL_RANK'])
        fsdp_loss = torch.zeros(2).to(local_rank)
    
        if sampler:
            sampler.set_epoch(epoch)
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

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        train_accuracy = fsdp_loss[0] / fsdp_loss[1]
        if rank == 0:
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
        return train_accuracy

2.3 Define a validation function 

.. code-block:: python

    def validation(model, rank, world_size, val_loader):
        model.eval()
        correct = 0
        local_rank = int(os.environ['LOCAL_RANK'])
        fsdp_loss = torch.zeros(3).to(local_rank)
        with torch.no_grad():
            for batch in val_loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)
                output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
                fsdp_loss[0] += output["loss"].item()  # sum up batch loss
                pred = output["logits"].argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                fsdp_loss[1] += pred.eq(batch["target_ids"].view_as(pred)).sum().item()
                fsdp_loss[2] += len(batch)

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)

        if rank == 0:
            val_loss = fsdp_loss[0] / fsdp_loss[2]
            print(f"Validation Loss: {val_loss:.4f}")
        return val_loss


2.4 Define a distributed train function that wraps the model in FSDP


.. code-block:: python

    
    def fsdp_main(args):

        model, tokenizer = setup_model("t5-large")

        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])


        dataset = load_dataset('wikihow', 'all', data_dir='data/')
        print(dataset.keys())
        print("Size of train dataset: ", dataset['train'].shape)
        print("Size of Validation dataset: ", dataset['validation'].shape)

        # tokenizer = T5Tokenizer.from_pretrained('t5-small')
        train_dataset = wikihow(tokenizer, 'train', None, 512, 150, True)
        val_dataset = wikihow(tokenizer, 'validation', None, 512, 150, True)
    
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

        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
        torch.cuda.set_device(local_rank)
    
    
        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)

        init_start_event.record()

    
        model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device())

        print(model)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        best_val_loss = float("inf")
        curr_val_loss = float("inf")
        file_save_name = "3B-model-"

        if rank == 0:
            time_of_run = get_date_of_run()
            dur = []
            train_acc_tracking = []
            val_acc_tracking = []
            training_start_time = time.time()

        if rank == 0 and args.track_memory:
            fn = "memory_tracking.txt"
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
            if rank == 0 and curr_val_loss < best_val_loss:

                best_val_loss = curr_val_loss
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

        init_end_event.record()

        if rank == 0:
            print(f"Cuda event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
            print(f"{model}")

        if args.save_model and curr_val_loss < best_val_loss:

            # save
            if rank == 0:
                print(f"--> entering save model state...")
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            print(f"saving process: rank {rank}  done w state_dict")

            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                save_name = file_save_name + "-" + time_of_run + "-" + currEpoch

                torch.save(cpu_state, save_name)
        if rank == 0:
            torch.save(states, "T5_checkpoint.pt")
        
        cleanup()



2.5 Finally parsing the arguments and setting the main function

.. code-block:: python

    
    if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_true', default=False,
                        help='track the gpy memory')
    parser.add_argument('--run_validation', action='store_true', default=False,
                        help='running the validation')
    parser.add_argument('--activation_checkpointing', action='store_true', default=False,
                        help='Checkpoint activations')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    fsdp_main(args)


To run the the training with torchrun:

.. code-block:: bash 

    torchrun --nnodes 1 --nproc_per_node 4  T5_training.py

.. _transformer_wrapping_policy:
Transformer Wrapping Policy
--------------
As discussed in the `previous tuotiral <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__, fsdp_auto_wrap_policy is one of the FSDP features that make it easier to put different model, optimizer and gradinet shards on different FSDP units.
However, for some of the architecutres such as Transformer encoder-decoders, some part of the model such as embedding table is being shared with both encoder and decoder.
In this case, we need to place the embedding table in the outer FSDP unit that could be accessed from both encoder and decoder. In Pytorch 1.12, FSDP added this support and now we have a wrapping policy for transfomers.

It can be deinfed as follows.


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

Applying the t5_auto_wrap_policy, the model would be as follows:
#TODO update with new wrapped units

.. code-block:: bash

    FullyShardedDataParallel(
  (_fsdp_wrapped_module): FlattenParamsWrapper(
    (_fpw_module): Net(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (dropout1): Dropout(p=0.25, inplace=False)
      (dropout2): Dropout(p=0.5, inplace=False)
      (fc1): FullyShardedDataParallel(
        (_fsdp_wrapped_module): FlattenParamsWrapper(
          (_fpw_module): Linear(in_features=9216, out_features=128, bias=True)
        )
      )
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )
  )





Mixed Percision
--------------
FSDP supports training with mixed percision with FP32, FP16 and BFloat16. Currently BFloat16 is only available on Ampre GPUs, so you need to make sure about its availbilty before you use it, otherwise it can result in slow downs.

To check if BFloat16 is ready you can use the following :

.. code-block:: python
    
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported() 
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

One of the advantages of mixed percision in FSDP is providing granular control over different communications for parameters, gradients and buffers as follows:

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


In 2.4 we just add it to the FSDP wrapper


.. code-block:: python

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen)

In our experiments, we have observed up to 4x speed up using BFloat16 for training.


Intializing FSDP Model on Device
--------------
There are multiple ways to initialize your model in FSDP:

Intialize the model on CPU then move it to device, this method would be slower compared to intializing the model directly on the device. 

In 2.4 we just add it to the FSDP wrapper

.. code-block:: python

    torch.cuda.set_device(local_rank)
    
     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen)
     model.to(local_rank)

This feature is available in PyTorch 1.12, that you could directly intialize model (FSDP units) on each device. This will speed up the model intialization.

.. code-block:: python

    torch.cuda.set_device(local_rank)

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device())
     
     
Activation Checkpointing
--------------
Activation checkpointing, is a technique to reduce the memory usage during training by clearing activations of certain layers and recomputing them during a backward pass. Using activation checkpointing, we could save up to .. memory in the running example and increase the batch size to .., this could increase the throughput and result in x speedups. Note: this feature is only available in PyTorch nightlies at this point.

We will need to import respective packages.

.. code-block:: python
   
   from transformers.models.t5.modeling_t5 import T5Block
   
   from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing_wrapper)
    
    
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    check_fn = lambda submodule: isinstance(submodule, T5Block)
    
    model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device())
            
    if args.activation_checkpointing:        
        apply_activation_checkpointing_wrapper(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )
    
#TODO make sure it works
    
Sharding Starategy
--------------
FSDP sharding strategy by default is set to Zero3, where model parameters, gradinets and optimizer states get sharded over DDP ranks. In case you are interested to have Zero2 sharding strategy, where only model parameters and gradinets are sharded, FSDP support this feature by passing the Sharding strategy by setting it to  "ShardingStrategy.SHARD_GRAD_OP" instead of "ShardingStrategy.FULL_SHARD" to the wrapper as follows:

.. code-block:: python

    torch.cuda.set_device(local_rank)

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP # FULL_SHARD)

This will reduce the communication in FSDP with the trade off a higher memory footprint. 

Backward Preftech
--------------
The other feature added to the FSDP in PyTorch 1.12 release. This can speedup the training in trade of with higher memory consumption. It can be in the wrapper as follows:

.. code-block:: python

    torch.cuda.set_device(local_rank)

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device(),
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE)
            
It has two settings, BACKWARD_PRE and BACKWARD_POST, (Add what each one does). Using BACKWARD_PRE, in the running HF T5 example, we could observer 2-10% speedup in training. 

Checkpoint Saving Streamed on CPU
--------------
To save the model checkpoints at the end of the training, if your model is larger than to fit into one gpu (e.g 3B and above), 
setting the FullStateDictConfig to to stream the model states to cpu,and using FSDP.state_dict_type context manager as shown below would help to avoid OOM errors. This, will stream model state dicts to CPU on each rank where on rank0 all the states dicts will be aggregated to build the full model state dict.

.. code-block:: python

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
    if rank == 0:
     save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
     torch.save(cpu_state, save_name)
