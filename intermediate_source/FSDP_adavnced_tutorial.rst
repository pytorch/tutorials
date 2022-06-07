Getting Started with Fully Sharded Data Parallel(FSDP)
=====================================================

**Author**: `Hamid Shojanazeri <https://github.com/HamidShojanazeri>`__, `Yanli Zhao <https://github.com/zhaojuanmao>`__, `Shen Li <https://mrshenli.github.io/>`__

.. note::
   View the source code for this tutorial in `github <https://github.com/pytorch/tutorials/blob/master/intermediate_source/FSDP_tutorial.rst>`__.

Training AI models at a large scale is a challenging task that requires a lot of compute power and resources. 
It also comes with considerable engineering complexity to handle the training of these very large models.
`Pytorch FSDP <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`__, released in PyTorch 1.11 makes this easier.

In this tutorial, we show how to use `FSDP APIs <https://pytorch.org/docs/1.11/fsdp.html>`__, for fine-tuning HuggingFace T5 model `HuggingFace BERT models <https://huggingface.co/blog/zero-deepspeed-fairscale>`__, 
`GPT 3 models up to 1T parameters <https://pytorch.medium.com/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff>`__ . This is follow up on the  `FSDP getting started tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__. 


How FSDP works
--------------
In `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__, (DDP) training, each process/ worker owns a replica of the model and processes a batch of data, finally it uses all-reduce to sum up gradients over different workers. In DDP the model weights and optimizer states are replicated across all workers. FSDP is a type of data parallelism that shards model parameters, optimizer states and gradients across DDP ranks. 

FSDP GPU memory footprint would be smaller than DDP across all workers. This makes the training of some very large models feasible and helps to fit larger models or batch sizes for our training job. This would come with the cost of increased communication volume. The communication overhead is reduced by internal optimizations like communication and computation overlapping.

.. figure:: /_static/img/distributed/fsdp_workflow.png
   :width: 100%
   :align: center
   :alt: FSDP workflow

   FSDP Workflow

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

How to use FSDP
--------------
Here we use a toy model to run training on MNIST dataset for demonstration purposes. Similarly the APIs and logic can be applied to larger models for training. 

*Setup*

1.1 Install Pytorch along with Torchvision

.. code-block:: bash 

    pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html

We add the following code snippets to a python script “FSDP_mnist.py”.

1.2  Import necessary packages

.. code-block:: python

    import os
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSequenceClassification, AutoModelForCausalLM
    from transformers import AutoTokenizer, GPT2TokenizerFast
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
        default_auto_wrap_policy,
        enable_wrap,
        wrap,
    )
    from torch.utils.data import DataLoader
    from pathlib import Path
    from nlp import load_metric
    from nlp import load_dataset
    from summerization_dataset import *
    from sklearn.model_selection import train_test_split
    from transformers.models.t5.modeling_t5 import T5Block
    from typing import Type

1.3 Distributed training setup. As we mentioned FSDP is a type of data parallelism which requires a distributed training environment, so here we use two helper functions to initialize the processes for distributed training and clean up.

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

        dist.reduce(fsdp_loss, 0, op=dist.ReduceOp.SUM)
        if rank == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, fsdp_loss[0] / fsdp_loss[1]))

2.3 Define a validation function 

.. code-block:: python

    def test(model, rank, world_size, test_loader):
        model.eval()
        correct = 0
        local_rank = int(os.environ['LOCAL_RANK'])
        fsdp_loss = torch.zeros(3).to(local_rank)
        with torch.no_grad():
            for batch in test_loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)
                output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
                fsdp_loss[0] += output["loss"].item()  # sum up batch loss
                pred = output["logits"].argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                fsdp_loss[1] += pred.eq(batch["target_ids"].view_as(pred)).sum().item()
                fsdp_loss[2] += len(batch)

        dist.reduce(fsdp_loss, 0, op=dist.ReduceOp.SUM)
        if rank == 0:
        test_loss = fsdp_loss[0] / fsdp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(fsdp_loss[1]), int(fsdp_loss[2]),
            100. * fsdp_loss[1] / fsdp_loss[2]))


2.4 Define a distributed train function that wraps the model in FSDP

**Note: to save the FSDP model, we need to call the state_dict on each rank then on Rank 0 save the overall states. This is only available in Pytorch nightlies, current Pytorch release is 1.11 at the moment.**

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
        test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
        my_auto_wrap_policy = functools.partial(
                auto_wrap_policy_transformer, min_num_params=20000, transformer_layer_cls=T5Block
            )
        torch.cuda.set_device(local_rank)
    
    
        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)

        init_start_event.record()

    
        model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy).to(local_rank)

        print(model)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
            test(model, rank, world_size, test_loader)
            scheduler.step()

        init_end_event.record()

        if rank == 0:
            print(f"Cuda event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
            print(f"{model}")

        if args.save_model:
            states = model.state_dict()
            dist.barrier()
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
        parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
        args = parser.parse_args()

        torch.manual_seed(args.seed)
        
        fsdp_main(args)


To run the the training with torchrun:



*Applying fsdp_auto_wrap_policy* in FSDP otherwise, FSDP will put the entire model in one FSDP unit, which will reduce computation efficiency and memory efficiency. 
The way it works is that, suppose your model contains 100 Linear layers. If you do FSDP(model), there will only be one FSDP unit which wraps the entire model. 
In that case, the allgather would collect the full parameters for all 100 linear layers, and hence won't save CUDA memory for parameter sharding.
Also, there is only one blocking allgather call for the all 100 linear layers, there will not be communication and computation overlapping between layers. 

To avoid that, you can pass in an fsdp_auto_wrap_policy, which will seal the current FSDP unit and start a new one automatically when the specified condition is met (e.g., size limit).
In that way you will have multiple FSDP units, and only one FSDP unit needs to collect full parameters at a time. E.g., suppose you have 5 FSDP units, and each wraps 20 linear layers.
Then, in the forward, the 1st FSDP unit will allgather parameters for the first 20 linear layers, do computation, discard the parameters and then move on to the next 20 linear layers. So, at any point in time, each rank only materializes parameters/grads for 20 linear layers instead of 100.


To do so in 2.4 we define the auto_wrap_policy and pass it to FSDP wrapper, in the following example, my_auto_wrap_policy defines that a layer could be wrapped or sharded by FSDP if the number of parameters in this layer is larger than 100.
If the number of parameters in this layer is smaller than 100, it will be wrapped with other small layers together by FSDP. 
Finding an optimal auto wrap policy is challenging, PyTorch will add auto tuning for this config in the future. Without an auto tuning tool, it is good to profile your workflow using different auto wrap policies experimentally and find the optimal one.

.. code-block:: python
    def auto_wrap_policy_transformer(module: nn.Module, recurse: bool, unwrapped_params: int, transformer_layer_cls: Type[nn.Module], min_num_params: int = int(1e8),) -> bool:
        is_large = unwrapped_params >= min_num_params
        if recurse:
    # always recurse
            return True
        else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return is_large and isinstance(module, transformer_layer_cls)

    my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=20000
        )
    torch.cuda.set_device(rank)
    model = Net().to(rank)

    model = FSDP(model,
        fsdp_auto_wrap_policy=my_auto_wrap_policy)

Applying the FSDP_auto_wrap_policy, the model would be as follows:
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


.. code-block:: bash

    python FSDP_mnist.py

    CUDA event elapsed time on training loop 41.89130859375sec

Following is the peak memory usage from FSDP with auto_wrap policy of MNIST training on g4dn.12.xlarge AWS EC2 instance with 4 gpus captured from Pytorch Profiler. 
It can be observed that the peak memory usage on each device is smaller compared to FSDP without auto wrap policy applied, from ~75 MB to 66 MB.

.. figure:: /_static/img/distributed/FSDP_autowrap.gif
   :width: 100%
   :align: center
   :alt: FSDP peak memory

   FSDP Peak Memory Usage using Auto_wrap policy

*CPU Off-loading*: In case the model is very large that even with FSDP wouldn't fit into gpus, then CPU offload can be helpful here. 

Currently, only parameter and gradient CPU offload is supported. It can be enabled via passing in cpu_offload=CPUOffload(offload_params=True).

Note that this currently implicitly enables gradient offloading to CPU in order for params and grads to be on the same device to work with the optimizer. This API is subject to change. Default is None in which case there will be no offloading.

Using this feature may slow down the training considerably, due to frequent copying of tensors from host to device, but it could help improve memory efficiency and train larger scale models. 

In 2.4 we just add it to the FSDP wrapper


.. code-block:: python

    model = FSDP(model,
        fsdp_auto_wrap_policy=my_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True))


Compare it with DDP, if in 2.4 we just normally wrap the model in ddp, saving the changes in “DDP_mnist.py”.

.. code-block:: python

    model = Net().to(rank)
    model = DDP(model)


.. code-block:: bash

    python DDP_mnist.py

    CUDA event elapsed time on training loop 39.77766015625sec

Following is the peak memory usage from DDP MNIST training on g4dn.12.xlarge AWS EC2 instance with 4 gpus captured from Pytorch profiler. 

.. figure:: /_static/img/distributed/DDP_memory.gif
   :width: 100%
   :align: center
   :alt: FSDP peak memory

   DDP Peak Memory Usage using Auto_wrap policy


Considering the toy example and tiny MNIST model we defined here, we can observe the difference between peak memory usage of DDP and FSDP. 
In DDP each process holds a replica of the model, so the memory footprint is higher compared to FSDP that shards the model parameter, optimizer states and gradients over DDP ranks.
The peak memory usage using FSDP with auto_wrap policy is the lowest followed by FSDP and DDP. 

Also, looking at timings, considering the small model and running the training on a single machine, FSDP with/out auto_wrap policy performed almost as fast as DDP.
This example does not represent most of the real applications, for detailed analysis and comparison between DDP and FSDP please refer to this `blog post  <https://pytorch.medium.com/6c8da2be180d>`__ .
