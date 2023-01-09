
Implementing a Parameter Server Using Distributed RPC Framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Author**\ : `Rohan Varma <https://github.com/rohan-varma>`_

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/rpc_param_server_tutorial.rst>`__.

Prerequisites:

-  `PyTorch Distributed Overview <../beginner/dist_overview.html>`__
-  `RPC API documents <https://pytorch.org/docs/master/rpc.html>`__

This tutorial walks through a simple example of implementing a parameter server using PyTorch's `Distributed RPC framework <https://pytorch.org/docs/stable/rpc.html>`_. The parameter server framework is a paradigm in which a set of servers store parameters, such as large embedding tables, and several trainers query the parameter servers in order to retrieve the most up to date parameters. These trainers can run a training loop locally and occasionally synchronize with the parameter server to get the latest parameters. For more reading on the parameter server approach, check out `this paper <https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf>`_.

Using the Distributed RPC Framework, we'll build an example where multiple trainers use RPC to communicate with the same parameter server and use `RRef <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef>`_ to access states on the remote parameter server instance. Each trainer will launch its dedicated backward pass in a distributed fashion through stitching of the autograd graph across multiple nodes using distributed autograd.

**Note**\ : This tutorial covers the use of the Distributed RPC Framework, which is useful for splitting a model onto multiple machines, or for implementing a parameter-server training strategy where network trainers fetch parameters hosted on a different machine. If instead you are looking for replicating your model across many GPUs, please see the `Distributed Data Parallel tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_. There is also another `RPC tutorial <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`_ that covers reinforcement learning and RNN use cases.

Let's start with the familiar: importing our required modules and defining a simple ConvNet that will train on the MNIST dataset. The below network is largely adopted from the network defined in the `pytorch/examples repo <https://github.com/pytorch/examples/tree/master/mnist>`_.

.. code-block:: python

   import argparse
   import os
   import time
   from threading import Lock

   import torch
   import torch.distributed.autograd as dist_autograd
   import torch.distributed.rpc as rpc
   import torch.multiprocessing as mp
   import torch.nn as nn
   import torch.nn.functional as F
   from torch import optim
   from torch.distributed.optim import DistributedOptimizer
   from torchvision import datasets, transforms

   # --------- MNIST Network to train, from pytorch/examples -----

   class Net(nn.Module):
       def __init__(self, num_gpus=0):
           super(Net, self).__init__()
           print(f"Using {num_gpus} GPUs to train")
           self.num_gpus = num_gpus
           device = torch.device(
               "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
           print(f"Putting first 2 convs on {str(device)}")
           # Put conv layers on the first cuda device, or CPU if no cuda device
           self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
           self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
           # Put rest of the network on the 2nd cuda device, if there is one
           if "cuda" in str(device) and num_gpus > 1:
               device = torch.device("cuda:1")

           print(f"Putting rest of layers on {str(device)}")
           self.dropout1 = nn.Dropout2d(0.25).to(device)
           self.dropout2 = nn.Dropout2d(0.5).to(device)
           self.fc1 = nn.Linear(9216, 128).to(device)
           self.fc2 = nn.Linear(128, 10).to(device)

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           x = self.conv2(x)
           x = F.max_pool2d(x, 2)

           x = self.dropout1(x)
           x = torch.flatten(x, 1)
           # Move tensor to next device if necessary
           next_device = next(self.fc1.parameters()).device
           x = x.to(next_device)

           x = self.fc1(x)
           x = F.relu(x)
           x = self.dropout2(x)
           x = self.fc2(x)
           output = F.log_softmax(x, dim=1)
           return output
Next, let's define some helper functions that will be useful for the rest of our script. The following uses `rpc_sync <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.rpc_sync>`_ and `RRef <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef>`_ in order to define a function that invokes a given method on an object living on a remote node. Below, our handle to the remote object is given by the ``rref`` argument, and we run it on its owning node: ``rref.owner()``. On the caller node, we run this command synchronously through the use of ``rpc_sync``\ , meaning that we will block until a response is received.

.. code-block:: python

   # --------- Helper Methods --------------------

   # On the local node, call a method with first arg as the value held by the
   # RRef. Other args are passed in as arguments to the function called.
   # Useful for calling instance methods. method could be any matching function, including
   # class methods.
   def call_method(method, rref, *args, **kwargs):
       return method(rref.local_value(), *args, **kwargs)

   # Given an RRef, return the result of calling the passed in method on the value
   # held by the RRef. This call is done on the remote node that owns
   # the RRef and passes along the given argument.
   # Example: If the value held by the RRef is of type Foo, then
   # remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
   # <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
   # back.

   def remote_method(method, rref, *args, **kwargs):
       args = [method, rref] + list(args)
       return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)
Now, we're ready to define our parameter server. We will subclass ``nn.Module`` and save a handle to our network defined above. We'll also save an input device which will be the device our input is transferred to before invoking the model.

.. code-block:: python

   # --------- Parameter Server --------------------
   class ParameterServer(nn.Module):
       def __init__(self, num_gpus=0):
           super().__init__()
           model = Net(num_gpus=num_gpus)
           self.model = model
           self.input_device = torch.device(
               "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
Next, we'll define our forward pass. Note that regardless of the device of the model output, we move the output to CPU, as the Distributed RPC Framework currently only supports sending CPU tensors over RPC. We have intentionally disabled sending CUDA tensors over RPC due to the potential for different devices (CPU/GPU) on on the caller/callee, but may support this in future releases.

.. code-block:: python

   class ParameterServer(nn.Module):
   ...
       def forward(self, inp):
           inp = inp.to(self.input_device)
           out = self.model(inp)
           # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
           # Tensors must be moved in and out of GPU memory due to this.
           out = out.to("cpu")
           return out
Next, we'll define a few miscellaneous functions useful for training and verification purposes. The first, ``get_dist_gradients``\ , will take in a Distributed Autograd context ID and call into the ``dist_autograd.get_gradients`` API in order to retrieve gradients computed by distributed autograd. More information can be found in the `distributed autograd documentation <https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework>`_. Note that we also iterate through the resulting dictionary and convert each tensor to a CPU tensor, as the framework currently only supports sending tensors over RPC. Next, ``get_param_rrefs`` will iterate through our model parameters and wrap them as a (local) `RRef <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef>`_. This method will be invoked over RPC by trainer nodes and will return a list of the parameters to be optimized. This is required as input to the `Distributed Optimizer <https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim>`_\ , which requires all parameters it must optimize as a list of ``RRef``\ s.

.. code-block:: python

   # Use dist autograd to retrieve gradients accumulated for this model.
   # Primarily used for verification.
   def get_dist_gradients(self, cid):
       grads = dist_autograd.get_gradients(cid)
       # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
       # Tensors must be moved in and out of GPU memory due to this.
       cpu_grads = {}
       for k, v in grads.items():
           k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
           cpu_grads[k_cpu] = v_cpu
       return cpu_grads

   # Wrap local parameters in a RRef. Needed for building the
   # DistributedOptimizer which optimizes paramters remotely.
   def get_param_rrefs(self):
       param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
       return param_rrefs
Finally, we'll create methods to initialize our parameter server. Note that there will only be one instance of a parameter server across all processes, and all trainers will talk to the same parameter server and update the same stored model. As seen in ``run_parameter_server``\ , the server itself does not take any independent actions; it waits for requests from trainers (which are yet to be defined) and responds to them by running the requested function.

.. code-block:: python

   # The global parameter server instance.
   param_server = None
   # A lock to ensure we only have one parameter server.
   global_lock = Lock()


   def get_parameter_server(num_gpus=0):
       """
       Returns a singleton parameter server to all trainer processes
       """
       global param_server
       # Ensure that we get only one handle to the ParameterServer.
       with global_lock:
           if not param_server:
               # construct it once
               param_server = ParameterServer(num_gpus=num_gpus)
           return param_server

   def run_parameter_server(rank, world_size):
       # The parameter server just acts as a host for the model and responds to
       # requests from trainers.
       # rpc.shutdown() will wait for all workers to complete by default, which
       # in this case means that the parameter server will wait for all trainers
       # to complete, and then exit.
       print("PS master initializing RPC")
       rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
       print("RPC initialized! Running parameter server...")
       rpc.shutdown()
       print("RPC shutdown on parameter server.")
Note that above, ``rpc.shutdown()`` will not immediately shut down the Parameter Server. Instead, it will wait for all workers (trainers in this case) to also call into ``rpc.shutdown()``. This gives us the guarantee that the parameter server will not go offline before all trainers (yet to be define) have completed their training process.

Next, we'll define our ``TrainerNet`` class. This will also be a subclass of ``nn.Module``\ , and our ``__init__`` method will use the ``rpc.remote`` API to obtain an RRef, or Remote Reference, to our parameter server. Note that here we are not copying the parameter server to our local process, instead, we can think of ``self.param_server_rref`` as a distributed shared pointer to the parameter server that lives on a separate process.

.. code-block:: python

   # --------- Trainers --------------------

   # nn.Module corresponding to the network trained by this trainer. The
   # forward() method simply invokes the network on the given parameter
   # server.
   class TrainerNet(nn.Module):
       def __init__(self, num_gpus=0):
           super().__init__()
           self.num_gpus = num_gpus
           self.param_server_rref = rpc.remote(
               "parameter_server", get_parameter_server, args=(num_gpus,))
Next, we'll define a method called ``get_global_param_rrefs``. To motivate the need for this method, it is worth it to read through the documentation on `DistributedOptimizer <https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim>`_, specifically the API signature.  The optimizer must be passed a list of ``RRef``\ s corresponding to the remote parameters to be optimized, so here we obtain the necessary ``RRef``\ s. Since the only remote worker that a given ``TrainerNet`` interacts with is the ``ParameterServer``\ , we simply invoke a ``remote_method`` on the ``ParameterServer``. We use the ``get_param_rrefs`` method which we defined in the ``ParameterServer`` class. This method will return a list of ``RRef``\ s to the parameters that need to be optimized. Note that in this case our ``TrainerNet`` does not define its own paramaters; if it did, we would need to wrap each parameter in an ``RRef`` as well and include it into our input to ``DistributedOptimizer``.

.. code-block:: python

   class TrainerNet(nn.Module):
   ...
       def get_global_param_rrefs(self):
           remote_params = remote_method(
               ParameterServer.get_param_rrefs,
               self.param_server_rref)
           return remote_params
Now, we're ready to define our ``forward`` method, which will invoke (synchronous) RPC to run the forward pass of the network defined on the ``ParameterServer``. Note that we pass in ``self.param_server_rref``\ , which is a remote handle to our ``ParameterServer``\ , to our RPC call. This call will send an RPC to the node on which our ``ParameterServer`` is running, invoke the ``forward`` pass, and return the ``Tensor`` corresponding to the model's output.

.. code-block:: python

   class TrainerNet(nn.Module):
   ...
       def forward(self, x):
           model_output = remote_method(
               ParameterServer.forward, self.param_server_rref, x)
           return model_output
With our trainer fully defined, it's now time to write our neural network training loop that will create our network and optimizer, run some inputs through the network and compute the loss. The training loop looks a lot like that of a local training program, with some modifications due to the nature of our network being distributed across machines.

Below, we initialize our ``TrainerNet`` and build a ``DistributedOptimizer``. Note that as mentioned above, we must pass in all of the global (across all nodes participating in distributed training) parameters that we want to be optimized. In addition, we pass in the local optimizer to be used, in this case, SGD. Note that we can configure the underlying optimizer algorithm in the same way as creating a local optimizer - all arguments for ``optimizer.SGD`` will be forwarded properly. As an example, we pass in a custom learning rate that will be used as the learning rate for all local optimizers.

.. code-block:: python

   def run_training_loop(rank, num_gpus, train_loader, test_loader):
       # Runs the typical nueral network forward + backward + optimizer step, but
       # in a distributed fashion.
       net = TrainerNet(num_gpus=num_gpus)
       # Build DistributedOptimizer.
       param_rrefs = net.get_global_param_rrefs()
       opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)
Next, we define our main training loop. We loop through iterables given by PyTorch's `DataLoader <https://pytorch.org/docs/stable/data.html>`_. Before writing our typical forward/backward/optimizer loop, we first wrap the logic within a `Distributed Autograd context <https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.context>`_. Note that this is needed to record RPCs invoked in the model's forward pass, so that an appropriate graph can be constructed which includes all participating distributed workers in the backward pass. The distributed autograd context returns a ``context_id`` which serves as an identifier for accumulating and optimizing gradients corresponding to a particular iteration.

As opposed to calling the typical ``loss.backward()`` which would kick off the backward pass on this local worker, we call ``dist_autograd.backward()`` and pass in our context_id as well as ``loss``\ , which is the root at which we want the backward pass to begin. In addition, we pass this ``context_id`` into our optimizer call, which is required to be able to look up the corresponding gradients computed by this particular backwards pass across all nodes.

.. code-block:: python

   def run_training_loop(rank, num_gpus, train_loader, test_loader):
   ...
       for i, (data, target) in enumerate(train_loader):
           with dist_autograd.context() as cid:
               model_output = net(data)
               target = target.to(model_output.device)
               loss = F.nll_loss(model_output, target)
               if i % 5 == 0:
                   print(f"Rank {rank} training batch {i} loss {loss.item()}")
               dist_autograd.backward(cid, [loss])
               # Ensure that dist autograd ran successfully and gradients were
               # returned.
               assert remote_method(
                   ParameterServer.get_dist_gradients,
                   net.param_server_rref,
                   cid) != {}
               opt.step(cid)

        print("Training complete!")
        print("Getting accuracy....")
        get_accuracy(test_loader, net)
The following simply computes the accuracy of our model after we're done training, much like a traditional local model. However, note that the ``net`` we pass into this function above is an instance of ``TrainerNet`` and therefore the forward pass invokes RPC in a transparent fashion.

.. code-block:: python

   def get_accuracy(test_loader, model):
       model.eval()
       correct_sum = 0
       # Use GPU to evaluate if possible
       device = torch.device("cuda:0" if model.num_gpus > 0
           and torch.cuda.is_available() else "cpu")
       with torch.no_grad():
           for i, (data, target) in enumerate(test_loader):
               out = model(data, -1)
               pred = out.argmax(dim=1, keepdim=True)
               pred, target = pred.to(device), target.to(device)
               correct = pred.eq(target.view_as(pred)).sum().item()
               correct_sum += correct

       print(f"Accuracy {correct_sum / len(test_loader.dataset)}")
Next, similar to how we defined ``run_parameter_server`` as the main loop for our ``ParameterServer`` that is responsible for initializing RPC, let's define a similar loop for our trainers. The difference will be that our trainers must run the training loop we defined above:

.. code-block:: python

   # Main loop for trainers.
   def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
       print(f"Worker rank {rank} initializing RPC")
       rpc.init_rpc(
           name=f"trainer_{rank}",
           rank=rank,
           world_size=world_size)

       print(f"Worker {rank} done initializing RPC")

       run_training_loop(rank, num_gpus, train_loader, test_loader)
       rpc.shutdown()
Note that similar to ``run_parameter_server``\ , ``rpc.shutdown()`` will by default wait for all workers, both trainers and ParameterServers, to call into ``rpc.shutdown()`` before this node exits. This ensures that nodes are terminated gracefully and no node goes offline while another is expecting it to be online.

We've now completed our trainer and parameter server specific code, and all that's left is to add code to launch trainers and parameter servers. First, we must take in various arguments that apply to our parameter server and trainers. ``world_size`` corresponds to the total number of nodes that will participate in training, and is the sum of all trainers and the parameter server. We also must pass in a unique ``rank`` for each individual process, from 0 (where we will run our single parameter server) to ``world_size - 1``. ``master_addr`` and ``master_port`` are arguments that can be used to identify where the rank 0 process is running, and will be used by individual nodes to discover each other. To test this example out locally, simply pass in ``localhost`` and the same ``master_port`` to all instances spawned. Note that for demonstration purposes, this example supports only between 0-2 GPUs, although the pattern can be extended to make use of additional GPUs.

.. code-block:: python

   if __name__ == '__main__':
       parser = argparse.ArgumentParser(
           description="Parameter-Server RPC based training")
       parser.add_argument(
           "--world_size",
           type=int,
           default=4,
           help="""Total number of participating processes. Should be the sum of
           master node and all training nodes.""")
       parser.add_argument(
           "rank",
           type=int,
           default=None,
           help="Global rank of this process. Pass in 0 for master.")
       parser.add_argument(
           "num_gpus",
           type=int,
           default=0,
           help="""Number of GPUs to use for training, Currently supports between 0
            and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
       parser.add_argument(
           "--master_addr",
           type=str,
           default="localhost",
           help="""Address of master, will default to localhost if not provided.
           Master must be able to accept network traffic on the address + port.""")
       parser.add_argument(
           "--master_port",
           type=str,
           default="29500",
           help="""Port that master is listening on, will default to 29500 if not
           provided. Master must be able to accept network traffic on the host and port.""")

       args = parser.parse_args()
       assert args.rank is not None, "must provide rank argument."
       assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
       os.environ['MASTER_ADDR'] = args.master_addr
       os.environ["MASTER_PORT"] = args.master_port
Now, we'll create a process corresponding to either a parameter server or trainer depending on our command line arguments. We'll create a ``ParameterServer`` if our passed in rank is 0, and a ``TrainerNet`` otherwise. Note that we're using ``torch.multiprocessing`` to launch a subprocess corresponding to the function that we want to execute, and waiting on this process's completion from the main thread with ``p.join()``. In the case of initializing our trainers, we also use PyTorch's `dataloaders <https://pytorch.org/docs/stable/data.html>`_ in order to specify train and test data loaders on the MNIST dataset.

.. code-block:: python

   processes = []
   world_size = args.world_size
   if args.rank == 0:
       p = mp.Process(target=run_parameter_server, args=(0, world_size))
       p.start()
       processes.append(p)
   else:
       # Get data to train on
       train_loader = torch.utils.data.DataLoader(
           datasets.MNIST('../data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
           batch_size=32, shuffle=True,)
       test_loader = torch.utils.data.DataLoader(
           datasets.MNIST(
               '../data',
               train=False,
               transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                           ])),
           batch_size=32,
           shuffle=True,
       )
       # start training worker on this node
       p = mp.Process(
           target=run_worker,
           args=(
               args.rank,
               world_size, args.num_gpus,
               train_loader,
               test_loader))
       p.start()
       processes.append(p)

   for p in processes:
       p.join()
To run the example locally, run the following command worker for the server and each worker you wish to spawn, in separate terminal windows: ``python rpc_parameter_server.py --world_size=WORLD_SIZE --rank=RANK``. For example, for a master node with world size of 2, the command would be ``python rpc_parameter_server.py --world_size=2 --rank=0``. The trainer can then be launched with the command ``python rpc_parameter_server.py --world_size=2 --rank=1`` in a separate window, and this will begin training with one server and a single trainer. Note that this tutorial assumes that training occurs using between 0 and 2 GPUs, and this argument can be configured by passing ``--num_gpus=N`` into the training script.

You can pass in the command line arguments ``--master_addr=ADDRESS`` and ``--master_port=PORT`` to indicate the address and port that the master worker is listening on, for example, to test functionality where trainers and master nodes run on different machines.
