Getting Started with Distributed RPC Framework
=================================================
**Author**: `Shen Li <https://mrshenli.github.io/>`_

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/rpc_tutorial.rst>`__.

Prerequisites:

-  `PyTorch Distributed Overview <../beginner/dist_overview.html>`__
-  `RPC API documents <https://pytorch.org/docs/master/rpc.html>`__

This tutorial uses two simple examples to demonstrate how to build distributed
training with the `torch.distributed.rpc <https://pytorch.org/docs/stable/rpc.html>`__
package which was first introduced as an experimental feature in PyTorch v1.4.
Source code of the two examples can be found in
`PyTorch examples <https://github.com/pytorch/examples>`__.

Previous tutorials,
`Getting Started With Distributed Data Parallel <ddp_tutorial.html>`__
and `Writing Distributed Applications With PyTorch <dist_tuto.html>`__,
described `DistributedDataParallel <https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html>`__
which supports a specific training paradigm where the model is replicated across
multiple processes and each process handles a split of the input data.
Sometimes, you might run into scenarios that require different training
paradigms. For example:

1) In reinforcement learning, it might be relatively expensive to acquire
   training data from environments while the model itself can be quite small. In
   this case, it might be useful to spawn multiple observers running in parallel
   and share a single agent. In this case, the agent takes care of the training
   locally, but the application would still need libraries to send and receive
   data between observers and the trainer.
2) Your model might be too large to fit in GPUs on a single machine, and hence
   would need a library to help split the model onto multiple machines. Or you
   might be implementing a `parameter server <https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf>`__
   training framework, where model parameters and trainers live on different
   machines.


The `torch.distributed.rpc <https://pytorch.org/docs/stable/rpc.html>`__ package
can help with the above scenarios. In case 1, `RPC <https://pytorch.org/docs/stable/rpc.html#rpc>`__
and `RRef <https://pytorch.org/docs/stable/rpc.html#rref>`__ allow sending data
from one worker to another while easily referencing remote data objects. In
case 2, `distributed autograd <https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework>`__
and `distributed optimizer <https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim>`__
make executing backward pass and optimizer step as if it is local training. In
the next two sections, we will demonstrate APIs of
`torch.distributed.rpc <https://pytorch.org/docs/stable/rpc.html>`__ using a
reinforcement learning example and a language model example. Please note, this
tutorial does not aim at building the most accurate or efficient models to
solve given problems, instead, the main goal here is to show how to use the
`torch.distributed.rpc <https://pytorch.org/docs/stable/rpc.html>`__ package to
build distributed training applications.



Distributed Reinforcement Learning using RPC and RRef
-----------------------------------------------------

This section describes steps to build a toy distributed reinforcement learning
model using RPC to solve CartPole-v1 from `OpenAI Gym <https://gym.openai.com>`__.
The policy code is mostly borrowed from the existing single-thread
`example <https://github.com/pytorch/examples/blob/master/reinforcement_learning>`__
as shown below. We will skip details of the ``Policy`` design, and focus on RPC
usages.

.. code:: python

    import torch.nn as nn
    import torch.nn.functional as F

    class Policy(nn.Module):

        def __init__(self):
            super(Policy, self).__init__()
            self.affine1 = nn.Linear(4, 128)
            self.dropout = nn.Dropout(p=0.6)
            self.affine2 = nn.Linear(128, 2)

        def forward(self, x):
            x = self.affine1(x)
            x = self.dropout(x)
            x = F.relu(x)
            action_scores = self.affine2(x)
            return F.softmax(action_scores, dim=1)


We are ready to present the observer. In this example, each observer creates its
own environment, and waits for the agent's command to run an episode. In each
episode, one observer loops at most ``n_steps`` iterations, and in each
iteration, it uses RPC to pass its environment state to the agent and gets an
action back. Then it applies that action to its environment, and gets the reward
and the next state from the environment. After that, the observer uses another
RPC to report the reward to the agent. Again, please note that, this is
obviously not the most efficient observer implementation. For example, one
simple optimization could be packing current state and last reward in one RPC to
reduce the communication overhead. However, the goal is to demonstrate RPC API
instead of building the best solver for CartPole. So, let's keep the logic
simple and the two steps explicit in this example.

.. code:: python

    import argparse
    import gym
    import torch.distributed.rpc as rpc

    parser = argparse.ArgumentParser(
        description="RPC Reinforcement Learning Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--world_size', default=2, type=int, metavar='W',
                        help='number of workers')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='how much to value future rewards')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed  for reproducibility')
    args = parser.parse_args()

    class Observer:

        def __init__(self):
            self.id = rpc.get_worker_info().id
            self.env = gym.make('CartPole-v1')
            self.env.seed(args.seed)

        def run_episode(self, agent_rref):
            state, ep_reward = self.env.reset(), 0
            for _ in range(10000):
                # send the state to the agent to get an action
                action = agent_rref.rpc_sync().select_action(self.id, state)

                # apply the action to the environment, and get the reward
                state, reward, done, _ = self.env.step(action)

                # report the reward to the agent for training purpose
                agent_rref.rpc_sync().report_reward(self.id, reward)

                # finishes after the number of self.env._max_episode_steps
                if done:
                    break


The code for agent is a little more complex, and we will break it into multiple
pieces. In this example, the agent serves as both the trainer and the master,
such that it sends command to multiple distributed observers to run episodes,
and it also records all actions and rewards locally which will be used during
the training phase after each episode. The code below shows ``Agent``
constructor where most lines are initializing various components. The loop at
the end initializes observers remotely on other workers, and holds ``RRefs`` to
those observers locally. The agent will use those observer ``RRefs`` later to
send commands. Applications don't need to worry about the lifetime of ``RRefs``.
The owner of each ``RRef`` maintains a reference counting map to track its
lifetime, and guarantees the remote data object will not be deleted as long as
there is any live user of that ``RRef``. Please refer to the ``RRef``
`design doc <https://pytorch.org/docs/master/notes/rref.html>`__ for details.


.. code:: python

    import gym
    import numpy as np

    import torch
    import torch.distributed.rpc as rpc
    import torch.optim as optim
    from torch.distributed.rpc import RRef, rpc_async, remote
    from torch.distributions import Categorical

    class Agent:
        def __init__(self, world_size):
            self.ob_rrefs = []
            self.agent_rref = RRef(self)
            self.rewards = {}
            self.saved_log_probs = {}
            self.policy = Policy()
            self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
            self.eps = np.finfo(np.float32).eps.item()
            self.running_reward = 0
            self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
            for ob_rank in range(1, world_size):
                ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
                self.ob_rrefs.append(remote(ob_info, Observer))
                self.rewards[ob_info.id] = []
                self.saved_log_probs[ob_info.id] = []


Next, the agent exposes two APIs to observers for selecting actions and
reporting rewards. Those functions only run locally on the agent, but will
be triggered by observers through RPC.


.. code:: python

    class Agent:
        ...
        def select_action(self, ob_id, state):
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs = self.policy(state)
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs[ob_id].append(m.log_prob(action))
            return action.item()

        def report_reward(self, ob_id, reward):
            self.rewards[ob_id].append(reward)


Let's add a ``run_episode`` function on agent which tells all observers
to execute an episode. In this function, it first creates a list to collect
futures from asynchronous RPCs, and then loop over all observer ``RRefs`` to
make asynchronous RPCs. In these RPCs, the agent also passes an ``RRef`` of
itself to the observer, so that the observer can call functions on the agent as
well. As shown above, each observer will make RPCs back to the agent, which are
nested RPCs. After each episode, the ``saved_log_probs`` and ``rewards`` will
contain the recorded action probs and rewards.


.. code:: python

    class Agent:
        ...
        def run_episode(self):
            futs = []
            for ob_rref in self.ob_rrefs:
                # make async RPC to kick off an episode on all observers
                futs.append(
                    rpc_async(
                        ob_rref.owner(),
                        ob_rref.rpc_sync().run_episode,
                        args=(self.agent_rref,)
                    )
                )

            # wait until all obervers have finished this episode
            for fut in futs:
                fut.wait()


Finally, after one episode, the agent needs to train the model, which
is implemented in the ``finish_episode`` function below. There is no RPCs in
this function and it is mostly borrowed from the single-thread
`example <https://github.com/pytorch/examples/blob/master/reinforcement_learning>`__.
Hence, we skip describing its contents.



.. code:: python

    class Agent:
        ...
        def finish_episode(self):
          # joins probs and rewards from different observers into lists
          R, probs, rewards = 0, [], []
          for ob_id in self.rewards:
              probs.extend(self.saved_log_probs[ob_id])
              rewards.extend(self.rewards[ob_id])

          # use the minimum observer reward to calculate the running reward
          min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
          self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

          # clear saved probs and rewards
          for ob_id in self.rewards:
              self.rewards[ob_id] = []
              self.saved_log_probs[ob_id] = []

          policy_loss, returns = [], []
          for r in rewards[::-1]:
              R = r + args.gamma * R
              returns.insert(0, R)
          returns = torch.tensor(returns)
          returns = (returns - returns.mean()) / (returns.std() + self.eps)
          for log_prob, R in zip(probs, returns):
              policy_loss.append(-log_prob * R)
          self.optimizer.zero_grad()
          policy_loss = torch.cat(policy_loss).sum()
          policy_loss.backward()
          self.optimizer.step()
          return min_reward


With ``Policy``, ``Observer``, and ``Agent`` classes, we are ready to launch
multiple processes to perform the distributed training. In this example, all
processes run the same ``run_worker`` function, and they use the rank to
distinguish their role. Rank 0 is always the agent, and all other ranks are
observers. The agent serves as master by repeatedly calling ``run_episode`` and
``finish_episode`` until the running reward surpasses the reward threshold
specified by the environment. All observers passively waiting for commands
from the agent. The code is wrapped by
`rpc.init_rpc <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.init_rpc>`__ and
`rpc.shutdown <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.shutdown>`__,
which initializes and terminates RPC instances respectively. More details are
available in the `API page <https://pytorch.org/docs/stable/rpc.html>`__.


.. code:: python

    import os
    from itertools import count

    import torch.multiprocessing as mp

    AGENT_NAME = "agent"
    OBSERVER_NAME="obs{}"

    def run_worker(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        if rank == 0:
            # rank0 is the agent
            rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

            agent = Agent(world_size)
            print(f"This will run until reward threshold of {agent.reward_threshold}"
                    " is reached. Ctrl+C to exit.")
            for i_episode in count(1):
                agent.run_episode()
                last_reward = agent.finish_episode()

                if i_episode % args.log_interval == 0:
                    print(f"Episode {i_episode}\tLast reward: {last_reward:.2f}\tAverage reward: "
                        f"{agent.running_reward:.2f}")
                if agent.running_reward > agent.reward_threshold:
                    print(f"Solved! Running reward is now {agent.running_reward}!")
                    break
        else:
            # other ranks are the observer
            rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
            # observers passively waiting for instructions from the agent

        # block until all rpcs finish, and shutdown the RPC instance
        rpc.shutdown()


    mp.spawn(
        run_worker,
        args=(args.world_size, ),
        nprocs=args.world_size,
        join=True
    )

Below are some sample outputs when training with `world_size=2`.

::

    This will run until reward threshold of 475.0 is reached. Ctrl+C to exit.
    Episode 10      Last reward: 26.00      Average reward: 10.01
    Episode 20      Last reward: 16.00      Average reward: 11.27
    Episode 30      Last reward: 49.00      Average reward: 18.62
    Episode 40      Last reward: 45.00      Average reward: 26.09
    Episode 50      Last reward: 44.00      Average reward: 30.03
    Episode 60      Last reward: 111.00     Average reward: 42.23
    Episode 70      Last reward: 131.00     Average reward: 70.11
    Episode 80      Last reward: 87.00      Average reward: 76.51
    Episode 90      Last reward: 86.00      Average reward: 95.93
    Episode 100     Last reward: 13.00      Average reward: 123.93
    Episode 110     Last reward: 33.00      Average reward: 91.39
    Episode 120     Last reward: 73.00      Average reward: 76.38
    Episode 130     Last reward: 137.00     Average reward: 88.08
    Episode 140     Last reward: 89.00      Average reward: 104.96
    Episode 150     Last reward: 97.00      Average reward: 98.74
    Episode 160     Last reward: 150.00     Average reward: 100.87
    Episode 170     Last reward: 126.00     Average reward: 104.38
    Episode 180     Last reward: 500.00     Average reward: 213.74
    Episode 190     Last reward: 322.00     Average reward: 300.22
    Episode 200     Last reward: 165.00     Average reward: 272.71
    Episode 210     Last reward: 168.00     Average reward: 233.11
    Episode 220     Last reward: 184.00     Average reward: 195.02
    Episode 230     Last reward: 284.00     Average reward: 208.32
    Episode 240     Last reward: 395.00     Average reward: 247.37
    Episode 250     Last reward: 500.00     Average reward: 335.42
    Episode 260     Last reward: 500.00     Average reward: 386.30
    Episode 270     Last reward: 500.00     Average reward: 405.29
    Episode 280     Last reward: 500.00     Average reward: 443.29
    Episode 290     Last reward: 500.00     Average reward: 464.65
    Solved! Running reward is now 475.3163778435275!


In this example, we show how to use RPC as the communication vehicle to pass
data across workers, and how to use RRef to reference remote objects. It is true
that you could build the entire structure directly on top of ``ProcessGroup``
``send`` and ``recv`` APIs or use other communication/RPC libraries. However,
by using `torch.distributed.rpc`, you can get the native support and
continuously optimized performance under the hood.

Next, we will show how to combine RPC and RRef with distributed autograd and
distributed optimizer to perform distributed model parallel training.



Distributed RNN using Distributed Autograd and Distributed Optimizer
--------------------------------------------------------------------

In this section, we use an RNN model to show how to build distributed model
parallel training with the RPC API. The example RNN model is very small and
can easily fit into a single GPU, but we still divide its layers onto two
different workers to demonstrate the idea. Developer can apply the similar
techniques to distribute much larger models across multiple devices and
machines.

The RNN model design is borrowed from the word language model in PyTorch
`example <https://github.com/pytorch/examples/tree/master/word_language_model>`__
repository, which contains three main components, an embedding table, an
``LSTM`` layer, and a decoder. The code below wraps the embedding table and the
decoder into sub-modules, so that their constructors can be passed to the RPC
API. In the ``EmbeddingTable`` sub-module, we intentionally put the
``Embedding`` layer on GPU to cover the use case. In v1.4, RPC always creates
CPU tensor arguments or return values on the destination worker. If the function
takes a GPU tensor, you need to move it to the proper device explicitly.


.. code:: python

    class EmbeddingTable(nn.Module):
        r"""
        Encoding layers of the RNNModel
        """
        def __init__(self, ntoken, ninp, dropout):
            super(EmbeddingTable, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp).cuda()
            self.encoder.weight.data.uniform_(-0.1, 0.1)

        def forward(self, input):
            return self.drop(self.encoder(input.cuda()).cpu()


    class Decoder(nn.Module):
        def __init__(self, ntoken, nhid, dropout):
            super(Decoder, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.decoder = nn.Linear(nhid, ntoken)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-0.1, 0.1)

        def forward(self, output):
            return self.decoder(self.drop(output))


With the above sub-modules, we can now piece them together using RPC to
create an RNN model. In the code below ``ps`` represents a parameter server,
which hosts parameters of the embedding table and the decoder. The constructor
uses the `remote <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.remote>`__
API to create an ``EmbeddingTable`` object and a ``Decoder`` object on the
parameter server, and locally creates the ``LSTM`` sub-module. During the
forward pass, the trainer uses the ``EmbeddingTable`` ``RRef`` to find the
remote sub-module and passes the input data to the ``EmbeddingTable`` using RPC
and fetches the lookup results. Then, it runs the embedding through the local
``LSTM`` layer, and finally uses another RPC to send the output to the
``Decoder`` sub-module. In general, to implement distributed model parallel
training, developers can divide the model into sub-modules, invoke RPC to create
sub-module instances remotely, and use on ``RRef`` to find them when necessary.
As you can see in the code below, it looks very similar to single-machine model
parallel training. The main difference is replacing ``Tensor.to(device)`` with
RPC functions.


.. code:: python

    class RNNModel(nn.Module):
        def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
            super(RNNModel, self).__init__()

            # setup embedding table remotely
            self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))
            # setup LSTM locally
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
            # setup decoder remotely
            self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))

        def forward(self, input, hidden):
            # pass input to the remote embedding table and fetch emb tensor back
            emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)
            output, hidden = self.rnn(emb, hidden)
            # pass output to the rremote decoder and get the decoded output back
            decoded = _remote_method(Decoder.forward, self.decoder_rref, output)
            return decoded, hidden

Before introducing the distributed optimizer, let's add a helper function to
generate a list of RRefs of model parameters, which will be consumed by the
distributed optimizer. In local training, applications could call
``Module.parameters()`` to grab references to all parameter tensors, and pass it
to the local optimizer for subsequent updates. However, the same API does not
work in distributed training scenarios as some parameters live on remote
machines. Therefore, instead of taking a list of parameter ``Tensors``, the
distributed optimizer takes a list of ``RRefs``, one ``RRef`` per model
parameter for both local and remote model parameters. The helper function is
pretty simple, just call ``Module.parameters()`` and creates a local ``RRef`` on
each of the parameters.


.. code:: python

    def _parameter_rrefs(module):
        param_rrefs = []
        for param in module.parameters():
            param_rrefs.append(RRef(param))
        return param_rrefs


Then, as the ``RNNModel`` contains three sub-modules, we need to call
``_parameter_rrefs`` three times, and wrap that into another helper function.


.. code:: python

    class RNNModel(nn.Module):
        ...
        def parameter_rrefs(self):
            remote_params = []
            # get RRefs of embedding table
            remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))
            # create RRefs for local parameters
            remote_params.extend(_parameter_rrefs(self.rnn))
            # get RRefs of decoder
            remote_params.extend(_remote_method(_parameter_rrefs, self.decoder_rref))
            return remote_params


Now, we are ready to implement the training loop. After initializing model
arguments, we create the ``RNNModel`` and the ``DistributedOptimizer``. The
distributed optimizer will take a list of parameter ``RRefs``, find all distinct
owner workers, and create the given local optimizer (i.e., ``SGD`` in this case,
you can use other local optimizers as well) on each of the owner worker using
the given arguments (i.e., ``lr=0.05``).

In the training loop, it first creates a distributed autograd context, which
will help the distributed autograd engine to find gradients and involved RPC
send/recv functions. The design details of the distributed autograd engine can
be found in its `design note <https://pytorch.org/docs/master/notes/distributed_autograd.html>`__.
Then, it kicks off the forward pass as if it is a local
model, and run the distributed backward pass. For the distributed backward, you
only need to specify a list of roots, in this case, it is the loss ``Tensor``.
The distributed autograd engine will traverse the distributed graph
automatically and write gradients properly. Next, it runs the ``step``
function on the distributed optimizer, which will reach out to all involved
local optimizers to update model parameters. Compared to local training, one
minor difference is that you don't need to run ``zero_grad()`` because each
autograd context has dedicated space to store gradients, and as we create a
context per iteration, those gradients from different iterations will not
accumulate to the same set of ``Tensors``.


.. code:: python

    def run_trainer():
        batch = 5
        ntoken = 10
        ninp = 2

        nhid = 3
        nindices = 3
        nlayers = 4
        hidden = (
            torch.randn(nlayers, nindices, nhid),
            torch.randn(nlayers, nindices, nhid)
        )

        model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)

        # setup distributed optimizer
        opt = DistributedOptimizer(
            optim.SGD,
            model.parameter_rrefs(),
            lr=0.05,
        )

        criterion = torch.nn.CrossEntropyLoss()

        def get_next_batch():
            for _ in range(5):
                data = torch.LongTensor(batch, nindices) % ntoken
                target = torch.LongTensor(batch, ntoken) % nindices
                yield data, target

        # train for 10 iterations
        for epoch in range(10):
            for data, target in get_next_batch():
                # create distributed autograd context
                with dist_autograd.context() as context_id:
                    hidden[0].detach_()
                    hidden[1].detach_()
                    output, hidden = model(data, hidden)
                    loss = criterion(output, target)
                    # run distributed backward pass
                    dist_autograd.backward(context_id, [loss])
                    # run distributed optimizer
                    opt.step(context_id)
                    # not necessary to zero grads since they are
                    # accumulated into the distributed autograd context
                    # which is reset every iteration.
            print("Training epoch {}".format(epoch))


Finally, let's add some glue code to launch the parameter server and the trainer
processes.


.. code:: python

    def run_worker(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        if rank == 1:
            rpc.init_rpc("trainer", rank=rank, world_size=world_size)
            _run_trainer()
        else:
            rpc.init_rpc("ps", rank=rank, world_size=world_size)
            # parameter server do nothing
            pass

        # block until all rpcs finish
        rpc.shutdown()


    if __name__=="__main__":
        world_size = 2
        mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)
