"""
Reinforcement Learning (A3C) Tutorial
=====================================
**Author**: `Andrea Schioppa <https://github.com/salayatana66>`_

This tutorial shows how to use PyTorch to build an agent solving the
CartPole-v0 task from the `OpenAI Gym <https://gym.openai.com/>`__
using the Asynchronous Advantage Actor-Critic (A3C) algorithm
described in `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/pdf/1602.01783.pdf>`__.

Because A3C requires multiple actor-learners running in parallel,
this tutorial showcases an example of using multiprocessing with PyTorch.

For an explanation of the task to solve we refer to the other tutorial
:doc:`/intermediate/reinforcement_q_learning` which deals with DQN for Deep Reinforcement Learning.

You will need to install the additional package `gym <https://gym.openai.com/docs>`__
e.g. using `pip install gym`.
"""

######################################################################
# Imports
# -------
#
# We start with importing components of PyTorch. Note in particular
# that we import ``multiprocessing``, which integrates PyTorch with
# Python's own ``multiprocessing`` module.
#
# As we will use multiprocessing, we will use the ``time`` module
# to delay the start of different processes.
#
# Finally, we believe the code is a bit more readable by using
# type annotations, via the ``typing`` module.


import torch
import torch.multiprocessing as mp
from torch.nn import Sequential, Linear, ReLU
from torch.optim.rmsprop import RMSprop
from torch.utils.tensorboard import SummaryWriter

import ctypes

import gym

import numpy as np
from datetime import datetime
import logging

import time

import argparse
from typing import List, Dict, Optional

from collections import OrderedDict

from pathlib import Path

# setting seeds
np.random.seed(1032)
torch.manual_seed(1032)

# A3C is meant to be run using CPUs so we will explicitly
# use the CPU device
device = torch.device("cpu")


######################################################################
# Setting up the Logger and Argument Parsing
# -------------------------------------------
#
# We solved the task running the script with the following arguments:
#
# .. code-block:: bash
#
#    $ python reinforcement_a3c_tutorial.py --num-iterations 50000 \
#    $          --lr .5e-4 --tmax 20 --num-processes 5
#
# We first create the ``logger`` and then the argument ``parser``.
# Here ``args.tmax`` corresponds to :math:`t_{max}`
# from the paper, which is the maximum number of consecutive
# actions an actor-learner is allowed to play before updating
# the model parameters; ``args.gamma`` is the discount rate; for the
# Cartpole-v0 in reality one should set :math:`\gamma = 1`
# but we want :math:`\gamma < 1`
# for better convergence properties, hence we choose a default
# value close to 1; finally ``args.num_processes`` is the number
# of actor-learners.

logger = logging.getLogger("a3c")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(message)s')
logger.addHandler(ch)
ch.setFormatter(formatter)

parser = argparse.ArgumentParser()
parser.add_argument("--tmax", type=int, required=True, help="""
From the paper, essentially size of the
buffer used to play the episodes
""")
parser.add_argument("--gamma", type=float, default=.995)
parser.add_argument("--num-processes", type=int, required=True)
parser.add_argument("--num-iterations", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)

# remove these values when running from shell
args = parser.parse_args(["--num-iterations", "50000", "--lr",
                          ".5e-4", "--tmax", "20", "--num-processes",
                          "5"])

state_size = 4
num_processes = args.num_processes
logger.info(f"Will spawn {num_processes} processes for sync updates")

print_every_T = 100

# this is used by tensorboard logging
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

######################################################################
# Overview of A3C
# -------------------------------------------
#
# In the `paper <https://arxiv.org/pdf/1602.01783.pdf>`__ the algorithm
# is named AS3 and occurs on page 14. The actor is represented by a
# policy function :math:`\pi(a_t | s_t, \theta)` parametrized as a
# neural network (parameters :math:`\theta`);
# here :math:`s_t` is the state at the current time-step and
# :math:`a_t` is an action, which in Cartpole-v0 can just be *move right* or
# *move left*. In order to learn to play better actions, the actor maintains
# also an estimate of the long term value that the current policy
# can achieve if the game starts in state :math:`s_t`;
# this estimate is a so-called advantage critic,
# :math:`V(s_t, \theta_v)`, also parametrized as a neural network
# (parameters :math:`\theta_v`).
#
# For the moment let us assume the loss function is a given black-box, and
# that a single actor-learner tries to solve the problem by playing consecutive
# episodes. Actions are sampled **on-policy** using :math:`\pi(a_t | s_t, \theta)`, rewards
# are then observed and the loss is updated. The problem with this method is that
# the sequence of states observed depends a lot on the policy; this introduces **correlation**
# and **non-stationary** that breaks the assumptions behind batch-learning. Because of this
# training gets unstable and the actor-learner might settle for a suboptimal policy.
# In the DQN tutorial you can see a possible way around this by using a *Replay Memory*
# which basically constructs batches using random samplings from past episodes; however a
# drawback of this approach is that learning is less efficient as it happens **off-policy**.
#
# The way A3C tackles these issues is to put different actor-learners on different processes;
# there is one model in *shared-memory* which each actor learner reads weights from when starting
# an iteration; the weights are used by the actor-learners' own model which is used to play a portion
# of the episode of length at most :math:`t_{max}`; then gradients are updated *asynchronously* on the
# shared model. The intuition is that different learners will experience different random states
# reducing correlation when estimating gradients.
#
# Let us have a look at the loss structure. From the Policy Gradient Theorem (e.g. [BS]_ Chapter 13)
# the gradient of the long-term reward with respect to the policy parameters satisfies:
#
# .. math:: \nabla_{\theta}J \sim \sum_s \mu(s) \sum_a Q_{\pi}(s, a)\nabla_{\theta}\pi(a | s, \theta)
#
# where :math:`Q_{\pi}(s, a)` is the **on-policy** Q-value, which can be estimated through the episodes.
# Specifically in each :math:`t_{max}` consecutive decisions if the episode terminates one just cumulates
# the discounted rewards and if the episode does not terminate one sums to the cumulated discounted rewards
# :math:`\gamma^{t_{max}}V(s,\theta_v)` where :math:`s` is the non-terminal state one lands into. Here
# :math:`\mu(s)` is the expected number of visits to :math:`s` under the policy.
# To reduce the variance in the previous estimation in A3C one uses :math:`V(s_t, \theta_v)` as baseline:
#
# .. math:: \nabla_{\theta}J \sim \sum_s \mu(s) \sum_a (Q_{\pi}(s, a) - V(s, \theta_v))\nabla_{\theta}\pi(a | s, \theta)
#
# where :math:`Q_{\pi}(s, a) - V(s, \theta_v)` is the advantage. Using MonteCarlo theory one can easily show
# that the gradient **ascent** steps following :math:`\nabla_{\theta}J` can be obtained performing
# gradient **descent** on a weighted version of the cross-entropy loss:
#
# .. math:: L(\theta) = -\sum_{s_t, a_t}(Q_{\pi}(s_t, a_t) - V(s_t, \theta_v))\log\pi(a_t | s_t, \theta).
#
# Finally the loss component for :math:`V(s_t, \theta_v)` is the standard square error loss to fit
# against the observed Q-values.


#############################################
# The Model
# ------------------
#
# The implementation of the model is straightforward; as suggested in the paper,
# we use the ``bottom`` neural network to map states to an internal
# layer; we then attach a softmax to build the policy :math:`\pi(a_t| s_t, \theta)`
# and a linear layer to build the value :math:`V(s_t, \theta_v)`.
#
# Some papers recommend adjustments to the loss function when policy and value
# share some parameters. In this problem the standard approach works well out of the
# box; however, because of the way ``Module.share_memory`` is implemented,
# we need to explicitly add the ``bottom`` neural network using ``Module.add_module``,
# otherwise the weights for ``bottom`` will not be available in shared memory.
#

class ActorCritic(torch.nn.Module):
    """
    Represents the actor Critic with Shared
    parameters
    """
    def __init__(self, layers: List[int]):
        super(ActorCritic, self).__init__()

        current_size = state_size

        self.layers_ = []
        for i, size in enumerate(layers):
            next_size = size
            self.layers_.append((f"layer_{i}", Linear(current_size, next_size)))
            self.layers_.append((f"relu_{i}", ReLU()))
            current_size = next_size

        # PyTorch has issues in declaring the parameter space
        # if we don't attribute layers to self, i.e. it cannot descend
        # into a list of layers; this was creating problems as the
        # parameters now in bottom were left out of the synchronization
        # mechanism
        # We solve this by using a submodule
        self.bottom = Sequential(OrderedDict(self.layers_))
        self.add_module("bottom", self.bottom)

        self.policy_layer_ = Linear(current_size, 2)
        self.value_layer_ = Linear(current_size, 1)

    def forward(self, x_state):

        y = self.bottom(x_state)

        logit = self.policy_layer_(y)
        value = self.value_layer_(y)

        return logit, value

    def prob(self, x_state):
        """
        Probabilities from the Actor
        """

        logit, _ = self.forward(x_state)
        if len(logit.shape) > 1:
            return logit.softmax(dim=1)
        else:
            return logit.softmax(dim=0)

    def value(self, x_state):
        """
        Value Estimations from the Critic
        """

        _, value_ = self.forward(x_state)
        return value_

    def loss(self, x_state, x_rewards, x_actions):
        """
        The loss combines the policy loss with
        the error for the critic; details are in
        the paper; important point different from the paper
        is that we just take .mean(); while in policy
        gradient updates it is often suggested to sum...
        """
        logit, value = self.forward(x_state)

        # residuals are used in both losses
        value_residual = x_rewards - value

        # component of the loss for the policy
        # - sign because of gradient ascent on policy
        # note that gradients from the value_residuals are
        # not back-propagated
        policy_loss = -torch.sum(
            logit.log_softmax(dim=1) * x_actions, dim=1) * \
            value_residual.clone().detach().reshape((-1,))

        # component of the loss from the value
        value_loss = value_residual.pow(2.0)

        return policy_loss.mean() + value_loss.mean()


#######################################################################
# The Player
# ------------
#
# The Player plays the game; when the buffer has been filled it
# returns a state and then on can use ``Player.returns`` to build
# the batch to train on.
#
# Note we explicitly tell PyTorch to use the CPU via ``device=device``.
#
class Player(object):
    """
    Plays multiple episodes returning a state
    when the buffer of size t_max has been filled
    """
    def __init__(self, gamma: float, t_max: int):
        self.gamma_ = gamma

        # maximal episode delay
        self.t_max_ = t_max
        # inner counter
        self.t = 1
        self.t_start = 1

        self.env_ = gym.make("CartPole-v0")
        # states during episode
        self.s_l_: List[np.array] = list()
        # current state
        self.s_ = self.env_.reset()
        # sequence of actions as a mask
        self.a_l_: List[List[float]] = list()
        # sequence of simple rewards
        self.r_l_: List[float] = list()
        # total reward in current episode
        self.total_reward_: float = 0.0
        self.terminated_: bool = False

    def current_state(self):
        return self.s_

    def terminated(self):
        return self.terminated_

    def reward(self):
        return self.total_reward_

    def restart(self):
        """
        Restart a terminated episode
        """
        self.total_reward_ = 0.0
        self.terminated_ = False
        self.s_l_: List[np.array] = list()
        self.a_l_: List[List[float]] = list()
        self.r_l_: List[float] = list()
        self.s_ = self.env_.reset()
        self.t = self.t_start

    def play(self, action: int) -> Optional[np.ndarray]:
        if self.terminated_:
            raise Exception("Cannot play in terminated state without restarting")
        s_next, r_, terminated, _ = self.env_.step(action)
        self.terminated_ = terminated
        self.total_reward_ += r_
        self.r_l_.append(r_)
        self.a_l_.append([0.0, 0.0])
        self.a_l_[-1][action] = 1.0
        self.s_l_.append(self.s_)

        self.s_ = s_next
        self.t += 1

        if self.terminated_ or (self.t - self.t_start == self.t_max_):
            self.t = self.t_start
            return s_next
        else:
            return None

    def returns(self, start_point: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        computing the returns "R" as in the paper
        and returning the states, actions, etc...
        to compute the different losses

        in reality "R" is more like a Q-value
        """
        if self.terminated() and (start_point is not None):
            raise ValueError("Cannot supply a start point when episode is terminated")
        v_ = torch.tensor([0.0], device=device)
        if start_point is not None:
            v_ = start_point
        R = torch.zeros(size=(len(self.r_l_), 1), dtype=torch.float32, device=device)
        R[-1, 0] = self.r_l_[-1] + self.gamma_ * v_
        idx = -2
        while idx >= -len(self.r_l_):
            R[idx, 0] = self.r_l_[idx] + self.gamma_ * R[idx+1, 0]
            idx -= 1

        out_s = torch.tensor(self.s_l_, dtype=torch.float32, device=device)
        out_a = torch.tensor(self.a_l_, device=device)
        self.s_l_ = list()
        self.a_l_ = list()
        self.r_l_ = list()
        return {
            "R": R,
            "s": out_s,
            "a": out_a,
        }


#####################################################
# Instantiating the Model and shared variables
# -------------------------
#
# We use a simple 2-layers architecture for the Model.
# We put the model parameters on the CPU and then in shared memory.
#
# We use two shared global variables:
#  - ``global_step`` is the :math:`T` from the paper which just
#    holds the count of gradient steps taken; this is used to decide
#    when to log data to Tensorboard and when to decide to stop.
# - ``stop_play``: processes will stop playing when this becomes ``True``;
#   this happens when one process reaches the maximum number of iterations
#

actor_critic_architecture = [64, 16]
logger.info(f"Using actor-critic neural network architecture: {actor_critic_architecture}")
actor_critic = ActorCritic(actor_critic_architecture)
actor_critic.to(device=device)

lr = args.lr
logger.info(f"Using learning rate: {lr}")

# it is important we make all the tensors available
# in shared memory to the processes
actor_critic.share_memory()

# global counter
global_step = mp.Value(ctypes.c_int, 1)
# this decides when we reach the desired number of iterations
# to terminate
stop_play = mp.Value(ctypes.c_bool, False)

player_length = args.tmax
logger.info(f"Each player will use a buffer of size {player_length}")
player_gamma = args.gamma
logger.info(f"Future will be discounted by using gamma = {player_gamma}")


################################################################################
# The Processes that play the game
# ---------------------------------
#
# The function ``playing_process`` executes a given actor-learner.
# At the beginning of each iteration weights are copied from the
# global model in shared memory to the local model.
# Then the game gets played *on policy* till the training
# buffer gets filled. Then gradients are computed and updates are applied
# to the model *in shared memory*.
#

def playing_process(pid: int):
    """
    A simple process that keeps playing the game
    and update model parameters
    """
    last_reward: float = -1.0
    player = Player(player_gamma, player_length)

    optimizer = RMSprop(actor_critic.parameters(), momentum=.9,
                        lr=lr)

    local_model = ActorCritic(actor_critic_architecture)
    local_model.to(device)

    Path("/tmp/tensorboard_logs").mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=f"/tmp/tensorboard_logs/{current_time}-a3c", flush_secs=1, max_queue=1)
    logger.info(f"Asynchronous training process[{pid}]: will train for {args.num_iterations} iterations")
    while not stop_play.value:
        # reinitialize the state dictionary
        local_model.load_state_dict(actor_critic.state_dict())
        while True:
            # choose an action
            prob = local_model.prob(torch.tensor(player.current_state(),
                                                 dtype=torch.float32)) \
                    .detach().numpy()

            a = np.random.choice([0, 1], p=prob)

            # when the outcome is not None either we filled
            # a batch with player_length or the episode terminated
            outcome = player.play(a)
            if outcome is not None:
                start_value = None
                if not player.terminated():
                    start_value = local_model.value(torch.tensor(outcome, dtype=torch.float32)).clone().detach()
                else:
                    last_reward = player.reward()
                # get the batch to train
                tensor_dict = player.returns(start_value)

                # compute the loss using the local model
                loss = local_model.loss(x_state=tensor_dict['s'],
                                        x_actions=tensor_dict['a'],
                                        x_rewards=tensor_dict['R'])

                # Gradient Descent on the global model
                optimizer.zero_grad()
                local_model.zero_grad()
                loss.backward()
                for p_glo, p_loc in zip(actor_critic.parameters(), local_model.parameters()):
                    p_glo.grad = p_loc.grad
                optimizer.step()

                if player.terminated():
                    player.restart()

                if global_step.value % print_every_T == 0:
                    global_step.acquire()
                    print(f"Asynchronous training process[{pid}]: Finishing iteration {global_step.value}")
                    writer.add_scalar("Loss", loss, global_step.value)
                    writer.add_scalar("Reward", last_reward, global_step.value)
                    global_step.release()

                global_step.value += 1

                if global_step.value > args.num_iterations:
                    print(f"Asynchronous training process[{pid}]: Reached maximum number of iterations, terminating")
                    stop_play.value = True

                if stop_play.value:
                    break
                break
    writer.flush()
    writer.close()


#################################################################
# Evaluation Function
# ---------------------
#
# This function is used to assess the quality of the final model.
# It is recommended to evaluate the rewards on at least 100 consecutive
# episodes; scores > 190 are good; note that episodes finish either
# when the pole is no longer in balance (more than 15 degrees from vertical),
# or the cart moves more than 2.4 units from the center or one has
# kept playing at least 200 actions.
#

def evaluation(num_iters: int = 100):
    """
    Evaluates on 100 episodes
    """

    player = Player(player_gamma, player_length)
    episode_rewards = list()
    while len(episode_rewards) < num_iters:
        # keep playing till the episode terminates
        while not player.terminated():
            # choose an action
            prob = actor_critic.prob(torch.tensor(player.current_state(),
                                                  dtype=torch.float32)) \
                .detach().numpy()

            a = np.random.choice([0, 1], p=prob)
            # a batch with player_length or the episode terminated
            _ = player.play(a)

        episode_rewards.append(player.reward())
        player.restart()

    return episode_rewards


#####################################################################
# The Training Loop
# ------------------------
#
# In the training loop we start the different processes with a slight
# delay. At the end of training we perform an evaluation.
#
processes = list()
for pid in range(num_processes):
    p = mp.Process(target=playing_process, kwargs={"pid": pid})
    p.start()
    time.sleep(.2)
    processes.append(p)

[p.join() for p in processes]
values = evaluation()
# This should give you at least 195 on average
print(f"Final Eval: {np.mean(values)}")

#####################################################################################
# References
# ********************
#
# .. [BS] Reinforcement Learning: An Introduction, Andrew Barto an Richard S. Sutton.
#
