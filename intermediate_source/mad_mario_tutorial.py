"""
Welcome to Mad Mario!
=====================

We put together this project to walk you through fundamentals of
reinforcement learning. Along the project, you will implement a smart
Mario that learns to complete levels on itself. To begin with, you don’t
need to know anything about Reinforcement Learning (RL). In case you
wanna peek ahead, here is a `cheatsheet on RL
basics <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N?usp=sharing>`__
that we will refer to throughout the project. At the end of the
tutorial, you will gain a solid understanding of RL fundamentals and
implement a classic RL algorithm, Q-learning, on yourself.

It’s recommended that you have familiarity with Python and high school
or equivalent level of math/statistics background – that said, don’t
worry if memory is blurry. Just leave comments anywhere you feel
confused, and we will explain the section in more details.

"""

# !pip install gym-super-mario-bros==7.3.0 > /dev/null 2>&1


######################################################################
# Let’s get started!
# ------------------
#
# First thing first, let’s look at what we will build: Just like when we
# first try the game, Mario enters the game not knowing anything about the
# game. It makes random action just to understand the game better. Each
# failure experience adds to Mario’s memory, and as failure accumulates,
# Mario starts to recognize the better action to take in a particular
# scenario. Eventually Mario learns a good strategy and completes the
# level.
#
# Let’s put the story into pseudo code.
#
# ::
#
#    for a total of N episodes:
#      for a total of M steps in each episode:
#        Mario makes an action
#        Game gives a feedback
#        Mario remembers the action and feedback
#        after building up some experiences:
#          Mario learns from experiences
#


######################################################################
# In RL terminology: agent (Mario) interacts with environment (Game) by
# choosing actions, and environment responds with reward and next state.
# Based on the collected (state, action, reward) information, agent learns
# to maximize its future return by optimizing its action policy.
#
# While these terms may sound scary, in a short while they will all make
# sense. It’d be helpful to review the
# `cheatsheet <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N?usp=sharing>`__,
# before we start coding. We begin our tutorial with the concept of
# Environment.
#


######################################################################
# Environment
# ===========
#
# `Environment <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=OMfuO883blEq>`__
# is a key concept in reinforcement learning. It’s the world that Mario
# interacts with and learns from. Environment is characterized by
# `state <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=36WmEZ-8bn9M>`__.
# In Mario, this is the game console consisting of tubes, mushrooms and
# other components. When Mario makes an action, environment responds with
# a
# `reward <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=rm0EqRQqbo09>`__
# and the `next
# state <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=_SnLbEzua1pv>`__.
#
# Code for running the Mario environment:
#
# ::
#
#    # Initialize Super Mario environment
#    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
#    # Limit Mario action to be 1. walk right or 2. jump right
#    env = JoypadSpace(
#        env,
#        [['right'],
#        ['right', 'A']]
#    )
#    # Start environment
#    env.reset()
#    for _ in range(1000):
#      # Render game output
#      env.render()
#      # Choose random action
#      action = env.action_space.sample()
#      # Perform action
#      env.step(action=action)
#    # Close environment
#    env.close()
#


######################################################################
# Wrappers
# --------
#
# A lot of times we want to perform some pre-processing to the environment
# before we feed its data to the agent. This introduces the idea of a
# wrapper.
#
# A common wrapper is one that transforms RGB images to grayscale. This
# reduces the size of state representation without losing much
# information. For the agent, its behavior doesn’t change whether it lives
# in a RGB world or grayscale world!
#
# **before wrapper**
#
# .. figure:: https://drive.google.com/uc?id=1c9-tUWFyk4u_vNNrkZo1Rg0e2FUcbF3N
#    :alt: picture
#
#    picture
#
# **after wrapper**
#
# .. figure:: https://drive.google.com/uc?id=1ED9brgnbPmUZL43Bl_x2FDmXd-hsHBQt
#    :alt: picture
#
#    picture
#
# We apply a wrapper to environment in this fashion:
#
# ::
#
#    env = wrapper(env, **args)
#
# Instructions
# ~~~~~~~~~~~~
#
# We want to apply 3 built-in wrappers to the given ``env``,
# ``GrayScaleObservation``, ``ResizeObservation``, and ``FrameStack``.
#
# https://github.com/openai/gym/tree/master/gym/wrappers
#
# ``FrameStack`` is a wrapper that will allow us to squash consecutive
# frames of the environment into a single observation point to feed to our
# learning model. This way, we can differentiate between when Mario was
# landing or jumping based on his direction of movement in the previous
# several frames.
#
# Let’s use the following arguments: ``GrayScaleObservation``:
# keep_dim=False ``ResizeObservation``: shape=84 ``FrameStack``:
# num_stack=4
#

import gym
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym.spaces import Box
import cv2
import numpy as np

class ResizeObservation(gym.ObservationWrapper):
    """Downsample the image observation to a square image. """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation

# the original environment object
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

# TODO wrap the given env with GrayScaleObservation
env = GrayScaleObservation(env, keep_dim=False)
# TODO wrap the given env with ResizeObservation
env = ResizeObservation(env, shape=84)
# TODO wrap the given env with FrameStack
env = FrameStack(env, num_stack=4)


######################################################################
# Custom Wrapper
# --------------
#
# We also would like you to get a taste of implementing an environment
# wrapper on your own, instead of calling off-the-shelf packages.
#
# Here is an idea: to speed up training, we can skip some frames and only
# show every n-th frame. While some frames are skipped, it’s important to
# accumulate all the rewards from those skipped frames. Sum all
# intermediate rewards and return on the n-th frame.
#
# Instruction
# ~~~~~~~~~~~
#
# Our custom wrapper ``SkipFrame`` inherits from ``gym.Wrapper`` and we
# need to implement the ``step()`` function.
#
# During each skipped frames inside the for loop, accumulate ``reward`` to
# ``total_reward``, and break if any step gives ``done=True``.
#

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # TODO accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

env = SkipFrame(env)


######################################################################
# **Final Wrapped State**
#
# .. figure:: https://drive.google.com/uc?id=1zZU63qsuOKZIOwWt94z6cegOF2SMEmvD
#    :alt: picture
#
#    picture
#
# After applying the above wrappers to the environment, the final wrapped
# state consists of 4 gray-scaled consecutive frames stacked together, as
# shown above in the image on the left. Each time mario makes an action,
# the environment responds with a state of this structure. The structure
# is represented by a 3-D array of size = (4 \* 84 \* 84).
#


######################################################################
# Agent
# =====
#
# Let’s now turn to the other core concept in reinforcement learning:
# `agent <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=OMfuO883blEq>`__.
# Agent interacts with the environment by making
# `actions <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=chyu7AVObwWP>`__
# following its `action
# policy <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=_SnLbEzua1pv>`__.
# Let’s review the pseudo code on how agent interacts with the
# environment:
#
# ::
#
#    for a total of N episodes:
#    for a total of M steps in each episode:
#      Mario makes an action
#      Game gives a feedback
#      Mario remembers the action and feedback
#      after building up some experiences:
#        Mario learns from experiences
#


######################################################################
# We create a class, ``Mario``, to represent our agent in the game.
# ``Mario`` should be able to:
#
# -  Choose the action to take. ``Mario`` acts following its `optimal
#    action
#    policy <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=SZ313skqbSjQ>`__,
#    based on the current environment
#    `state <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=36WmEZ-8bn9M>`__.
#
# -  Remember experiences. The experience consists of current environment
#    state, current agent action, reward from environment and next
#    environment state. ``Mario`` later uses all these experience to
#    update its `action
#    policy <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=_SnLbEzua1pv>`__.
#
# -  Improve action policy over time. ``Mario`` updates its action policy
#    using
#    `Q-learning <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=bBny3BgNbcmh>`__.
#
# In following sections, we use Mario and agent interchangeably.
#

class Mario:
    def __init__(self, state_dim, action_dim):
        pass

    def act(self, state):
        """Given a state, choose an epsilon-greedy action
        """
        pass

    def remember(self, experience):
        """Add the observation to memory
        """
        pass

    def learn(self):
        """Update online action value (Q) function with a batch of experiences
        """
        pass



######################################################################
# Initialize
# ----------
#
# Before implementing any of the above functions, let’s define some key
# parameters.
#
# Instruction
# ~~~~~~~~~~~
#
# Initialize these key parameters inside ``__init__()``.
#
# ::
#
#    exploration_rate: float = 1.0
#
# Random Exploration Prabability. Under `some
# probability <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=_SnLbEzua1pv>`__,
# agent does not follow the `optimal action
# policy <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=SZ313skqbSjQ>`__,
# but instead chooses a random action to explore the environment. A high
# exploration rate is important at the early stage of learning to ensure
# proper exploration and not falling to local optima. The exploration rate
# should decrease as agent improves its policy.
#
# ::
#
#    exploration_rate_decay: float = 0.99999975
#
# Decay rate of ``exploration_rate``. Agent rigorously explores space at
# the early stage, but gradually reduces its exploration rate to maintain
# action quality. In the later stage, agent already learns a fairly good
# policy, so we want it to follow its policy more frequently. Decrease
# ``exploration_rate`` by the factor of ``exploration_rate_decay`` each
# time the agent acts.
#
# ::
#
#    exploration_rate_min: float = 0.1
#
# Minimum ``exploration_rate`` that Mario can decays into. Note that this
# value could either be ``0``, in which case Mario acts completely
# deterministiclly, or a very small number.
#
# ::
#
#    discount_factor: float = 0.9
#
# Future reward discount factor. This is :math:`\gamma` in the definition
# of
# `return <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=6lmIKuxsb6qu>`__.
# It serves to make agent give higher weight on the short-term rewards
# over future reward.
#
# ::
#
#    batch_size: int = 32
#
# Number of experiences used to update each time.
#
# ::
#
#    state_dim
#
# State space dimension. In Mario, this is 4 consecutive snapshots of the
# enviroment stacked together, where each snapshot is a 84*84 gray-scale
# image. This is passed in from the environment,
# ``self.state_dim = (4, 84, 84)``.
#
# ::
#
#    action_dim
#
# Action space dimension. In Mario, this is the number of total possible
# actions. This is passed in from environment as well.
#
# ::
#
#    memory
#
# ``memory`` is a queue structure filled with Mario’s past experiences.
# Each experience consists of (state, next_state, action, reward, done).
# As Mario collects more experiences, old experiences are popped to make
# room for most recent ones. We initialize the memory queue with
# ``maxlen=100000``.
#

from collections import deque

class Mario(object):
    def __init__(self, state_dim, action_dim):
       # state space dimension
      self.state_dim = state_dim
      # action space dimension
      self.action_dim = action_dim
      # replay buffer
      self.memory = deque(maxlen=100000)
      # current step, updated everytime the agent acts
      self.step = 0

      # TODO: Please initialize other variables as described above
      self.exploration_rate = 1.0
      self.exploration_rate_decay = 0.99999975
      self.exploration_rate_min = 0.1
      self.discount_factor = 0.9
      self.batch_size = 32



######################################################################
# Predict :math:`Q^*`
# -------------------
#
# `Optimal value action
# function <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=snRMrCIccEx8>`__,
# :math:`Q^*(s, a)`, is the single most important function in this
# project. ``Mario`` uses it to choose the `optimal
# action <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=SZ313skqbSjQ>`__
#
# .. math::
#
#
#    a^*(s) = argmax_{a}Q^*(s, a)
#
# and `update its action
# policy <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=bBny3BgNbcmh>`__
#
# .. math::
#
#
#    Q^*(s, a) \leftarrow Q^*(s, a)+\alpha (r + \gamma \max_{a'} Q^*(s', a') -Q^*(s, a))
#
# In this section, let’s implement ``agent.predict()`` to calculate
# :math:`Q^*(s, a)`.
#


######################################################################
# :math:`Q^*_{online}` and :math:`Q^*_{target}`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let’s review the inputs to :math:`Q^*(s, a)` function.
#
# :math:`s` is the observed state from environment. After our wrappers,
# :math:`s` is a stack of grayscale images. :math:`a` is a single integer
# representing the action taken. To deal with image/video signal, we often
# use a `convolution neural
# network <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`__.
# To save you time, we have created a simple ``ConvNet`` for you.
#
# Instead of passing action :math:`a` together with state :math:`s` into
# :math:`Q^*` function, we pass only the state. ``ConvNet`` returns a list
# of real values representing :math:`Q^*` for *all actions*. Later we can
# choose the :math:`Q^*` for any specific action :math:`a`.
#
# .. raw:: html
#
#    <!-- Let's now look at Q-learning more closely.
#
#    $$
#    Q^*(s, a) \leftarrow Q^*(s, a)+\alpha (r + \gamma \max_{a'} Q^*(s', a') -Q^*(s, a))
#    $$
#
#    $(r + \gamma \max_{a'} Q^*(s', a'))$ is the *TD target* (cheatsheet) and $Q^*(s, a)$ is the *TD estimate* (cheatsheet). $s$ and $a$ are the current state and action, and $s'$ and $a'$ are next state and next action.  -->
#
# In this section, we define two functions: :math:`Q_{online}` and
# :math:`Q_{target}`. *Both* represent the optimal value action function
# :math:`Q^*`. Intuitively, we use :math:`Q_{online}` to make action
# decisions, and :math:`Q_{target}` to improve :math:`Q_{online}`. We will
# explain further in details in `later
# sections <https://colab.research.google.com/drive/1kptUkdESbxBC-yOfSYngynjV5Hge_-t-#scrollTo=BOALqrSC5VIf>`__.
#
# Instructions
# ~~~~~~~~~~~~
#
# Use our provided ``ConvNet`` to define ``self.online_q`` and
# ``self.target_q`` separately. Intialize ``ConvNet`` with
# ``input_dim=self.state_dim`` and ``output_dim=self.action_dim`` for both
# :math:`Q^*` functions.
#

import torch.nn as nn

class ConvNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim):
        super(ConvNet, self).__init__()
        c, h, w = input_dim
        self.conv_1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(3136, 512)
        self.output = nn.Linear(512, output_dim)

    def forward(self, input):
        # input: B x C x H x W
        x = input
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.output(x)

        return x

class Mario(Mario):
    def __init__(self, state_dim, action_dim):
      super().__init__(state_dim, action_dim)
      # TODO: define online action value function
      self.online_q = ConvNet(input_dim=self.state_dim, output_dim=self.action_dim)
      # TODO: define target action value function
      self.target_q = ConvNet(input_dim=self.state_dim, output_dim=self.action_dim)


######################################################################
# Calling :math:`Q^*`
# ~~~~~~~~~~~~~~~~~~~
#
# Instruction
# ~~~~~~~~~~~
#
# Both ``self.online_q`` and ``self.target_q`` are optimal value action
# function :math:`Q^*`, which take a single input :math:`s`.
#
# Implement ``Mario.predict()`` to calculate the :math:`Q^*` of input
# :math:`s`. Here, :math:`s` is a batch of states, i.e.
#
# ::
#
#    shape(state) = batch_size, 4, 84, 84
#
# Return :math:`Q^*` for all possible actions for the entire batch of
# states.
#

class Mario(Mario):
      def predict(self, state, model):
        """Given a state, predict Q values of all possible actions using specified model (either online or target)
        Input:
          state
           dimension of (batch_size * state_dim)
          model
           either 'online' or 'target'
        Output
          pred_q_values (torch.tensor)
            dimension of (batch_size * action_dim), predicted Q values for all possible actions given the state
        """
        # LazyFrame -> np array -> torch tensor
        state_float = torch.FloatTensor(np.array(state))
        # normalize
        state_float = state_float / 255.

        if model == 'online':
          # TODO return the predicted Q values using self.online_q
          pred_q_values = self.online_q(state_float)
        elif model == 'target':
          # TODO return the predicted Q values using self.target_q
          pred_q_values = self.target_q(state_float)

        return pred_q_values


######################################################################
# Act
# ---
#
# Let’s now look at how Mario should ``act()`` in the environment.
#
# Given a state, Mario mostly `chooses the action with the highest Q
# value <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=SZ313skqbSjQ>`__.
# There is an *epislon* chance that Mario acts randomly instead, which
# encourages environment exploration.
#
# Instruction
# ~~~~~~~~~~~
#
# We will use ``torch.tensor`` and ``numpy.array`` in this section.
# Familiarize yourself with `basic syntax with some
# examples <https://colab.research.google.com/drive/1D8k6i-TIKfqEHVkzKwYMjJvZRAKe9IuH?usp=sharing>`__.
#
# We will now implement ``Mario.act()``. Recall that we have defined
# :math:`Q_{online}` above, which we will use here to calculate Q values
# for all actions given *state*. We then need to select the action that
# results in largest Q value. We have set up the logic for epsilon-greedy
# policy, and leave it to you to determine the optimal and random action.
#
# Before implementing ``Mario.act()``, let’s first get used to basic
# operations on *torch.tensor*, which is the data type returned in
# ``Mario.predict()``
#

class Mario(Mario):
    def act(self, state):
        """Given a state, choose an epsilon-greedy action and update value of step
        Input
          state(np.array)
            A single observation of the current state, dimension is (state_dim)
        Output
          action
            An integer representing which action agent will perform
        """
        if np.random.rand() < self.exploration_rate:
          # TODO: choose a random action from all possible actions (self.action_dim)
          action = np.random.randint(self.action_dim)
        else:
          state = np.expand_dims(state, 0)
          # TODO: choose the best action based on self.online_q
          action_values = self.predict(state, model='online')
          action = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        # increment step
        self.step += 1
        return action


######################################################################
# Remember
# --------
#
# In order to improve policy, Mario need to collect and save past
# experiences. Each time agent performs an action, it collects an
# experience which includes the current state, action it performs, the
# next state after performing the action, the reward it collected, and
# whether the game is finished or not.
#
# We use the ``self.memory`` defined above to store past experiences,
# consisting of (state, next_state, action, reward, done).
#
# Instruction
# ~~~~~~~~~~~
#
# Implement ``Mario.remember()`` to save the experience to Mario’s memory.
#

class Mario(Mario):
    def remember(self, experience):
        """Add the experience to self.memory
        Input
          experience =  (state, next_state, action, reward, done) tuple
        Output
          None
        """
        # TODO Add the experience to memory
        self.memory.append(experience)


######################################################################
# Learn
# -----
#
# The entire learning process is based on `Q-learning
# algorithm <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=bBny3BgNbcmh>`__.
# By learning, we mean updating our :math:`Q^*` function to better predict
# the optimal value of current state-action pair. We will use both
# :math:`Q^*_{online}` and :math:`Q^*_{target}` in this section.
#
# Some key steps to perform: - **Experience Sampling:** We will sample
# experiences from memory as the *training data* to update
# :math:`Q^*_{online}`.
#
# -  **Evaluating TD Estimate:** Calculate the `TD
#    estimate <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=2abP5k2kcRnn>`__
#    of sampled experiences, using current states and actions. We use
#    :math:`Q^*_{online}` in this step to directly predict
#    :math:`Q^*(s, a)`.
#
# -  **Evaluating TD Target:** Calculate the `TD
#    target <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=Q072-fLecSkb>`__
#    of sampled experiences, using next states and rewards. We use both
#    :math:`Q^*_{online}` and :math:`Q^*_{target}` to calculate
#    :math:`r + \gamma \max_{a'} Q^*_{target}(s', a')`, where the
#    :math:`\max_{a'}` part is determined by :math:`Q^*_{online}`.
#
# -  **Loss between TD Estimate and TD Target:** Calculate the mean
#    squared loss between TD estimate and TD target.
#
# -  **Updating :math:`Q^*_{online}`:** Perform an optimization step with
#    the above calculated loss to update :math:`Q^*_{online}`.
#
# Summarizing the above in pseudo code for ``Mario.learn()``:
#
# ::
#
#    if enough experiences are collected:
#      sample a batch of experiences
#      calculate the predicted Q values using Q_online
#      calculate the target Q values using Q_target and reward
#      calculate loss between prediction and target Q values
#      update Q_online based on loss
#


######################################################################
# Experience Sampling
# ~~~~~~~~~~~~~~~~~~~
#
# Mario learns by drawing past experiences from its memory. The memory is
# a queue data structure that stores each individual experience in the
# format of
#
# ::
#
#    state, next_state, action, reward, done
#
# Examples of some experiences in Mario’s memory:
#
# -  state: |pic| next_state: |pic| action: jump reward: 0.0 done: False
#
# -  state: |pic| next_state: |pic| action: right reward: -10.0 done: True
#
# -  state: |pic| next_state: |pic| action: right reward: -10.0 done: True
#
# -  state: |pic| next_state: |pic| action: jump_right reward: 0.0 done:
#    False
#
# -  state: |pic| next_state: |pic| action: right reward: 10.0 done: False
#
# State/next_state: Observation at timestep *t*/*t+1*. They are both of
# type ``LazyFrame``.
#
# Action: Mario’s action during state transition.
#
# Reward: Environment’s reward during state transition.
#
# Done: Boolean indicating if next_state is a terminal state (end of
# game). Terminal state has a known Q value of 0.
#
# Instruction
# -----------
#
# Sample a batch of experiences from ``self.memory`` of size
# ``self.batch_size``.
#
# Return a tuple of numpy arrays, in the order of (state, next_state,
# action, reward, done). Each numpy array should have its first dimension
# equal to ``self.batch_size``.
#
# To convert a ``LazyFrame`` to numpy array, do
#
# ::
#
#    state_np_array = np.array(state_lazy_frame)
#
# .. |pic| image:: https://drive.google.com/uc?id=1D34QpsmJSwHrdzszROt405ZwNY9LkTej
# .. |pic| image:: https://drive.google.com/uc?id=13j2TzRd1SGmFru9KJImZsY9DMCdqcr_J
# .. |pic| image:: https://drive.google.com/uc?id=1ByUKXf967Z6C9FBVtsn_QRnJTr9w-18v
# .. |pic| image:: https://drive.google.com/uc?id=1hmGGVO1cS7N7kdcM99-K3Y2sxrAFd0Oh
# .. |pic| image:: https://drive.google.com/uc?id=10MHERSI6lap79VcZfHtIzCS9qT45ksk-
# .. |pic| image:: https://drive.google.com/uc?id=1VFNOwQHGAf9pH_56_w0uRO4WUJTIXG90
# .. |pic| image:: https://drive.google.com/uc?id=1T6CAIMzNxeZlBTUdz3sB8t_GhDFbNdUO
# .. |pic| image:: https://drive.google.com/uc?id=1aZlA0EnspQdcSQcVxuVmaqPW_7jT3lfW
# .. |pic| image:: https://drive.google.com/uc?id=1bPRnGRx2c1HJ_0y_EEOFL5GOG8sUBdIo
# .. |pic| image:: https://drive.google.com/uc?id=1qtR4qCURBq57UCrmObM6A5-CH26NYaHv
#
#

import random

class Mario(Mario):
  def sample_batch(self):
    """
    Input
      self.memory (FIFO queue)
        a queue where each entry has five elements as below
        state: LazyFrame of dimension (state_dim)
        next_state: LazyFrame of dimension (state_dim)
        action: integer, representing the action taken
        reward: float, the reward from state to next_state with action
        done: boolean, whether state is a terminal state
      self.batch_size (int)
        size of the batch to return

    Output
      state, next_state, action, reward, done (tuple)
        a tuple of numpy arrays: state, next_state, action, reward, done
        state: numpy array of dimension (batch_size x state_dim)
        next_state: numpy array of dimension (batch_size x state_dim)
        action: numpy array of dimension (batch_size)
        reward: numpy array of dimension (batch_size)
        done: numpy array of dimension (batch_size)
    """
    # TODO convert everything into numpy array
    batch = random.sample(self.memory, self.batch_size)
    state, next_state, action, reward, done = map(np.array, zip(*batch))
    return state, next_state, action, reward, done


######################################################################
# TD Estimate
# ~~~~~~~~~~~
#
# `TD
# estimate <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=2abP5k2kcRnn>`__
# is the estimated :math:`Q^*(s, a)` based on *current state-action pair
# :math:`s, a`*.
#
# .. raw:: html
#
#    <!-- It represents the best estimate we have so far, and the goal is to keep updating it using TD target (link to Q learning equation) We will use $Q_{online}$ to calculate this.  -->
#
# Recall our defined ``Mario.predict()`` above:
#
# ::
#
#    q_values = self.predict(state, model='online')
#
# Instruction
# -----------
#
# Using our defined ``Mario.predict()`` above, calculate the *TD Estimate*
# of given ``state`` and ``action`` with ``online`` model. Return the
# results in ``torch.tensor`` format.
#
# Note that returned values from ``Mario.predict()`` are :math:`Q^*` for
# all actions. To locate :math:`Q^*` values for specific actions, use
# `tensor
# indexing <https://colab.research.google.com/drive/1D8k6i-TIKfqEHVkzKwYMjJvZRAKe9IuH?usp=sharing>`__.
#

class Mario(Mario):
  def calculate_prediction_q(self, state, action):
    """
    Input
      state (np.array)
        dimension is (batch_size x state_dim), each item is an observation
        for the current state
      action (np.array)
        dimension is (batch_size), each item is an integer representing the
        action taken for current state

    Output
      pred_q (torch.tensor)
        dimension of (batch_size), each item is a predicted Q value of the
        current state-action pair
    """
    curr_state_q = self.predict(state, model='online')
    # TODO select specific Q values based on input actions
    curr_state_q = curr_state_q[np.arange(0, self.batch_size), action]

    return curr_state_q


######################################################################
# TD Target
# ~~~~~~~~~
#
# `TD
# target <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=Q072-fLecSkb>`__
# is the estimated :math:`Q^*(s, a)` *based on next state-action pair
# :math:`s', a'` and reward :math:`r`*.
#
# *TD target* is in the form of
#
# .. math::
#
#
#    r + \gamma \max_{a'} Q^*(s', a')
#
# :math:`r` is the current reward, :math:`s'` is the next state,
# :math:`a'` is the next action.
#
# Caveats
# ~~~~~~~
#
# **Getting best next action**
#
# Because we don’t know what next action :math:`a'` will be, we estimate
# it using next state :math:`s'` and :math:`Q_{online}`. Specifically,
#
# .. math::
#
#
#    a' = argmax_a Q_{online}(s', a)
#
# That is, we apply :math:`Q_{online}` on the next_state :math:`s'`, and
# pick the action which will yield the largest Q value, and use that
# action to index into :math:`Q_{target}` to calculate *TD target* . This
# is why, if you compare the function signatures of ``calculate_target_q``
# and ``calculate_pred_q``, while in ``calculate_prediction_q`` we have
# ``action`` and ``state`` as an input parameter, in
# ``calculate_target_q`` we only have ``reward`` and ``next_state``.
#
# **Terminal state**
#
# Another small caveat is the terminal state, as recorded with the
# variable ``done``, which is 1 when Mario is dead or the game finishes.
#
# Hence, we need to make sure we don’t keep adding future rewards when
# “there is no future”, i.e. when the game reaches terminal state. Since
# ``done`` is a boolean, we can multiply ``1.0 - done`` with future
# reward. This way, future reward after the terminal state is not taken
# into account in TD target.
#
# Therefore, the complete *TD target* is in the form of
#
# .. math::
#
#
#    r + (1.0 - done) \gamma \max_{a'} Q^*_{target}(s', a')
#
#  where :math:`a'` is determined by
#
# .. math::
#
#
#    a' = argmax_a Q_{online}(s', a)
#
# Let’s calculate *TD Target* now.
#
# Instruction
# -----------
#
# For a batch of experiences consisting of next_states :math:`s'` and
# rewards :math:`r`, calculate the *TD target*. Note that :math:`a'` is
# not explicitly given, so we will need to first obtain that using
# :math:`Q_{online}` and next state :math:`s'`.
#
# Return the results in ``torch.tensor`` format.
#

class Mario(Mario):
  def calculate_target_q(self, next_state, reward):
    """
    Input
      next_state (np.array)
        dimension is (batch_size x state_dim), each item is an observation
        for the next state
      reward (np.array)
        dimension is (batch_size), each item is a float representing the
        reward collected from (state -> next state) transition

    Output
      target_q (torch.tensor)
        dimension of (batch_size), each item is a target Q value of the current
        state-action pair, calculated based on reward collected and
        estimated Q value for next state
    """
    next_state_q = self.predict(next_state, 'target')

    online_q = self.predict(next_state, 'online')
    # TODO select the best action at next state based on online Q function
    action_idx = torch.argmax(online_q, axis=1)

    # TODO calculate target Q values based on action_idx and reward
    target_q = torch.tensor(reward) + (1. - done) * next_state_q[np.arange(0, self.batch_size), action_idx] * self.gamma

    return target_q


######################################################################
# Loss
# ~~~~
#
# Let’s now calculate the loss between TD target and TD estimate. Loss is
# what drives the optimization and updates :math:`Q^*_{online}` to better
# predict :math:`Q^*` in the future. We will calculate the mean squared
# loss in the form of:
#
# :math:`MSE = \frac{1}{n}\sum_{i=0}^n( y_i - \hat{y}_i)^2`
#
# PyTorch already has an implementation of this loss:
#
# ::
#
#    loss = nn.functional.mse_loss(pred_q, target_q)
#
# Instruction
# -----------
#
# Given *TD Estimate* (``pred_q``) and *TD Target* (``target_q``) for the
# batch of experiences, return the Mean Squared Error.
#

import torch.nn as nn

class Mario(Mario):
  def calculate_loss(self, pred_q, target_q):
    """
    Input
      pred_q (torch.tensor)
        dimension is (batch_size), each item is an observation
        for the next state
      target_q (torch.tensor)
        dimension is (batch_size), each item is a float representing the
        reward collected from (state -> next state) transition

    Output
      loss (torch.tensor)
        a single value representing the MSE loss of pred_q and target_q
    """
    # TODO calculate mean squared error loss
    loss = nn.functional.mse_loss(pred_q, target_q)
    return loss


######################################################################
# Update :math:`Q^*_{online}`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As the final step to complete ``Mario.learn()``, we use Adam optimizer
# to optimize upon the above calculated ``loss``. This updates the
# parameters inside :math:`Q^*_{online}` function so that TD estimate gets
# closer to TD target.
#
# You’ve coded a lot so far. We got this section covered for you.
#

import torch

class Mario(Mario):
  def __init__(self, state_dim, action_dim):
    super().__init__(state_dim, action_dim)
    # optimizer updates parameters in online_q using backpropagation
    self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=0.00025)

  def update_online_q(self, loss):
    '''
    Input
      loss (torch.tensor)
        a single value representing the Huber loss of pred_q and target_q
      optimizer
        optimizer updates parameter in our online_q neural network to reduce
        the loss
    '''
    # update online_q
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()




######################################################################
# Update :math:`Q^*_{target}`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We need to sync :math:`Q^*_{target}` with :math:`Q^*_{online}` every
# once in a while, to make sure our :math:`Q^*_{target}` is up-to-date. We
# use ``self.copy_every`` to control how often we do the sync-up.
#

class Mario(Mario):
    def sync_target_q(self):
      """Update target action value (Q) function with online action value (Q) function
      """
      self.target_q.load_state_dict(self.online_q.state_dict())


######################################################################
# Put them Together
# ~~~~~~~~~~~~~~~~~
#
# With all the helper methods implemented, let’s revisit our
# ``Mario.learn()`` function.
#
# Instructions
# ~~~~~~~~~~~~
#
# We’ve added some logic on checking learning criterion. For the rest, use
# the helper methods defined above to complete ``Mario.learn()`` function.
#

import os
import datetime

class Mario(Mario):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        # number of experiences to collect before training
        self.burnin = 1e5
        # number of experiences between updating online q
        self.learn_every = 3
        # number of experiences between updating target q with online q
        self.sync_every = 1e4
        # number of experiences between saving the current agent
        self.save_every = 1e5
        self.save_dir = os.path.join(
            "checkpoints",
            f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_model(self):
        """Save the current agent
        """
        save_path = os.path.join(self.save_dir, f"online_q_{self.step}.chkpt")
        torch.save(self.online_q.state_dict(), save_path)


    def learn(self):
        """Update prediction action value (Q) function with a batch of experiences
        """
        # sync target network
        if self.step % self.sync_every == 0:
            self.sync_target_q()
        # checkpoint model
        if self.step % self.save_every == 0:
            self.save_model()
        # break if burn-in
        if self.step < self.burnin:
            return
        # break if no training
        if self.step % self.learn_every != 0:
            return

        # TODO: sample a batch of experiences from self.memory
        state, next_state, action, reward, done = self.sample_batch()

        # TODO: calculate prediction Q values for the batch
        pred_q = self.calculate_prediction_q(state, action)

        # TODO: calculate target Q values for the batch
        target_q = self.calculate_target_q(next_state, reward)

        # TODO: calculate huber loss of target and prediction values
        loss = self.calculate_loss(pred_q, target_q)

        # TODO: update target network
        self.update_online_q(loss)
        print('udpating')



######################################################################
# Start Learning!
# ===============
#
# With the agent and environment wrappers implemented, we are ready to put
# Mario in the game and start learning! We will wrap the learning process
# in a big ``for`` loop that repeats the process of acting, remembering
# and learning by Mario.
#
# The meat of the algorithm is in the loop, let’s take a closer look:
#
# Instruction
# ~~~~~~~~~~~
#
# 1. At the beginning of a new episode, we need to reinitialize the
#    ``state`` by calling ``env.reset()``
#
# 2. Then we need several variables to hold the logging information we
#    collected in this episode:
#
# -  ``ep_reward``: reward collected in this episode
# -  ``ep_length``: total length of this episode
#
# 3. Now we are inside the while loop that plays the game, and we can call
#    ``env.render()`` to display the visual
#
# 4. We want to act by calling ``Mario.act(state)`` now. Remember our
#    action follows the `action
#    policy <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=SZ313skqbSjQ>`__,
#    which is determined by :math:`Q^*_{online}`.
#
# 5. Perform the above selected action on env by calling
#    ``env.step(action)``. Collect the environment feedback: next_state,
#    reward, if Mario is dead (done) and info.
#
# 6. Store the current experience into Mario’s memory, by calling
#    ``Mario.remember(exp)``.
#
# 7. Learn by drawing experiences from Mario’s memory and update the
#    action policy, by calling ``Mario.learn()``.
#
# 8. Update logging info.
#
# 9. Update state to prepare for next step.
#

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n)
episodes = 10

### for Loop that train the model num_episodes times by playing the game

for e in range(episodes):

    # 1. Reset env/restart the game
    state = env.reset()

    # 2. Logging
    ep_reward = 0.0
    ep_length = 0

    # Play the game!
    while True:

        # 3. Show environment (the visual)

        # 4. Run agent on the state
        action = mario.act(state)

        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)

        # 6. Remember
        mario.remember(experience=(state, next_state, action, reward, done))

        # 7. Learn
        mario.learn()

        # 8. Logging
        ep_reward += reward
        ep_length += 1

        # 9. Update state
        state = next_state

        # If done break loop
        if done or info['flag_get']:
            print(f"episode length: {ep_length}, reward: {ep_reward}")
            break


######################################################################
# Discussion
# ==========
#


######################################################################
# Off-policy
# ----------
#
# Two major categories of RL algorithms are on-policy and off-policy. The
# algorithm we used, Q-learning, is an example of off-policy algorithm.
#
# What this means is that the experiences that Mario learns from, do not
# need to be generated from the current action policy. Mario is able to
# learn from very distant memory that are generated with an outdated
# action policy. In our case, how *distant* this memory could extend to is
# decided by ``Mario.max_memory``.
#
# On-policy algorithm, on the other hand, requires that Mario learns from
# fresh experiences generated with current action policy. Examples include
# `policy gradient
# method <https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html>`__.
#
# **Why do we want to sample data points from all past experiences rather
# than the most recent ones(for example, from the previous episode), which
# are newly trained with higher accuracy?**
#
# The intuition is behind the tradeoff between these two approaches:
#
# Do we want to train on data that are generated from a small-size dataset
# with relatively high quality or a huge-size dataset with relatively
# lower quality?
#
# The answer is the latter, because the more data we have, the more of a
# wholistic, comprehensive point of view we have on the overall behavior
# of the system we have, in our case, the Mario game. Limited size dataset
# has the danger of overfitting and overlooking bigger pictures of the
# entire action/state space.
#
# Remember, Reinforcement Learning is all about exploring different
# scenarios(state) and keeping improving based on trial and errors,
# generated from the interactions between the **agent**\ (action) and the
# **environmental feedback**\ (reward).
#


######################################################################
# Why two :math:`Q^*` functions?
# ------------------------------
#
# We defined two :math:`Q^*` functions, :math:`Q^*_{online}` and
# :math:`Q^*_{target}`. Both represent the exact same thing: `optimal
# value action
# function <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=snRMrCIccEx8>`__.
# We use :math:`Q^*_{online}` in the `TD
# estimate <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=2abP5k2kcRnn>`__
# and :math:`Q^*_{target}` in the `TD
# target <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=Q072-fLecSkb>`__.
# This is to prevent the optimization divergence during Q-learning update:
#
# .. math::
#
#
#    Q^*(s, a) \leftarrow Q^*(s, a)+\alpha (r + \gamma \max_{a'} Q^*(s', a') -Q^*(s, a))
#
# where 1st, 2nd and 4th :math:`Q^*(s, a)` are using :math:`Q^*_{online}`,
# and 3rd :math:`Q^*(s', a')` is using :math:`Q^*_{target}`.
#
