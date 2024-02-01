# -*- coding: utf-8 -*-

"""
Pendulum: Writing your environment and transforms with TorchRL
==============================================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

Creating an environment (a simulator or an interface to a physical control system)
is an integrative part of reinforcement learning and control engineering.

TorchRL provides a set of tools to do this in multiple contexts.
This tutorial demonstrates how to use PyTorch and TorchRL code a pendulum
simulator from the ground up.
It is freely inspired by the Pendulum-v1 implementation from `OpenAI-Gym/Farama-Gymnasium
control library <https://github.com/Farama-Foundation/Gymnasium>`__.

.. figure:: /_static/img/pendulum.gif
   :alt: Pendulum
   :align: center

   Simple Pendulum

Key learnings:

- How to design an environment in TorchRL:
  - Writing specs (input, observation and reward);
  - Implementing behavior: seeding, reset and step.
- Transforming your environment inputs and outputs, and writing your own
  transforms;
- How to use :class:`~tensordict.TensorDict` to carry arbitrary data structures 
  through the ``codebase``.

  In the process, we will touch three crucial components of TorchRL:

* `environments <https://pytorch.org/rl/reference/envs.html>`__
* `transforms <https://pytorch.org/rl/reference/envs.html#transforms>`__
* `models (policy and value function) <https://pytorch.org/rl/reference/modules.html>`__

"""

######################################################################
# To give a sense of what can be achieved with TorchRL's environments, we will
# be designing a *stateless* environment. While stateful environments keep track of
# the latest physical state encountered and rely on this to simulate the state-to-state
# transition, stateless environments expect the current state to be provided to
# them at each step, along with the action undertaken. TorchRL supports both
# types of environments, but stateless environments are more generic and hence
# cover a broader range of features of the environment API in TorchRL.
#
# Modeling stateless environments gives users full control over the input and
# outputs of the simulator: one can reset an experiment at any stage or actively
# modify the dynamics from the outside. However, it assumes that we have some control
# over a task, which may not always be the case: solving a problem where we cannot
# control the current state is more challenging but has a much wider set of applications.
#
# Another advantage of stateless environments is that they can enable
# batched execution of transition simulations. If the backend and the
# implementation allow it, an algebraic operation can be executed seamlessly on
# scalars, vectors, or tensors. This tutorial gives such examples.
#
# This tutorial will be structured as follows:
#
# * We will first get acquainted with the environment properties:
#   its shape (``batch_size``), its methods (mainly :meth:`~torchrl.envs.EnvBase.step`,
#   :meth:`~torchrl.envs.EnvBase.reset` and :meth:`~torchrl.envs.EnvBase.set_seed`)
#   and finally its specs.
# * After having coded our simulator, we will demonstrate how it can be used
#   during training with transforms.
# * We will explore new avenues that follow from the TorchRL's API,
#   including: the possibility of transforming inputs, the vectorized execution
#   of the simulation and the possibility of backpropagation through the
#   simulation graph.
# * Finally, we will train a simple policy to solve the system we implemented.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
from torch import multiprocessing

# TorchRL prefers spawn method, that restricts creation of  ``~torchrl.envs.ParallelEnv`` inside
# `__main__` method call, but for the easy of reading the code switch to fork
# which is also a default spawn method in Google's Colaboratory
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

# sphinx_gallery_end_ignore

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

######################################################################
# There are four things you must take care of when designing a new environment
# class:
#
# * :meth:`EnvBase._reset`, which codes for the resetting of the simulator
#   at a (potentially random) initial state;
# * :meth:`EnvBase._step` which codes for the state transition dynamic;
# * :meth:`EnvBase._set_seed`` which implements the seeding mechanism;
# * the environment specs.
#
# Let us first describe the problem at hand: we would like to model a simple
# pendulum over which we can control the torque applied on its fixed point.
# Our goal is to place the pendulum in upward position (angular position at 0
# by convention) and having it standing still in that position.
# To design our dynamic system, we need to define two equations: the motion
# equation following an action (the torque applied) and the reward equation
# that will constitute our objective function.
#
# For the motion equation, we will update the angular velocity following:
#
# .. math::
#
#    \dot{\theta}_{t+1} = \dot{\theta}_t + (3 * g / (2 * L) * \sin(\theta_t) + 3 / (m * L^2) * u) * dt
#
# where :math:`\dot{\theta}` is the angular velocity in rad/sec, :math:`g` is the
# gravitational force, :math:`L` is the pendulum length, :math:`m` is its mass,
# :math:`\theta` is its angular position and :math:`u` is the torque. The
# angular position is then updated according to
#
# .. math::
#
#    \theta_{t+1} = \theta_{t} + \dot{\theta}_{t+1} dt
#
# We define our reward as
#
# .. math::
#
#    r = -(\theta^2 + 0.1 * \dot{\theta}^2 + 0.001 * u^2)
#
# which will be maximized when the angle is close to 0 (pendulum in upward
# position), the angular velocity is close to 0 (no motion) and the torque is
# 0 too.
#
# Coding the effect of an action: :func:`~torchrl.envs.EnvBase._step`
# -------------------------------------------------------------------
#
# The step method is the first thing to consider, as it will encode
# the simulation that is of interest to us. In TorchRL, the
# :class:`~torchrl.envs.EnvBase` class has a :meth:`EnvBase.step`
# method that receives a :class:`tensordict.TensorDict`
# instance with an ``"action"`` entry indicating what action is to be taken.
#
# To facilitate the reading and writing from that ``tensordict`` and to make sure
# that the keys are consistent with what's expected from the library, the
# simulation part has been delegated to a private abstract method :meth:`_step`
# which reads input data from a ``tensordict``, and writes a *new*  ``tensordict``
# with the output data.
#
# The :func:`_step` method should do the following:
#
#   1. Read the input keys (such as ``"action"``) and execute the simulation
#      based on these;
#   2. Retrieve observations, done state and reward;
#   3. Write the set of observation values along with the reward and done state
#      at the corresponding entries in a new :class:`TensorDict`.
#
# Next, the :meth:`~torchrl.envs.EnvBase.step` method will merge the output
# of :meth:`~torchrl.envs.EnvBase.step` in the input ``tensordict`` to enforce
# input/output consistency.
#
# Typically, for stateful environments, this will look like this:
#
# .. code-block::
#
#   >>> policy(env.reset())
#   >>> print(tensordict)
#   TensorDict(
#       fields={
#           action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
#           done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
#           observation: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
#       batch_size=torch.Size([]),
#       device=cpu,
#       is_shared=False)
#   >>> env.step(tensordict)
#   >>> print(tensordict)
#   TensorDict(
#       fields={
#           action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
#           done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
#           next: TensorDict(
#               fields={
#                   done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
#                   observation: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
#                   reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False)},
#               batch_size=torch.Size([]),
#               device=cpu,
#               is_shared=False),
#           observation: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
#       batch_size=torch.Size([]),
#       device=cpu,
#       is_shared=False)
#
# Notice that the root ``tensordict`` has not changed, the only modification is the
# appearance of a new ``"next"`` entry that contains the new information.
#
# In the Pendulum example, our :meth:`_step` method will read the relevant
# entries from the input ``tensordict`` and compute the position and velocity of
# the pendulum after the force encoded by the ``"action"`` key has been applied
# onto it. We compute the new angular position of the pendulum
# ``"new_th"`` as the result of the previous position ``"th"`` plus the new
# velocity ``"new_thdot"`` over a time interval ``dt``.
#
# Since our goal is to turn the pendulum up and maintain it still in that
# position, our ``cost`` (negative reward) function is lower for positions
# close to the target and low speeds.
# Indeed, we want to discourage positions that are far from being "upward"
# and/or speeds that are far from 0.
#
# In our example, :meth:`EnvBase._step` is encoded as a static method since our
# environment is stateless. In stateful settings, the ``self`` argument is
# needed as the state needs to be read from the environment.
#


def _step(tensordict):
    th, thdot = tensordict["th"], tensordict["thdot"]  # th := theta

    g_force = tensordict["params", "g"]
    mass = tensordict["params", "m"]
    length = tensordict["params", "l"]
    dt = tensordict["params", "dt"]
    u = tensordict["action"].squeeze(-1)
    u = u.clamp(-tensordict["params", "max_torque"], tensordict["params", "max_torque"])
    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

    new_thdot = (
        thdot
        + (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length**2) * u) * dt
    )
    new_thdot = new_thdot.clamp(
        -tensordict["params", "max_speed"], tensordict["params", "max_speed"]
    )
    new_th = th + new_thdot * dt
    reward = -costs.view(*tensordict.shape, 1)
    done = torch.zeros_like(reward, dtype=torch.bool)
    out = TensorDict(
        {
            "th": new_th,
            "thdot": new_thdot,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
    )
    return out


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


######################################################################
# Resetting the simulator: :func:`~torchrl.envs.EnvBase._reset`
# -------------------------------------------------------------
#
# The second method we need to care about is the
# :meth:`~torchrl.envs.EnvBase._reset` method. Like
# :meth:`~torchrl.envs.EnvBase._step`, it should write the observation entries
# and possibly a done state in the ``tensordict`` it outputs (if the done state is
# omitted, it will be filled as ``False`` by the parent method
# :meth:`~torchrl.envs.EnvBase.reset`). In some contexts, it is required that
# the ``_reset`` method receives a command from the function that called
# it (for example, in multi-agent settings we may want to indicate which agents need
# to be reset). This is why the :meth:`~torchrl.envs.EnvBase._reset` method
# also expects a ``tensordict`` as input, albeit it may perfectly be empty or
# ``None``.
#
# The parent :meth:`EnvBase.reset` does some simple checks like the
# :meth:`EnvBase.step` does, such as making sure that a ``"done"`` state
# is returned in the output ``tensordict`` and that the shapes match what is
# expected from the specs.
#
# For us, the only important thing to consider is whether
# :meth:`EnvBase._reset` contains all the expected observations. Once more,
# since we are working with a stateless environment, we pass the configuration
# of the pendulum in a nested ``tensordict`` named ``"params"``.
#
# In this example, we do not pass a done state as this is not mandatory
# for :meth:`_reset` and our environment is non-terminating, so we always
# expect it to be ``False``.
#


def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        # if no ``tensordict`` is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input ``tensordict`` contains all the relevant
        # parameters to get started.
        tensordict = self.gen_params(batch_size=self.batch_size)

    high_th = torch.tensor(DEFAULT_X, device=self.device)
    high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
    low_th = -high_th
    low_thdot = -high_thdot

    # for non batch-locked environments, the input ``tensordict`` shape dictates the number
    # of simulators run simultaneously. In other contexts, the initial
    # random state's shape will depend upon the environment batch-size instead.
    th = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_th - low_th)
        + low_th
    )
    thdot = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_thdot - low_thdot)
        + low_thdot
    )
    out = TensorDict(
        {
            "th": th,
            "thdot": thdot,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    return out


######################################################################
# Environment metadata: ``env.*_spec``
# ------------------------------------
#
# The specs define the input and output domain of the environment.
# It is important that the specs accurately define the tensors that will be
# received at runtime, as they are often used to carry information about
# environments in multiprocessing and distributed settings. They can also be
# used to instantiate lazily defined neural networks and test scripts without
# actually querying the environment (which can be costly with real-world
# physical systems for instance).
#
# There are four specs that we must code in our environment:
#
# * :obj:`EnvBase.observation_spec`: This will be a :class:`~torchrl.data.CompositeSpec`
#   instance where each key is an observation (a :class:`CompositeSpec` can be
#   viewed as a dictionary of specs).
# * :obj:`EnvBase.action_spec`: It can be any type of spec, but it is required
#   that it corresponds to the ``"action"`` entry in the input ``tensordict``;
# * :obj:`EnvBase.reward_spec`: provides information about the reward space;
# * :obj:`EnvBase.done_spec`: provides information about the space of the done
#   flag.
#
# TorchRL specs are organized in two general containers: ``input_spec`` which
# contains the specs of the information that the step function reads (divided
# between ``action_spec`` containing the action and ``state_spec`` containing
# all the rest), and ``output_spec`` which encodes the specs that the
# step outputs (``observation_spec``, ``reward_spec`` and ``done_spec``).
# In general, you should not interact directly with ``output_spec`` and
# ``input_spec`` but only with their content: ``observation_spec``,
# ``reward_spec``, ``done_spec``, ``action_spec`` and ``state_spec``.
# The reason if that the specs are organized in a non-trivial way
# within ``output_spec`` and
# ``input_spec`` and neither of these should be directly modified.
#
# In other words, the ``observation_spec`` and related properties are
# convenient shortcuts to the content of the output and input spec containers.
#
# TorchRL offers multiple :class:`~torchrl.data.TensorSpec`
# `subclasses <https://pytorch.org/rl/reference/data.html#tensorspec>`_ to
# encode the environment's input and output characteristics.
#
# Specs shape
# ^^^^^^^^^^^
#
# The environment specs leading dimensions must match the
# environment batch-size. This is done to enforce that every component of an
# environment (including its transforms) have an accurate representation of
# the expected input and output shapes. This is something that should be
# accurately coded in stateful settings.
#
# For non batch-locked environments, such as the one in our example (see below),
# this is irrelevant as the environment batch size will most likely be empty.
#


def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = CompositeSpec(
        th=BoundedTensorSpec(
            low=-torch.pi,
            high=torch.pi,
            shape=(),
            dtype=torch.float32,
        ),
        thdot=BoundedTensorSpec(
            low=-td_params["params", "max_speed"],
            high=td_params["params", "max_speed"],
            shape=(),
            dtype=torch.float32,
        ),
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    self.action_spec = BoundedTensorSpec(
        low=-td_params["params", "max_torque"],
        high=td_params["params", "max_torque"],
        shape=(1,),
        dtype=torch.float32,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


######################################################################
# Reproducible experiments: seeding
# ---------------------------------
#
# Seeding an environment is a common operation when initializing an experiment.
# The only goal of :func:`EnvBase._set_seed` is to set the seed of the contained
# simulator. If possible, this operation should not call ``reset()`` or interact
# with the environment execution. The parent :func:`EnvBase.set_seed` method
# incorporates a mechanism that allows seeding multiple environments with a
# different pseudo-random and reproducible seed.
#


def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng


######################################################################
# Wrapping things together: the :class:`~torchrl.envs.EnvBase` class
# ------------------------------------------------------------------
#
# We can finally put together the pieces and design our environment class.
# The specs initialization needs to be performed during the environment
# construction, so we must take care of calling the :func:`_make_spec` method
# within :func:`PendulumEnv.__init__`.
#
# We add a static method :meth:`PendulumEnv.gen_params` which deterministically
# generates a set of hyperparameters to be used during execution:
#


def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_speed": 8,
                    "max_torque": 2.0,
                    "dt": 0.05,
                    "g": g,
                    "m": 1.0,
                    "l": 1.0,
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


######################################################################
# We define the environment as non-``batch_locked`` by turning the ``homonymous``
# attribute to ``False``. This means that we will **not** enforce the input
# ``tensordict`` to have a ``batch-size`` that matches the one of the environment.
#
# The following code will just put together the pieces we have coded above.
#


class PendulumEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed


######################################################################
# Testing our environment
# -----------------------
#
# TorchRL provides a simple function :func:`~torchrl.envs.utils.check_env_specs`
# to check that a (transformed) environment has an input/output structure that
# matches the one dictated by its specs.
# Let us try it out:
#

env = PendulumEnv()
check_env_specs(env)

######################################################################
# We can have a look at our specs to have a visual representation of the environment
# signature:
#

print("observation_spec:", env.observation_spec)
print("state_spec:", env.state_spec)
print("reward_spec:", env.reward_spec)

######################################################################
# We can execute a couple of commands too to check that the output structure
# matches what is expected.

td = env.reset()
print("reset tensordict", td)

######################################################################
# We can run the :func:`env.rand_step` to generate
# an action randomly from the ``action_spec`` domain. A ``tensordict`` containing
# the hyperparameters and the current state **must** be passed since our
# environment is stateless. In stateful contexts, ``env.rand_step()`` works
# perfectly too.
#
td = env.rand_step(td)
print("random step tensordict", td)

######################################################################
# Transforming an environment
# ---------------------------
#
# Writing environment transforms for stateless simulators is slightly more
# complicated than for stateful ones: transforming an output entry that needs
# to be read at the following iteration requires to apply the inverse transform
# before calling :func:`meth.step` at the next step.
# This is an ideal scenario to showcase all the features of TorchRL's
# transforms!
#
# For instance, in the following transformed environment we ``unsqueeze`` the entries
# ``["th", "thdot"]`` to be able to stack them along the last
# dimension. We also pass them as ``in_keys_inv`` to squeeze them back to their
# original shape once they are passed as input in the next iteration.
#
env = TransformedEnv(
    env,
    # ``Unsqueeze`` the observations that we will concatenate
    UnsqueezeTransform(
        unsqueeze_dim=-1,
        in_keys=["th", "thdot"],
        in_keys_inv=["th", "thdot"],
    ),
)

######################################################################
# Writing custom transforms
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TorchRL's transforms may not cover all the operations one wants to execute
# after an environment has been executed.
# Writing a transform does not require much effort. As for the environment
# design, there are two steps in writing a transform:
#
# - Getting the dynamics right (forward and inverse);
# - Adapting the environment specs.
#
# A transform can be used in two settings: on its own, it can be used as a
# :class:`~torch.nn.Module`. It can also be used appended to a
# :class:`~torchrl.envs.transforms.TransformedEnv`. The structure of the class allows to
# customize the behavior in the different contexts.
#
# A :class:`~torchrl.envs.transforms.Transform` skeleton can be summarized as follows:
#
# .. code-block::
#
#   class Transform(nn.Module):
#       def forward(self, tensordict):
#           ...
#       def _apply_transform(self, tensordict):
#           ...
#       def _step(self, tensordict):
#           ...
#       def _call(self, tensordict):
#           ...
#       def inv(self, tensordict):
#           ...
#       def _inv_apply_transform(self, tensordict):
#           ...
#
# There are three entry points (:func:`forward`, :func:`_step` and :func:`inv`)
# which all receive :class:`tensordict.TensorDict` instances. The first two
# will eventually go through the keys indicated by :obj:`~tochrl.envs.transforms.Transform.in_keys`
# and call :meth:`~torchrl.envs.transforms.Transform._apply_transform` to each of these. The results will
# be written in the entries pointed by :obj:`Transform.out_keys` if provided
# (if not the ``in_keys`` will be updated with the transformed values).
# If inverse transforms need to be executed, a similar data flow will be
# executed but with the :func:`Transform.inv` and
# :func:`Transform._inv_apply_transform` methods and across the ``in_keys_inv``
# and ``out_keys_inv`` list of keys.
# The following figure summarized this flow for environments and replay
# buffers.
#
#
#    Transform API
#
# In some cases, a transform will not work on a subset of keys in a unitary
# manner, but will execute some operation on the parent environment or
# work with the entire input ``tensordict``.
# In those cases, the :func:`_call` and :func:`forward` methods should be
# re-written, and the :func:`_apply_transform` method can be skipped.
#
# Let us code new transforms that will compute the ``sine`` and ``cosine``
# values of the position angle, as these values are more useful to us to learn
# a policy than the raw angle value:


class SinTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.sin()

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


class CosTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.cos()

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


t_sin = SinTransform(in_keys=["th"], out_keys=["sin"])
t_cos = CosTransform(in_keys=["th"], out_keys=["cos"])
env.append_transform(t_sin)
env.append_transform(t_cos)

######################################################################
# Concatenates the observations onto an "observation" entry.
# ``del_keys=False`` ensures that we keep these values for the next
# iteration.
cat_transform = CatTensors(
    in_keys=["sin", "cos", "thdot"], dim=-1, out_key="observation", del_keys=False
)
env.append_transform(cat_transform)

######################################################################
# Once more, let us check that our environment specs match what is received:
check_env_specs(env)

######################################################################
# Executing a rollout
# -------------------
#
# Executing a rollout is a succession of simple steps:
#
# * reset the environment
# * while some condition is not met:
#
#   * compute an action given a policy
#   * execute a step given this action
#   * collect the data
#   * make a ``MDP`` step
#
# * gather the data and return
#
# These operations have been conveniently wrapped in the :meth:`~torchrl.envs.EnvBase.rollout`
# method, from which we provide a simplified version here below.


def simple_rollout(steps=100):
    # preallocate:
    data = TensorDict({}, [steps])
    # reset
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand()
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data


print("data from rollout:", simple_rollout(100))

######################################################################
# Batching computations
# ---------------------
#
# The last unexplored end of our tutorial is the ability that we have to
# batch computations in TorchRL. Because our environment does not
# make any assumptions regarding the input data shape, we can seamlessly
# execute it over batches of data. Even better: for non-batch-locked
# environments such as our Pendulum, we can change the batch size on the fly
# without recreating the environment.
# To do this, we just generate parameters with the desired shape.
#

batch_size = 10  # number of environments to be executed in batch
td = env.reset(env.gen_params(batch_size=[batch_size]))
print("reset (batch size of 10)", td)
td = env.rand_step(td)
print("rand step (batch size of 10)", td)

######################################################################
# Executing a rollout with a batch of data requires us to reset the environment
# out of the rollout function, since we need to define the batch_size
# dynamically and this is not supported by :meth:`~torchrl.envs.EnvBase.rollout`:
#

rollout = env.rollout(
    3,
    auto_reset=False,  # we're executing the reset out of the ``rollout`` call
    tensordict=env.reset(env.gen_params(batch_size=[batch_size])),
)
print("rollout of len 3 (batch size of 10):", rollout)


######################################################################
# Training a simple policy
# ------------------------
#
# In this example, we will train a simple policy using the reward as a
# differentiable objective, such as a negative loss.
# We will take advantage of the fact that our dynamic system is fully
# differentiable to backpropagate through the trajectory return and adjust the
# weights of our policy to maximize this value directly. Of course, in many
# settings many of the assumptions we make do not hold, such as
# differentiable system and full access to the underlying mechanics.
#
# Still, this is a very simple example that showcases how a training loop can
# be coded with a custom environment in TorchRL.
#
# Let us first write the policy network:
#
torch.manual_seed(0)
env.set_seed(0)

net = nn.Sequential(
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(1),
)
policy = TensorDictModule(
    net,
    in_keys=["observation"],
    out_keys=["action"],
)

######################################################################
# and our optimizer:
#

optim = torch.optim.Adam(policy.parameters(), lr=2e-3)

######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# We will successively:
#
# * generate a trajectory
# * sum the rewards
# * backpropagate through the graph defined by these operations
# * clip the gradient norm and make an optimization step
# * repeat
#
# At the end of the training loop, we should have a final reward close to 0
# which demonstrates that the pendulum is upward and still as desired.
#
batch_size = 32
pbar = tqdm.tqdm(range(20_000 // batch_size))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20_000)
logs = defaultdict(list)

for _ in pbar:
    init_td = env.reset(env.gen_params(batch_size=[batch_size]))
    rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
    traj_return = rollout["next", "reward"].mean()
    (-traj_return).backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    pbar.set_description(
        f"reward: {traj_return: 4.4f}, "
        f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
    )
    logs["return"].append(traj_return.item())
    logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
    scheduler.step()


def plot():
    import matplotlib
    from matplotlib import pyplot as plt

    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    with plt.ion():
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(logs["return"])
        plt.title("returns")
        plt.xlabel("iteration")
        plt.subplot(1, 2, 2)
        plt.plot(logs["last_reward"])
        plt.title("last reward")
        plt.xlabel("iteration")
        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        plt.show()


plot()


######################################################################
# Conclusion
# ----------
#
# In this tutorial, we have learned how to code a stateless environment from
# scratch. We touched the subjects of:
#
# * The four essential components that need to be taken care of when coding
#   an environment (``step``, ``reset``, seeding and building specs).
#   We saw how these methods and classes interact with the
#   :class:`~tensordict.TensorDict` class;
# * How to test that an environment is properly coded using
#   :func:`~torchrl.envs.utils.check_env_specs`;
# * How to append transforms in the context of stateless environments and how
#   to write custom transformations;
# * How to train a policy on a fully differentiable simulator.
#
