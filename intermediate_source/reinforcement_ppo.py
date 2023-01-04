# -*- coding: utf-8 -*-
"""
Reinforcement Learning (PPO) with TorchRL Tutorial
==================================================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

This tutorial shows how to use PyTorch and TorchRL to train a parametric policy
network to solve the cheetah-run task from the `DeepMind control library
<https://github.com/deepmind/dm_control>`__.

Key learning items:
- What is PPO and how to code the loss;
- How to create an environment in TorchRL, transform its outputs, and collect data from this env;
- How to compute the advantage signal for policy gradient methods;
- How to create a stochastic policy using a probabilistic neural network;
- How to create a dynamic replay buffer and sample from it without repetition.

.. code-block:: bash

   %%bash
   pip3 install torchrl
   pip3 install dm_control
   pip3 install tqdm

Proximal Policy Optimization (PPO) is a policy-gradient algorithm (think of it
as an elaborated version of REINFORCE) where after a batch of data has been
collected, it is used in an inner loop to optimize the policy to maximise
the expected return given some constraints.

Ref: https://arxiv.org/abs/1707.06347

In this tutorial, you will learn how to use TorchRL to code a PPO algorithm,
including instructions on how to collect data, run multiple environments in parallel
or normalize the data.

PPO is usually regarded as a fast and efficient method online reinforcement
algorithm.
The algorithm works as follows: we will sample a batch of data by playing the
policy in the environment for a given number of steps. Then, we will perform a
given number of optimization steps with random sub-samples of this batch using
a clipped version of the REINFORCE loss.
The clipping will put a pessimistic bound on our loss.
The precise formula of the loss is:

.. math::

  L(s,a,\theta_k,\theta) = \min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
g(\epsilon, A^{\pi_{\theta_k}}(s,a))
\right),

There are two components in that loss: in the first part of the minimum operator,
we simply compute an importance-weighted version of the REINFORCE loss (i.e. a
REINFORCE loss that we have corrected for the fact that the current policy
configuration lags the one that was used for the data collection).
The second part of that minimum operator is a similar loss where we have clipped
the ratios when they exceeded or were below a given pair of thresholds.

This loss ensures that whether the advantage is positive or negative, policy
updates that would produce significant shifts from the previous configuration
are being discouraged.

"""
import os
os.environ["MUJOCO_GL"] = "egl"

import math

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import MultiSyncDataCollector
from torchrl.envs import ParallelEnv, TransformedEnv, ObservationNorm, Compose, \
    CatTensors, DoubleToFloat, EnvCreator, CatFrames
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import NormalParamWrapper, TanhNormal, ProbabilisticActor, ValueOperator
from torchrl.objectives.value import GAE
from torchrl.objectives.utils import distance_loss
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.rb_prototype import ReplayBuffer
from tqdm import tqdm

######################################################################
# Hyperparameters
# ---------------
#
# We set the hyperparameters for our algorithm. Depending on the resources
# available, one may choose to execute the policy on CUDA or on another
# device.
# The :obj:`frame_skip` will control how for how many frames is a single
# action being executed. The rest of the arguments that count frames
# must be corrected for this value (since one environment step will
# actually return :obj:`frame_skip` frames).
#
num_procs = 4
num_cells = 128
frame_skip = 2
frames_per_batch = 2000 // frame_skip
max_frames_per_traj = 1000 // frame_skip
total_frames = 100_000 // frame_skip
device = "cpu"
gamma = 0.99
lmbda = 0.95
batch_size = 64  # total batch-size to compute the loss
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = 0.2  # clip value for PPO loss
lr_policy = 3e-4
lr_value = 1e-3
max_grad_norm = 1.0
n_entropy_samples = 10
entropy_eps = 1e-4
seed = 0

######################################################################
# Define a parallel environment
# -----------------------------
#
# A parallel environment is defined with a custom function that returns an environment
# instance and a number of workers to spawn. The environment creator function should be
# wrapped in an :obj:`EnvCreator` instance to keep track of the meta-data.
# For this task, we'll focus on the cheetah-run task from the DeepMind control suite
# (https://linkinghub.elsevier.com/retrieve/pii/S2665963820300099)
#
torch.manual_seed(seed)

def env_maker(from_pixels=False):
    return DMControlEnv("cheetah", "run", device=device,
                        frame_skip=frame_skip, from_pixels=from_pixels)


parallel_env = ParallelEnv(num_procs, EnvCreator(env_maker))
parallel_env.set_seed(seed)

######################################################################
# Transforms
# ----------
#
# We will append some transforms to our environments to prepare the data for
# the policy. The :obj:`CatTensors` transform will concatenate the two observation
# keys ("position" and "velocity") into a single state named "observation".
# We will then normalize the data. In general, it is preferable to have data
# that loosely match a unit Gaussian distribution: to obtain this, we will
# run a certain number of random steps in the environment and compute
# the summary statistics of these observations.
# Since we're executing the environments in parallel, we can execute this
# normalization in batch.
# Finally, dm_control returns values in float64 and expects the actions to
# be in the same format. Since our network is parameterized with single
# precision numbers, we must transform the actions to float64 and then transform
# the reward and the observation back to float32. The :obj:`DoubleToFloat`
# has, as all TorchRL transforms do, a forward and an inverse method.
# The inverse is being called before the environment step is executed, the
# forward after. Hence, we just need to specify which entry has to be
# transformed when the data (i.e. the action) gets in and which when the
# data (i.e. the observation) is returned.
#

env = TransformedEnv(
    parallel_env,
    Compose(
        CatTensors(["position", "velocity"], "observation"),
        # concatenates the observations in a single vector
        ObservationNorm(in_keys=["observation"]),
        # dm_control returns float64 observations and expects float64 actions
        DoubleToFloat(in_keys=["observation", "reward"],
                      in_keys_inv=["action"]),
    )
)

######################################################################
# :obj:`ObservationNorm` can automatically gather the summary statistics of
# our environment:
#
env.transform[1].init_stats(max_frames_per_traj // frame_skip * env.batch_size[0], (0, 1), 1)
# check the shape of our summary stats
print("normalization constant shape:", env.transform[1].loc.shape)

######################################################################
# We create an evaluation environment to check the policy performance in
# pure exploitation mode:
#
eval_env = TransformedEnv(
    ParallelEnv(
        num_procs,
        EnvCreator(lambda: env_maker(from_pixels=True))
    ),
    env.transform.clone())
assert (eval_env.transform[1].loc == env.transform[1].loc).all()

######################################################################
# Policy
# ------
#
# PPO utilizes a stochastic policy to handle exploration.
# As the data is continuous, we use a Tanh-Normal distribution to respect the
# distribution bounds.
# We design the policy in three steps:
# 1. Define a neural network D_obs -> 2 D_action (loc and scale both have dimension D_action)
# 2. Wrap this neural network in a NormalParamWrapper to extract a location and a scale
# 3. Create a probabilistic TensorDictModule that can create this distribution and sample from it.
#

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1]),
)
value_net = nn.Sequential(
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(num_cells),
    nn.Tanh(),
    nn.LazyLinear(1),
)

######################################################################
# The output of the network must be a location and a non-negative scale.
# The NormalParamWrapper class does just that transformation:
#
actor_net = NormalParamWrapper(actor_net)
policy_module = TensorDictModule(actor_net, in_keys=["observation"],
                                 out_keys=["loc", "scale"])

######################################################################
# We now need to build a distribution out of the location and scale of our normal
# distribution. As the action space is bounded, we'll use a TanhNormal distribution
# to limit the values of the samples withing the accepted range. Here's how to code
# this in TorchRL:
#
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={"min": env.action_spec.space.minimum, "max": env.action_spec.space.maximum},
    return_log_prob=True,  # we'll need the log-prob for the numerator of the importance weights
)
value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

######################################################################
# let's try our policy:
#
print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

######################################################################
# Data collector
#
# TorchRL provides a set of DataCollector classes. Briefly, these classes
# allow you to control how many frames to collect at each iteration,
# when to reset the environment, on which device the policy should be executed etc.

collector = MultiSyncDataCollector(
    # we'll just run one ParallelEnvironment. Adding elements to the list would increase the number of envs run in parallel
    [env,],
    policy_module,
    frames_per_batch=frames_per_batch,
    max_frames_per_traj=max_frames_per_traj,
    total_frames=total_frames,
)
collector.set_seed(seed)

######################################################################
# Replay buffer
#
# We store the data in a replay buffer after every collection

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

######################################################################
# Loss function
# -------------
#
# The PPO loss can be directly imported from torchrl for convenience using
#
# .. code-block:: python
#
#    from torchrl.objectives import PPOClipLoss
#    loss_module = ClipPPOLoss(policy, critic, gamma=gamma)
#
# For a more explicit description, we unwrap this code in the training loop.
# Simply put, the objective function of PPO consists in maximising the expected
# value of an advantage function for the policy parameters given some proximity
# constraints. Let us first build the advantage module. PPO usually uses the GAE
# (Generalized Advantage Estimator) function (https://arxiv.org/abs/1506.02438).
# TorchRL provides a vectorized implementation of this function.
# The GAE module will update the input tensordict with new "advantage" and "value_target" keys.
# The value_target is a gradient-free tensor that represents the empirical
# value that the value network should represent with the input observation.

advantage_module = GAE(gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True)

######################################################################
# Training loop
# -------------
#

optim_policy = torch.optim.Adam(policy_module.parameters(), lr_policy)
optim_value = torch.optim.Adam(value_module.parameters(), lr_value)
scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optim_policy, total_frames // frames_per_batch, lr_policy/2)
scheduler_value = torch.optim.lr_scheduler.CosineAnnealingLR(optim_value, total_frames // frames_per_batch, lr_value/2)

logs = []
logs_eval_rewards = []
# log-ratio bounds: log(1-eps) and log(1+eps)
log_clip_bounds = (math.log1p(-clip_epsilon), math.log1p(clip_epsilon))
pbar = tqdm(total=total_frames * frame_skip)
eval_str = ""
for i, data in enumerate(collector):
    logs.append(data["reward"].sum(-2).mean().item())
    pbar.update(data["mask"].sum().item() * frame_skip)
    cum_reward_str = f"cumulative reward={logs[-1]: 4.4f} (init={logs[0]: 4.4f})"
    lr_str = f"lr policy: {optim_policy.param_groups[0]['lr']: 4.4f}"
    pbar.set_description(", ".join([
        eval_str,
        cum_reward_str,
        lr_str
    ]))
    for k in range(num_epochs):
        # we place the data in the replay buffer after removing the time dimension
        # the "mask" key represents the valid data in the batch: by indexing with "mask"
        # we make sure that we eliminate all the 0-padding values.
        advantage_module(data)
        replay_buffer.extend(data[data["mask"]])
        for j in range(frames_per_batch // batch_size):
            subdata, *_ = replay_buffer.sample(batch_size)
            # loss (1): Objective
            action = subdata.get("action")
            advantage = subdata.get("advantage")
            dist = policy_module.get_dist(subdata.clone(recurse=False))
            log_prob = dist.log_prob(action)
            prev_log_prob = subdata.get("sample_log_prob")
            # we need to unsqueeze the log_weight for the last dim, as
            # the advantage has shape [batch, 1] but the log-probability has
            # just size [batch]
            log_weight = (log_prob - prev_log_prob).unsqueeze(-1)

            gain1 = log_weight.exp() * advantage

            log_weight_clip = log_weight.clamp(*log_clip_bounds)
            gain2 = log_weight_clip.exp() * advantage

            gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
            loss_objective = -gain.mean()

            # Entropy bonus: we add a small entropy bonus to favour exploration
            entropy_bonus = entropy_eps * dist.log_prob(dist.rsample((n_entropy_samples,))).mean()

            # Optim
            loss = loss_objective + entropy_bonus
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_policy.param_groups[0]["params"], max_grad_norm)
            optim_policy.step()
            optim_policy.zero_grad()

            # loss (2): Value.
            # A simpler implementation would be to use GAE(..., gradient_mode=True), which
            # would compute a differentiable "value_error" key-value pair to be used for the
            # value network training.
            value_module(subdata)
            value = subdata.get("state_value")
            loss_value = distance_loss(
                value,
                subdata["value_target"],
                loss_function="smooth_l1",
            ).mean()

            # Optim
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(optim_value.param_groups[0]["params"], max_grad_norm)
            optim_value.step()
            optim_value.zero_grad()


    collector.update_policy_weights_()
    scheduler_policy.step()
    scheduler_value.step()
    if i % 10 == 0:
        with set_exploration_mode("mean"), torch.no_grad():
            eval_rollout = eval_env.rollout(max_frames_per_traj//frame_skip, policy_module)
            eval_reward = eval_rollout["reward"].sum(-2).mean().item()
            logs_eval_rewards.append(eval_reward)
            eval_str = f"eval cumulative reward: {logs_eval_rewards[-1]: 4.4f} (init: {logs_eval_rewards[0]: 4.4f})"

plt.plot(logs)
plt.savefig("training.png")
collector.shutdown()
del collector
