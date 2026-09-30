Note

Go to the end
to download the full example code.

# Reinforcement Learning (PPO) with TorchRL Tutorial

**Author**: [Vincent Moens](https://github.com/vmoens)

This tutorial demonstrates how to use PyTorch and `torchrl` to train a parametric policy
network to solve the Inverted Pendulum task from the [OpenAI-Gym/Farama-Gymnasium
control library](https://github.com/Farama-Foundation/Gymnasium).

![Inverted pendulum](../_images/invpendulum.gif)

Inverted pendulum

Key learnings:

- How to create an environment in TorchRL, transform its outputs, and collect data from this environment;
- How to make your classes talk to each other using [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict);
- The basics of building your training loop with TorchRL:

- How to compute the advantage signal for policy gradient methods;
- How to create a stochastic policy using a probabilistic neural network;
- How to create a dynamic replay buffer and sample from it without repetition.

We will cover six crucial components of TorchRL:

- [environments](https://docs.pytorch.org/rl/stable/reference/envs.html)
- [transforms](https://docs.pytorch.org/rl/stable/reference/envs.html#transforms)
- [models (policy and value function)](https://docs.pytorch.org/rl/stable/reference/modules.html)
- [loss modules](https://docs.pytorch.org/rl/stable/reference/objectives.html)
- [data collectors](https://docs.pytorch.org/rl/stable/reference/collectors.html)
- [replay buffers](https://docs.pytorch.org/rl/stable/reference/data.html#replay-buffers)

If you are running this in Google Colab, make sure you install the following dependencies:

```
!pip3 install torchrl
!pip3 install gym[mujoco]
!pip3 install tqdm
```

Proximal Policy Optimization (PPO) is a policy-gradient algorithm where a
batch of data is being collected and directly consumed to train the policy to maximise
the expected return given some proximality constraints. You can think of it
as a sophisticated version of [REINFORCE](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf),
the foundational policy-optimization algorithm. For more information, see the
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) paper.

PPO is usually regarded as a fast and efficient method for online, on-policy
reinforcement algorithm. TorchRL provides a loss-module that does all the work
for you, so that you can rely on this implementation and focus on solving your
problem rather than re-inventing the wheel every time you want to train a policy.

For completeness, here is a brief overview of what the loss computes, even though
this is taken care of by our [`ClipPPOLoss`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss) module--the algorithm works as follows:
1. we will sample a batch of data by playing the
policy in the environment for a given number of steps.
2. Then, we will perform a given number of optimization steps with random sub-samples of this batch using
a clipped version of the REINFORCE loss.
3. The clipping will put a pessimistic bound on our loss: lower return estimates will
be favored compared to higher ones.
The precise formula of the loss is:

\[L(s,a,\theta_k,\theta) = \min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a), \;\;
g(\epsilon, A^{\pi_{\theta_k}}(s,a))
\right),\]

There are two components in that loss: in the first part of the minimum operator,
we simply compute an importance-weighted version of the REINFORCE loss (for example, a
REINFORCE loss that we have corrected for the fact that the current policy
configuration lags the one that was used for the data collection).
The second part of that minimum operator is a similar loss where we have clipped
the ratios when they exceeded or were below a given pair of thresholds.

This loss ensures that whether the advantage is positive or negative, policy
updates that would produce significant shifts from the previous configuration
are being discouraged.

This tutorial is structured as follows:

1. First, we will define a set of hyperparameters we will be using for training.
2. Next, we will focus on creating our environment, or simulator, using TorchRL's
wrappers and transforms.
3. Next, we will design the policy network and the value model,
which is indispensable to the loss function. These modules will be used
to configure our loss module.
4. Next, we will create the replay buffer and data loader.
5. Finally, we will run our training loop and analyze the results.

Throughout this tutorial, we'll be using the `tensordict` library.
[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) is the lingua franca of TorchRL: it helps us abstract
what a module reads and writes and care less about the specific data
description and more about the algorithm itself.

```

```

## Define Hyperparameters

We set the hyperparameters for our algorithm. Depending on the resources
available, one may choose to execute the policy on GPU or on another
device.
The `frame_skip` will control how for how many frames is a single
action being executed. The rest of the arguments that count frames
must be corrected for this value (since one environment step will
actually return `frame_skip` frames).

### Data collection parameters

When collecting data, we will be able to choose how big each batch will be
by defining a `frames_per_batch` parameter. We will also define how many
frames (such as the number of interactions with the simulator) we will allow ourselves to
use. In general, the goal of an RL algorithm is to learn to solve the task
as fast as it can in terms of environment interactions: the lower the `total_frames`
the better.

```
# For a complete training, bring the number of frames up to 1M
```

### PPO parameters

At each data collection (or batch collection) we will run the optimization
over a certain number of *epochs*, each time consuming the entire data we just
acquired in a nested training loop. Here, the `sub_batch_size` is different from the
`frames_per_batch` here above: recall that we are working with a "batch of data"
coming from our collector, which size is defined by `frames_per_batch`, and that
we will further split in smaller sub-batches during the inner training loop.
The size of these sub-batches is controlled by `sub_batch_size`.

## Define an environment

In RL, an *environment* is usually the way we refer to a simulator or a
control system. Various libraries provide simulation environments for reinforcement
learning, including Gymnasium (previously OpenAI Gym), DeepMind control suite, and
many others.
As a general library, TorchRL's goal is to provide an interchangeable interface
to a large panel of RL simulators, allowing you to easily swap one environment
with another. For example, creating a wrapped gym environment can be achieved with few characters:

There are a few things to notice in this code: first, we created
the environment by calling the `GymEnv` wrapper. If extra keyword arguments
are passed, they will be transmitted to the `gym.make` method, hence covering
the most common environment construction commands.
Alternatively, one could also directly create a gym environment using `gym.make(env_name, **kwargs)`
and wrap it in a GymWrapper class.

Also the `device` argument: for gym, this only controls the device where
input action and observed states will be stored, but the execution will always
be done on CPU. The reason for this is simply that gym does not support on-device
execution, unless specified otherwise. For other libraries, we have control over
the execution device and, as much as we can, we try to stay consistent in terms of
storing and execution backends.

### Transforms

We will append some transforms to our environments to prepare the data for
the policy. In Gym, this is usually achieved via wrappers. TorchRL takes a different
approach, more similar to other pytorch domain libraries, through the use of transforms.
To add transforms to an environment, one should simply wrap it in a [`TransformedEnv`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv)
instance and append the sequence of transforms to it. The transformed environment will inherit
the device and meta-data of the wrapped environment, and transform these depending on the sequence
of transforms it contains.

### Normalization

The first to encode is a normalization transform.
As a rule of thumbs, it is preferable to have data that loosely
match a unit Gaussian distribution: to obtain this, we will
run a certain number of random steps in the environment and compute
the summary statistics of these observations.

We'll append two other transforms: the [`DoubleToFloat`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.DoubleToFloat.html#torchrl.envs.transforms.DoubleToFloat) transform will
convert double entries to single-precision numbers, ready to be read by the
policy. The [`StepCounter`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.StepCounter.html#torchrl.envs.transforms.StepCounter) transform will be used to count the steps before
the environment is terminated. We will use this measure as a supplementary measure
of performance.

As we will see later, many of the TorchRL's classes rely on [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
to communicate. You could think of it as a python dictionary with some extra
tensor features. In practice, this means that many modules we will be working
with need to be told what key to read (`in_keys`) and what key to write
(`out_keys`) in the `tensordict` they will receive. Usually, if `out_keys`
is omitted, it is assumed that the `in_keys` entries will be updated
in-place. For our transforms, the only entry we are interested in is referred
to as `"observation"` and our transform layers will be told to modify this
entry and this entry only:

As you may have noticed, we have created a normalization layer but we did not
set its normalization parameters. To do this, [`ObservationNorm`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.ObservationNorm.html#torchrl.envs.transforms.ObservationNorm) can
automatically gather the summary statistics of our environment:

The [`ObservationNorm`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.ObservationNorm.html#torchrl.envs.transforms.ObservationNorm) transform has now been populated with a
location and a scale that will be used to normalize the data.

Let us do a little sanity check for the shape of our summary stats:

An environment is not only defined by its simulator and transforms, but also
by a series of metadata that describe what can be expected during its
execution.
For efficiency purposes, TorchRL is quite stringent when it comes to
environment specs, but you can easily check that your environment specs are
adequate.
In our example, the `GymWrapper` and
`GymEnv` that inherits
from it already take care of setting the proper specs for your environment so
you should not have to care about this.

Nevertheless, let's see a concrete example using our transformed
environment by looking at its specs.
There are three specs to look at: `observation_spec` which defines what
is to be expected when executing an action in the environment,
`reward_spec` which indicates the reward domain and finally the
`input_spec` (which contains the `action_spec`) and which represents
everything an environment requires to execute a single step.

the `check_env_specs()` function runs a small rollout and compares its output against the environment
specs. If no error is raised, we can be confident that the specs are properly defined:

For fun, let's see what a simple random rollout looks like. You can
call env.rollout(n_steps) and get an overview of what the environment inputs
and outputs look like. Actions will automatically be drawn from the action spec
domain, so you don't need to care about designing a random sampler.

Typically, at each step, an RL environment receives an
action as input, and outputs an observation, a reward and a done state. The
observation may be composite, meaning that it could be composed of more than one
tensor. This is not a problem for TorchRL, since the whole set of observations
is automatically packed in the output [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict). After executing a rollout
(for example, a sequence of environment steps and random action generations) over a given
number of steps, we will retrieve a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) instance with a shape
that matches this trajectory length:

Our rollout data has a shape of `torch.Size([3])`, which matches the number of steps
we ran it for. The `"next"` entry points to the data coming after the current step.
In most cases, the `"next"` data at time t matches the data at `t+1`, but this
may not be the case if we are using some specific transformations (for example, multi-step).

## Policy

PPO utilizes a stochastic policy to handle exploration. This means that our
neural network will have to output the parameters of a distribution, rather
than a single value corresponding to the action taken.

As the data is continuous, we use a Tanh-Normal distribution to respect the
action space boundaries. TorchRL provides such distribution, and the only
thing we need to care about is to build a neural network that outputs the
right number of parameters for the policy to work with (a location, or mean,
and a scale):

\[f_{\theta}(\text{observation}) = \mu_{\theta}(\text{observation}), \sigma^{+}_{\theta}(\text{observation})\]

The only extra-difficulty that is brought up here is to split our output in two
equal parts and map the second to a strictly positive space.

We design the policy in three steps:

1. Define a neural network `D_obs` -> `2 * D_action`. Indeed, our `loc` (mu) and `scale` (sigma) both have dimension `D_action`.
2. Append a [`NormalParamExtractor`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.NormalParamExtractor.html#tensordict.nn.distributions.NormalParamExtractor) to extract a location and a scale (for example, splits the input in two equal parts and applies a positive transformation to the scale parameter).
3. Create a probabilistic [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) that can generate this distribution and sample from it.

To enable the policy to "talk" with the environment through the `tensordict`
data carrier, we wrap the `nn.Module` in a [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule). This
class will simply ready the `in_keys` it is provided with and write the
outputs in-place at the registered `out_keys`.

We now need to build a distribution out of the location and scale of our
normal distribution. To do so, we instruct the
[`ProbabilisticActor`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.tensordict_module.ProbabilisticActor.html#torchrl.modules.tensordict_module.ProbabilisticActor)
class to build a [`TanhNormal`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.TanhNormal.html#torchrl.modules.TanhNormal) out of the location and scale
parameters. We also provide the minimum and maximum values of this
distribution, which we gather from the environment specs.

The name of the `in_keys` (and hence the name of the `out_keys` from
the [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) above) cannot be set to any value one may
like, as the [`TanhNormal`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.TanhNormal.html#torchrl.modules.TanhNormal) distribution constructor will expect the
`loc` and `scale` keyword arguments. That being said,
[`ProbabilisticActor`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.tensordict_module.ProbabilisticActor.html#torchrl.modules.tensordict_module.ProbabilisticActor) also accepts
`Dict[str, str]` typed `in_keys` where the key-value pair indicates
what `in_key` string should be used for every keyword argument that is to be used.

## Value network

The value network is a crucial component of the PPO algorithm, even though it
won't be used at inference time. This module will read the observations and
return an estimation of the discounted return for the following trajectory.
This allows us to amortize learning by relying on the some utility estimation
that is learned on-the-fly during training. Our value network share the same
structure as the policy, but for simplicity we assign it its own set of
parameters.

let's try our policy and value modules. As we said earlier, the usage of
[`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) makes it possible to directly read the output
of the environment to run these modules, as they know what information to read
and where to write it:

## Data collector

TorchRL provides a set of [DataCollector classes](https://docs.pytorch.org/rl/stable/reference/collectors.html).
Briefly, these classes execute three operations: reset an environment,
compute an action given the latest observation, execute a step in the environment,
and repeat the last two steps until the environment signals a stop (or reaches
a done state).

They allow you to control how many frames to collect at each iteration
(through the `frames_per_batch` parameter),
when to reset the environment (through the `max_frames_per_traj` argument),
on which `device` the policy should be executed, etc. They are also
designed to work efficiently with batched and multiprocessed environments.

The simplest data collector is the `SyncDataCollector`:
it is an iterator that you can use to get batches of data of a given length, and
that will stop once a total number of frames (`total_frames`) have been
collected.
Other data collectors (`MultiSyncDataCollector` and
`MultiaSyncDataCollector`) will execute
the same operations in synchronous and asynchronous manner over a
set of multiprocessed workers.

As for the policy and environment before, the data collector will return
[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) instances with a total number of elements that will
match `frames_per_batch`. Using [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) to pass data to the
training loop allows you to write data loading pipelines
that are 100% oblivious to the actual specificities of the rollout content.

## Replay buffer

Replay buffers are a common building piece of off-policy RL algorithms.
In on-policy contexts, a replay buffer is refilled every time a batch of
data is collected, and its data is repeatedly consumed for a certain number
of epochs.

TorchRL's replay buffers are built using a common container
[`ReplayBuffer`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) which takes as argument the components
of the buffer: a storage, a writer, a sampler and possibly some transforms.
Only the storage (which indicates the replay buffer capacity) is mandatory.
We also specify a sampler without repetition to avoid sampling multiple times
the same item in one epoch.
Using a replay buffer for PPO is not mandatory and we could simply
sample the sub-batches from the collected batch, but using these classes
make it easy for us to build the inner training loop in a reproducible way.

## Loss function

The PPO loss can be directly imported from TorchRL for convenience using the
[`ClipPPOLoss`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss) class. This is the easiest way of utilizing PPO:
it hides away the mathematical operations of PPO and the control flow that
goes with it.

PPO requires some "advantage estimation" to be computed. In short, an advantage
is a value that reflects an expectancy over the return value while dealing with
the bias / variance tradeoff.
To compute the advantage, one just needs to (1) build the advantage module, which
utilizes our value operator, and (2) pass each batch of data through it before each
epoch.
The GAE module will update the input `tensordict` with new `"advantage"` and
`"value_target"` entries.
The `"value_target"` is a gradient-free tensor that represents the empirical
value that the value network should represent with the input observation.
Both of these will be used by [`ClipPPOLoss`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss) to
return the policy and value losses.

## Training loop

We now have all the pieces needed to code our training loop.
The steps include:

- Collect data

- Compute advantage

- Loop over the collected to compute loss values
- Back propagate
- Optimize
- Repeat
- Repeat
- Repeat

```
# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
```

## Results

Before the 1M step cap is reached, the algorithm should have reached a max
step count of 1000 steps, which is the maximum number of steps before the
trajectory is truncated.

## Conclusion and next steps

In this tutorial, we have learned:

1. How to create and customize an environment with `torchrl`;
2. How to write a model and a loss function;
3. How to set up a typical training loop.

If you want to experiment with this tutorial a bit more, you can apply the following modifications:

- From an efficiency perspective,
we could run several simulations in parallel to speed up data collection.
Check [`ParallelEnv`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) for further information.
- From a logging perspective, one could add a `torchrl.record.VideoRecorder` transform to
the environment after asking for rendering to get a visual rendering of the
inverted pendulum in action. Check `torchrl.record` to
know more.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: reinforcement_ppo.ipynb`](../_downloads/4065a985b933a4377d3c7d93557e2282/reinforcement_ppo.ipynb)

[`Download Python source code: reinforcement_ppo.py`](../_downloads/7ed508ed54ec36ee5c1d3fa1e8ceede0/reinforcement_ppo.py)

[`Download zipped: reinforcement_ppo.zip`](../_downloads/67b7ac26f2ad3d834e0413c0bff24803/reinforcement_ppo.zip)