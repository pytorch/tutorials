Note

Go to the end
to download the full example code.

# Pendulum: Writing your environment and transforms with TorchRL

**Author**: [Vincent Moens](https://github.com/vmoens)

Creating an environment (a simulator or an interface to a physical control system)
is an integrative part of reinforcement learning and control engineering.

TorchRL provides a set of tools to do this in multiple contexts.
This tutorial demonstrates how to use PyTorch and TorchRL code a pendulum
simulator from the ground up.
It is freely inspired by the Pendulum-v1 implementation from [OpenAI-Gym/Farama-Gymnasium
control library](https://github.com/Farama-Foundation/Gymnasium).

![Pendulum](../_images/pendulum.gif)

Simple Pendulum

Key learnings:

- How to design an environment in TorchRL:
- Writing specs (input, observation and reward);
- Implementing behavior: seeding, reset and step.
- Transforming your environment inputs and outputs, and writing your own
transforms;
- How to use [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) to carry arbitrary data structures
through the `codebase`.

In the process, we will touch three crucial components of TorchRL:

- [environments](https://pytorch.org/rl/stable/reference/envs.html)
- [transforms](https://pytorch.org/rl/stable/reference/envs.html#transforms)
- [models (policy and value function)](https://pytorch.org/rl/stable/reference/modules.html)

To give a sense of what can be achieved with TorchRL's environments, we will
be designing a *stateless* environment. While stateful environments keep track of
the latest physical state encountered and rely on this to simulate the state-to-state
transition, stateless environments expect the current state to be provided to
them at each step, along with the action undertaken. TorchRL supports both
types of environments, but stateless environments are more generic and hence
cover a broader range of features of the environment API in TorchRL.

Modeling stateless environments gives users full control over the input and
outputs of the simulator: one can reset an experiment at any stage or actively
modify the dynamics from the outside. However, it assumes that we have some control
over a task, which may not always be the case: solving a problem where we cannot
control the current state is more challenging but has a much wider set of applications.

Another advantage of stateless environments is that they can enable
batched execution of transition simulations. If the backend and the
implementation allow it, an algebraic operation can be executed seamlessly on
scalars, vectors, or tensors. This tutorial gives such examples.

This tutorial will be structured as follows:

- We will first get acquainted with the environment properties:
its shape (`batch_size`), its methods (mainly [`step()`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#id4),
[`reset()`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#id1) and [`set_seed()`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#id3))
and finally its specs.
- After having coded our simulator, we will demonstrate how it can be used
during training with transforms.
- We will explore new avenues that follow from the TorchRL's API,
including: the possibility of transforming inputs, the vectorized execution
of the simulation and the possibility of backpropagation through the
simulation graph.
- Finally, we will train a simple policy to solve the system we implemented.

```

```

There are four things you must take care of when designing a new environment
class:

- `EnvBase._reset()`, which codes for the resetting of the simulator
at a (potentially random) initial state;
- `EnvBase._step()` which codes for the state transition dynamic;
- EnvBase._set_seed`() which implements the seeding mechanism;
- the environment specs.

Let us first describe the problem at hand: we would like to model a simple
pendulum over which we can control the torque applied on its fixed point.
Our goal is to place the pendulum in upward position (angular position at 0
by convention) and having it standing still in that position.
To design our dynamic system, we need to define two equations: the motion
equation following an action (the torque applied) and the reward equation
that will constitute our objective function.

For the motion equation, we will update the angular velocity following:

\[\dot{\theta}_{t+1} = \dot{\theta}_t + (3 * g / (2 * L) * \sin(\theta_t) + 3 / (m * L^2) * u) * dt\]

where \(\dot{\theta}\) is the angular velocity in rad/sec, \(g\) is the
gravitational force, \(L\) is the pendulum length, \(m\) is its mass,
\(\theta\) is its angular position and \(u\) is the torque. The
angular position is then updated according to

\[\theta_{t+1} = \theta_{t} + \dot{\theta}_{t+1} dt\]

We define our reward as

\[r = -(\theta^2 + 0.1 * \dot{\theta}^2 + 0.001 * u^2)\]

which will be maximized when the angle is close to 0 (pendulum in upward
position), the angular velocity is close to 0 (no motion) and the torque is
0 too.

## Coding the effect of an action: `_step()`

The step method is the first thing to consider, as it will encode
the simulation that is of interest to us. In TorchRL, the
[`EnvBase`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) class has a `EnvBase.step()`
method that receives a [`tensordict.TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
instance with an `"action"` entry indicating what action is to be taken.

To facilitate the reading and writing from that `tensordict` and to make sure
that the keys are consistent with what's expected from the library, the
simulation part has been delegated to a private abstract method `_step()`
which reads input data from a `tensordict`, and writes a *new* `tensordict`
with the output data.

The `_step()` method should do the following:

> 1. Read the input keys (such as `"action"`) and execute the simulation
> based on these;
> 2. Retrieve observations, done state and reward;
> 3. Write the set of observation values along with the reward and done state
> at the corresponding entries in a new `TensorDict`.

Next, the [`step()`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#id4) method will merge the output
of [`step()`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#id4) in the input `tensordict` to enforce
input/output consistency.

Typically, for stateful environments, this will look like this:

```
>>> policy(env.reset())
>>> print(tensordict)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
>>> env.step(tensordict)
>>> print(tensordict)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
```

Notice that the root `tensordict` has not changed, the only modification is the
appearance of a new `"next"` entry that contains the new information.

In the Pendulum example, our `_step()` method will read the relevant
entries from the input `tensordict` and compute the position and velocity of
the pendulum after the force encoded by the `"action"` key has been applied
onto it. We compute the new angular position of the pendulum
`"new_th"` as the result of the previous position `"th"` plus the new
velocity `"new_thdot"` over a time interval `dt`.

Since our goal is to turn the pendulum up and maintain it still in that
position, our `cost` (negative reward) function is lower for positions
close to the target and low speeds.
Indeed, we want to discourage positions that are far from being "upward"
and/or speeds that are far from 0.

In our example, `EnvBase._step()` is encoded as a static method since our
environment is stateless. In stateful settings, the `self` argument is
needed as the state needs to be read from the environment.

## Resetting the simulator: `_reset()`

The second method we need to care about is the
`_reset()` method. Like
`_step()`, it should write the observation entries
and possibly a done state in the `tensordict` it outputs (if the done state is
omitted, it will be filled as `False` by the parent method
[`reset()`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#id1)). In some contexts, it is required that
the `_reset` method receives a command from the function that called
it (for example, in multi-agent settings we may want to indicate which agents need
to be reset). This is why the `_reset()` method
also expects a `tensordict` as input, albeit it may perfectly be empty or
`None`.

The parent `EnvBase.reset()` does some simple checks like the
`EnvBase.step()` does, such as making sure that a `"done"` state
is returned in the output `tensordict` and that the shapes match what is
expected from the specs.

For us, the only important thing to consider is whether
`EnvBase._reset()` contains all the expected observations. Once more,
since we are working with a stateless environment, we pass the configuration
of the pendulum in a nested `tensordict` named `"params"`.

In this example, we do not pass a done state as this is not mandatory
for `_reset()` and our environment is non-terminating, so we always
expect it to be `False`.

## Environment metadata: `env.*_spec`

The specs define the input and output domain of the environment.
It is important that the specs accurately define the tensors that will be
received at runtime, as they are often used to carry information about
environments in multiprocessing and distributed settings. They can also be
used to instantiate lazily defined neural networks and test scripts without
actually querying the environment (which can be costly with real-world
physical systems for instance).

There are four specs that we must code in our environment:

- `EnvBase.observation_spec`: This will be a `CompositeSpec`
instance where each key is an observation (a `CompositeSpec` can be
viewed as a dictionary of specs).
- `EnvBase.action_spec`: It can be any type of spec, but it is required
that it corresponds to the `"action"` entry in the input `tensordict`;
- `EnvBase.reward_spec`: provides information about the reward space;
- `EnvBase.done_spec`: provides information about the space of the done
flag.

TorchRL specs are organized in two general containers: `input_spec` which
contains the specs of the information that the step function reads (divided
between `action_spec` containing the action and `state_spec` containing
all the rest), and `output_spec` which encodes the specs that the
step outputs (`observation_spec`, `reward_spec` and `done_spec`).
In general, you should not interact directly with `output_spec` and
`input_spec` but only with their content: `observation_spec`,
`reward_spec`, `done_spec`, `action_spec` and `state_spec`.
The reason if that the specs are organized in a non-trivial way
within `output_spec` and
`input_spec` and neither of these should be directly modified.

In other words, the `observation_spec` and related properties are
convenient shortcuts to the content of the output and input spec containers.

TorchRL offers multiple [`TensorSpec`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)
[subclasses](https://pytorch.org/rl/stable/reference/data.html#tensorspec) to
encode the environment's input and output characteristics.

### Specs shape

The environment specs leading dimensions must match the
environment batch-size. This is done to enforce that every component of an
environment (including its transforms) have an accurate representation of
the expected input and output shapes. This is something that should be
accurately coded in stateful settings.

For non batch-locked environments, such as the one in our example (see below),
this is irrelevant as the environment batch size will most likely be empty.

## Reproducible experiments: seeding

Seeding an environment is a common operation when initializing an experiment.
The only goal of `EnvBase._set_seed()` is to set the seed of the contained
simulator. If possible, this operation should not call `reset()` or interact
with the environment execution. The parent `EnvBase.set_seed()` method
incorporates a mechanism that allows seeding multiple environments with a
different pseudo-random and reproducible seed.

## Wrapping things together: the [`EnvBase`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) class

We can finally put together the pieces and design our environment class.
The specs initialization needs to be performed during the environment
construction, so we must take care of calling the `_make_spec()` method
within `PendulumEnv.__init__()`.

We add a static method `PendulumEnv.gen_params()` which deterministically
generates a set of hyperparameters to be used during execution:

We define the environment as non-`batch_locked` by turning the `homonymous`
attribute to `False`. This means that we will **not** enforce the input
`tensordict` to have a `batch-size` that matches the one of the environment.

The following code will just put together the pieces we have coded above.

## Testing our environment

TorchRL provides a simple function `check_env_specs()`
to check that a (transformed) environment has an input/output structure that
matches the one dictated by its specs.
Let us try it out:

We can have a look at our specs to have a visual representation of the environment
signature:

We can execute a couple of commands too to check that the output structure
matches what is expected.

We can run the `env.rand_step()` to generate
an action randomly from the `action_spec` domain. A `tensordict` containing
the hyperparameters and the current state **must** be passed since our
environment is stateless. In stateful contexts, `env.rand_step()` works
perfectly too.

## Transforming an environment

Writing environment transforms for stateless simulators is slightly more
complicated than for stateful ones: transforming an output entry that needs
to be read at the following iteration requires to apply the inverse transform
before calling `meth.step()` at the next step.
This is an ideal scenario to showcase all the features of TorchRL's
transforms!

For instance, in the following transformed environment we `unsqueeze` the entries
`["th", "thdot"]` to be able to stack them along the last
dimension. We also pass them as `in_keys_inv` to squeeze them back to their
original shape once they are passed as input in the next iteration.

### Writing custom transforms

TorchRL's transforms may not cover all the operations one wants to execute
after an environment has been executed.
Writing a transform does not require much effort. As for the environment
design, there are two steps in writing a transform:

- Getting the dynamics right (forward and inverse);
- Adapting the environment specs.

A transform can be used in two settings: on its own, it can be used as a
[`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). It can also be used appended to a
[`TransformedEnv`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv). The structure of the class allows to
customize the behavior in the different contexts.

A [`Transform`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) skeleton can be summarized as follows:

```
class Transform(nn.Module):
 def forward(self, tensordict):
 ...
 def _apply_transform(self, tensordict):
 ...
 def _step(self, tensordict):
 ...
 def _call(self, tensordict):
 ...
 def inv(self, tensordict):
 ...
 def _inv_apply_transform(self, tensordict):
 ...
```

There are three entry points (`forward()`, `_step()` and `inv()`)
which all receive [`tensordict.TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) instances. The first two
will eventually go through the keys indicated by `in_keys`
and call `_apply_transform()` to each of these. The results will
be written in the entries pointed by `Transform.out_keys` if provided
(if not the `in_keys` will be updated with the transformed values).
If inverse transforms need to be executed, a similar data flow will be
executed but with the `Transform.inv()` and
`Transform._inv_apply_transform()` methods and across the `in_keys_inv`
and `out_keys_inv` list of keys.
The following figure summarized this flow for environments and replay
buffers.

> Transform API

In some cases, a transform will not work on a subset of keys in a unitary
manner, but will execute some operation on the parent environment or
work with the entire input `tensordict`.
In those cases, the `_call()` and `forward()` methods should be
re-written, and the `_apply_transform()` method can be skipped.

Let us code new transforms that will compute the `sine` and `cosine`
values of the position angle, as these values are more useful to us to learn
a policy than the raw angle value:

Concatenates the observations onto an "observation" entry.
`del_keys=False` ensures that we keep these values for the next
iteration.

Once more, let us check that our environment specs match what is received:

## Executing a rollout

Executing a rollout is a succession of simple steps:

- reset the environment
- while some condition is not met:

- compute an action given a policy
- execute a step given this action
- collect the data
- make a `MDP` step
- gather the data and return

These operations have been conveniently wrapped in the [`rollout()`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#id2)
method, from which we provide a simplified version here below.

## Batching computations

The last unexplored end of our tutorial is the ability that we have to
batch computations in TorchRL. Because our environment does not
make any assumptions regarding the input data shape, we can seamlessly
execute it over batches of data. Even better: for non-batch-locked
environments such as our Pendulum, we can change the batch size on the fly
without recreating the environment.
To do this, we just generate parameters with the desired shape.

Executing a rollout with a batch of data requires us to reset the environment
out of the rollout function, since we need to define the batch_size
dynamically and this is not supported by [`rollout()`](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#id2):

## Training a simple policy

In this example, we will train a simple policy using the reward as a
differentiable objective, such as a negative loss.
We will take advantage of the fact that our dynamic system is fully
differentiable to backpropagate through the trajectory return and adjust the
weights of our policy to maximize this value directly. Of course, in many
settings many of the assumptions we make do not hold, such as
differentiable system and full access to the underlying mechanics.

Still, this is a very simple example that showcases how a training loop can
be coded with a custom environment in TorchRL.

Let us first write the policy network:

and our optimizer:

### Training loop

We will successively:

- generate a trajectory
- sum the rewards
- backpropagate through the graph defined by these operations
- clip the gradient norm and make an optimization step
- repeat

At the end of the training loop, we should have a final reward close to 0
which demonstrates that the pendulum is upward and still as desired.

## Conclusion

In this tutorial, we have learned how to code a stateless environment from
scratch. We touched the subjects of:

- The four essential components that need to be taken care of when coding
an environment (`step`, `reset`, seeding and building specs).
We saw how these methods and classes interact with the
[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) class;
- How to test that an environment is properly coded using
`check_env_specs()`;
- How to append transforms in the context of stateless environments and how
to write custom transformations;
- How to train a policy on a fully differentiable simulator.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: pendulum.ipynb`](../_downloads/8016e5cfa285bd92b9684c45552fffcc/pendulum.ipynb)

[`Download Python source code: pendulum.py`](../_downloads/0c4dc681209d8c964aae9de5e477d280/pendulum.py)

[`Download zipped: pendulum.zip`](../_downloads/426af85b1017ae7d4a0d1204359c1852/pendulum.zip)