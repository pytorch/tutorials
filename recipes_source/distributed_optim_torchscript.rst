Distributed Optimizer with TorchScript support
==============================================================

.. note:: Distributed Optimizer with TorchScript support is introduced in PyTorch 1.8
    as a beta feature. This API is subject to change.

In this recipe, you will learn:

- The high-level idea of distributed optimizer with TorchScript support and what this feature brings
- How to write customized distributed optimizer that enables TorchScript support


Requirements
------------

- PyTorch 1.8+
- `Getting Started With Distributed RPC Framework <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`_


What is Distributed Optimizer?
------------------------------------

`DistributedOptimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`_ takes a list of remote
parameters (RRef) and runs the optimizer locally on the workers where the parameters live, which is commonly used together
with Distributed RPC/Autograd to do model parallel training. It could use any of the local optimizer algorithms (either
pre-defined algorithms provided in ``torch.optim`` or custom defined ones) to apply the gradients on each worker.


What is Distributed Optimizer with TorchScript support?
-------------------------------------------------------

Distributed Optimizer are widely used in distributed model parallel training, and in some
common use cases, training need to be done in multithreaded manner instead of multiprocess
due to performance concern and resource utilizations (or at least partially multithreaded,
i.e. Parameter Server hosting part of the model and parameters, with new thread updating the
parameters per request). PyTorch itself does not support multithreaded training natively as
it suffers from the Python's Global Interpreter Lock (GIL), but it could leverage 
`TorchScript <https://pytorch.org/docs/stable/jit.html>`_ to get rid of GIL and run the
model in a multithreaded way. 

For critical model training workloads, improving the training performance is an
important topic. Researchers often would like to implement different optimization strategies
with the graph representation (i.e. via operator fusion) or implement custom operator kernels
in order to speed up training.

Distributed Optimizer with TorchScript support could help getting rid of GIL, thus improve
PyTorch's training performance in the multithreaded environment, it also unlocks the potential
to further enhance the performance by using advanced compiler technologies that TorchScript
offers (i.e. CPU/GPU fusion).


How to write a customized distributed optimizer with TorchScript support?
-------------------------------------------------------------------------

The code below shows how to write a customized distributed optimizer given an existing local
optimizer implementation, which unlocks the TorchScript benefits including GIL removal and
performance improvement opportunities.

Suppose that you already have a local optimizer that is currently used during training,
In this case we will use `quasi-hyperbolic momentum (QHM) <https://github.com/facebookresearch/qhoptim/blob/e81dea3f2765780cf4fbb90b87b22ba7604b8625/qhoptim/pyt/qhm.py#L12>`_
as an example to show how to enable the TorchScript support, note that it also applies
to any custom optimizers that inherits from ``torch.optim.Optimizer``.

First, we need to separate the computation and state management from the optimizer implementation,
this is so that we could extract the computation part and make it a free function, which is
TorchScript friendly. It has two benefits: 1. The computation logic becomes easier to inspect,
it allows us to quickly turn the parameter update/computation part into TorchScript, and utilize
TorchScript IR to do further optimizations (operator fusion, etc.) 2. Distributed Optimizer
underlying is using a different mechanisms to get gradients and update parameters (we store
gradients separately instead of directly populating the ``param.grad`` field during backward).
Separating the computation allows distributed optimizer to enable the possibility of optimizer
update in multithreaded mode, as it eliminates the possible race condition to ``param.grad``.


::

    import torch
    from torch import Tensor
    from typing import List


    def qhm_update(params: List[Tensor],
                dp_list: List[Tensor],
                momentum_buffer_list: List[Tensor],
                lr: float,
                nu: float,
                weight_decay: float,
                weight_decay_type: str,
                momentum: float):

        for p, d_p, momentum_buffer in zip(params, dp_list, momentum_buffer_list):
            if weight_decay != 0:
                if weight_decay_type == "grad":
                    d_p.add_(weight_decay, p)
                elif weight_decay_type == "direct":
                    p.mul_(1.0 - lr * weight_decay)
                else:
                    raise ValueError("Invalid weight decay type provided")

            momentum_buffer.mul_(momentum).add_(1.0 - momentum, d_p)

            p.data.add_(-lr * nu, momentum_buffer)
            p.data.add_(-lr * (1.0 - nu), d_p)



Next we will define a distributed functional optimizer with TorchScript compatability to manage
the optimizer states and calls into the TorchScript compatible update function we defined above. 
Note that a few conventions are different from normal custom optimizers: 1. We don't inherit
``torch.optim.Optimizer`` as TorchScript does not support polymorphism 2. ``step`` takes gradients
list instead of the loss closure.

::

    import torch
    from torch import Tensor
    from typing import List, Optional, Dict

    # define this as a TorchScript class
    @torch.jit.script
    class FunctionalQHM(object):
        def __init__(self,
                    params: List[Tensor],
                    lr: float,
                    momentum: float,
                    nu: float,
                    weight_decay: float = 0.0,
                    weight_decay_type: str = "grad"):
            if lr < 0.0:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if momentum < 0.0:
                raise ValueError("Invalid momentum value: {}".format(momentum))
            if weight_decay < 0.0:
                raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
            if weight_decay_type not in ("grad", "direct"):
                raise ValueError("Invalid weight_decay_type value: {}".format(weight_decay_type))

            self.defaults = {
                "lr": lr,
                "momentum": momentum,
                "nu": nu,
                "weight_decay": weight_decay,
            }
            self.weight_decay_type = weight_decay_type

            # NOTE: we only have one param_group here and don't allow user to add additional
            # param group as it's not a common use case.
            self.param_group = {"params": params}

            self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        def step(self, gradients: List[Optional[Tensor]]):
            params = self.param_group['params']
            params_with_grad = []
            grads = []
            momentum_buffer_list: List[Tensor] = []

            if len(params) != len(gradients):
                raise ValueError(
                    "the gradients passed in does not equal to the size of the parameters!"
                    + f"Params length: {len(params)}. "
                    + f"Gradients length: {len(gradients)}"
                )

            for param, gradient in zip(self.param_group['params'], gradients):
                if gradient is not None:
                    params_with_grad.append(param)
                    grads.append(gradient)
                    state = self.state[param]
                    state['momentum_buffer'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    momentum_buffer_list.append(state['momentum_buffer'])

            # calls into the update function we just defined
            with torch.no_grad():
                qhm_update(params_with_grad,
                        grads,
                        momentum_buffer_list,
                        self.defaults['lr'],
                        self.defaults['nu'],
                        self.defaults['weight_decay'],
                        self.weight_decay_type,
                        self.defaults['momentum'])



Finally, we register our newly defined distributed functional optimizer into the ``functional_optim_map``
This is so that the ``DistributedOptimizer`` will try to pick up our custom implementation instead of the
pre-defined default ones.

::

    from torch.distributed.optim import DistributedOptimizer

    DistributedOptimizer.functional_optim_map[QHM] = FunctionalQHM

Now you can use the ``QHM`` optimizer as normal in distributed training by passing it to
`DistributedOptimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`_


::

    ...
    remote_params_list = [...]
    dist_optim = DistributedOptimizer(
        QHM, remote_params_list, *args, **kwargs
    )

DistributedOptimizer will automatically transform the QHM optimizer into the ``FunctionalQHM`` under the hood,
and enable the TorchScript support. This will unlock the performance that boosted by multithreaded training
and also give more potentials for further improvements (i.e. TorchScript fusion, etc.)

Note that majority of PyTorch built-in optimizers are already using this methodology to speed up distributed
training. If you see warning about some optimizers haven't been converted yet, you can write your own conversion
by following this recipe.
