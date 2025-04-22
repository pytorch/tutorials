"""
Explicit horizontal fusion with foreach_map and torch.compile
============================================================

**Author:** `Michael Lazos <https://github.com/mlazos>`_
"""

#########################################################
#  Horizontal fusion is a key optimization in ML compilers. In eager,
#  this is typically expressed using the torch._foreach* ops which parallelizes
#  operations across a list of tensors. However, supporting all possible permutations
#  of arguments is quite difficult (e.g. mixtures of scalars and lists). Foreach_map
#  allows conversion of any pointwise op in ``torch`` to a horiztonally fused foreach
#  variant. In this tutorial, we will demonstrate how to implement the Adam optimizer
#  with ``foreach_map`` to generate a fully fused kernel.  
#
# .. note::
#
#    This recipe describes a prototype feature. Prototype features are typically
#    at an early stage for feedback and testing and are subject to change.
#
# Prerequisites
# -------------
#
# * PyTorch v2.7.0 or later
#

#####################################################################
# Model Setup
# ~~~~~~~~~~~~~~~~~~~~~
# For this example, we'll use a simple sequence of linear layers.
# We instantiate an independent copy to compare the two optimizer implementations.
#
import torch

# exit cleanly if we are on a device that doesn't support ``torch.compile``
if torch.cuda.get_device_capability() < (7, 0):
    print("Exiting because torch.compile is not supported on this device.")
    import sys
    sys.exit(0)

# Create simple model
model = torch.nn.Sequential(
    *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
)
model_copy = torch.nn.Sequential(
    *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
)
input = torch.rand(1024, device="cuda")

# run forward pass
output = model(input)
output_copy = model_copy(input)

# run backward to populate the grads for our optimizer below
output.sum().backward()
output_copy.sum().backward()

#####################################################################
# Helper functions for foreach_map implementation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this section, we'll begin our implementation of the Adam optimizer.
#
from torch._higher_order_ops.foreach_map import foreach_map

# Helper function to extract optimizer states from a torch.optim.Adam instance
def get_inputs(optim):
    steps = []
    params = []
    grads = []
    exp_avgs = []
    exp_avg_sqs = []
    for group in optim.param_groups:
        for p in group["params"]:
            params.append(p)
            grads.append(p.grad)
            state = optim.state[p]
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            steps.append(state["step"])

    return steps, params, exp_avgs, exp_avg_sqs


# Functions to update the different optimizer states
def update_exp_avg_sq(exp_avg_sq, grad, beta2):
    return exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)

def update_param(param, step, exp_avg, exp_avg_sq, beta1, beta2, lr, eps):
    bias_correction1 = 1 - torch.pow(beta1, step)
    bias_correction2 = (1 - torch.pow(beta2, step)).sqrt()
    step_size = (lr / bias_correction1).neg()
    denom = (exp_avg_sq.sqrt() / (bias_correction2 * step_size)).add(eps / step_size)
    return torch.add(param, torch.div(exp_avg, denom))

# Our full Adam implementation
def foreach_map_adam(
    steps,
    params,
    exp_avgs,
    exp_avg_sqs,
    weight_decay=0,
    beta1=0.9,
    beta2=0.999,
    lr=1e-3,
    eps=1e-8,
):
    with torch.no_grad():
        grads = [param.grad for param in params]
        # update step
        updated_steps = foreach_map(lambda x: x + 1, steps)
        torch._foreach_copy_(steps, updated_steps)

        if weight_decay != 0:
            foreach_map(torch.add, (grads,), alpha=weight_decay)

        # Higher-order operators (HOPs) cannot have multiple outputs at the moment
        # need to call foreach_map once for each output
        exp_avgs_updated = foreach_map(torch.lerp, exp_avgs, grads, 1 - beta1)
        exp_avgs_sq_updated = foreach_map(update_exp_avg_sq, exp_avg_sqs, grads, beta2)
        params_updated = foreach_map(
            update_param,
            params,
            steps,
            exp_avgs_updated,
            exp_avgs_sq_updated,
            beta1,
            beta2,
            lr,
            eps,
        )
        # Higher-order operators (HOPs) don't support input mutation today
        # so manually  update the states in-place
        torch._foreach_copy_(exp_avgs, exp_avgs_updated)
        torch._foreach_copy_(exp_avg_sqs, exp_avgs_sq_updated)
        torch._foreach_copy_(params, params_updated)
    return

#####################################################################
# Setting up and running the compiled kernel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this section, we'll run our Adam optimizer 
# and compare the results
#
# .. note::
#
#    ``torch.compile`` is only supported on CUDA devices that have a compute capability of 7.0 or higher.
opt_eager = torch.optim.Adam(model.parameters(), lr=torch.tensor(0.01))
opt_eager_copy = torch.optim.Adam(model_copy.parameters(), lr=torch.tensor(0.01))

# warm up the optimizer state dict
opt_eager.step()
opt_eager_copy.step()

inputs = get_inputs(opt_eager_copy)
compiled_adam = torch.compile(foreach_map_adam)

# optionally view the output code
torch._logging.set_logs(output_code=True)

# Warmup runs to compile the function
for _ in range(5):
    opt_eager.step()
    compiled_adam(*inputs)

for eager_p, compile_p in zip(opt_eager.param_groups[0]["params"], opt_eager_copy.param_groups[0]["params"]):
    torch.allclose(eager_p, compile_p)

# Benchmark performance

 # Let's define a helpful benchmarking function:
import torch.utils.benchmark as benchmark

def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

eager_runtime = benchmark_torch_function_in_microseconds(opt_eager.step)
compiled_runtime = benchmark_torch_function_in_microseconds(lambda: compiled_adam(*inputs))

assert eager_runtime > compiled_runtime
   
print(f"eager runtime: {eager_runtime}us")
print(f"compiled runtime: {compiled_runtime}us")



######################################################################
# Conclusion
# ~~~~~~~~~~
# In this tutorial, we successfully implemented a custom fully-fused Adam optimizer using foreach_map. 
# By leveraging the power of foreach_map and torch.compile, we were able to create an optimized version of the Adam 
# optimizer that can be used in various machine learning applications. This tutorial provides a comprehensive guide 
# on how to use foreach_map and torch.compile to optimize machine learning models, and serves as a 
# valuable resource for developers looking to improve the performance of their models with horizontal fusion.
#
# See also:
#
# * `Compiled optimizer tutorial <https://pytorch.org/tutorials/recipes/compiling_optimizer.html>`__ - an intro into the compiled optimizer.
# * `Compiling the optimizer with PT2 <https://dev-discuss.pytorch.org/t/compiling-the-optimizer-with-pt2/1669>`__ - deeper technical details on the compiled optimizer. 
