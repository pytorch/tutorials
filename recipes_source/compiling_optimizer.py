"""
(beta) Compiling the optimizer with torch.compile
==========================================================================================


**Author:** `Michale Lazos <https://github.com/mlazos>`_
"""

######################################################################
# Summary
# ~~~~~~~~
#
# In this tutorial we will apply torch.compile to the optimizer to observe
# the GPU performance improvement
#
# .. note::
#
#   This tutorial requires PyTorch 2.2.0 or later.
#


######################################################################
# Model Setup
# ~~~~~~~~~~~~~~~~~~~~~
# For this example we'll use a simple sequence of linear layers.
# Since we are only benchmarking the optimizer, choice of model doesn't matter
# because optimizer performance is a function of the number of parameters.
#
# Depending on what machine you are using, your exact results may vary.

import torch

model = torch.nn.Sequential(
    *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
)
input = torch.rand(1024, device="cuda")
output = model(input)
output.sum().backward()

#############################################################################
# Setting up and running the optimizer benchmark
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# In this example, we'll use the Adam optimizer
# and create a helper function to wrap the step()
# in torch.compile()

opt = torch.optim.Adam(model.parameters(), lr=0.01)


@torch.compile()
def fn():
    opt.step()


# Lets define a helpful benchmarking function:
import torch.utils.benchmark as benchmark


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


# Warmup runs to compile the function
for _ in range(5):
    fn()

print(f"eager runtime: {benchmark_torch_function_in_microseconds(opt.step)}us")
print(f"compiled runtime: {benchmark_torch_function_in_microseconds(fn)}us")
