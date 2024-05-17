"""
(beta) Running the compiled optimizer with an LR Scheduler

**Author:** `Michael Lazos <https://github.com/mlazos>`_
"""

#########################################################
# The optimizer is a key algorithm for training any deep learning model.
# In this example, we will show how to pair the ``torch.compile``d optimizer
# with the LR schedulers to accelerate training convergence
#
# .. note::
#
#    This tutorial requires PyTorch 2.2.0 or later.

#####################################################################
# Model Setup
# ~~~~~~~~~~~~~~~~~~~~~
# For this example, we'll use a simple sequence of linear layers.
#

import torch

# Create simple model
model = torch.nn.Sequential(
    *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
)
input = torch.rand(1024, device="cuda")

# run forward pass
output = model(input)

# run backward to populate the grads for our optimizer below
output.sum().backward()


#####################################################################
# Setting up and running the compiled optimizer with LR Scheduler
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this section, we'll use the Adam optimizer with LinearLR Scheduler
# and create a helper function to wrap the ``step()`` call for each
# in ``torch.compile()``.
#
# .. note::
#
#    ``torch.compile`` is only supported on cuda devices with compute capability >= 7.0


# exit cleanly if we are on a device that doesn't support torch.compile
if torch.cuda.get_device_capability() < (7, 0):
    print("Exiting because torch.compile is not supported on this device.")
    import sys
    sys.exit(0)

# !!! IMPORTANT !!! Wrap the lr in a tensor if we are pairing the
# the optimizer with an LR Scheduler.
# Without this, torch.compile will recompile as the value of the LR
# changes.
opt = torch.optim.Adam(model.parameters(), lr=torch.tensor(0.01))
sched = torch.optim.lr_scheduler.LinearLR(opt, total_iters=5)

@torch.compile(fullgraph=False)
def fn():
    opt.step()
    sched.step()


# Warmup runs to compile the function
for _ in range(5):
    fn()
    print(opt.param_groups[0]["lr"])

########################################################################
# Sample Output:
#
# >> tensor(0.0047)
# >> tensor(0.0060)
# >> tensor(0.0073)
# >> tensor(0.0087)
# >> tensor(0.0100)

######################################################################
# Conclusion
# ~~~~~~~~~~
#
# In this tutorial we showed how to pair the ``torch.compile``d optimizer
# with an LR Scheduler to accelerate training convergence. We used a model consisting
# of a simple sequence of linear layers with the Adam optimizer paired
# with a LinearLR scheduler to demonstrate the LR changing across iterations.
#
# See also:
# * tutorial on the compiled optimizer - `Compiled optimizer tutorial`_
# * deeper technical details on the compiled optimizer see `Compiling the optimizer with PT2`_
#
# .. Compiled optimizer tutorial: https://pytorch.org/tutorials/recipes/compiling_optimizer.html
# .. Compiling the optimizer with PT2: https://dev-discuss.pytorch.org/t/compiling-the-optimizer-with-pt2/1669

######################################################################
# Extension: What happens with a non-tensor LR?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For the curious, we will show how to peek into what happens with ``torch.compile`` when we don't wrap the
# LR in a tensor.

# No longer wrap the LR in a tensor here
opt = torch.optim.Adam(model.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.LinearLR(opt, total_iters=5)

@torch.compile(fullgraph=False)
def fn():
    opt.step()
    sched.step()

# Setup logging to view recompiles
torch._logging.set_logs(recompiles=True)

# Warmup runs to compile the function
# We will now recompile on each iteration
# as the value of the lr is mutated.
for _ in range(5):
    fn()


######################################################################
# Sample Output:
#
# >>[DEBUG]:Recompiling function step in /data/users/mlazos/pytorch/torch/optim/adam.py:191
# >>    triggered by the following guard failure(s):
# >>    - L['self'].param_groups[0]['lr'] == 0.003333333333333333
# >>[DEBUG]:Recompiling function step in /data/users/mlazos/pytorch/torch/optim/adam.py:191
# >>    triggered by the following guard failure(s):
# >>    - L['self'].param_groups[0]['lr'] == 0.004666666666666667
# >>    - L['self'].param_groups[0]['lr'] == 0.003333333333333333
# >>[DEBUG]:Recompiling function step in /data/users/mlazos/pytorch/torch/optim/adam.py:191
# >>    triggered by the following guard failure(s):
# >>    - L['self'].param_groups[0]['lr'] == 0.006000000000000001
# >>    - L['self'].param_groups[0]['lr'] == 0.004666666666666667
# >>    - L['self'].param_groups[0]['lr'] == 0.003333333333333333
# >>[DEBUG]:Recompiling function step in /data/users/mlazos/pytorch/torch/optim/adam.py:191
# >>    triggered by the following guard failure(s):
# >>    - L['self'].param_groups[0]['lr'] == 0.007333333333333335
# >>    - L['self'].param_groups[0]['lr'] == 0.006000000000000001
# >>    - L['self'].param_groups[0]['lr'] == 0.004666666666666667
# >>    - L['self'].param_groups[0]['lr'] == 0.003333333333333333
#
# With this example, we can see that we recompile the optimizer 4 additional
# due to the guard failure on the 'lr' in param_groups[0]
