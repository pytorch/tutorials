"""
Changing default device
=======================
It is common to want to write PyTorch code in a device agnostic way,
and then switch between CPU and CUDA depending on what hardware is available.
Traditionally, to do this you might have used if-statements and cuda() calls
to do this:
"""

import torch

USE_CUDA = False

mod = torch.nn.Linear(20, 30)
if USE_CUDA:
    mod.cuda()

device = 'cpu'
if USE_CUDA:
    device = 'cuda'
inp = torch.randn(128, 20, device=device)
print(mod(inp).device)

# PyTorch now also has a context manager which can take care of the
# device transfer automatically.

with torch.device('cuda'):
    mod = torch.nn.Linear(20, 30)
    print(mod.weight.device)
    print(mod(torch.randn(128, 20)).device)

# You can also set it globally

torch.set_default_device('cuda')

mod = torch.nn.Linear(20, 30)
print(mod.weight.device)
print(mod(torch.randn(128, 20)).device)

# This function imposes a slight performance cost on every Python
# call to the torch API (not just factory functions).  If this
# is causing problems for you, please comment on
# https://github.com/pytorch/pytorch/issues/92701
