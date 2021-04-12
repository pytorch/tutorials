import torch

torch.ops.load_library("build/libdispatcher.so")
print(torch.ops.myops.myadd(torch.randn(32, 32), torch.rand(32, 32)))
"""
# Doesn't currently work, because Python frontend on torch.ops doesn't
# support names (for not a good reason?)
x = torch.randn(32, 32, names=('A', 'B'))
y = torch.rand(32, 32, names=('A', 'B'))
print(torch.ops.myops.myadd(x, y))
"""
