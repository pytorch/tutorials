import torch


print("BEGIN preamble")
torch.ops.load_library("build/libwarp_perspective.so")
print(torch.ops.my_ops.warp_perspective(torch.randn(32, 32), torch.rand(3, 3)))
print("END preamble")


# BEGIN compute
def compute(x, y, z):
    return x.matmul(y) + torch.relu(z)
# END compute


print("BEGIN trace")
inputs = [torch.randn(4, 8), torch.randn(8, 5), torch.randn(4, 5)]
trace = torch.jit.trace(compute, inputs)
print(trace.graph)
print("END trace")


# BEGIN compute2
def compute(x, y, z):
    x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
    return x.matmul(y) + torch.relu(z)
# END compute2


print("BEGIN trace2")
inputs = [torch.randn(4, 8), torch.randn(8, 5), torch.randn(8, 5)]
trace = torch.jit.trace(compute, inputs)
print(trace.graph)
print("END trace2")
