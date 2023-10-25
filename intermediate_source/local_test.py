import torch
from torch.export import dynamic_dim, export

def fn(x, y):
    z = x.clone()
    z.copy_(y)
    return z

inp1 = torch.randn(10, 10)
inp2 = torch.randn(1, 10)
constraints = (
    [dynamic_dim(inp1, i) for i in range(inp1.dim())] +
    [dynamic_dim(inp2, i) for i in range(inp2.dim())]
)
exp1 = export(fn, (inp1, inp2))
# exp1 = export(fn, (inp1, inp2), constraints=constraints)
exp1.graph_module.print_readable()
# exp(torch.randn(10, 10), torch.randn(10, 10))
exp2 = export(fn, (torch.randn(10, 10), torch.randn(10, 10)))
exp2.graph_module.print_readable()
