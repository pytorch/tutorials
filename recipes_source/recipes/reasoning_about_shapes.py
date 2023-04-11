"""
Reasoning about Shapes in PyTorch
=================================

When writing models with PyTorch, it is commonly the case that the parameters
to a given layer depend on the shape of the output of the previous layer. For
example, the ``in_features`` of an ``nn.Linear`` layer must match the
``size(-1)`` of the input. For some layers, the shape computation involves
complex equations, for example convolution operations.

One way around this is to run the forward pass with random inputs, but this is
wasteful in terms of memory and compute.

Instead, we can make use of the ``meta`` device to determine the output shapes
of a layer without materializing any data.
"""

import torch
import timeit

t = torch.rand(2, 3, 10, 10, device="meta")
conv = torch.nn.Conv2d(3, 5, 2, device="meta")
start = timeit.default_timer()
out = conv(t)
end = timeit.default_timer()

print(out)
print(f"Time taken: {end-start}")


##########################################################################
# Observe that since data is not materialized, passing arbitrarily large
# inputs will not significantly alter the time taken for shape computation.

t_large = torch.rand(2**10, 3, 2**16, 2**16, device="meta")
start = timeit.default_timer()
out = conv(t_large)
end = timeit.default_timer()

print(out)
print(f"Time taken: {end-start}")


######################################################
# Consider an arbitrary network such as the following:

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


###############################################################################
# We can view the intermediate shapes within an entire network by registering a
# forward hook to each layer that prints the shape of the output.

def fw_hook(module, input, output):
    print(f"Shape of output to {module} is {output.shape}.")


# Any tensor created within this torch.device context manager will be
# on the meta device.
with torch.device("meta"):
    net = Net()
    inp = torch.randn((1024, 3, 32, 32))

for name, layer in net.named_modules():
    layer.register_forward_hook(fw_hook)

out = net(inp)
