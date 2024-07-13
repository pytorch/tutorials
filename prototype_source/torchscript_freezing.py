"""
Model Freezing in TorchScript
=============================

In this tutorial, we introduce the syntax for *model freezing* in TorchScript.
Freezing is the process of inlining Pytorch module parameters and attributes
values into the TorchScript internal representation. Parameter and attribute
values are treated as final values and they cannot be modified in the resulting
Frozen module.

Basic Syntax
------------
Model freezing can be invoked using API below:

 ``torch.jit.freeze(mod : ScriptModule, names : str[]) -> ScriptModule``

Note the input module can either be the result of scripting or tracing.
See https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html

Next, we demonstrate how freezing works using an example:
"""

import torch, time

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

    @torch.jit.export
    def version(self):
        return 1.0

net = torch.jit.script(Net())
fnet = torch.jit.freeze(net)

print(net.conv1.weight.size())
print(net.conv1.bias)

try:
    print(fnet.conv1.bias)
    # without exception handling, prints:
    # RuntimeError: __torch__.z.___torch_mangle_3.Net does not have a field
    # with name 'conv1'
except RuntimeError:
    print("field 'conv1' is inlined. It does not exist in 'fnet'")

try:
    fnet.version()
    # without exception handling, prints:
    # RuntimeError: __torch__.z.___torch_mangle_3.Net does not have a field
    # with name 'version'
except RuntimeError:
    print("method 'version' is not deleted in fnet. Only 'forward' is preserved")

fnet2 = torch.jit.freeze(net, ["version"])

print(fnet2.version())

B=1
warmup = 1
iter = 1000
input = torch.rand(B, 1,28, 28)

start = time.time()
for i in range(warmup):
    net(input)
end = time.time()
print("Scripted - Warm up time: {0:7.4f}".format(end-start), flush=True)

start = time.time()
for i in range(warmup):
    fnet(input)
end = time.time()
print("Frozen   - Warm up time: {0:7.4f}".format(end-start), flush=True)

start = time.time()
for i in range(iter):
    input = torch.rand(B, 1,28, 28)
    net(input)
end = time.time()
print("Scripted - Inference: {0:5.2f}".format(end-start), flush=True)

start = time.time()
for i in range(iter):
    input = torch.rand(B, 1,28, 28)
    fnet2(input)
end = time.time()
print("Frozen    - Inference time: {0:5.2f}".format(end-start), flush =True)

###############################################################
# On my machine, I measured the time:
#
# * Scripted - Warm up time:  0.0107
# * Frozen   - Warm up time:  0.0048
# * Scripted - Inference:  1.35
# * Frozen   - Inference time:  1.17

###############################################################
# In our example, warm up time measures the first two runs. The frozen model
# is 50% faster than the scripted model. On some more complex models, we
# observed even higher speed up of warm up time. freezing achieves this speed up
# because it is doing some the work TorchScript has to do when the first couple
# runs are initiated.
#
# Inference time measures inference execution time after the model is warmed up.
# Although we observed significant variation in execution time, the
# frozen model is often about 15% faster than the scripted model. When input is larger,
# we observe a smaller speed up because the execution is dominated by tensor operations.

###############################################################
# Conclusion
# -----------
# In this tutorial, we learned about model freezing. Freezing is a useful technique to
# optimize models for inference and it also can significantly reduce TorchScript warmup time.
