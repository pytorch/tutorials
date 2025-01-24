# -*- coding: utf-8 -*-
"""
`Introduction to ONNX <intro_onnx.html>`_ ||
`Exporting a PyTorch model to ONNX <export_simple_model_to_onnx_tutorial.html>`_ ||
`Extending the ONNX exporter operator support <onnx_registry_tutorial.html>`_ ||
**`Export a model with control flow to ONNX**

Export a model with control flow to ONNX
==========================================

**Author**: `Xavier Dupré <https://github.com/xadupre>`_.

Conditional logic cannot be exported into ONNX unless they refactored
to use :func:`torch.cond`. Let's start with a simple model
implementing a test.
"""

import torch

class ForwardWithControlFlowTest(torch.nn.Module):
    def forward(self, x):
        if x.sum():
            return x * 2
        return -x


class ModelWithControlFlowTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 2),
            torch.nn.Linear(2, 1),
            ForwardWithControlFlowTest(),
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


model = ModelWithControlFlowTest()

# %%
# Let's check it runs.
x = torch.randn(3)
model(x)

# %%
# As expected, it does not export.
try:
    torch.export.export(model, (x,), strict=False)
    raise AssertionError("This export should failed unless pytorch now supports this model.")
except Exception as e:
    print(e)

# %%
# It does export with :func:`torch.onnx.export` because
# the exporter falls back to use JIT tracing as the graph capturing strategy.
# But the model is not exactly the same as the initial model.
onnx_program = torch.onnx.export(model, (x,), dynamo=True) 
print(onnx_program.model)


# %%
# Suggested Patch
# +++++++++++++++
#
# Let's avoid the graph break by replacing the forward.


def new_forward(x):
    def identity2(x):
        return x * 2

    def neg(x):
        return -x

    return torch.cond(x.sum() > 0, identity2, neg, (x,))


print("the list of submodules")
for name, mod in model.named_modules():
    print(name, type(mod))
    if isinstance(mod, ForwardWithControlFlowTest):
        mod.forward = new_forward

# %%
# Let's see what the fx graph looks like.

print(torch.export.export(model, (x,), strict=False))  

# %%
# Let's export again.

onnx_program = torch.onnx.export(model, (x,), dynamo=True)  
print(onnx_program.model) 


# %%  
# We can optimize the model and get rid of the model local functions created to capture the control flow branches.  

onnx_program = torch.onnx.export(model, (x,), dynamo=True)  
onnx_program.optimize()  
print(onnx_program.model)  