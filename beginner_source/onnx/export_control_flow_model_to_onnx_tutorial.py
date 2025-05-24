"""
`Introduction to ONNX <intro_onnx.html>`_ ||
`Exporting a PyTorch model to ONNX <export_simple_model_to_onnx_tutorial.html>`_ ||
`Extending the ONNX exporter operator support <onnx_registry_tutorial.html>`_ ||
**`Export a model with control flow to ONNX**

Export a model with control flow to ONNX
========================================

**Author**: `Xavier Dupré <https://github.com/xadupre>`_
"""


###############################################################################
# Overview
# --------
# 
# This tutorial demonstrates how to handle control flow logic while exporting
# a PyTorch model to ONNX. It highlights the challenges of exporting
# conditional statements directly and provides solutions to circumvent them.
#
# Conditional logic cannot be exported into ONNX unless they refactored
# to use :func:`torch.cond`. Let's start with a simple model
# implementing a test.
# 
# What you will learn:
#
# - How to refactor the model to use :func:`torch.cond` for exporting.
# - How to export a model with control flow logic to ONNX.
# - How to optimize the exported model using the ONNX optimizer.
#
# Prerequisites
# ~~~~~~~~~~~~~
#
# * ``torch >= 2.6``


import torch

###############################################################################
# Define the Models
# -----------------
#
# Two models are defined:
#
# ``ForwardWithControlFlowTest``: A model with a forward method containing an
# if-else conditional.
#
# ``ModelWithControlFlowTest``: A model that incorporates ``ForwardWithControlFlowTest``
# as part of a simple MLP. The models are tested with
# a random input tensor to confirm they execute as expected.

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


###############################################################################
# Exporting the Model: First Attempt
# ----------------------------------
#
# Exporting this model using torch.export.export fails because the control
# flow logic in the forward pass creates a graph break that the exporter cannot
# handle. This behavior is expected, as conditional logic not written using
# :func:`torch.cond` is unsupported.
# 
# A try-except block is used to capture the expected failure during the export
# process. If the export unexpectedly succeeds, an ``AssertionError`` is raised.

x = torch.randn(3)
model(x)

try:
    torch.export.export(model, (x,), strict=False)
    raise AssertionError("This export should failed unless PyTorch now supports this model.")
except Exception as e:
    print(e)

###############################################################################
# Using :func:`torch.onnx.export` with JIT Tracing
# ----------------------------------------
#
# When exporting the model using :func:`torch.onnx.export` with the dynamo=True
# argument, the exporter defaults to using JIT tracing. This fallback allows
# the model to export, but the resulting ONNX graph may not faithfully represent
# the original model logic due to the limitations of tracing.


onnx_program = torch.onnx.export(model, (x,), dynamo=True) 
print(onnx_program.model)


###############################################################################
# Suggested Patch: Refactoring with :func:`torch.cond`
# --------------------------------------------
#
# To make the control flow exportable, the tutorial demonstrates replacing the
# forward method in ``ForwardWithControlFlowTest`` with a refactored version that
# uses :func:`torch.cond``.
#
# Details of the Refactoring:
#
# Two helper functions (identity2 and neg) represent the branches of the conditional logic:
# * :func:`torch.cond`` is used to specify the condition and the two branches along with the input arguments.
# * The updated forward method is then dynamically assigned to the ``ForwardWithControlFlowTest`` instance within the model. A list of submodules is printed to confirm the replacement.

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

###############################################################################
# Let's see what the FX graph looks like.

print(torch.export.export(model, (x,), strict=False))  

###############################################################################
# Let's export again.

onnx_program = torch.onnx.export(model, (x,), dynamo=True)  
print(onnx_program.model) 


###############################################################################
# We can optimize the model and get rid of the model local functions created to capture the control flow branches.  

onnx_program.optimize()  
print(onnx_program.model)  

###############################################################################
# Conclusion
# ----------
#
# This tutorial demonstrates the challenges of exporting models with conditional
# logic to ONNX and presents a practical solution using :func:`torch.cond`.
# While the default exporters may fail or produce imperfect graphs, refactoring the
# model's logic ensures compatibility and generates a faithful ONNX representation.
#
# By understanding these techniques, we can overcome common pitfalls when
# working with control flow in PyTorch models and ensure smooth integration with ONNX workflows.
#
# Further reading
# ---------------
#
# The list below refers to tutorials that ranges from basic examples to advanced scenarios,
# not necessarily in the order they are listed.
# Feel free to jump directly to specific topics of your interest or
# sit tight and have fun going through all of them to learn all there is about the ONNX exporter.
#
# .. include:: /beginner_source/onnx/onnx_toc.txt
#
# .. toctree::
#    :hidden:
#