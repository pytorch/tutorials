# -*- coding: utf-8 -*-
"""
(beta) Building a Convolution/Batch Norm fuser in FX
*******************************************************
**Author**: `Horace He <https://github.com/chillee>`_

In this tutorial, we are going to use FX, a toolkit for composable function
transformations of PyTorch, to do the following:

1) Find patterns of conv/batch norm in the data dependencies.
2) For the patterns found in 1), fold the batch norm statistics into the convolution weights.

Note that this optimization only works for models in inference mode (i.e. `mode.eval()`)

We will be building the fuser that exists here:
https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/fx/experimental/fuser.py

"""


######################################################################
# First, let's get some imports out of the way (we will be using all
# of these later in the code).

from typing import Type, Dict, Any, Tuple, Iterable
import copy
import torch.fx as fx
import torch
import torch.nn as nn

######################################################################
# For this tutorial, we are going to create a model consisting of convolutions
# and batch norms. Note that this model has some tricky components - some of
# the conv/batch norm patterns are hidden within Sequentials and one of the
# ``BatchNorms`` is wrapped in another Module.

class WrappedBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = nn.BatchNorm2d(1)
    def forward(self, x):
        return self.mod(x)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.nested = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, 1),
        )
        self.wrapped = WrappedBatchNorm()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.nested(x)
        x = self.wrapped(x)
        return x

model = M()

model.eval()

######################################################################
# Fusing Convolution with Batch Norm
# -----------------------------------------
# One of the primary challenges with trying to automatically fuse convolution
# and batch norm in PyTorch is that PyTorch does not provide an easy way of
# accessing the computational graph. FX resolves this problem by symbolically
# tracing the actual operations called, so that we can track the computations
# through the `forward` call, nested within Sequential modules, or wrapped in
# an user-defined module.

traced_model = torch.fx.symbolic_trace(model)
print(traced_model.graph)

######################################################################
# This gives us a graph representation of our model. Note that both the modules
# hidden within the sequential as well as the wrapped Module have been inlined
# into the graph. This is the default level of abstraction, but it can be
# configured by the pass writer. More information can be found at the FX
# overview https://pytorch.org/docs/master/fx.html#module-torch.fx


####################################
# Fusing Convolution with Batch Norm
# ----------------------------------
# Unlike some other fusions, fusion of convolution with batch norm does not
# require any new operators. Instead, as batch norm during inference
# consists of a pointwise add and multiply, these operations can be "baked"
# into the preceding convolution's weights. This allows us to remove the batch
# norm entirely from our model! Read
# https://nenadmarkus.com/p/fusing-batchnorm-and-conv/ for further details. The
# code here is copied from
# https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/nn/utils/fusion.py
# clarity purposes.
def fuse_conv_bn_eval(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


####################################
# FX Fusion Pass
# ----------------------------------
# Now that we have our computational graph as well as a method for fusing
# convolution and batch norm, all that remains is to iterate over the FX graph
# and apply the desired fusions.


def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a ``qualname`` into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    # The first step of most FX passes is to symbolically trace our model to
    # obtain a `GraphModule`. This is a representation of our original model
    # that is functionally identical to our original model, except that we now
    # also have a graph representation of our forward pass.
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    # The primary representation for working with FX are the `Graph` and the
    # `Node`. Each `GraphModule` has a `Graph` associated with it - this
    # `Graph` is also what generates `GraphModule.code`.
    # The `Graph` itself is represented as a list of `Node` objects. Thus, to
    # iterate through all of the operations in our graph, we iterate over each
    # `Node` in our `Graph`.
    for node in fx_model.graph.nodes:
        # The FX IR contains several types of nodes, which generally represent
        # call sites to modules, functions, or methods. The type of node is
        # determined by `Node.op`.
        if node.op != 'call_module': # If our current node isn't calling a Module then we can ignore it.
            continue
        # For call sites, `Node.target` represents the module/function/method
        # that's being called. Here, we check `Node.target` to see if it's a
        # batch norm module, and then check `Node.args[0].target` to see if the
        # input `Node` is a convolution.
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            # As we've folded the batch nor into the conv, we need to replace all uses
            # of the batch norm with the conv.
            node.replace_all_uses_with(node.args[0])
            # Now that all uses of the batch norm have been replaced, we can
            # safely remove the batch norm.
            fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    # After we've modified our graph, we need to recompile our graph in order
    # to keep the generated code in sync.
    fx_model.recompile()
    return fx_model


######################################################################
# .. note::
#       We make some simplifications here for demonstration purposes, such as only
#       matching 2D convolutions. View
#       https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/fuser.py
#       for a more usable pass.

######################################################################
# Testing out our Fusion Pass
# -----------------------------------------
# We can now run this fusion pass on our initial toy model and verify that our
# results are identical. In addition, we can print out the code for our fused
# model and verify that there are no more batch norms.


fused_model = fuse(model)
print(fused_model.code)
inp = torch.randn(5, 1, 1, 1)
torch.testing.assert_allclose(fused_model(inp), model(inp))


######################################################################
# Benchmarking our Fusion on ResNet18
# -----------------------------------
# We can test our fusion pass on a larger model like ResNet18 and see how much
# this pass improves inference performance.
import torchvision.models as models
import time

rn18 = models.resnet18()
rn18.eval()

inp = torch.randn(10, 3, 224, 224)
output = rn18(inp)

def benchmark(model, iters=20):
    for _ in range(10):
        model(inp)
    begin = time.time()
    for _ in range(iters):
        model(inp)
    return str(time.time()-begin)

fused_rn18 = fuse(rn18)
print("Unfused time: ", benchmark(rn18))
print("Fused time: ", benchmark(fused_rn18))
######################################################################
# As we previously saw, the output of our FX transformation is
# ("torchscriptable") PyTorch code, we can easily ``jit.script`` the output to try
# and increase our performance even more. In this way, our FX model
# transformation composes with TorchScript with no issues.
jit_rn18 = torch.jit.script(fused_rn18)
print("jit time: ", benchmark(jit_rn18))


############
# Conclusion
# ----------
# As we can see, using FX we can easily write static graph transformations on
# PyTorch code.
#
# Since FX is still in beta, we would be happy to hear any
# feedback you have about using it. Please feel free to use the
# PyTorch Forums (https://discuss.pytorch.org/) and the issue tracker
# (https://github.com/pytorch/pytorch/issues) to provide any feedback
# you might have.
