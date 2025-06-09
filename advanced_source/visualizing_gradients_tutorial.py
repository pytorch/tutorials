"""
Visualizing Gradients
=====================

**Author**: `Justin Silver <https://github.com/j-silv>`_

By performance and efficiency reasons, PyTorch does not save the
intermediate gradients when running back-propagation. To visualize the
gradients of these internal layer tensor, we have to explicitly tell
PyTorch to retain those values with the ``retain_grad`` parameter.

By the end of this tutorial, you will be able to:

-  Visualize gradients after backward propagation in a neural network
-  Differentiate between *leaf* and *non-leaf* tensors
-  Know when to use\ ``retain_grad`` vs. ``require_grad``

"""


######################################################################
# Introduction
# ------------
# 
# When training neural networks with PyTorch, it is easy to disregard the
# internal mechanisms of the PyTorch library. For example, to run
# back-propagation the API requires a single call to ``loss.backward()``.
# This tutorial will dive into how exactly those gradients are calculated
# and stored in two different kinds of PyTorch tensors: *leaf*, and
# *non-leaf*. It will also cover how we can extract and visualize
# gradients at any neuron in the computational graph. Some important
# barriers to efficient neural network training are vanishing/exploding
# gradients, which lead to slow training progress and/or broken
# optimization pipelines. Thus, it is important to understand how
# information flows from one end of the network, through the computational
# graph, and finally to the parameters we want to optimize.
# 


######################################################################
# Setup
# -----
# 
# First, make sure PyTorch is installed and then import the necessary
# libraries
# 

import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# Next, we will instantiate an extremely simple network so that we can
# focus on the gradients. This will be an affine layer followed by a ReLU
# activation. Note that the ``requires_grad=True`` is necessary for the
# parameters (``W`` and ``b``) so that PyTorch tracks operations involving
# those tensors. We’ll discuss more about this attribute shortly.
# 

# tensor setup
x = torch.ones(1, 3)                      # input with shape: (1, 3)
W = torch.ones(3, 2, requires_grad=True)  # weights with shape: (3, 2)               
b = torch.ones(1, 2, requires_grad=True)  # bias with shape: (1, 2)
y = torch.ones(1, 2)                      # output with shape: (1, 2) 

# forward pass
z = (x @ W) + b                           # pre-activation with shape: (1, 2)
y_pred = F.relu(z)                        # activation with shape: (1, 2)
loss = F.mse_loss(y_pred, y)              # scalar loss


######################################################################
# Before we perform back-propagation on this network, we need to know the
# difference between *leaf* and *non-leaf* nodes. This is important
# because the distinction affects how gradients are calculated and stored.
# 


######################################################################
# Leaf vs. non-leaf tensors
# -------------------------
# 
# The backbone for PyTorch Autograd is a dynamic computational graph which
# keeps a record of input tensor data, all subsequent operations on those
# tensors, and finally the resulting new tensors. It is a directed acyclic
# graph (DAG) which can be used to compute gradients along every node all
# the way from the roots (output tensors) to the leaves (input tensors)
# using the chain rule from calculus.
# 
# In the context of a generic DAG then, a *leaf* is simply a node which is
# at the input (beginning) of the graph, and *non-leaf* nodes are
# everything else.
# 
# To start the generation of the computational graph which can be used for
# gradient calculation, we need to pass in the ``requires_grad=True``
# parameter to the tensor constructors. That is because by default,
# PyTorch is not tracking gradients on any created tensors. To verify
# this, try removing the parameter above and then run back-propagation:
# 
# ::
# 
#    >>> loss.backward()
#    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# 
# This runtime error is telling us that the tensor is not tracking
# gradients and has no associated gradient function. Thus, it cannot
# back-propagate to the leaf tensors and calculate the gradients for each
# node.
# 
# From the above discussion, we can see that ``x``, ``W``, ``b``, and
# ``y`` are leaf tensors, whereas ``z``, ``y_pred``, and ``loss`` are
# non-leaf tensors. We can verify this with the class attribute
# ``is_leaf()``:
# 

# prints all True because new tensors are leafs by convention
print(f"{x.is_leaf=}")
print(f"{W.is_leaf=}")      
print(f"{b.is_leaf=}")      
print(f"{y.is_leaf=}")      

# prints all False because tensors are the result of an operation
# with at least one tensor having requires_grad=True
print(f"{z.is_leaf=}")      
print(f"{y_pred.is_leaf=}") 
print(f"{loss.is_leaf=}")  


######################################################################
# The distinction between leaf and non-leaf is important, because that
# attribute determines whether the tensor’s gradient will be stored in the
# ``grad`` property after the backward pass, and thus be usable for
# gradient descent optimization. We’ll cover this some more in the
# following section.
# 
# Also note that by convention, when the user creates a new tensor,
# PyTorch automatically makes it a leaf node. This is the case even though
# is no computational graph associated with the tensor. For example:
# 

a = torch.tensor([1.0, 5.0, 2.0])
a.is_leaf


######################################################################
# Now that we understand what makes a tensor a leaf vs. non-leaf, the
# second piece of the puzzle is knowing when PyTorch calculates and stores
# gradients for the tensors in its computational graph.
# 


######################################################################
# ``requires_grad``
# =================
# 
# To tell PyTorch to explicitly start tracking gradients, when we create
# the tensor, we can pass in the parameter ``requires_grad=True`` to the
# class constructor (by default it is ``False``). This tells PyTorch to
# treat the tensor as a leaf tensor, and all the subsequent operations
# will generate results which also need to require the gradient for
# back-propagation to work. This is because the backward pass uses the
# chain rule from calculus, where intermediate gradients ‘flow’ backward
# through the network.
# 
# We already did this for the parameters we want to optimize, so we’re
# good. If you need to change the property though, you can call
# ``requires_grad_()`` on the tensor to change it (notice the ``_``
# suffix).
# 
# Similar to the analysis above, we can sanity-check which nodes in our
# network have to calculate the gradient for back-propagation to work.
# 

# prints all False because tensors are leaf nodes
print(f"{x.requires_grad=}")       
print(f"{y.requires_grad=}")       

# prints all True because requires_grad=True in constructor
print(f"{W.requires_grad=}")       
print(f"{b.requires_grad=}")       

# prints all True because tensors are non-leaf nodes
print(f"{z.requires_grad=}")      
print(f"{y_pred.requires_grad=}") 
print(f"{loss.requires_grad=}")  


######################################################################
# A useful heuristic to remember is that whenever a tensor is a non-leaf,
# it **has** to have ``requires_grad=True``, otherwise back-propagation
# would fail. If the tensor is a leaf, then it will only have
# ``requires_grad=True`` if it was specifically set by the user. Another
# way to phrase this is that if at least one of the inputs to the tensor
# requires the gradient, then it will require the gradient as well.
# 
# There are two exceptions to the above guideline:
# 
# 1. Using ``nn.Module`` and ``nn.Parameter``
# 2. `Locally disabling gradient computation with context
#    managers <https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation>`__
# 
# For the first case, if you subclass the ``nn.Module`` base class, then
# by default all of the parameters of that module will have
# ``requires_grad`` automatically set to ``True``. e.g.:
# 

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
    
m = Model()

for name, param in m.named_parameters():
    print(name, param.requires_grad)


######################################################################
# For the second case, if you wrap one of the gradient context managers
# around a tensor, then computations behave as if none of the inputs
# require grad.
# 

z = (x @ W) + b # same as before

with torch.no_grad(): # could also use torch.inference_mode() 
    z2 = (x @ W) + b
    
print(f"{z.requires_grad=}")
print(f"{z2.requires_grad=}")


######################################################################
# In summary, ``requires_grad`` tells autograd which tensors need to have
# their gradients calculated for back-propagation to work. This is
# different from which gradients have to be stored inside the tensor,
# which is the topic of the next section.
# 


######################################################################
# Back-propagation
# ----------------
# 
# To actually perform optimization (e.g. SGD, Adam, etc.), we need to run
# the backward pass so that we can extract the gradients.
# 

loss.backward()


######################################################################
# This single function call populated the ``grad`` property of all leaf
# tensors which had their ``requires_grad=True``. The ``grad`` is the
# gradient of the loss with respect to the tensor we are probing.
# 

print(f"{W.grad=}")
print(f"{b.grad=}")     


######################################################################
# You might be wondering about the other tensors in our network. Let’s
# check the remaining leaf nodes:
# 

print(f"{x.grad=}")
print(f"{y.grad=}")


######################################################################
# Interestingly, these gradients haven’t been populated into the ``grad``
# property and they default to ``None``. This is expected behavior though
# because we did not explicitly tell PyTorch to calculate gradient with
# the ``requires_grad`` parameter.
# 
# Let’s now look at an intermediate non-leaf node:
# 

print(f"{z.grad=}")


######################################################################
# We also get ``None`` for the gradient, but now PyTorch warns us that a
# non-leaf node’s ``grad`` attribute is being accessed. It might come as a
# surprise that we can’t access the gradient for intermediate tensors in
# the computational graph, since they **have** to calculate the gradient
# for back-propagation to work. PyTorch errs on the side of performance
# and assumes that you don’t need to access intermediate gradients if
# you’re trying to optimize leaf tensors. To change this behavior, we can
# use the ``retain_grad()`` function.
# 


######################################################################
# ``retain_grad``
# ---------------
# 
# When we call ``retain_grad()`` on a tensor, this signals to the autograd
# engine that we want to have that tensor’s ``grad`` populated after
# calling ``backward()``.
# 
# We can verify that PyTorch is not storing gradients for non-leaf tensors
# by accessing the ``retains_grad`` flag:
# 

# Prints all False because we didn't tell PyTorch to store gradients with `retain_grad()`
print(f"{z.retains_grad=}")      
print(f"{y_pred.retains_grad=}")
print(f"{loss.retains_grad=}")


######################################################################
# We can also check the other leaf tensors, but note that by convention,
# this attribute will print ``False`` for any leaf node, even if that
# tensor was set to require its gradient. This is true even if you call
# ``retain_grad()`` on a leaf node that has ``requires_grad=True``, which
# results in a no-op.
# 

# Prints all False because these are leaf tensors
print(f"{x.retains_grad=}")  
print(f"{y.retains_grad=}")
print(f"{b.retains_grad=}")
print(f"{W.retains_grad=}")

W.retain_grad()
print(f"{W.retains_grad=}") # still False


######################################################################
# If we try calling ``retain_grad()`` on a node that has
# ``require_grad=False``, PyTorch actually throws an error.
# 
# ::
# 
#    >>> x.retain_grad()
#    RuntimeError: can't retain_grad on Tensor that has requires_grad=False
# 


######################################################################
# In summary, using ``retain_grad()`` and ``retains_grad`` only make sense
# for non-leaf nodes, since the ``grad`` attribute has to be populated for
# leaf tensors that have ``requires_grad=True``. By default, these
# non-leaf nodes do not retain (store) their gradient after
# back-propagation.
# 
# We can change that by rerunning the forward pass, telling PyTorch to
# store the gradients, and then performing back-propagation.
# 

# forward pass
z = (x @ W) + b                          
y_pred = F.relu(z)               
loss = F.mse_loss(y_pred, y)

# tell PyTorch to store the gradients after backward()
z.retain_grad()
y_pred.retain_grad()
loss.retain_grad()

# have to zero out gradients otherwise they would accumulate
W.grad = None
b.grad = None

# back-propagation
loss.backward()

# print gradients for all tensors that have requires_grad=True
print(f"{W.grad=}")
print(f"{b.grad=}")
print(f"{z.grad=}")      
print(f"{y_pred.grad=}")
print(f"{loss.grad=}")


######################################################################
# Note we get the same result for ``W.grad`` as before. Also note that
# because the loss is scalar, the gradient of the loss with respect to
# itself is simply ``1.0``.
# 


######################################################################
# (work-in-progress) Real-world example - visualizing gradient flow
# -----------------------------------------------------------------
# 
# We used a toy example above, but let’s now apply the concepts we learned
# to the visualization of intermediate gradients in a more powerful neural
# network: ResNet.
# 


######################################################################
# (work-in-progress) Conclusion
# -----------------------------
# 
# This table can be used as a cheat-sheet which summarizes the above
# discussions. The following scenarios are the only ones that are valid
# for PyTorch tensors.
# 
# ============  ==================  ================  ===================================  =============================
# ``is_leaf``   ``requires_grad``   ``retains_grad``  ``require_grad()``                   ``retain_grad()``
# ============  ==================  ================  ===================================  =============================
# True          False               False             sets ``require_grad`` to True/False  no-op                                            
# True          True                False             sets ``require_grad`` to True/False  no-op                                                                       
# False         True                False             no-op                                sets ``retains_grad`` to True
# False         True                True              no-op                                no-op
# ============  ==================  ================  ===================================  =============================


######################################################################
# References
# ----------
# 
# https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial
# 
# https://docs.pytorch.org/docs/stable/notes/autograd.html#setting-requires-grad
# 