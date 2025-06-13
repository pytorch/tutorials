"""
Visualizing Gradients
=====================

**Author:** `Justin Silver <https://github.com/j-silv>`__

When training neural networks with PyTorch, it’s possible to ignore some
of the library’s internal mechanisms. For example, running
backpropagation requires a simple call to ``backward()``. This tutorial
dives into how those gradients are calculated and stored in two
different kinds of PyTorch tensors: leaf vs. non-leaf. It will also
cover how we can extract and visualize gradients at any layer in the
network’s computational graph. By inspecting how information flows from
the end of the network to the parameters we want to optimize, we can
debug issues that occur during training such as `vanishing or exploding
gradients <https://arxiv.org/abs/1211.5063>`__.

By the end of this tutorial, you will be able to:

-  Differentiate leaf vs. non-leaf tensors
-  Know when to use ``requires_grad`` vs. ``retain_grad``
-  Visualize gradients after backpropagation in a neural network

We will start off with a simple network to understand how PyTorch
calculates and stores gradients, and then build on this knowledge to
visualize the gradient flow of a `ResNet
model <https://docs.pytorch.org/vision/2.0/models/resnet.html>`__.

Before starting, it is recommended to have a solid understanding of
`tensors and how to manipulate
them <https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>`__.
A basic knowledge of `how autograd
works <https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html>`__
would also be useful.

"""


######################################################################
# Setup
# -----
# 
# First, make sure `PyTorch is
# installed <https://pytorch.org/get-started/locally/>`__ and then import
# the necessary libraries.
# 

import torch
import torchvision
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


######################################################################
# Next, we will instantiate a simple network so that we can focus on the
# gradients. This will be an affine layer, followed by a ReLU activation,
# and ending with a MSE loss between the prediction and label tensors.
# 
# .. math::
# 
#    \mathbf{y}_{\text{pred}} = \text{ReLU}(\mathbf{x} \mathbf{W} + \mathbf{b})
# 
# .. math::
# 
#    L = \text{MSE}(\mathbf{y}_{\text{pred}}, \mathbf{y})
# 
# Note that the ``requires_grad=True`` is necessary for the parameters
# (``W`` and ``b``) so that PyTorch tracks operations involving those
# tensors. We’ll discuss more about this in a future
# `section <#requires-grad>`__.
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
# Leaf vs. non-leaf tensors
# -------------------------
# 
# After running the forward pass, PyTorch autograd has built up a `dynamic
# computational
# graph <https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#computational-graph>`__
# which is shown below. This is a `Directed Acyclic Graph
# (DAG) <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`__ which
# keeps a record of input tensors (leaf nodes), all subsequent operations
# on those tensors, and the intermediate/output tensors (non-leaf nodes).
# The graph is used to compute gradients for each tensor starting from the
# graph roots (outputs) to the leaves (inputs) using the `chain
# rule <https://en.wikipedia.org/wiki/Chain_rule>`__ from calculus:
# 
# .. math::
# 
#    \mathbf{y} = \mathbf{f}_k\bigl(\mathbf{f}_{k-1}(\dots \mathbf{f}_1(\mathbf{x}) \dots)\bigr)
# 
# .. math::
# 
#    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
#    \frac{\partial \mathbf{f}_k}{\partial \mathbf{f}_{k-1}} \cdot
#    \frac{\partial \mathbf{f}_{k-1}}{\partial \mathbf{f}_{k-2}} \cdot
#    \cdots \cdot
#    \frac{\partial \mathbf{f}_1}{\partial \mathbf{x}}
# 
# .. figure:: /_static/img/visualizing_gradients_tutorial/comp-graph-1.png
#    :alt: Computational graph after forward pass
# 
#    Computational graph after forward pass
# 


######################################################################
# PyTorch considers a node to be a *leaf* if it is not the result of a
# tensor operation with at least one input having ``requires_grad=True``
# (e.g. ``x``, ``W``, ``b``, and ``y``), and everything else to be
# *non-leaf* (e.g. ``z``, ``y_pred``, and ``loss``). You can verify this
# programmatically by probing the ``is_leaf`` attribute of the tensors:
# 

# prints True because new tensors are leafs by convention
print(f"{x.is_leaf=}")

# prints False because tensor is the result of an operation with at
# least one input having requires_grad=True
print(f"{z.is_leaf=}")      


######################################################################
# The distinction between leaf and non-leaf determines whether the
# tensor’s gradient will be stored in the ``grad`` property after the
# backward pass, and thus be usable for gradient descent optimization.
# We’ll cover this some more in the `following section <#retain-grad>`__.
# 
# Let’s now investigate how PyTorch calculates and stores gradients for
# the tensors in its computational graph.
# 


######################################################################
# ``requires_grad``
# -----------------
# 
# To start the generation of the computational graph which can be used for
# gradient calculation, we need to pass in the ``requires_grad=True``
# parameter to a tensor constructor. By default, the value is ``False``,
# and thus PyTorch does not track gradients on any created tensors. To
# verify this, try not setting ``requires_grad``, re-run the forward pass,
# and then run backpropagation. You will see:
# 
# ::
# 
#    >>> loss.backward()
#    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# 
# PyTorch is telling us that because the tensor is not tracking gradients,
# autograd can’t backpropagate to any leaf tensors. If you need to change
# the property, you can call ``requires_grad_()`` on the tensor (notice
# the ’_’ suffix).
# 
# We can sanity-check which nodes require gradient calculation, just like
# we did above with the ``is_leaf`` attribute:
# 

print(f"{x.requires_grad=}") # prints False because requires_grad=False by default     
print(f"{W.requires_grad=}") # prints True because we set requires_grad=True in constructor       
print(f"{z.requires_grad=}") # prints True because tensor is a non-leaf node


######################################################################
# It’s useful to remember that by definition a non-leaf tensor has
# ``requires_grad=True``. Backpropagation would fail if this wasn’t the
# case. If the tensor is a leaf, then it will only have
# ``requires_grad=True`` if it was specifically set by the user. Another
# way to phrase this is that if at least one of the inputs to the tensor
# requires the gradient, then it will require the gradient as well.
# 
# There are two exceptions to this rule:
# 
# 1. Any ``nn.Module`` that has ``nn.Parameter`` will have
#    ``requires_grad=True`` for its parameters (see
#    `here <https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#creating-models>`__)
# 2. Locally disabling gradient computation with context managers (see
#    `here <https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation>`__)
# 


######################################################################
# In summary, ``requires_grad`` tells autograd which tensors need to have
# their gradients calculated for backpropagation to work. This is
# different from which gradients have to be stored inside the tensor,
# which is the topic of the next section.
# 


######################################################################
# ``retain_grad``
# ---------------
# 
# To actually perform optimization (e.g. SGD, Adam, etc.), we need to run
# the backward pass so that we can extract the gradients.
# 

loss.backward()


######################################################################
# This single function call populated the ``grad`` property of all leaf
# tensors which had ``requires_grad=True``. The ``grad`` is the gradient
# of the loss with respect to the tensor we are probing. Before running
# ``backward()``, this attribute is set to ``None``.
# 

print(f"{W.grad=}")
print(f"{b.grad=}")     


######################################################################
# You might be wondering about the other tensors in our network. Let’s
# check the remaining leaf nodes:
# 

# prints all None because requires_grad=False
print(f"{x.grad=}")
print(f"{y.grad=}") 


######################################################################
# The gradients for these tensors haven’t been populated because we did
# not explicitly tell PyTorch to calculate their gradient
# (``requires_grad=False``).
# 
# Let’s now look at an intermediate non-leaf node:
# 

print(f"{z.grad=}")


######################################################################
# We also get ``None`` for the gradient, but now PyTorch warns us that a
# non-leaf node’s ``grad`` attribute is being accessed. Although autograd
# has to calculate intermediate gradients for backpropagation to work, it
# assumes you don’t need to access the values afterwards. To change this
# behavior, we can use the ``retain_grad()`` function on a tensor. This
# tells the autograd engine to populate that tensor’s ``grad`` after
# calling ``backward()``.
# 

# we have to re-run the forward pass
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

# backpropagation
loss.backward()

# print gradients for all tensors that have requires_grad=True
print(f"{W.grad=}")
print(f"{b.grad=}")
print(f"{z.grad=}")      
print(f"{y_pred.grad=}")
print(f"{loss.grad=}")


######################################################################
# We get the same result for ``W.grad`` as before. Also note that because
# the loss is scalar, the gradient of the loss with respect to itself is
# simply ``1.0``.
# 
# If we look at the state of the computational graph now, we see that the
# ``retains_grad`` attribute has changed for the intermediate tensors. By
# convention, this attribute will print ``False`` for any leaf node, even
# if it requires its gradient.
# 
# .. figure:: /_static/img/visualizing_gradients_tutorial/comp-graph-2.png
#    :alt: Computational graph after backward pass
# 
#    Computational graph after backward pass
# 


######################################################################
# If you call ``retain_grad()`` on a non-leaf node, it results in a no-op.
# If we call ``retain_grad()`` on a node that has ``requires_grad=False``,
# PyTorch actually throws an error, since it can’t store the gradient if
# it is never calculated.
# 
# ::
# 
#    >>> x.retain_grad()
#    RuntimeError: can't retain_grad on Tensor that has requires_grad=False
# 
# In summary, using ``retain_grad()`` and ``retains_grad`` only make sense
# for non-leaf nodes, since the ``grad`` attribute will already be
# populated for leaf tensors that have ``requires_grad=True``. By default,
# these non-leaf nodes do not retain (store) their gradient after
# backpropagation. We can change that by rerunning the forward pass,
# telling PyTorch to store the gradients, and then performing
# backpropagation.
# 
# The following table can be used as a cheat-sheet which summarizes the
# above discussions. The following scenarios are the only ones that are
# valid for PyTorch tensors.
# 
# 
# 
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
# |  ``is_leaf``   |   ``requires_grad``    |   ``retains_grad``     |  ``require_grad()``                               |   ``retain_grad()``                 |
# +================+========================+========================+===================================================+=====================================+
# | ``True``       | ``False``              | ``False``              | sets ``requires_grad`` to ``True`` or ``False``   | no-op                               |
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
# | ``True``       | ``True``               | ``False``              | sets ``requires_grad`` to ``True`` or ``False``   | no-op                               |
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
# | ``False``      | ``True``               | ``False``              | no-op                                             | sets ``retains_grad`` to ``True``   |
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
# | ``False``      | ``True``               | ``True``               | no-op                                             | no-op                               |
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
# 


######################################################################
# (work-in-progress) Real world example with ResNet
# -------------------------------------------------
# 
# Let’s move on from the toy example above and study a realistic network:
# `ResNet <https://docs.pytorch.org/vision/2.0/models/resnet.html>`__.
# 
# To illustrate the importance of gradient visualization, we will
# instantiate two versions of ResNet: one without batch normalization
# (``BatchNorm``), and one with it. `Batch
# normalization <https://arxiv.org/abs/1502.03167>`__ is an extremely
# effective technique to resolve the vanishing/exploding gradients issue,
# and we will be verifying that experimentally.
# 
# We first initiate the models without ``BatchNorm`` following the
# `documentation <https://docs.pytorch.org/vision/2.0/models/generated/torchvision.models.resnet18.html>`__.
# 

# set up dummy data
x = torch.randn(1, 3, 224, 224)
y = torch.randn(1, 1000)

# init model
# model = resnet18(norm_layer=nn.Identity)
model = resnet18()
model.train()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


######################################################################
# Because we are using a ``nn.Module`` instead of individual tensors for
# our forward pass, we need another adopt our method to access the
# intermediate gradients. This is done by `registering a
# hook <https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging>`__.
# 
# Note that using backward pass hooks to probe an intermediate nodes
# gradient is preferred over using ``retain_grad()``. It avoids the memory
# retention overhead if gradients aren’t needed after backpropagation. It
# also lets you modify and/or clamp gradients during the backward pass, so
# they don’t vanish or explode.
# 
# The following code defines our forward pass hook (notice the call to
# ``retain_grad()``) and also collects names of all parameters and layers.
# 

def hook_forward(module, args, output):
    output.retain_grad() # store gradient in ouput tensors
    
    # grads and layers are global variables
    outputs.append((layers[module], output))
    
def get_all_layers(layer, hook_fn):
    """Returns dict where keys are children modules and values are layer names"""
    layers = dict()
    for name, layer in model.named_modules():
        if any(layer.children()) is False:
            # skip Sequential and/or wrapper modules
            layers[layer] = name
            layer.register_forward_hook(hook_fn) # hook_forward 
    return layers

def get_all_params(model):
    """return list of all leaf tensors with requires_grad=True and which are not bias terms"""
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            params.append((name, param))
    return params

# register hooks 
layers = get_all_layers(model, hook_forward)

# get parameter gradients
params = get_all_params(model)


######################################################################
# Let’s check a few of the layers and parameters to make sure things are
# as expected:
# 

num_layers = 5 
print("<--------Params-------->")
for name, param in params[0:num_layers]:
    print(name, param.shape)

count = 0
print("<--------Layers-------->")
for layer in layers.values():
    print(layer)
    count += 1
    if count >= num_layers:
        break


######################################################################
# Now let’s run a forward pass and verify our output tensor values were
# populated.
# 

outputs = [] # list with layer name, output tensor tuple
optimizer.zero_grad()
y_pred = model(x)
loss = F.mse_loss(y_pred, y) 

print("<--------Outputs-------->")
for name, output in outputs[0:num_layers]:
    print(name, output.shape)


######################################################################
# Everything looks good so far, so let’s call ``backward()``, populate the
# ``grad`` values for all intermediate tensors, and get the average
# gradient for each layer.
# 

loss.backward()

def get_grads():
    layer_idx = []
    avg_grads = []
    print("<--------Grads-------->")
    for idx, (name, output) in enumerate(outputs[0:-2]):
        if output.grad is not None:
            avg_grad = output.grad.abs().mean()
            if idx < num_layers:
                print(name, avg_grad)
            avg_grads.append(avg_grad)
            layer_idx.append(idx)
    return layer_idx, avg_grads    
    
layer_idx, avg_grads = get_grads()


######################################################################
# Now that we have all our gradients stored in ``grads``, we can plot them
# and see how the average gradient values change as a function of the
# network depth.
# 

def plot_grads(layer_idx, avg_grads):    
    plt.plot(layer_idx, avg_grads)
    plt.xlabel("Layer depth")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    
plot_grads(layer_idx, avg_grads)


######################################################################
# Upon initialization, this is not very interesting. Let’s try running for
# several epochs, use gradient descent, and then see how the values
# change.
# 

epochs = 20

for epoch in range(epochs):
    outputs = [] # list with layer name, output tensor tuple
    optimizer.zero_grad()
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y) 
    loss.backward()
    optimizer.step()
    
layer_idx, avg_grads = get_grads()
plot_grads(layer_idx, avg_grads)


######################################################################
# Still not very interesting… surprised that the gradients don’t
# accumulate. Let’s check the leaf tensors… those tensors are probably
# just recreated whenever I rerun the forward pass, and thus they don’t
# accumulate. Let’s see if that’s the case with the parameters.
# 

def get_param_grads():
    layer_idx = []
    avg_grads = []
    print("<--------Params-------->")
    for idx, (name, param) in enumerate(params):
        if param.grad is not None:
            avg_grad = param.grad.abs().mean()
            if idx < num_layers:
                print(name, avg_grad)
            avg_grads.append(avg_grad)
            layer_idx.append(idx)
    return layer_idx, avg_grads    
    
layer_idx, avg_grads = get_param_grads()

    
plot_grads(layer_idx, avg_grads)


######################################################################
# (work-in-progress) Conclusion
# -----------------------------
# 
# If you would like to learn more about how PyTorch’s autograd system
# works, please visit the `references <#references>`__ below. If you have
# any feedback for this tutorial (improvements, typo fixes, etc.) then
# please use the `PyTorch Forums <https://discuss.pytorch.org/>`__ and/or
# the `issue tracker <https://github.com/pytorch/tutorials/issues>`__ to
# reach out.
# 


######################################################################
# References
# ----------
# 
# -  `A Gentle Introduction to
#    torch.autograd <https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`__
# -  `Automatic Differentiation with
#    torch.autograd <https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial>`__
# -  `Autograd
#    mechanics <https://docs.pytorch.org/docs/stable/notes/autograd.html>`__
# 