# Learning PyTorch with Examples

**Author**: [Justin Johnson](https://github.com/jcjohnson/pytorch-examples)

Note

This is one of our older PyTorch tutorials. You can view our latest
beginner content in
[Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html).

This tutorial introduces the fundamental concepts of
[PyTorch](https://github.com/pytorch/pytorch) through self-contained
examples.

At its core, PyTorch provides two main features:

- An n-dimensional Tensor, similar to numpy but can run on GPUs
- Automatic differentiation for building and training neural networks

We will use a problem of fitting \(y=\sin(x)\) with a third order polynomial
as our running example. The network will have four parameters, and will be trained with
gradient descent to fit random data by minimizing the Euclidean distance
between the network output and the true output.

Note

You can browse the individual examples at the
end of this page.

To run the tutorials below, make sure you have the [torch](https://github.com/pytorch/pytorch)
and [numpy](https://github.com/numpy/numpy) packages installed.

## Tensors

### Warm-up: numpy

Before introducing PyTorch, we will first implement the network using
numpy.

Numpy provides an n-dimensional array object, and many functions for
manipulating these arrays. Numpy is a generic framework for scientific
computing; it does not know anything about computation graphs, or deep
learning, or gradients. However we can easily use numpy to fit a
third order polynomial to sine function by manually implementing the forward
and backward passes through the network using numpy operations:

```
# -*- coding: utf-8 -*-

# Create random input and output data

# Randomly initialize weights

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

### PyTorch: Tensors

Numpy is a great framework, but it cannot utilize GPUs to accelerate its
numerical computations. For modern deep neural networks, GPUs often
provide speedups of [50x or greater](https://github.com/jcjohnson/cnn-benchmarks), so
unfortunately numpy won't be enough for modern deep learning.

Here we introduce the most fundamental PyTorch concept: the **Tensor**.
A PyTorch Tensor is conceptually identical to a numpy array: a Tensor is
an n-dimensional array, and PyTorch provides many functions for
operating on these Tensors. Behind the scenes, Tensors can keep track of
a computational graph and gradients, but they're also useful as a
generic tool for scientific computing.

Also unlike numpy, PyTorch Tensors can utilize GPUs to accelerate
their numeric computations. To run a PyTorch Tensor on GPU, you simply
need to specify the correct device.

Here we use PyTorch Tensors to fit a third order polynomial to sine function.
Like the numpy example above we need to manually implement the forward
and backward passes through the network:

```
# -*- coding: utf-8 -*-

# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data

# Randomly initialize weights

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

## Autograd

### PyTorch: Tensors and autograd

In the above examples, we had to manually implement both the forward and
backward passes of our neural network. Manually implementing the
backward pass is not a big deal for a small two-layer network, but can
quickly get very hairy for large complex networks.

Thankfully, we can use [automatic
differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
to automate the computation of backward passes in neural networks. The
**autograd** package in PyTorch provides exactly this functionality.
When using autograd, the forward pass of your network will define a
**computational graph**; nodes in the graph will be Tensors, and edges
will be functions that produce output Tensors from input Tensors.
Backpropagating through this graph then allows you to easily compute
gradients.

This sounds complicated, it's pretty simple to use in practice. Each Tensor
represents a node in a computational graph. If `x` is a Tensor that has
`x.requires_grad=True` then `x.grad` is another Tensor holding the
gradient of `x` with respect to some scalar value.

Here we use PyTorch Tensors and autograd to implement our fitting sine wave
with third order polynomial example; now we no longer need to manually
implement the backward pass through the network:

```
# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

### PyTorch: Defining new autograd functions

Under the hood, each primitive autograd operator is really two functions
that operate on Tensors. The **forward** function computes output
Tensors from input Tensors. The **backward** function receives the
gradient of the output Tensors with respect to some scalar value, and
computes the gradient of the input Tensors with respect to that same
scalar value.

In PyTorch we can easily define our own autograd operator by defining a
subclass of `torch.autograd.Function` and implementing the `forward`
and `backward` functions. We can then use our new autograd operator by
constructing an instance and calling it like a function, passing
Tensors containing input data.

In this example we define our model as \(y=a+b P_3(c+dx)\) instead of
\(y=a+bx+cx^2+dx^3\), where \(P_3(x)=\frac{1}{2}\left(5x^3-3x\right)\)
is the [Legendre polynomial](https://en.wikipedia.org/wiki/Legendre_polynomials) of degree three. We write our own custom autograd
function for computing forward and backward of \(P_3\), and use it to implement
our model:

```
# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.

# Create random Tensors for weights. For this example, we need
# 4 weights: y = a + b * P3(c + d * x), these weights need to be initialized
# not too far from the correct result to ensure convergence.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

## `nn` module

### PyTorch: `nn`

Computational graphs and autograd are a very powerful paradigm for
defining complex operators and automatically taking derivatives; however
for large neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the
computation into **layers**, some of which have **learnable parameters**
which will be optimized during learning.

In TensorFlow, packages like
[Keras](https://github.com/fchollet/keras),
[TensorFlow-Slim](https://github.com/google-research/tf-slim),
and [TFLearn](http://tflearn.org/) provide higher-level abstractions
over raw computational graphs that are useful for building neural
networks.

In PyTorch, the `nn` package serves this same purpose. The `nn`
package defines a set of **Modules**, which are roughly equivalent to
neural network layers. A Module receives input Tensors and computes
output Tensors, but may also hold internal state such as Tensors
containing learnable parameters. The `nn` package also defines a set
of useful loss functions that are commonly used when training neural
networks.

In this example we use the `nn` package to implement our polynomial model
network:

```
# -*- coding: utf-8 -*-

# Create Tensors to hold input and outputs.

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).

# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3) 

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.

# You can access the first layer of `model` like accessing the first item of a list

# For linear layer, its parameters are stored as `weight` and `bias`.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

### PyTorch: optim

Up to this point we have updated the weights of our models by manually
mutating the Tensors holding learnable parameters with `torch.no_grad()`.
This is not a huge burden for simple optimization algorithms like stochastic
gradient descent, but in practice we often train neural networks using more
sophisticated optimizers like `AdaGrad`, `RMSProp`, `Adam`, and other.

The `optim` package in PyTorch abstracts the idea of an optimization
algorithm and provides implementations of commonly used optimization
algorithms.

In this example we will use the `nn` package to define our model as
before, but we will optimize the model using the `RMSprop` algorithm provided
by the `optim` package:

```
# -*- coding: utf-8 -*-

# Create Tensors to hold input and outputs.

# Prepare the input tensor (x, x^2, x^3).

# Use the nn package to define our model and loss function.

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms. The first argument to the RMSprop constructor tells the
# optimizer which Tensors it should update.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

### PyTorch: Custom `nn` Modules

Sometimes you will want to specify models that are more complex than a
sequence of existing Modules; for these cases you can define your own
Modules by subclassing `nn.Module` and defining a `forward` which
receives input Tensors and produces output Tensors using other
modules or other autograd operations on Tensors.

In this example we implement our third order polynomial as a custom Module
subclass:

```
# -*- coding: utf-8 -*-
import torch
import math

class Polynomial3(torch.nn.Module):
 def __init__(self):
 """
 In the constructor we instantiate four parameters and assign them as
 member parameters.
 """
 super().__init__()
 self.a = torch.nn.Parameter(torch.randn(()))
 self.b = torch.nn.Parameter(torch.randn(()))
 self.c = torch.nn.Parameter(torch.randn(()))
 self.d = torch.nn.Parameter(torch.randn(()))

 def forward(self, x):
 """
 In the forward function we accept a Tensor of input data and we must return
 a Tensor of output data. We can use Modules defined in the constructor as
 well as arbitrary operators on Tensors.
 """
 return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

 def string(self):
 """
 Just like any class in Python, you can also define custom method on PyTorch modules
 """
 return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Construct our model by instantiating the class defined above
model = Polynomial3()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined 
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
 # Forward pass: Compute predicted y by passing x to the model
 y_pred = model(x)

 # Compute and print loss
 loss = criterion(y_pred, y)
 if t % 100 == 99:
 print(t, loss.item())

 # Zero gradients, perform a backward pass, and update the weights.
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

print(f'Result: {model.string()}')
```

### PyTorch: Control Flow + Weight Sharing

As an example of dynamic graphs and weight sharing, we implement a very
strange model: a third-fifth order polynomial that on each forward pass
chooses a random number between 3 and 5 and uses that many orders, reusing
the same weights multiple times to compute the fourth and fifth order.

For this model we can use normal Python flow control to implement the loop,
and we can implement weight sharing by simply reusing the same parameter multiple
times when defining the forward pass.

We can easily implement this model as a Module subclass:

```
# -*- coding: utf-8 -*-

# Create Tensors to hold input and outputs.

# Construct our model by instantiating the class defined above

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

## Examples

You can browse the above examples here.

### Tensors

- [Warm-up: numpy](examples_tensor/polynomial_numpy.html)
- [PyTorch: Tensors](examples_tensor/polynomial_tensor.html)

### Autograd

- [PyTorch: Tensors and autograd](examples_autograd/polynomial_autograd.html)
- [PyTorch: Defining New autograd Functions](examples_autograd/polynomial_custom_function.html)

### `nn` module

- [PyTorch: nn](examples_nn/polynomial_nn.html)
- [PyTorch: optim](examples_nn/polynomial_optim.html)
- [PyTorch: Custom nn Modules](examples_nn/polynomial_module.html)
- [PyTorch: Control Flow + Weight Sharing](examples_nn/dynamic_net.html)