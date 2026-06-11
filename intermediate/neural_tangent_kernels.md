Note

Go to the end
to download the full example code.

# Neural Tangent Kernels

The neural tangent kernel (NTK) is a kernel that describes
[how a neural network evolves during training](https://en.wikipedia.org/wiki/Neural_tangent_kernel).
There has been a lot of research around it [in recent years](https://arxiv.org/abs/1806.07572).
This tutorial, inspired by the implementation of [NTKs in JAX](https://github.com/google/neural-tangents)
(see [Fast Finite Width Neural Tangent Kernel](https://arxiv.org/abs/2206.08720) for details),
demonstrates how to easily compute this quantity using `torch.func`,
composable function transforms for PyTorch.

Note

This tutorial requires PyTorch 2.6.0 or later.

## Setup

First, some setup. Let's define a simple CNN that we wish to compute the NTK of.

```
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, vjp, jvp, jacrev

if torch.accelerator.is_available() and torch.accelerator.device_count() > 0:
 device = torch.accelerator.current_accelerator()
else:
 device = torch.device("cpu")

class CNN(nn.Module):
 def __init__(self):
 super(CNN, self).__init__()
 self.conv1 = nn.Conv2d(3, 32, (3, 3))
 self.conv2 = nn.Conv2d(32, 32, (3, 3))
 self.conv3 = nn.Conv2d(32, 32, (3, 3))
 self.fc = nn.Linear(21632, 10)

 def forward(self, x):
 x = self.conv1(x)
 x = x.relu()
 x = self.conv2(x)
 x = x.relu()
 x = self.conv3(x)
 x = x.flatten(1)
 x = self.fc(x)
 return x
```

And let's generate some random data

```
x_train = torch.randn(20, 3, 32, 32, device=device)
x_test = torch.randn(5, 3, 32, 32, device=device)
```

## Create a function version of the model

`torch.func` transforms operate on functions. In particular, to compute the NTK,
we will need a function that accepts the parameters of the model and a single
input (as opposed to a batch of inputs!) and returns a single output.

We'll use `torch.func.functional_call`, which allows us to call an `nn.Module`
using different parameters/buffers, to help accomplish the first step.

Keep in mind that the model was originally written to accept a batch of input
data points. In our CNN example, there are no inter-batch operations. That
is, each data point in the batch is independent of other data points. With
this assumption in mind, we can easily generate a function that evaluates the
model on a single data point:

```
net = CNN().to(device)

# Detaching the parameters because we won't be calling Tensor.backward().
params = {k: v.detach() for k, v in net.named_parameters()}

def fnet_single(params, x):
 return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)
```

## Compute the NTK: method 1 (Jacobian contraction)

We're ready to compute the empirical NTK. The empirical NTK for two data
points \(x_1\) and \(x_2\) is defined as the matrix product between the Jacobian
of the model evaluated at \(x_1\) and the Jacobian of the model evaluated at
\(x_2\):

\[J_{net}(x_1) J_{net}^T(x_2)\]

In the batched case where \(x_1\) is a batch of data points and \(x_2\) is a
batch of data points, then we want the matrix product between the Jacobians
of all combinations of data points from \(x_1\) and \(x_2\).

The first method consists of doing just that - computing the two Jacobians,
and contracting them. Here's how to compute the NTK in the batched case:

```
def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
 # Compute J(x1)
 jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
 jac1 = jac1.values()
 jac1 = [j.flatten(2) for j in jac1]

 # Compute J(x2)
 jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
 jac2 = jac2.values()
 jac2 = [j.flatten(2) for j in jac2]

 # Compute J(x1) @ J(x2).T
 result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
 result = result.sum(0)
 return result

result = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_test)
print(result.shape)
```

```
torch.Size([20, 5, 10, 10])
```

In some cases, you may only want the diagonal or the trace of this quantity,
especially if you know beforehand that the network architecture results in an
NTK where the non-diagonal elements can be approximated by zero. It's easy to
adjust the above function to do that:

```
def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute='full'):
 # Compute J(x1)
 jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
 jac1 = jac1.values()
 jac1 = [j.flatten(2) for j in jac1]

 # Compute J(x2)
 jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
 jac2 = jac2.values()
 jac2 = [j.flatten(2) for j in jac2]

 # Compute J(x1) @ J(x2).T
 einsum_expr = None
 if compute == 'full':
 einsum_expr = 'Naf,Mbf->NMab'
 elif compute == 'trace':
 einsum_expr = 'Naf,Maf->NM'
 elif compute == 'diagonal':
 einsum_expr = 'Naf,Maf->NMa'
 else:
 assert False

 result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
 result = result.sum(0)
 return result

result = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_test, 'trace')
print(result.shape)
```

```
torch.Size([20, 5])
```

The asymptotic time complexity of this method is \(N O [FP]\) (time to
compute the Jacobians) + \(N^2 O^2 P\) (time to contract the Jacobians),
where \(N\) is the batch size of \(x_1\) and \(x_2\), \(O\)
is the model's output size, \(P\) is the total number of parameters, and
\([FP]\) is the cost of a single forward pass through the model. See
section 3.2 in
[Fast Finite Width Neural Tangent Kernel](https://arxiv.org/abs/2206.08720)
for details.

## Compute the NTK: method 2 (NTK-vector products)

The next method we will discuss is a way to compute the NTK using NTK-vector
products.

This method reformulates NTK as a stack of NTK-vector products applied to
columns of an identity matrix \(I_O\) of size \(O\times O\)
(where \(O\) is the output size of the model):

\[J_{net}(x_1) J_{net}^T(x_2) = J_{net}(x_1) J_{net}^T(x_2) I_{O} = \left[J_{net}(x_1) \left[J_{net}^T(x_2) e_o\right]\right]_{o=1}^{O},\]

where \(e_o\in \mathbb{R}^O\) are column vectors of the identity matrix
\(I_O\).

- Let \(\textrm{vjp}_o = J_{net}^T(x_2) e_o\). We can use
a vector-Jacobian product to compute this.
- Now, consider \(J_{net}(x_1) \textrm{vjp}_o\). This is a
Jacobian-vector product!
- Finally, we can run the above computation in parallel over all
columns \(e_o\) of \(I_O\) using `vmap`.

This suggests that we can use a combination of reverse-mode AD (to compute
the vector-Jacobian product) and forward-mode AD (to compute the
Jacobian-vector product) to compute the NTK.

Let's code that up:

```
def empirical_ntk_ntk_vps(func, params, x1, x2, compute='full'):
 def get_ntk(x1, x2):
 def func_x1(params):
 return func(params, x1)

 def func_x2(params):
 return func(params, x2)

 output, vjp_fn = vjp(func_x1, params)

 def get_ntk_slice(vec):
 # This computes ``vec @ J(x2).T``
 # `vec` is some unit vector (a single slice of the Identity matrix)
 vjps = vjp_fn(vec)
 # This computes ``J(X1) @ vjps``
 _, jvps = jvp(func_x2, (params,), vjps)
 return jvps

 # Here's our identity matrix
 basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
 return vmap(get_ntk_slice)(basis)

 # ``get_ntk(x1, x2)`` computes the NTK for a single data point x1, x2
 # Since the x1, x2 inputs to ``empirical_ntk_ntk_vps`` are batched,
 # we actually wish to compute the NTK between every pair of data points
 # between {x1} and {x2}. That's what the ``vmaps`` here do.
 result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)

 if compute == 'full':
 return result
 if compute == 'trace':
 return torch.einsum('NMKK->NM', result)
 if compute == 'diagonal':
 return torch.einsum('NMKK->NMK', result)

# Disable TensorFloat-32 for convolutions on Ampere+ GPUs to sacrifice performance in favor of accuracy
with torch.backends.cudnn.flags(allow_tf32=False):
 result_from_jacobian_contraction = empirical_ntk_jacobian_contraction(fnet_single, params, x_test, x_train)
 result_from_ntk_vps = empirical_ntk_ntk_vps(fnet_single, params, x_test, x_train)

assert torch.allclose(result_from_jacobian_contraction, result_from_ntk_vps, atol=1e-5)
```

Our code for `empirical_ntk_ntk_vps` looks like a direct translation from
the math above! This showcases the power of function transforms: good luck
trying to write an efficient version of the above by only using
`torch.autograd.grad`.

The asymptotic time complexity of this method is \(N^2 O [FP]\), where
\(N\) is the batch size of \(x_1\) and \(x_2\), \(O\) is the
model's output size, and \([FP]\) is the cost of a single forward pass
through the model. Hence this method performs more forward passes through the
network than method 1, Jacobian contraction (\(N^2 O\) instead of
\(N O\)), but avoids the contraction cost altogether (no \(N^2 O^2 P\)
term, where \(P\) is the total number of model's parameters). Therefore,
this method is preferable when \(O P\) is large relative to \([FP]\),
such as fully-connected (not convolutional) models with many outputs \(O\).
Memory-wise, both methods should be comparable. See section 3.3 in
[Fast Finite Width Neural Tangent Kernel](https://arxiv.org/abs/2206.08720)
for details.

**Total running time of the script:** (0 minutes 0.740 seconds)

[`Download Jupyter notebook: neural_tangent_kernels.ipynb`](../_downloads/412c6fac9e4f7432b11f6e67d066ee2f/neural_tangent_kernels.ipynb)

[`Download Python source code: neural_tangent_kernels.py`](../_downloads/57e71bcf0a6c2280481b4e79ca070e22/neural_tangent_kernels.py)

[`Download zipped: neural_tangent_kernels.zip`](../_downloads/880d5a7a3c5b85594caa78fa0808639c/neural_tangent_kernels.zip)