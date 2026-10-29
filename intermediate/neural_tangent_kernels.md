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

And let's generate some random data

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
# Detaching the parameters because we won't be calling Tensor.backward().
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

In some cases, you may only want the diagonal or the trace of this quantity,
especially if you know beforehand that the network architecture results in an
NTK where the non-diagonal elements can be approximated by zero. It's easy to
adjust the above function to do that:

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
# Disable TensorFloat-32 for convolutions on Ampere+ GPUs to sacrifice performance in favor of accuracy
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

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: neural_tangent_kernels.ipynb`](../_downloads/412c6fac9e4f7432b11f6e67d066ee2f/neural_tangent_kernels.ipynb)

[`Download Python source code: neural_tangent_kernels.py`](../_downloads/57e71bcf0a6c2280481b4e79ca070e22/neural_tangent_kernels.py)

[`Download zipped: neural_tangent_kernels.zip`](../_downloads/880d5a7a3c5b85594caa78fa0808639c/neural_tangent_kernels.zip)