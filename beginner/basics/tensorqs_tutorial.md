Note

Go to the end
to download the full example code.

[Learn the Basics](intro.html) ||
[Quickstart](quickstart_tutorial.html) ||
**Tensors** ||
[Datasets & DataLoaders](data_tutorial.html) ||
[Transforms](transforms_tutorial.html) ||
[Build Model](buildmodel_tutorial.html) ||
[Autograd](autogradqs_tutorial.html) ||
[Optimization](optimization_tutorial.html) ||
[Save & Load Model](saveloadrun_tutorial.html)

# Tensors

Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model's parameters.

Tensors are similar to [NumPy's](https://numpy.org/) ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and
NumPy arrays can often share the same underlying memory, eliminating the need to copy data (see [Bridge with NumPy](../blitz/tensor_tutorial.html#bridge-to-np-label)). Tensors
are also optimized for automatic differentiation (we'll see more about that later in the [Autograd](autogradqs_tutorial.html)
section). If you're familiar with ndarrays, you'll be right at home with the Tensor API. If not, follow along!

```
import torch
import numpy as np
```

## Initializing a Tensor

Tensors can be initialized in various ways. Take a look at the following examples:

**Directly from data**

Tensors can be created directly from data. The data type is automatically inferred.

```
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

**From a NumPy array**

Tensors can be created from NumPy arrays (and vice versa - see [Bridge with NumPy](../blitz/tensor_tutorial.html#bridge-to-np-label)).

```
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

**From another tensor:**

The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

```
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

```
Ones Tensor:
 tensor([[1, 1],
 [1, 1]])

Random Tensor:
 tensor([[0.0516, 0.1291],
 [0.2764, 0.2862]])
```

**With random or constant values:**

`shape` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

```
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

```
Random Tensor:
 tensor([[0.5556, 0.2018, 0.2390],
 [0.4958, 0.3193, 0.7638]])

Ones Tensor:
 tensor([[1., 1., 1.],
 [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
 [0., 0., 0.]])
```

---

## Attributes of a Tensor

Tensor attributes describe their shape, datatype, and the device on which they are stored.

```
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

```
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

---

## Operations on Tensors

Over 1200 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing,
indexing, slicing), sampling and more are
comprehensively described [here](https://pytorch.org/docs/stable/torch.html).

Each of these operations can be run on the CPU and [Accelerator](https://pytorch.org/docs/stable/torch.html#accelerators)
such as CUDA, MPS, MTIA, or XPU. If you're using Colab, allocate an accelerator by going to Runtime > Change runtime type > GPU.

By default, tensors are created on the CPU. We need to explicitly move tensors to the accelerator using
`.to` method (after checking for accelerator availability). Keep in mind that copying large tensors
across devices can be expensive in terms of time and memory!

```
# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
 tensor = tensor.to(torch.accelerator.current_accelerator())
```

Try out some of the operations from the list.
If you're familiar with the NumPy API, you'll find the Tensor API a breeze to use.

**Standard numpy-like indexing and slicing:**

```
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

```
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.]])
```

**Joining tensors** You can use `torch.cat` to concatenate a sequence of tensors along a given dimension.
See also [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html),
another tensor joining operator that is subtly different from `torch.cat`.

```
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

```
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
 [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
 [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
 [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

**Arithmetic operations**

```
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

```
tensor([[1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.]])
```

**Single-element tensors** If you have a one-element tensor, for example by aggregating all
values of a tensor into one value, you can convert it to a Python
numerical value using `item()`:

```
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

```
12.0 <class 'float'>
```

**In-place operations**
Operations that store the result into the operand are called in-place. They are denoted by a `_` suffix.
For example: `x.copy_(y)`, `x.t_()`, will change `x`.

```
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```

```
tensor([[1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
 [6., 5., 6., 6.],
 [6., 5., 6., 6.],
 [6., 5., 6., 6.]])
```

Note

In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss
of history. Hence, their use is discouraged.

---

## Bridge with NumPy

Tensors on the CPU and NumPy arrays can share their underlying memory
locations, and changing one will change the other.

### Tensor to NumPy array

```
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

```
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

A change in the tensor reflects in the NumPy array.

```
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

```
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

### NumPy array to Tensor

```
n = np.ones(5)
t = torch.from_numpy(n)
```

Changes in the NumPy array reflects in the tensor.

```
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

```
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```

**Total running time of the script:** (0 minutes 0.470 seconds)

[`Download Jupyter notebook: tensorqs_tutorial.ipynb`](../../_downloads/0e6615c5a7bc71e01ff3c51217ea00da/tensorqs_tutorial.ipynb)

[`Download Python source code: tensorqs_tutorial.py`](../../_downloads/3fb82dc8278b08d5e5dee31ec1c16170/tensorqs_tutorial.py)

[`Download zipped: tensorqs_tutorial.zip`](../../_downloads/825966dd6a92d097539bce1ab8fbb3fa/tensorqs_tutorial.zip)