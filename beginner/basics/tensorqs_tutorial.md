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

## Initializing a Tensor

Tensors can be initialized in various ways. Take a look at the following examples:

**Directly from data**

Tensors can be created directly from data. The data type is automatically inferred.

**From a NumPy array**

Tensors can be created from NumPy arrays (and vice versa - see [Bridge with NumPy](../blitz/tensor_tutorial.html#bridge-to-np-label)).

**From another tensor:**

The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

**With random or constant values:**

`shape` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

---

## Attributes of a Tensor

Tensor attributes describe their shape, datatype, and the device on which they are stored.

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
```

Try out some of the operations from the list.
If you're familiar with the NumPy API, you'll find the Tensor API a breeze to use.

**Standard numpy-like indexing and slicing:**

**Joining tensors** You can use `torch.cat` to concatenate a sequence of tensors along a given dimension.
See also [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html),
another tensor joining operator that is subtly different from `torch.cat`.

**Arithmetic operations**

```
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor

# This computes the element-wise product. z1, z2, z3 will have the same value
```

**Single-element tensors** If you have a one-element tensor, for example by aggregating all
values of a tensor into one value, you can convert it to a Python
numerical value using `item()`:

**In-place operations**
Operations that store the result into the operand are called in-place. They are denoted by a `_` suffix.
For example: `x.copy_(y)`, `x.t_()`, will change `x`.

Note

In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss
of history. Hence, their use is discouraged.

---

## Bridge with NumPy

Tensors on the CPU and NumPy arrays can share their underlying memory
locations, and changing one will change the other.

### Tensor to NumPy array

A change in the tensor reflects in the NumPy array.

### NumPy array to Tensor

Changes in the NumPy array reflects in the tensor.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: tensorqs_tutorial.ipynb`](../../_downloads/0e6615c5a7bc71e01ff3c51217ea00da/tensorqs_tutorial.ipynb)

[`Download Python source code: tensorqs_tutorial.py`](../../_downloads/3fb82dc8278b08d5e5dee31ec1c16170/tensorqs_tutorial.py)

[`Download zipped: tensorqs_tutorial.zip`](../../_downloads/825966dd6a92d097539bce1ab8fbb3fa/tensorqs_tutorial.zip)