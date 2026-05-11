Note

Go to the end
to download the full example code.

# Tensors

Tensors are a specialized data structure that are very similar to arrays
and matrices. In PyTorch, we use tensors to encode the inputs and
outputs of a model, as well as the model's parameters.

Tensors are similar to NumPy's ndarrays, except that tensors can run on
GPUs or other specialized hardware to accelerate computing. If you're familiar with ndarrays, you'll
be right at home with the Tensor API. If not, follow along in this quick
API walkthrough.

## Tensor Initialization

Tensors can be initialized in various ways. Take a look at the following examples:

**Directly from data**

Tensors can be created directly from data. The data type is automatically inferred.

**From a NumPy array**

Tensors can be created from NumPy arrays (and vice versa - see Bridge with NumPy).

**From another tensor:**

The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

**With random or constant values:**

`shape` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

---

## Tensor Attributes

Tensor attributes describe their shape, datatype, and the device on which they are stored.

---

## Tensor Operations

Over 100 tensor operations, including transposing, indexing, slicing,
mathematical operations, linear algebra, random sampling, and more are
comprehensively described
[here](https://pytorch.org/docs/stable/torch.html).

Each of them can be run on the GPU (at typically higher speeds than on a
CPU). If you're using Colab, allocate a GPU by going to Edit > Notebook
Settings.

```
# We move our tensor to the GPU if available
```

Try out some of the operations from the list.
If you're familiar with the NumPy API, you'll find the Tensor API a breeze to use.

**Standard numpy-like indexing and slicing:**

**Joining tensors** You can use `torch.cat` to concatenate a sequence of tensors along a given dimension.
See also [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html),
another tensor joining op that is subtly different from `torch.cat`.

**Multiplying tensors**

```
# This computes the element-wise product

# Alternative syntax:
```

This computes the matrix multiplication between two tensors

```
# Alternative syntax:
```

**In-place operations**
Operations that have a `_` suffix are in-place. For example: `x.copy_(y)`, `x.t_()`, will change `x`.

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

[`Download Jupyter notebook: tensor_tutorial.ipynb`](../../_downloads/3dbbd6931d76adb0dc37d4e88b328852/tensor_tutorial.ipynb)

[`Download Python source code: tensor_tutorial.py`](../../_downloads/6133d4c3ca687bdecb6dda6d3a243c24/tensor_tutorial.py)

[`Download zipped: tensor_tutorial.zip`](../../_downloads/3ceb90d4c33acc714529078189bbde96/tensor_tutorial.zip)