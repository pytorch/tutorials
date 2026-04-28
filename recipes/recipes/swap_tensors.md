Note

Go to the end
to download the full example code.

# Extension points in `nn.Module` for `load_state_dict` and tensor subclasses

**Author:** [Mikayla Gawarecki](https://github.com/mikaylagawarecki)

This recipe introduces a new utility function `torch.utils.swap_tensors`
as well as two new extension points where it has been integrated in
`nn.Module`:

- `nn.Module.to()` and related methods
- `nn.Module.load_state_dict()`

Note

This recipe requires PyTorch 2.3.0 or later.

## `torch.utils.swap_tensors`

`torch.utils.swap_tensors` (hereafter referred to as `swap_tensors`) is a
utility function that takes in two Python tensors and swaps them.

More specifically, `swap_tensors` swaps the Python `__class__`, `__dict__`
and `__slots__` of the two tensors, as well as their associated `at::Tensor`.

## Application to `nn.Module`

This utility is pertinent to `nn.Module` when a Python object outside
of the module holds a reference to parameters of the module. If an `nn.Module`
modifies any of its parameters out of place, the object holding references to
the parameters will not see the change. A classic example of this is the
optimizer, which holds a reference to the parameters of the `nn.Module`.
This leads to a silent correctness issue where the `optimizer.step()` will
run without error but the weights of the `nn.Module` will not be updated.

## `nn.Module.to()` and related methods

This includes methods that change the device of the module (such as `nn.Module.cpu()`),
methods that change the `dtype` of the module (such as `nn.Module.float()`)
as well as methods that allow the module to be materialized
(such as `nn.Module.to_empty()`).

At first glance, it might be non-intuitive that these methods are able to
modify the parameters of the module in-place. The existing approach has been
to use a nasty hack dating back from the first days of PyTorch.

Notably, the existing approach does not work in these cases:

- when using `__torch_dispatch__` subclasses
- when `param` and `new_param` do not have the same Python `type()`
- For tensors with special C++ representations (such as sparse tensors and `XLA` tensors)

In the following part of this recipe, we will define a toy `__torch_dispatch__`
subclass `MyQuantizedLinearWeight` that represents quantized linear weights.
This subclass will be used for illustration purposes throughout the rest of
the tutorial. For brevity, we omit most of the `__torch_dispatch__`
implementation.

Let us create an `nn.Linear` layer of `dtype` `torch.float32` where the weight is
a `MyQuantizedLinearWeight` and try to convert it to `torch.bfloat16`.
Observe that the weight's `dtype` changes as expected. However, the `dtype`
of the subclass' payload (`elem`) does not change.

To this end, we introduce a global config
`torch.__future__.set_swap_module_params_on_conversion` that will use
`swap_tensors` to swap the parameters of the module while preserving
references in place of `.data` setting. When this config is set,
`swap_tensors` will be used during the conversion, which ensures that
the `dtype` of the payload is properly converted.

## `nn.Module.load_state_dict()`

Depending on the value of the `assign` keyword argument passed
to `load_state_dict()`, there are two ways to load the `state_dict`:

- `assign=False`: preserves the properties of `module.param` and only takes the values
from `state_dict['param_name']`
- `assign=True`: preserves the properties and values of `state_dict['param_name']`.

Previously, these were implemented with in-place `copy_` and `__setattr__` respectively.
With the existing implementation, each approach had its own limitations - `assign=False`
imposes the constraint that the type of the parameter in the `state_dict` must
be the same as the type of the parameter in the module while `assign=True` imposes
the constraint that anything that holds references to the module's parameters must
be initialized after `nn.Module.load_state_dict()`.

Now, we address both constraints by adding a `swap_tensors` path to `load_state_dict()`
and introducing a new extension point `torch.Tensor.module_load(self, other, assign=False)`.
When the `swap_tensors` path is enabled via the `__future__` mentioned above,
we can use a `__torch_function__` handler for `module_load` to apply a
custom transformation to the value in the `state_dict`. The result of this
transformation will be swapped with the parameter in the module.

In the following example, we will use the `MyQuantizedLinearWeight` subclass
defined above to illustrate how we can use these features to apply a
custom quantization scheme to the weights of a linear layer when
loading the `state_dict`.

Recall that the `__torch_function__` handler for `module_load` will be
invoked if either `self` or `other` (in this case `param` or
`state_dict[param_key]`) are `MyQuantizedLinearWeight` subclasses.

Assume that we expect the `state_dict` to contain plain tensors and the
module to contain `MyQuantizedLinearWeight` parameters where we want the
tensors in the `state_dict` to be transformed into the subclass. Then we
can define a `__torch_function__` handler for `torch.Tensor.module_load`
as such:

First, let us create a skeleton of a model on the meta device to avoid
materializing storages. We convert all weights in the modules to
`MyQuantizedLinearWeight` subclasses while leaving biases intact.

We can then load the `state_dict`. Observe that we use `assign=True` because
for biases, we want to preserve the properties of the tensor in the `state_dict`
(for example, we do not want the bias to be on the `meta` device after loading).

The above is a toy example of how we can use the new extension point in
`nn.Module.load_state_dict()`. One can also imagine alternate scenarios such
as when we have tensor subclasses in the `state_dict` and plain `nn.Parameters`/
tensors in the module or when both are tensor subclasses. Based on the use
case, we can define the `__torch_function__` handler for `module_load`
to apply the transforms as needed.

## Conclusion

In this recipe, we learned about `swap_tensors`, the importance
of preserving references for parameters in `nn.Module` as well as how to
use the two new extension points that are gated by
`torch.__future__.set_swap_module_params_on_conversion`.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: swap_tensors.ipynb`](../../_downloads/8ec147fe4546ad23cb0cefdb015f3352/swap_tensors.ipynb)

[`Download Python source code: swap_tensors.py`](../../_downloads/db0de66558b1ca13e18495862bf4b024/swap_tensors.py)

[`Download zipped: swap_tensors.zip`](../../_downloads/cae9d0bee3f9b533499fec4df455e784/swap_tensors.zip)