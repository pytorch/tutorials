Note

Go to the end
to download the full example code.

# Using User-Defined Triton Kernels with `torch.compile`

**Author:** [Oguz Ulgen](https://github.com/oulgen)

User-defined Triton kernels can be used to optimize specific parts of your
model's computation. These kernels are written in Triton's language, which is designed
to make it easier to achieve peak hardware performance. By using user-defined Triton
kernels with `torch.compile`, you can integrate these optimized computations into
your PyTorch model, potentially achieving significant performance improvements.

This recipes demonstrates how you can use user-defined Triton kernels with `torch.compile`.

## Prerequisites

Before starting this recipe, make sure that you have the following:

- Basic understanding of `torch.compile` and Triton. See:

- [torch.compiler API documentation](https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler)
- [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Triton language documentation](https://triton-lang.org/main/index.html)
- PyTorch 2.3 or later
- A GPU that supports Triton

## Basic Usage

In this example, we will use a simple vector addition kernel from the Triton documentation
with `torch.compile`.
For reference, see [Triton documentation](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html).

## Advanced Usage

Triton's autotune feature is a powerful tool that automatically optimizes the configuration
parameters of your Triton kernels. It explores a range of possible configurations and
selects the one that delivers the best performance for your specific use case.

When used with `torch.compile`, `triton.autotune` can help ensure that your PyTorch
model is running as efficiently as possible. Here is an example of using `torch.compile`
and `triton.autotune`.

Note

`torch.compile` only supports configs and key arguments to `triton.autotune`.

## Composability

User-defined Triton kernels do not automatically support all PyTorch
subsystems. This can be seen in the following use cases:

- Adding a CPU fallback
- Adding a `FlopCounter` formula
- Composing with Tensor Subclasses

To compose with additional PyTorch subsystems, use `torch.library.triton_op`.

`triton_op is` a structured way of defining a custom operator that is backed by one
or more Triton kernels: like regular custom operators (`torch.library.custom_op`),
you are able to specify the interactions with PyTorch subsystems via `torch.library`.
However, unlike `torch.library.custom_op`, which creates opaque callables with respect to
`torch.compile`, `torch.compile` traces into `triton_op` to apply optimizations.

Here's a chart of which API to use when integrating Triton kernels with PyTorch.

| | Triton kernel (no explicit `torch.library` wrapper) | `torch.library.triton_op` | `torch.library.custom_op` |
| --- | --- | --- | --- |
| Supports inference | Yes | Yes | Yes |
| Supports training | In the majority of cases | Yes | Yes |
| Supports `torch.compile` | Yes | Yes | Yes |
| Supports `torch.compile(fullgraph=True)` | In the majority of cases | In the majority of cases | In all cases |
| Does torch.compile trace into the implementation? | Yes | Yes | No |
| Supports AOTInductor | Yes | Yes | No |
| Supports PyTorch Subsystems like FlopCounterMode, CPU Fallback, Tensor Subclasses | No | Yes | Yes |

### Wrapping Triton kernels with `triton_op`

Use `torch.library.triton_op` to wrap a function that may invoke one or more Triton kernels.
Use `torch.library.wrap_triton` to wrap the calls to the Triton kernel.

You can invoke the `triton_op` in one of the following two ways.

The resulting `triton_op` works with `torch.compile` and `AOTInductor`.

### Adding training support

Use `register_autograd` to add an autograd formula for the `triton_op`.
Prefer this to using `torch.autograd.Function` (which has various composability footguns
with `torch.compile`).

Note that the backward must be a composition of PyTorch-understood operators.
If you want the backward to call Triton kernels, then those must be wrapped in `triton_op` as well:

### Adding a CPU Fallback

Triton kernels don't run on CPU. Use `register_kernel` to add a CPU (or any other device) fallback for the `triton_op`:

The fallback must be composed of PyTorch operators.

### Adding a FlopCounter Formula

To specify how many flops the triton kernel reports under PyTorch's flop counter,
use `register_flop_formula`.

`FlopCounterMode` requires [tabulate](https://pypi.org/project/tabulate/).
Before running the code below, make sure you have `tabulate` installed or install by
running `pip install tabulate`.

```
>>> with FlopCounterMode() as flop_counter:
>>> y = mysin(x)
```

## Limitations

As of PyTorch 2.3, the support for user-defined Triton kernels in `torch.compile`
includes dynamic shapes, `torch.autograd.Function`, JIT inductor, and AOT inductor.
You can use these features together to build complex, high-performance models.

PyTorch 2.6 added `torch.library.triton_op`, which adds support for
user-defined Triton kernels in tensor subclasses and other advanced features.

However, there are certain limitations to be aware of:

- **Triton Features:** While `triton.heuristics` can be used either standalone or
before `triton.autotune`, it cannot be used after `triton.autotune`. This
implies that if `triton.heuristics` and `triton.autotune` are to be used
together, `triton.heuristics` must be used first.

## Conclusion

In this recipe, we explored how to utilize user-defined Triton kernels
with `torch.compile`. We delved into the basic usage of a simple
vector addition kernel and advanced usage involving Triton's autotune
feature. We also discussed the composability of user-defined Triton
kernels with other PyTorch features and highlighted some current limitations.

## See Also

- [Compiling the Optimizers](https://pytorch.org/tutorials/recipes/compiling_optimizer.html)
- [Implementing High-Performance Transformers with Scaled Dot Product Attention](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: torch_compile_user_defined_triton_kernel_tutorial.ipynb`](../_downloads/f827f181506a79226f4ffbcf7c9a5a50/torch_compile_user_defined_triton_kernel_tutorial.ipynb)

[`Download Python source code: torch_compile_user_defined_triton_kernel_tutorial.py`](../_downloads/0ccffddcfee1f815c02241b985844376/torch_compile_user_defined_triton_kernel_tutorial.py)

[`Download zipped: torch_compile_user_defined_triton_kernel_tutorial.zip`](../_downloads/76f953c387e263af080eabba6f4e8a76/torch_compile_user_defined_triton_kernel_tutorial.zip)