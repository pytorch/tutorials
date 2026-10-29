Note

Go to the end
to download the full example code.

# Dynamic Compilation Control with `torch.compiler.set_stance`

**Author:** [William Wen](https://github.com/williamwen42)

`torch.compiler.set_stance` is a `torch.compiler` API that
enables you to change the behavior of `torch.compile` across different
calls to your model without having to reapply `torch.compile` to your model.

This recipe provides some examples on how to use `torch.compiler.set_stance`.

## Prerequisites

- `torch >= 2.6`

## Description

`torch.compile.set_stance` can be used as a decorator, context manager, or raw function
to change the behavior of `torch.compile` across different calls to your model.

In the example below, the `"force_eager"` stance ignores all `torch.compile` directives.

Sample decorator usage

Sample context manager usage

Sample raw function usage

`torch.compile` stance can only be changed **outside** of any `torch.compile` region. Attempts
to do otherwise will result in an error.

Other stances include:

- `"default"`: The default stance, used for normal compilation.
- `"eager_on_recompile"`: Run code eagerly when a recompile is necessary. If there is cached compiled code valid for the input, it will still be used.
- `"fail_on_recompile"`: Raise an error when recompiling a function.

See the `torch.compiler.set_stance` [doc page](https://pytorch.org/docs/main/generated/torch.compiler.set_stance.html#torch.compiler.set_stance)
for more stances and options. More stances/options may also be added in the future.

## Examples

### Preventing recompilation

Some models do not expect any recompilations - for example, you may always have inputs with the same shape.
Since recompilations may be expensive, we may wish to error out when we attempt to recompile so we can detect and fix recompilation cases.
The `"fail_on_recompilation"` stance can be used for this.

```
# first compilation
```

If erroring out is too disruptive, we can use `"eager_on_recompile"` instead,
which will cause `torch.compile` to fall back to eager instead of erroring out.
This may be useful if we don't expect recompilations to happen frequently, but
when one is required, we'd rather pay the cost of running eagerly over the cost of recompilation.

```
# first compilation
```

# Measuring performance gains

`torch.compiler.set_stance` can be used to compare eager vs. compiled performance
without having to define a separate eager model.

```
# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.

# warmups
```

# Crashing sooner

Running an eager iteration first before a compiled iteration using the `"force_eager"` stance
can help us to catch errors unrelated to `torch.compile` before attempting a very long compile.

## Conclusion

In this recipe, we have learned how to use the `torch.compiler.set_stance` API
to modify the behavior of `torch.compile` across different calls to a model
without needing to reapply it. The recipe demonstrates using
`torch.compiler.set_stance` as a decorator, context manager, or raw function
to control compilation stances like `force_eager`, `default`,
`eager_on_recompile`, and "fail_on_recompile."

For more information, see: [torch.compiler.set_stance API documentation](https://pytorch.org/docs/main/generated/torch.compiler.set_stance.html#torch.compiler.set_stance).

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: torch_compiler_set_stance_tutorial.ipynb`](../_downloads/60868f136246d48ff6f18abf35fdf18e/torch_compiler_set_stance_tutorial.ipynb)

[`Download Python source code: torch_compiler_set_stance_tutorial.py`](../_downloads/12d57dfb5886880e71752ea149c1dc6d/torch_compiler_set_stance_tutorial.py)

[`Download zipped: torch_compiler_set_stance_tutorial.zip`](../_downloads/20f245312699c4ccc60f9b83d2a0bc18/torch_compiler_set_stance_tutorial.zip)