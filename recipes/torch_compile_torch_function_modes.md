Note

Go to the end
to download the full example code.

# (beta) Utilizing Torch Function modes with torch.compile

**Author:** [Michael Lazos](https://github.com/mlazos)

This recipe covers how to use a key torch extensibility point,

torch function modes, in tandem with `torch.compile` to override
the behavior of torch operators, also know as **ops**, at trace time, with no runtime overhead.

Note

This recipe requires PyTorch 2.7.0 or later.

## Rewriting a torch op (torch.add -> torch.mul)

For this example, we'll use torch function modes to rewrite occurences
of addition with multiply instead. This type of override can be common
if a certain backend has a custom implementation that should be dispatched
for a given op.

```
# exit cleanly if we are on a device that doesn't support ``torch.compile``

# Define our mode, Note: ``BaseTorchFunctionMode``
# implements the actual invocation of func(..)

# The mode can also be used within the compiled region as well like this:
```

## Conclusion

In this recipe we demonstrated how to override the behavior of `torch.*` operators
using torch function modes from within `torch.compile`. This enables users to utilize
the extensibility benefits of torch function modes without the runtime overhead
of calling torch function on every op invocation.

- See [Extending Torch API with Modes](https://pytorch.org/docs/stable/notes/extending.html#extending-all-torch-api-with-modes) for other examples and background on Torch Function modes.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: torch_compile_torch_function_modes.ipynb`](../_downloads/d036a4d61563d6157e333f7e5b20b091/torch_compile_torch_function_modes.ipynb)

[`Download Python source code: torch_compile_torch_function_modes.py`](../_downloads/822c89760620e7273256cb7f63647167/torch_compile_torch_function_modes.py)

[`Download zipped: torch_compile_torch_function_modes.zip`](../_downloads/844220185c7c132a54399ed6ea7a43fb/torch_compile_torch_function_modes.zip)