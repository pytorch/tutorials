Note

Go to the end
to download the full example code.

# Reasoning about Shapes in PyTorch

When writing models with PyTorch, it is commonly the case that the parameters
to a given layer depend on the shape of the output of the previous layer. For
example, the `in_features` of an `nn.Linear` layer must match the
`size(-1)` of the input. For some layers, the shape computation involves
complex equations, for example convolution operations.

One way around this is to run the forward pass with random inputs, but this is
wasteful in terms of memory and compute.

Instead, we can make use of the `meta` device to determine the output shapes
of a layer without materializing any data.

Observe that since data is not materialized, passing arbitrarily large
inputs will not significantly alter the time taken for shape computation.

Consider an arbitrary network such as the following:

We can view the intermediate shapes within an entire network by registering a
forward hook to each layer that prints the shape of the output.

```
# Any tensor created within this torch.device context manager will be
# on the meta device.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: reasoning_about_shapes.ipynb`](../../_downloads/1bba1c0153db192997cdb32f9c312b2c/reasoning_about_shapes.ipynb)

[`Download Python source code: reasoning_about_shapes.py`](../../_downloads/9f4fb47ef3d58524029d86df50e90a08/reasoning_about_shapes.py)

[`Download zipped: reasoning_about_shapes.zip`](../../_downloads/a9292711edeeba8848f0d8bf20464f1e/reasoning_about_shapes.zip)