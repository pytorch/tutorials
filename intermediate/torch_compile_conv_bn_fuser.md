Note

Go to the end
to download the full example code.

# Building a Convolution/Batch Norm fuser with torch.compile

**Author:** [Horace He](https://github.com/chillee), [Will Feng](https://github.com/yf225)

 What you will learn

- How to register custom fusion patterns with torch.compile's pattern matcher

 Prerequisites

- PyTorch v2.7.0

Note

This optimization only works for models in inference mode (i.e. `model.eval()`).
However, torch.compile's pattern matching system works for both training and inference.

First, let's get some imports out of the way (we will be using all
of these later in the code).

For this tutorial, we are going to create a model consisting of convolutions
and batch norms. Note that this model has some tricky components - some of
the conv/batch norm patterns are hidden within Sequentials and one of the
`BatchNorms` is wrapped in another Module.

## Fusing Convolution with Batch Norm

One of the primary challenges with trying to automatically fuse convolution
and batch norm in PyTorch is that PyTorch does not provide an easy way of
accessing the computational graph. torch.compile resolves this problem by
capturing the computational graph during compilation, allowing us to apply
pattern-based optimizations across the entire model, including operations
nested within Sequential modules or wrapped in custom modules.

torch.compile will capture a graph representation of our model. During
compilation, modules hidden within Sequential containers and wrapped
modules are all inlined into the graph, making them available for
pattern matching and optimization.

## Fusing Convolution with Batch Norm

Unlike some other fusions, fusion of convolution with batch norm does not
require any new operators. Instead, as batch norm during inference
consists of a pointwise add and multiply, these operations can be "baked"
into the preceding convolution's weights. This allows us to remove the batch
norm entirely from our model! Read
[https://nenadmarkus.com/p/fusing-batchnorm-and-conv/](https://nenadmarkus.com/p/fusing-batchnorm-and-conv/) for further details. The
code here is copied from
[pytorch/pytorch](https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/nn/utils/fusion.py)
clarity purposes.

## Pattern Matching with torch.compile

Now that we have our fusion logic, we need to register a pattern that
torch.compile's pattern matcher will recognize and replace during
compilation.

```
# Define the pattern we want to match: conv2d followed by batch_norm

# Example inputs are needed to trace the pattern functions.
# The inputs should match the function signatures of conv_bn_pattern and conv_bn_replacement.
# These are used to trace the pattern functions to create the match template.
# IMPORTANT: The pattern matcher is shape-agnostic! The specific shapes you use here
# don't limit what shapes will be matched - any valid conv2d->batch_norm sequence
# will be matched regardless of channels, kernel size, or spatial dimensions.
# - x: input tensor (batch_size, channels, height, width)
# - conv_weight: (out_channels, in_channels, kernel_h, kernel_w)
# - conv_bias: (out_channels,)
# - bn_mean, bn_var, bn_weight, bn_bias: all have shape (num_features,) matching out_channels

# Create a pattern matcher pass and register our pattern

# Create a custom pass function that applies our patterns

# Set our custom pass in the config
```

Note

We make some simplifications here for demonstration purposes, such as only
matching 2D convolutions. The pattern matcher in torch.compile
can handle more complex patterns.

## Testing out our Fusion Pass

We can now run this fusion pass on our initial toy model and verify that our
results are identical. In addition, we can print out the code for our fused
model and verify that there are no more batch norms.

```
# Clear the counters before compilation

# Ensure pattern matcher is enabled

# Run the model to trigger compilation and pattern matching

# Check how many patterns were matched

# Create a model with different shapes than our example_inputs

# Check how many patterns were matched
```

## Benchmarking our Fusion on ResNet18

We can test our fusion pass on a larger model like ResNet18 and see how much
this pass improves inference performance.

```
# Benchmark original model

# Compile with our custom pattern

# Benchmark compiled model

############
# Conclusion
# ----------
# As we can see, torch.compile provides a powerful way to implement
# graph transformations and optimizations through pattern matching.
# By registering custom patterns, we can extend torch.compile's
# optimization capabilities to handle domain-specific transformations.
#
# The conv-bn fusion demonstrated here is just one example of what's
# possible with torch.compile's pattern matching system.
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: torch_compile_conv_bn_fuser.ipynb`](../_downloads/0f07da1cd37d6cff0aa8a34f53cf282d/torch_compile_conv_bn_fuser.ipynb)

[`Download Python source code: torch_compile_conv_bn_fuser.py`](../_downloads/ca219776ab5f53cd5d489866f364e11b/torch_compile_conv_bn_fuser.py)

[`Download zipped: torch_compile_conv_bn_fuser.zip`](../_downloads/a90154adcde8725e10cc04b1de415a58/torch_compile_conv_bn_fuser.zip)