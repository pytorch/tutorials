Note

Go to the end
to download the full example code.

# Reducing AoT cold start compilation time with regional compilation

**Author:** [Sayak Paul](https://huggingface.co/sayakpaul), [Charles Bensimon](https://huggingface.co/cbensimon), [Angela Yi](https://github.com/angelayi)

In the [regional compilation recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html), we showed
how to reduce cold start compilation times while retaining (almost) full compilation benefits. This was demonstrated for
just-in-time (JIT) compilation.

This recipe shows how to apply similar principles when compiling a model ahead-of-time (AoT). If you
are not familiar with AOTInductor and `torch.export`, we recommend you to check out [this tutorial](https://docs.pytorch.org/tutorials/recipes/torch_export_aoti_python.html).

## Prerequisites

- Pytorch 2.6 or later
- Familiarity with regional compilation
- Familiarity with AOTInductor and `torch.export`

## Setup

Before we begin, we need to install `torch` if it is not already
available.

```
pip install torch
```

## Steps

In this recipe, we will follow the same steps as the regional compilation recipe mentioned above:

1. Import all necessary libraries.
2. Define and initialize a neural network with repeated regions.
3. Measure the compilation time of the full model and the regional compilation with AoT.

First, let's import the necessary libraries for loading our data:

## Defining the Neural Network

We will use the same neural network structure as the regional compilation recipe.

We will use a network, composed of repeated layers. This mimics a
large language model, that typically is composed of many Transformer blocks. In this recipe,
we will create a `Layer` using the `nn.Module` class as a proxy for a repeated region.
We will then create a `Model` which is composed of 64 instances of this
`Layer` class.

## Compiling the model ahead-of-time

Since we're compiling the model ahead-of-time, we need to prepare representative
input examples, that we expect the model to see during actual deployments.

Let's create an instance of `Model` and pass it some sample input data.

Now, let's compile our model ahead-of-time. We will use `input` created above to pass
to `torch.export`. This will yield a `torch.export.ExportedProgram` which we can compile.

We can load from this `path` and use it to perform inference.

## Compiling _regions_ of the model ahead-of-time

Compiling model regions ahead-of-time, on the other hand, requires a few key changes.

Since the compute pattern is shared by all the blocks that
are repeated in a model (`Layer` instances in this cases), we can just
compile a single block and let the inductor reuse it.

An exported program (`torch.export.ExportedProgram`) contains the Tensor computation,
a `state_dict` containing tensor values of all lifted parameters and buffer alongside
other metadata. We specify the `aot_inductor.package_constants_in_so` to be `False` to
not serialize the model parameters in the generated artifact.

Now, when loading the compiled binary, we can reuse the existing parameters of
each block. This lets us take advantage of the compiled binary obtained above.

Just like JIT regional compilation, compiling regions within a model ahead-of-time
leads to significantly reduced cold start times. The actual number will vary from
model to model.

Even though full model compilation offers the fullest scope of optimizations,
for practical purposes and depending on the type of model, we have seen regional
compilation (both JiT and AoT) providing similar speed benefits, while drastically
reducing the cold start times.

## Measuring compilation time

Next, let's measure the compilation time of the full model and the regional compilation.

There may also be layers in a model incompatible with compilation. So,
full compilation will result in a fragmented computation graph resulting
in potential latency degradation. In these case, regional compilation
can be beneficial.

## Conclusion

This recipe shows how to control the cold start time when compiling your
model ahead-of-time. This becomes effective when your model has repeated
blocks, which is typically seen in large generative models. We used this
recipe on various models to speed up real-time performance. Learn more
[here](https://huggingface.co/blog/zerogpu-aoti).

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: regional_aot.ipynb`](../_downloads/358714d0b9f9354d8e8cd3af8154ba50/regional_aot.ipynb)

[`Download Python source code: regional_aot.py`](../_downloads/dc7a7c633c87d05b3db480d6ea12dedf/regional_aot.py)

[`Download zipped: regional_aot.zip`](../_downloads/d9e0c071eae77a408515d7f524ce5159/regional_aot.zip)