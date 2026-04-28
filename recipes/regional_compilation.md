Note

Go to the end
to download the full example code.

# Reducing torch.compile cold start compilation time with regional compilation

**Author:** [Animesh Jain](https://github.com/anijain2305)

As deep learning models get larger, the compilation time of these models also
increases. This extended compilation time can result in a large startup time in
inference services or wasted resources in large-scale training. This recipe
shows an example of how to reduce the cold start compilation time by choosing to
compile a repeated region of the model instead of the entire model.

## Prerequisites

- Pytorch 2.5 or later

## Setup

Before we begin, we need to install `torch` if it is not already
available.

```
pip install torch
```

Note

This feature is available starting with the 2.5 release. If you are using version 2.4,
you can enable the configuration flag `torch._dynamo.config.inline_inbuilt_nn_modules=True`
to prevent recompilations during regional compilation. In version 2.5, this flag is enabled by default.

## Steps

In this recipe, we will follow these steps:

1. Import all necessary libraries.
2. Define and initialize a neural network with repeated regions.
3. Understand the difference between the full model and the regional compilation.
4. Measure the compilation time of the full model and the regional compilation.

First, let's import the necessary libraries for loading our data:

Next, let's define and initialize a neural network with repeated regions.

Typically, neural networks are composed of repeated layers. For example, a
large language model is composed of many Transformer blocks. In this recipe,
we will create a `Layer` using the `nn.Module` class as a proxy for a repeated region.
We will then create a `Model` which is composed of 64 instances of this
`Layer` class.

Next, let's review the difference between the full model and the regional compilation.

In full model compilation, the entire model is compiled as a whole. This is the common approach
most users take with `torch.compile`. In this example, we apply `torch.compile` to
the `Model` object. This will effectively inline the 64 layers, producing a
large graph to compile. You can look at the full graph by running this recipe
with `TORCH_LOGS=graph_code`.

The regional compilation, on the other hand, compiles a region of the model.
By strategically choosing to compile a repeated region of the model, we can compile a
much smaller graph and then reuse the compiled graph for all the regions.
In the example, `torch.compile` is applied only to the `layers` and not the full model.

Applying compilation to a repeated region, instead of full model, leads to
large savings in compile time. Here, we will just compile a layer instance and
then reuse it 64 times in the `Model` object.

Note that with repeated regions, some part of the model might not be compiled.
For example, the `self.linear` in the `Model` is outside of the scope of
regional compilation.

Also, note that there is a tradeoff between performance speedup and compile
time. Full model compilation involves a larger graph and,
theoretically, offers more scope for optimizations. However, for practical
purposes and depending on the model, we have observed many cases with minimal
speedup differences between the full model and regional compilation.

Next, let's measure the compilation time of the full model and the regional compilation.

`torch.compile` is a JIT compiler, which means that it compiles on the first invocation.
In the code below, we measure the total time spent in the first invocation. While this method is not
precise, it provides a good estimate since the majority of the time is spent in
compilation.

## Conclusion

This recipe shows how to control the cold start compilation time if your model
has repeated regions. This approach requires user modifications to apply torch.compile to
the repeated regions instead of more commonly used full model compilation. We
are continually working on reducing cold start compilation time.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: regional_compilation.ipynb`](../_downloads/cbd5804c4553cb4a23dc24137bde6077/regional_compilation.ipynb)

[`Download Python source code: regional_compilation.py`](../_downloads/1ac8a049de0513cb49a0e834e4c27a20/regional_compilation.py)

[`Download zipped: regional_compilation.zip`](../_downloads/9e958043a51d6477abc403a70ccb3646/regional_compilation.zip)