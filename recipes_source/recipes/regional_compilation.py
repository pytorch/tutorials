"""
Reducing torch.compile cold start compilation time with regional compilation
============================================================================

Introduction
------------
As deep learning models get larger, the compilation time of these models also
increase. This increase in compilation time can lead to a large startup time in
inference services or wasted resources in large scale training. This recipe
shows an example of how to reduce the cold start compilation time by choosing to
compile a repeated region of the model instead of the entire model.

Setup
-----
Before we begin, we need to install ``torch`` if it is not already
available.

.. code-block:: sh

   pip install torch

"""



######################################################################
# Steps
# -----
# 
# 1. Import all necessary libraries
# 2. Define and initialize a neural network with repeated regions.
# 3. Understand the difference between the full model and the regional compilation.
# 4. Measure the compilation time of the full model and the regional compilation.
# 
# 1. Import necessary libraries for loading our data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 

import torch
import torch.nn as nn
from time import perf_counter

# 
# 2. Define and initialize a neural network with repeated regions.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Typically neural networks are composed of repeated layers. For example, a
# large language model is composed of many Transformer blocks. In this recipe,
# we will create a `Layer` `nn.Module` class as a proxy for a repeated region.
# We will then create a `Model` which is composed of 64 instances of this
# `Layer` class.
# 
class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 10)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        a = self.linear1(x)
        a = self.relu1(a)
        a = torch.sigmoid(a)
        b = self.linear2(a)
        b = self.relu2(b)
        return b

class Model(torch.nn.Module):
    def __init__(self, apply_regional_compilation):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        # Apply compile only to the repeated layers.
        if apply_regional_compilation:
            self.layers = torch.nn.ModuleList([torch.compile(Layer()) for _ in range(64)])
        else:
            self.layers = torch.nn.ModuleList([Layer() for _ in range(64)])

    def forward(self, x):
        # In regional compilation, the self.linear is outside of the scope of `torch.compile`.
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x)
        return x

#
# 3. Understand the difference between the full model and the regional compilation.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In full model compilation, the full model is compiled as a whole. This is how
# most users use torch.compile. In this example, we can apply torch.compile to
# the `model` object. This will effectively inline the 64 layers, producing a
# large graph to compile. You can look at the full graph by running this recipe
# with `TORCH_LOGS=graph_code`.
# 
#

model = Model(apply_regional_compilation=False).cuda()
full_compiled_model = torch.compile(model)


# 
# The regional compilation, on the other hand, compiles a region of the model.
# By wisely choosing to compile a repeated region of the model, we can compile a
# much smaller graph and then reuse the compiled graph for all the regions. We
# can apply regional compilation in the example as follows. `torch.compile` is
# applied only to the `layers` and not the full model.
# 

regional_compiled_model = Model(apply_regional_compilation=True).cuda()

# Applying compilation to a repeated region, instead of full model, leads to
# large savings in compile time. Here, we will just compile a layer instance and
# then reuse it 64 times in the `model` object.
# 
# Note that with repeated regions, some part of the model might not be compiled.
# For example, the `self.linear` in the `Model` is outside of the scope of
# regional compilation.
# 
# Also, note that there is a tradeoff between performance speedup and compile
# time.  The full model compilation has larger graph and therefore,
# theoretically, has more scope for optimizations. However for practical
# purposes and depending on the model, we have observed many cases with minimal
# speedup differences between the full model and regional compilation.


#
# 4. Measure the compilation time of the full model and the regional compilation.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.compile` is a JIT compiler, i.e., it compiles on the first invocation.
# Here, we measure the total time spent in the first invocation. This is not
# precise, but it gives a good idea because the majority of time is spent in
# compilation.

def measure_latency(fn, input):
    # Reset the compiler caches to ensure no reuse between different runs
    torch.compiler.reset()
    with torch._inductor.utils.fresh_inductor_cache():
        start = perf_counter()
        fn(input)
        torch.cuda.synchronize()
        end = perf_counter()
        return end - start

input = torch.randn(10, 10, device="cuda")
full_model_compilation_latency = measure_latency(full_compiled_model, input)
print(f"Full model compilation time = {full_model_compilation_latency:.2f} seconds")

regional_compilation_latency = measure_latency(regional_compiled_model, input)
print(f"Regional compilation time = {regional_compilation_latency:.2f} seconds")

############################################################################
# This recipe shows how to control the cold start compilation time if your model
# has repeated regions. This requires user changes to apply `torch.compile` to
# the repeated regions instead of more commonly used full model compilation. We
# are continually working on reducing cold start compilation time. So, please
# stay tuned for our next tutorials.
# 
# This feature is available with 2.5 release. If you are on 2.4, you can use a
# config flag - `torch._dynamo.config.inline_inbuilt_nn_modules=True` to avoid
# recompilations on the regional compilation. In 2.5, this flag is turned on by
# default.
