
"""
Reducing AoT cold start compilation time with regional compilation
============================================================================

**Author:** `Sayak Paul <https://github.com/sayakpaul>`_, `Charles Bensimon <https://github.com/cbensimon>`_, `Angela Yi <https://github.com/angelayi>`_

In the [regional compilation recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html), we showed
how to reduce cold start compilation times while retaining (almost) full compilation benefits. This was demonstrated for
just-in-time (JIT) compilation.

This recipe shows how to apply similar principles when compiling a model ahead-of-time (AoT). If you
are not familiar with AOTInductor and ``torch.export``, we recommend you to check out [this tutorial](https://docs.pytorch.org/tutorials/recipes/torch_export_aoti_python.html).

Prerequisites
----------------

* Pytorch 2.6 or later
* Familiarity with regional compilation
* Familiarity with AOTInductor and ``torch.export``

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
# In this recipe, we will follow the same steps as the regional compilation recipe mentioned above:
#
# 1. Import all necessary libraries.
# 2. Define and initialize a neural network with repeated regions.
# 3. Measure the compilation time of the full model and the regional compilation with AoT.
#
# First, let's import the necessary libraries for loading our data:
#

import torch
torch.set_grad_enabled(False)

from time import perf_counter

###################################################################################
# Defining the Neural Network
# ---------------------------
# 
# We will use the same neural network structure as the regional compilation recipe.
#
# We will use a network, composed of repeated layers. This mimics a
# large language model, that typically is composed of many Transformer blocks. In this recipe,
# we will create a ``Layer`` using the ``nn.Module`` class as a proxy for a repeated region.
# We will then create a ``Model`` which is composed of 64 instances of this
# ``Layer`` class.
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
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.layers = torch.nn.ModuleList([Layer() for _ in range(64)])

    def forward(self, x):
        # In regional compilation, the self.linear is outside of the scope of ``torch.compile``.
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Compiling the model ahead-of-time
# ---------------------------------
# 
# Since we're compiling the model ahead-of-time, we need to prepare representative
# input examples, that we expect the model to see during actual deployments.
# 
# Let's create an instance of ``Model`` and pass it some sample input data.
# 

model = Model().cuda()
input = torch.randn(10, 10, device="cuda")
output = model(input)
print(f"{output.shape=}")

###############################################################################################
# Now, let's compile our model ahead-of-time. We will use ``input`` created above to pass
# to ``torch.export``. This will yield a ``torch.export.ExportedProgram`` which we can compile.

path = torch._inductor.aoti_compile_and_package(
    torch.export.export(model, args=(input,))
)

#################################################################
# We can load from this ``path`` and use it to perform inference.

compiled_binary = torch._inductor.aoti_load_package(path)
output_compiled = compiled_binary(input)
print(f"{output_compiled.shape=}")

######################################################################################
# Compiling _regions_ of the model ahead-of-time
# ----------------------------------------------
# 
# Compiling model regions ahead-of-time, on the other hand, requires a few key changes.
#
# Since the compute pattern is shared by all the blocks that
# are repeated in a model (``Layer`` instances in this cases), we can just
# compile a single block and let the inductor reuse it.

model = Model().cuda()
path = torch._inductor.aoti_compile_and_package(
    torch.export.export(model.layers[0], args=(input,)),
    inductor_configs={
        # compile artifact w/o saving params in the artifact
        "aot_inductor.package_constants_in_so": False,
    }
)

###################################################
# An exported program (``torch.export.ExportedProgram``) contains the Tensor computation,
# a ``state_dict`` containing tensor values of all lifted parameters and buffer alongside 
# other metadata. We specify the ``aot_inductor.package_constants_in_so`` to be ``False`` to
# not serialize the model parameters in the generated artifact.
#
# Now, when loading the compiled binary, we can reuse the existing parameters of
# each block. This lets us take advantage of the compiled binary obtained above.
# 

for layer in model.layers:
    compiled_layer = torch._inductor.aoti_load_package(path)
    compiled_layer.load_constants(
        layer.state_dict(), check_full_update=True, user_managed=True
    )
    layer.forward = compiled_layer

output_regional_compiled = model(input)
print(f"{output_regional_compiled.shape=}")

#####################################################
# Just like JIT regional compilation, compiling regions within a model ahead-of-time
# leads to significantly reduced cold start times. The actual number will vary from
# model to model.
#
# Even though full model compilation offers the fullest scope of optimizations,
# for practical purposes and depending on the type of model, we have seen regional
# compilation (both JiT and AoT) providing similar speed benefits, while drastically
# reducing the cold start times.

###################################################
# Measuring compilation time
# --------------------------
# Next, let's measure the compilation time of the full model and the regional compilation.
#

def measure_compile_time(input, regional=False):
    start = perf_counter()
    model = aot_compile_load_model(regional=regional)
    torch.cuda.synchronize()
    end = perf_counter()
    # make sure the model works.
    _ = model(input)
    return end - start

def aot_compile_load_model(regional=False) -> torch.nn.Module:
    input = torch.randn(10, 10, device="cuda")
    model = Model().cuda()
    
    inductor_configs = {}
    if regional:
        inductor_configs = {"aot_inductor.package_constants_in_so": False}
    
    # Reset the compiler caches to ensure no reuse between different runs
    torch.compiler.reset()
    with torch._inductor.utils.fresh_inductor_cache():
        path = torch._inductor.aoti_compile_and_package(
            torch.export.export(
                model.layers[0] if regional else model, 
                args=(input,)
            ),
            inductor_configs=inductor_configs,
        )

        if regional:
            for layer in model.layers:
                compiled_layer = torch._inductor.aoti_load_package(path)
                compiled_layer.load_constants(
                    layer.state_dict(), check_full_update=True, user_managed=True
                )
                layer.forward = compiled_layer
        else:
            model = torch._inductor.aoti_load_package(path)
    return model

input = torch.randn(10, 10, device="cuda")
full_model_compilation_latency = measure_compile_time(input, regional=False)
print(f"Full model compilation time = {full_model_compilation_latency:.2f} seconds")

regional_compilation_latency = measure_compile_time(input, regional=True)
print(f"Regional compilation time = {regional_compilation_latency:.2f} seconds")

assert regional_compilation_latency < full_model_compilation_latency

############################################################################
# There may also be layers in a model incompatible with compilation. So, 
# full compilation will result in a fragmented computation graph resulting
# in potential latency degradation. In these case, regional compilation
# can be beneficial.
# 

############################################################################
# Conclusion
# -----------
#
# This recipe shows how to control the cold start time when compiling your 
# model ahead-of-time. This becomes effective when your model has repeated
# blocks, which is typically seen in large generative models.
