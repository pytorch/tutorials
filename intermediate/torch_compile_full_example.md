Note

Go to the end
to download the full example code.

# `torch.compile` End-to-End Tutorial

**Author:** William Wen

`torch.compile` is the new way to speed up your PyTorch code!
`torch.compile` makes PyTorch code run faster by
JIT-compiling PyTorch code into optimized kernels,
while requiring minimal code changes.

This tutorial covers an end-to-end example of training and evaluating a
real model with `torch.compile`. For a gentle introduction to `torch.compile`,
please check out [the introduction to torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).

**Required pip Dependencies**

- `torch >= 2.0`
- `torchvision`

 What you will learn

- How to apply `torch.compile` to a real model
- `torch.compile` speedups on a real model
- `torch.compile`'s first few iterations are expected to be slower due to compilation overhead

 Prerequisites

- [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

```
# NOTE: a modern NVIDIA GPU (H100, A100, or V100) is recommended for this tutorial in
# order to reproduce the speedup numbers shown below and documented elsewhere.

import torch
import warnings

gpu_ok = False
if torch.cuda.is_available():
 device_cap = torch.cuda.get_device_capability()
 if device_cap in ((7, 0), (8, 0), (9, 0)):
 gpu_ok = True

if not gpu_ok:
 warnings.warn(
 "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
 "than expected."
 )
```

```
/var/lib/workspace/intermediate_source/torch_compile_full_example.py:51: UserWarning: GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected.
 warnings.warn(
```

Let's demonstrate how using `torch.compile` can speed up a real model.
We will compare standard eager mode and
`torch.compile` by evaluating and training a `torchvision` model on random data.

Before we start, we need to define some utility functions.

```
# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
 start = torch.cuda.Event(enable_timing=True)
 end = torch.cuda.Event(enable_timing=True)
 start.record()
 result = fn()
 end.record()
 torch.cuda.synchronize()
 return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
 return (
 torch.randn(b, 3, 128, 128).cuda(),
 torch.randint(1000, (b,)).cuda(),
 )

N_ITERS = 10

from torchvision.models import densenet121

def init_model():
 return densenet121().cuda()
```

First, let's compare inference.

Note that in the call to `torch.compile`, we have the additional
`mode` argument, which we will discuss below.

```
model = init_model()

# Note that we generally recommend directly compiling a torch.nn.Module by calling
# its .compile() method.
model_opt = init_model()
model_opt.compile(mode="reduce-overhead")

inp = generate_data(16)[0]
with torch.no_grad():
 print("eager:", timed(lambda: model(inp))[1])
 print("compile:", timed(lambda: model_opt(inp))[1])
```

```
eager: 0.34860647583007814
/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py:320: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
 warnings.warn(
compile: 52.69946484375
```

Notice that `torch.compile` takes a lot longer to complete
compared to eager. This is because `torch.compile` compiles
the model into optimized kernels as it executes. In our example, the
structure of the model doesn't change, and so recompilation is not
needed. So if we run our optimized model several more times, we should
see a significant improvement compared to eager.

```
eager_times = []
for i in range(N_ITERS):
 inp = generate_data(16)[0]
 with torch.no_grad():
 _, eager_time = timed(lambda: model(inp))
 eager_times.append(eager_time)
 print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

compile_times = []
for i in range(N_ITERS):
 inp = generate_data(16)[0]
 with torch.no_grad():
 _, compile_time = timed(lambda: model_opt(inp))
 compile_times.append(compile_time)
 print(f"compile eval time {i}: {compile_time}")
print("~" * 10)

import numpy as np

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert speedup > 1
print(
 f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
)
print("~" * 10)
```

```
eager eval time 0: 0.01802239990234375
eager eval time 1: 0.017136480331420897
eager eval time 2: 0.01659903907775879
eager eval time 3: 0.01668908882141113
eager eval time 4: 0.016578559875488282
eager eval time 5: 0.016482303619384766
eager eval time 6: 0.01660016059875488
eager eval time 7: 0.016507904052734376
eager eval time 8: 0.016656383514404297
eager eval time 9: 0.0164800968170166
~~~~~~~~~~
compile eval time 0: 0.08504518127441406
compile eval time 1: 0.008691712379455567
compile eval time 2: 0.008939519882202148
compile eval time 3: 0.00808448028564453
compile eval time 4: 0.008076288223266602
compile eval time 5: 0.008031231880187988
compile eval time 6: 0.008011775970458984
compile eval time 7: 0.008022015571594238
compile eval time 8: 0.00801587200164795
compile eval time 9: 0.008005632400512695
~~~~~~~~~~
(eval) eager median: 0.016599599838256835, compile median: 0.008053760051727296, speedup: 2.061099378630818x
~~~~~~~~~~
```

And indeed, we can see that running our model with `torch.compile`
results in a significant speedup. Speedup mainly comes from reducing Python overhead and
GPU read/writes, and so the observed speedup may vary on factors such as model
architecture and batch size. For example, if a model's architecture is simple
and the amount of data is large, then the bottleneck would be
GPU compute and the observed speedup may be less significant.

You may also see different speedup results depending on the chosen `mode`
argument. The `"reduce-overhead"` mode uses CUDA graphs to further reduce
the overhead of Python. For your own models,
you may need to experiment with different modes to maximize speedup. You can
read more about modes [here](https://pytorch.org/get-started/pytorch-2.0/#user-experience).

You may might also notice that the second time we run our model with `torch.compile` is significantly
slower than the other runs, although it is much faster than the first run. This is because the `"reduce-overhead"`
mode runs a few warm-up iterations for CUDA graphs.

Now, let's consider comparing training.

```
model = init_model()
opt = torch.optim.Adam(model.parameters())

def train(mod, data):
 opt.zero_grad(True)
 pred = mod(data[0])
 loss = torch.nn.CrossEntropyLoss()(pred, data[1])
 loss.backward()
 opt.step()

eager_times = []
for i in range(N_ITERS):
 inp = generate_data(16)
 _, eager_time = timed(lambda: train(model, inp))
 eager_times.append(eager_time)
 print(f"eager train time {i}: {eager_time}")
print("~" * 10)

model = init_model()
opt = torch.optim.Adam(model.parameters())

# Note that because we are compiling a regular Python function, we do not
# call any .compile() method.
train_opt = torch.compile(train, mode="reduce-overhead")

compile_times = []
for i in range(N_ITERS):
 inp = generate_data(16)
 _, compile_time = timed(lambda: train_opt(model, inp))
 compile_times.append(compile_time)
 print(f"compile train time {i}: {compile_time}")
print("~" * 10)

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert speedup > 1
print(
 f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
)
print("~" * 10)
```

```
eager train time 0: 0.3442607727050781
eager train time 1: 0.051601375579833984
eager train time 2: 0.04911718368530273
eager train time 3: 0.048892929077148435
eager train time 4: 0.04935782241821289
eager train time 5: 0.04922265625
eager train time 6: 0.04884787368774414
eager train time 7: 0.04923699188232422
eager train time 8: 0.04926054382324219
eager train time 9: 0.04871558380126953
~~~~~~~~~~
compile train time 0: 160.570125
compile train time 1: 2.62533837890625
compile train time 2: 0.023964672088623046
compile train time 3: 0.020967424392700194
compile train time 4: 0.020320255279541014
compile train time 5: 0.020311040878295897
compile train time 6: 0.02030182456970215
compile train time 7: 0.020279296875
compile train time 8: 0.020281343460083007
compile train time 9: 0.020281343460083007
~~~~~~~~~~
(train) eager median: 0.04922982406616211, compile median: 0.020315648078918454, speedup: 2.4232465474359093x
~~~~~~~~~~
```

Again, we can see that `torch.compile` takes longer in the first
iteration, as it must compile the model, but in subsequent iterations, we see
significant speedups compared to eager.

We remark that the speedup numbers presented in this tutorial are for
demonstration purposes only. Official speedup values can be seen at the
[TorchInductor performance dashboard](https://hud.pytorch.org/benchmark/compilers).

## Conclusion

In this tutorial, we applied `torch.compile` to training and inference on a real model,
demonstrating speedups.

Importantly, we note that the first few iterations of a compiled model
are slower than eager mode due to compilation overhead, but subsequent iterations are expected to
have speedups.

For a gentle introduction to `torch.compile`, please check out [the introduction to torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).

To troubleshoot issues and to gain a deeper understanding of how to apply `torch.compile` to your code, check out [the torch.compile programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html).

We hope that you will give `torch.compile` a try!

**Total running time of the script:** (3 minutes 39.127 seconds)

[`Download Jupyter notebook: torch_compile_full_example.ipynb`](../_downloads/cf1148cb3c2260353d407c20256391cd/torch_compile_full_example.ipynb)

[`Download Python source code: torch_compile_full_example.py`](../_downloads/9563ce28e48bb09db22ec0071539f25a/torch_compile_full_example.py)

[`Download zipped: torch_compile_full_example.zip`](../_downloads/43b18f26a804521120218f3a9d5ec40c/torch_compile_full_example.zip)