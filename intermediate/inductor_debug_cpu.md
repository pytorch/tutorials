Note

Go to the end
to download the full example code.

# Inductor CPU backend debugging and profiling

**Authors**: [Xuan Liao](https://github.com/Valentine233), [Haozhe Zhu](https://github.com/zhuhaozhe), [Jiong Gong](https://github.com/jgong5), [Weihan Wang](https://github.com/EikanWang)

## Overview

PyTorch 2.0 introduced the compilation API called `torch.compile`.
This new feature offers a significant speedup over eager mode execution through graph-level optimization powered by the default Inductor backend.

This tutorial is intended to provide an in-depth introduction on the debugging
and performance profiling on Inductor CPU backend by delving into the intricacies of `torch.compile`.

Meanwhile, you may also find related tutorials about `torch.compile`
around [basic usage](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html),
comprehensive [troubleshooting](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
and GPU-specific knowledge like [GPU performance profiling](https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html).

We will start debugging with a motivating example that triggers compilation issues and accuracy problems
by demonstrating the process of debugging to pinpoint the problems.

By enabling logging and exploring the underlying generated code,
you can learn how to narrow down the failure step by step and finally figure out the route cause.

Following that, we will proceed to discuss how to profile the compiled code and,
through a performance comparison with eager mode,
elaborate on the reasons why `torch.compile` can provide an additional performance boost compared to its eager counterpart.

## Debugging

Here is a simple example to run the `torch.compile` using Inductor and compare its result with eager mode:

The correct implementation of `neg` in the `cpp` codegen is as follows:

In order to demonstrate the debugging, we will modify the function to a wrong one later.

### Get more logging information

No debugging information would be provided if you run this simple example by default. In order to get more useful debugging and logging information, we usually add a `TORCH_COMPILE_DEBUG` environment variable like below:

```
TORCH_COMPILE_DEBUG=1 python xx.py
```

This would print more debug information in the output logs and also dump the intermediate IRs generated during the codegen process. You can find the dumped file paths in the log like below:

```
torch._inductor.debug: [WARNING] model___20 debug trace: /tmp/torchinductor_root/rx/crxfi2ybd7yp5sbj2pnhw33wfhtdw7wumvrobyp5sjvdui5ktjc2.debug
```

In this directory, the following files are saved for debugging purposes:

| File | Description |
| --- | --- |
| `fx_graph_runnable.py` | Executable FX graph, after decomposition, before pattern match |
| `fx_graph_transformed.py` | Transformed FX graph, after pattern match |
| `ir_pre_fusion.txt` | Inductor IR before fusion |
| `ir_post_fusion.txt` | Inductor IR after fusion |
| `output_code.py` | Generated Python code for graph, with C++/Triton kernels |

Note that `fx_graph_runnable.py` and `output_code.py` are both runnable and editable in order to make debugging easier.
Here are the main parts of code extracted from the files and we correlate the C++ generated line with the FX code line.

`fx_graph_runnable`:

C++ kernel in `output_code`:

```
#include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
```

### Determine component of error

When encountering errors or accuracy problems, a straightforward solution to find the bug is to narrow down the problem. The first thing to do is to determine the component where the error occurs. Luckily, it can be simply achieved by changing the backend of `torch.compile`.

| Code | Description |
| --- | --- |
| `torch.compile(fn, backend="eager")` | Enable Dynamo |
| `torch.compile(fn, backend="aot_eager")` | Enable Dynamo + AOT Autograd |
| `torch.compile(fn, backend="inductor")` | Enable Dynamo + AOT Autograd + Inductor |

If the model can successfully run when the backend is set to `eager` or `aot_eager` while it fails with `inductor`, we can narrow down the failure to Inductor.

### Compilation error

As we know, the evolved chain of graph-level optimization is like:

```
torch.neg (Python) -> torch.ops.aten.neg.default (within FX graph) -> ops.neg (within IR node) -> tmp2 = -tmp1 (within C++ kernel)
```

If you encounter a compilation error, there is something wrong when compiling C++ kernels in the output code.
This type of error indicates that bugs are introduced when lowering IR nodes to output code.
The root cause of compilation error is usually shown in the traceback log.

For example, the `neg` function is modified like this:

The logging gives the following compile error with a rather clear reason.

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
 CppCompileError: C++ compile error
 /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp: In function 'void kernel(const unsigned char*, const unsigned char*, unsigned char*)':
 /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp:17:57: error: no matching function for call to 'max_propagate_nan(unsigned char&, int&)'
 17 | auto tmp3 = max_propagate_nan(tmp0, tmp2);
 | ^
 In file included from /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp:2:
 /tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note: candidate: 'template<class scalar_t> scalar_t max_propagate_nan(scalar_t, scalar_t)'
 27 | inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
 | ^~~~~~~~~~~~~~~~~
 /tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note: template argument deduction/substitution failed:
/tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp:17:57: note: deduced conflicting types for parameter 'scalar_t' ('unsigned char' and 'int')
 17 | auto tmp3 = max_propagate_nan(tmp0, tmp2);
 | ^
```

Let us also see the corresponding C++ kernel in output code and IR node.

C++ kernel:

```
include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
extern "C" void kernel(const unsigned char* in_ptr0,
 const unsigned char* in_ptr1,
 unsigned char* out_ptr0)
{
 {
 #pragma GCC ivdep
 for(long i0=static_cast<long>(0L); i0<static_cast<long>(8390L); i0+=static_cast<long>(1L))
 {
 #pragma GCC ivdep
 for(long i1=static_cast<long>(0L); i1<static_cast<long>(8L); i1+=static_cast<long>(1L))
 {
 auto tmp0 = in_ptr0[static_cast<long>(i1 + (8L*i0))];
 auto tmp1 = in_ptr1[static_cast<long>(i1)];
 auto tmp2 = -tmp1;
 auto tmp3 = max_propagate_nan(tmp0, tmp2);
 out_ptr0[static_cast<long>(i1 + (8L*i0))] = tmp3;
 }
 }
 }
}
```

IR node:

```
buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep('buf0', c0, {c0: 67120})]
buf0.unmet_dependencies = []
buf0.met_dependencies =
 [ MemoryDep('arg0_1', c1, {c0: 8390, c1: 8}),
 MemoryDep('arg1_1', c0, {c0: 67120})]
buf0.users = [NodeUser(node=OUTPUT, can_inplace=False)]
buf0.group.device = cpu
buf0.group.iteration = ((8390, 8), ())
buf0.sizes = ([8390, 8], [])
class buf0_loop_body:
 var_ranges = {z0: 8390, z1: 8}
 index0 = 8*z0 + z1
 index1 = z1
 def body(self, ops):
 get_index = self.get_index('index0')
 load = ops.load('arg1_1', get_index)
 get_index_1 = self.get_index('index1')
 load_1 = ops.load('arg0_1', get_index_1)
 neg = ops.neg(load_1)
 maximum = ops.maximum(load, neg)
 get_index_2 = self.get_index('index0')
 store = ops.store('buf0', get_index_2, maximum, None)
 return store
```

According to the traceback logging, the compilation error is caused by the data type inconsistency of `max_propagate_nan`'s inputs.
By checking the C++ kernel, we know that `tmp2` is no longer `long` after doing `-` as `tmp0` is `long`.
We can easily match `-` and `max_propagate_nan` in C++ kernel with `ops.neg` and `ops.maximum` in IR node respectively.

Now we successfully find that the root cause is the implementation of `ops.neg` in `cpp` codegen, which silently changes the data type when doing `neg`.

### Accuracy debugging

Otherwise, if the model runs with other errors or accuracy problem, you can use the PyTorch debugging tool called [Minifier](https://pytorch.org/functorch/stable/notebooks/minifier.html).

The core idea of `Minifier` is to keep removing the nodes and inputs of graph until finding the minimal graph with problem.
It helps to automatically generate a minified problematic graph through 4 strategies: truncating suffix, delta debugging, eliminating dead code and removing unused inputs.

We will now show the debugging process for the accuracy problem with the help of `Minifer`.
The accuracy problem refers to the case where the outputs of backends eager and inductor are different.

For instance, we modify the example like this:

And also modify the `neg` function:

An accuracy problem would be raised as follows:

```
torch._dynamo.utils: [ERROR] Accuracy failed: allclose not within tol=0.0001
Traceback (most recent call last):
 File "test_script.py", line 18, in <module>
 assert same(expected_result, actual_result) == True
AssertionError
```

To debug an accuracy problem with Minifier, two environment variables are needed:

```
TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4 python xx.py
```

Which gives us logging information that demonstrates the steps of minifying:

```
Started off with 6 nodes

Trying granularity 2
Strategy: Truncate suffix (G: 2) (6 nodes, 2 inputs)
SUCCESS: Went from 6 to 4 nodes

Trying granularity 4
Strategy: Remove unused inputs (G: 4) (4 nodes, 2 inputs)
SUCCESS: Went from 4 to 3 nodes
```

After running, we get the final minified graph with the target node `neg`:

For more usage details about Minifier, please refer to [Troubleshooting](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html).

## Performance profiling

Within this section, we will demonstrate the process of conducting performance analysis for a model that has been compiled using the Inductor CPU backend.
In the example below, we benchmark a Hugging Face Transformer model `MobileBertForQuestionAnswering` with both the eager mode and the Inductor graph mode.
The execution time and the speedup ratio of Inductor are printed after the benchmark.
We use Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz and run benchmark on the first socket to demonstrate the optimization within this section.
We set following environment variable as a best practice to benchmark on Intel(R) CPU.

```
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
numactl -C 0-31 -m 0 python bench.py
```

```
# bench.py

# Initialize an eager model

# Initialize the inductor model

# print(f"eager use: {eager_t * 1000 / NUM_ITERS} ms/iter")
# print(f"inductor use: {inductor_t * 1000 / NUM_ITERS} ms/iter")
# print(f"speed up ratio: {eager_t / inductor_t}")
```

Output:

```
eager use: 802.1023553796113 ms/iter
inductor use: 339.95180135127157 ms/iter
speed up ratio: 2.359459053287382
```

In our own testing, we find the Inductor CPU backend speed up the model by around 2.355x.

Next, let's dive deep into the performance at the operation level to understand where the speed-up comes from.
[Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) is a good tool to help us.
Inductor CPU backend has the support to report the time of the fusion kernels to the profiler with the `enable_kernel_profile` configuration option:

Following the steps in [Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
We are able to get the profiling table and trace files.

```
# bench.py
```

We get the following performance profiling table for the eager-mode model (omitting some columns):

```
------------------------- ------------ ------------ ------------
 Name CPU total % CPU total # of Calls
------------------------- ------------ ------------ ------------
 aten::addmm 45.73% 370.814ms 362
 aten::add 19.89% 161.276ms 363
 aten::copy_ 14.97% 121.416ms 488
 aten::mul 9.02% 73.154ms 194
 aten::clamp_min 8.81% 71.444ms 96
 aten::bmm 5.46% 44.258ms 48
 ProfilerStep* 100.00% 810.920ms 1
 aten::div 2.89% 23.447ms 24
 aten::_softmax 1.00% 8.087ms 24
 aten::linear 46.48% 376.888ms 362
 aten::clone 2.77% 22.430ms 98
 aten::t 0.31% 2.502ms 362
 aten::view 0.14% 1.161ms 850
 aten::transpose 0.17% 1.377ms 386
 aten::index_select 0.12% 952.000us 3
 aten::expand 0.12% 986.000us 458
 aten::matmul 8.31% 67.420ms 48
 aten::cat 0.09% 703.000us 1
 aten::as_strided 0.08% 656.000us 963
 aten::relu 8.86% 71.864ms 96
------------------------- ------------ ------------ ------------
Self CPU time total: 810.920ms
```

Similarly, we also get the table for the compiled model with Inductor (omitting some columns):

```
----------------------------------------------- ------------ ------------ ------------
 Name CPU total % CPU total # of Calls
----------------------------------------------- ------------ ------------ ------------
 mkl::_mkl_linear 68.79% 231.573ms 362
 aten::bmm 8.02% 26.992ms 48
 ProfilerStep* 100.00% 336.642ms 1
 graph_0_cpp_fused_constant_pad_nd_embedding_0 0.27% 915.000us 1
 aten::empty 0.27% 911.000us 362
 graph_0_cpp_fused__mkl_linear_add_mul_relu_151 0.27% 901.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_226 0.27% 899.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_361 0.27% 898.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_121 0.27% 895.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_31 0.27% 893.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_76 0.26% 892.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_256 0.26% 892.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_346 0.26% 892.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_241 0.26% 891.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_316 0.26% 891.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_91 0.26% 890.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_106 0.26% 890.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_211 0.26% 890.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_61 0.26% 889.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_286 0.26% 889.000us 1
----------------------------------------------- ------------ ------------ ------------
Self CPU time total: 336.642ms
```

From the profiling table of the eager model, we can see the most time consumption ops are [`aten::addmm`, `aten::add`, `aten::copy_`, `aten::mul`, `aten::clamp_min`, `aten::bmm`].
Comparing with the inductor model profiling table, we notice an `mkl::_mkl_linear` entry and multiple fused kernels in the form `graph_0_cpp_fused_*`. They are the major
optimizations that the inductor model is doing. Let us discuss them separately.

(1) Regarding `mkl::_mkl_linear`: You may notice the number of calls to this kernel is 362, which is exactly the same as `aten::linear` in the eager model profiling table.
The CPU total of `aten::linear` is 376.888ms, while it is 231.573ms for `mkl::_mkl_linear`. This suggests a ~1.63x for the "linear" part.
The speedup mainly comes from [packing the weight tensor to block memory format](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/cblas-gemm-pack-002.html)
and invoking [cblas_sgemm_compute](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/cblas-gemm-compute-002.html) within the Inductor CPU backend
to have a better cache behavior during GEMM computation.

(2) Regarding other memory-intensive ops: The end-to-end latency for the eager/inductor model is 802/339ms in our testing. So we can roughly infer that the speed up for the other memory-intensive ops is around 3.94x.
Let's read the generated code to understand how the inductor achieves this impressive optimization. You can find the generated code by
searching `cpp_fused__mkl_linear_add_mul_relu_151` in `output_code.py`

```
#include <ATen/record_function.h>
#include "/tmp/torchinductor_root/lr/clrlgu27q4ggd472umdzwsu6qcpqxcuusjxqvx2hwitjbujiiz7z.h"
```

From the generated code above, we can see this kernel has done a typical [Loop Fusion](https://en.wikipedia.org/wiki/Loop_fission_and_fusion) on `[add, add, mul, add]`.
This is a memory-bound bottle neck preventing good performance. To get a more intuitive feeling about this optimization,
we can infer the sizes and stride of the inputs and further benchmark this `[add, add, mul, add]` pattern.

```
# bench.py

# print(f"eager use: {eager_t * 1000 / NUM_ITERS} ms/iter")
# print(f"inductor use: {inductor_t * 1000 / NUM_ITERS} ms/iter")
# print(f"speed up ratio: {eager_t / inductor_t}")
```

Output:

```
eager use: 5.780875144992024 ms/iter
inductor use: 0.9588955780491233 ms/iter
speed up ratio: 6.0286805751604735
```

This is just an example. The profiling table shows all element-wise op are fused within the inductor automatically in this model. You can read more kernels in
output_code.py

## Conclusion

The document gives an in-depth tutorial for the Inductor CPU backend.

With motivating examples, we walk through the process of debugging and profiling.
The main idea is to narrow down the problem.

We demonstrate step by step the way to delve deeper the issue and find the root cause of failures, with the help of debugging logging and the tool Minifier.
Firstly determine which component the failure occurs in and then try to generate the smallest snippet of code that can reproduce the failure.

When the performance with Inductor is better than that of eager mode, we provide a solid analytical method for performance profiling.
We show how to find the time-consuming hotspot with PyTorch Profiler and figure out the operator-level or kernel-level reason to explain the phenomenon.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: inductor_debug_cpu.ipynb`](../_downloads/57fbbe6265e9c97da47580b6e60037ac/inductor_debug_cpu.ipynb)

[`Download Python source code: inductor_debug_cpu.py`](../_downloads/864b90f09a798ba06b420b737cd463b1/inductor_debug_cpu.py)

[`Download zipped: inductor_debug_cpu.zip`](../_downloads/060c4e307cd150879aabace05fa1e16d/inductor_debug_cpu.zip)