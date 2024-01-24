# -*- coding: utf-8 -*-

"""
Inductor CPU backend debugging and profiling
============================================

**Authors**: `Xuan Liao <https://github.com/Valentine233>`_, `Haozhe Zhu <https://github.com/zhuhaozhe>`_, `Jiong Gong <https://github.com/jgong5>`_, `Weihan Wang <https://github.com/EikanWang>`_
"""

#########################################################################
# Overview
# --------
# 
# PyTorch 2.0 introduced the compilation API called ``torch.compile``. 
# This new feature offers a significant speedup over eager mode execution through graph-level optimization powered by the default Inductor backend.
#
# This tutorial is intended to provide an in-depth introduction on the debugging 
# and performance profiling on Inductor CPU backend by delving into the intricacies of ``torch.compile``. 
#
# Meanwhile, you may also find related tutorials about ``torch.compile`` 
# around `basic usage <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_, 
# comprehensive `troubleshooting <https://pytorch.org/docs/stable/dynamo/troubleshooting.html>`_ 
# and GPU-specific knowledge like `GPU performance profiling <https://github.com/pytorch/pytorch/blob/main/docs/source/compile/profiling_torch_compile.rst>`_.
#
# We will start debugging with a motivating example that triggers compilation issues and accuracy problems 
# by demonstrating the process of debugging to pinpoint the problems.
#
# By enabling logging and exploring the underlying generated code, 
# you can learn how to narrow down the failure step by step and finally figure out the route cause.
#
# Following that, we will proceed to discuss how to profile the compiled code and, 
# through a performance comparison with eager mode, 
# elaborate on the reasons why ``torch.compile`` can provide an additional performance boost compared to its eager counterpart.


######################################################################
# Debugging
# ---------
#
# Here is a simple example to run the ``torch.compile`` using Inductor and compare its result with eager mode:

import torch

def foo1(x1, x2):
    a = torch.neg(x1)
    b = torch.maximum(x2, a)
    y = torch.cat([b], dim=0)
    return y

x1 = torch.randint(256, (1, 8), dtype=torch.uint8)
x2 = torch.randint(256, (8390, 8), dtype=torch.uint8)

compiled_foo1 = torch.compile(foo1)
result = compiled_foo1(x1, x2)

######################################################################
# The correct implementation of ``neg`` in the ``cpp`` codegen is as follows:

def neg1(x):
    return f"decltype({x})(-{x})"

######################################################################
# In order to demonstrate the debugging, we will modify the function to a wrong one later.
#
#
# Get more logging information
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# No debugging information would be provided if you run this simple example by default. In order to get more useful debugging and logging information, we usually add a ``TORCH_COMPILE_DEBUG`` environment variable like below:
#
# .. code-block:: shell
#
# 	TORCH_COMPILE_DEBUG=1 python xx.py
#
# This would print more debug information in the output logs and also dump the intermediate IRs generated during the codegen process. You can find the dumped file paths in the log like below:
#
# .. code-block:: shell
#
# 	torch._inductor.debug: [WARNING] model___20 debug trace: /tmp/torchinductor_root/rx/crxfi2ybd7yp5sbj2pnhw33wfhtdw7wumvrobyp5sjvdui5ktjc2.debug
#
# In this directory, the following files are saved for debugging purposes:
#
# +-----------------------------+----------------------------------------------------------------+
# | File                        | Description                                                    |
# +=============================+================================================================+
# | ``fx_graph_runnable.py``    | Executable FX graph, after decomposition, before pattern match |
# +-----------------------------+----------------------------------------------------------------+
# | ``fx_graph_transformed.py`` | Transformed FX graph, after pattern match                      |
# +-----------------------------+----------------------------------------------------------------+
# | ``ir_post_fusion.txt``      | Inductor IR before fusion                                      |
# +-----------------------------+----------------------------------------------------------------+
# | ``ir_pre_fusion.txt``       | Inductor IR after fusion                                       |
# +-----------------------------+----------------------------------------------------------------+
# | ``output_code.py``          | Generated Python code for graph, with C++/Triton kernels       |
# +-----------------------------+----------------------------------------------------------------+
#
# Note that ``fx_graph_runnable.py`` and ``output_code.py`` are both runnable and editable in order to make debugging easier. 
# Here are the main parts of code extracted from the files and we correlate the C++ generated line with the FX code line.
#
# ``fx_graph_runnable``:
#

def forward1(self, arg0_1, arg1_1):
    neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
    maximum = torch.ops.aten.maximum.default(arg1_1, neg);  arg1_1 = neg = None
    clone = torch.ops.aten.clone.default(maximum);  maximum = None
    return (clone,)

######################################################################
# C++ kernel in ``output_code``:
#

from torch._inductor.codecache import AsyncCompile
async_compile = AsyncCompile()

cpp_fused_cat_maximum_neg_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
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
                // Corresponding FX code line: neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
                auto tmp2 = decltype(tmp1)(-tmp1);
                // Corresponding FX code line: maximum = torch.ops.aten.maximum.default(arg1_1, neg);  arg1_1 = neg = None
                auto tmp3 = max_propagate_nan(tmp0, tmp2);
                // Corresponding FX code line: clone = torch.ops.aten.clone.default(maximum);  maximum = None
                out_ptr0[static_cast<long>(i1 + (8L*i0))] = tmp3;
            }
        }
    }
}''')


######################################################################
# Determine component of error
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When encountering errors or accuracy problems, a straightforward solution to find the bug is to narrow down the problem. The first thing to do is to determine the component where the error occurs. Luckily, it can be simply achieved by changing the backend of ``torch.compile``.
#
# +--------------------------------------------+-----------------------------------------+
# | Code                                       | Description                             |
# +============================================+=========================================+
# | ``torch.compile(fn, backend="eager")``     | Enable Dynamo                           |
# +--------------------------------------------+-----------------------------------------+
# | ``torch.compile(fn, backend="aot_eager")`` | Enable Dynamo + AOT Autograd            |
# +--------------------------------------------+-----------------------------------------+
# | ``torch.compile(fn, backend="inductor")``  | Enable Dynamo + AOT Autograd + Inductor |
# +--------------------------------------------+-----------------------------------------+
#
# If the model can successfully run when the backend is set to ``eager`` or ``aot_eager`` while it fails with ``inductor``, we can narrow down the failure to Inductor.
#
#
# Compilation error
# ^^^^^^^^^^^^^^^^^
#
# As we know, the evolved chain of graph-level optimization is like:
#
# .. code-block:: sh
#
# 	torch.neg (Python) -> torch.ops.aten.neg.default (within FX graph) -> ops.neg (within IR node) -> tmp2 = -tmp1 (within C++ kernel)
#
# If you encounter a compilation error, there is something wrong when compiling C++ kernels in the output code.
# This type of error indicates that bugs are introduced when lowering IR nodes to output code.
# The root cause of compilation error is usually shown in the traceback log.
#
# For example, the ``neg`` function is modified like this:

def neg2(x):
    return f"-{x}"

######################################################################
# The logging gives the following compile error with a rather clear reason.
#
# .. code-block::
#
#    torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
#    CppCompileError: C++ compile error
#    /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp: In function ‘void kernel(const unsigned char*, const unsigned char*, unsigned char*)’:
#    /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp:17:57: error: no matching function for call to ‘max_propagate_nan(unsigned char&, int&)’
#      17 |                 auto tmp3 = max_propagate_nan(tmp0, tmp2);
#           |                                                         ^
#    In file included from /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp:2:
#    /tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note: candidate: ‘template<class scalar_t> scalar_t max_propagate_nan(scalar_t, scalar_t)’
#    27 | inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
#         |                 ^~~~~~~~~~~~~~~~~
#    /tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note:   template argument deduction/substitution failed:
#   /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp:17:57: note:   deduced conflicting types for parameter ‘scalar_t’ (‘unsigned char’ and ‘int’)
#    17 |                 auto tmp3 = max_propagate_nan(tmp0, tmp2);
#         |                                                         ^
#
#
# Let us also see the corresponding C++ kernel in output code and IR node.
#
# C++ kernel:
#
# .. code:: c
#
#     include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
#     extern "C" void kernel(const unsigned char* in_ptr0,
#                         const unsigned char* in_ptr1,
#                         unsigned char* out_ptr0)
#     {
#         {
#             #pragma GCC ivdep
#             for(long i0=static_cast<long>(0L); i0<static_cast<long>(8390L); i0+=static_cast<long>(1L))
#             {
#                 #pragma GCC ivdep
#                 for(long i1=static_cast<long>(0L); i1<static_cast<long>(8L); i1+=static_cast<long>(1L))
#                 {
#                     auto tmp0 = in_ptr0[static_cast<long>(i1 + (8L*i0))];
#                     auto tmp1 = in_ptr1[static_cast<long>(i1)];
#                     auto tmp2 = -tmp1;
#                     auto tmp3 = max_propagate_nan(tmp0, tmp2);
#                     out_ptr0[static_cast<long>(i1 + (8L*i0))] = tmp3;
#                 }
#             }
#         }
#     }
#

######################################################################
# IR node:
#
# .. code-block:: sh
#
#     buf0: SchedulerNode(ComputedBuffer)
#     buf0.writes = [MemoryDep('buf0', c0, {c0: 67120})]
#     buf0.unmet_dependencies = []
#     buf0.met_dependencies = 
#         [   MemoryDep('arg0_1', c1, {c0: 8390, c1: 8}),
#             MemoryDep('arg1_1', c0, {c0: 67120})]
#     buf0.users = [NodeUser(node=OUTPUT, can_inplace=False)]
#     buf0.group.device = cpu
#     buf0.group.iteration = ((8390, 8), ())
#     buf0.sizes = ([8390, 8], [])
#     class buf0_loop_body:
#         var_ranges = {z0: 8390, z1: 8}
#         index0 = 8*z0 + z1
#         index1 = z1
#         def body(self, ops):
#             get_index = self.get_index('index0')
#             load = ops.load('arg1_1', get_index)
#             get_index_1 = self.get_index('index1')
#             load_1 = ops.load('arg0_1', get_index_1)
#             neg = ops.neg(load_1)
#             maximum = ops.maximum(load, neg)
#             get_index_2 = self.get_index('index0')
#             store = ops.store('buf0', get_index_2, maximum, None)
#             return store
#

######################################################################
# According to the traceback logging, the compilation error is caused by the data type inconsistency of ``max_propagate_nan``'s inputs. 
# By checking the C++ kernel, we know that ``tmp2`` is no longer ``long`` after doing ``-`` as ``tmp0`` is ``long``.
# We can easily match ``-`` and ``max_propagate_nan`` in C++ kernel with ``ops.neg`` and ``ops.maximum`` in IR node respectively.
#
# Now we successfully find that the root cause is the implementation of ``ops.neg`` in ``cpp`` codegen, which silently changes the data type when doing ``neg``. 
#
#
# Accuracy debugging
# ^^^^^^^^^^^^^^^^^^^
#
# Otherwise, if the model runs with other errors or accuracy problem, you can use the PyTorch debugging tool called `Minifier <https://pytorch.org/functorch/stable/notebooks/minifier.html>`_. 
#
# The core idea of ``Minifier`` is to keep removing the nodes and inputs of graph until finding the minimal graph with problem.
# It helps to automatically generate a minified problematic graph through 4 strategies: truncating suffix, delta debugging, eliminating dead code and removing unused inputs.
#
#
# We will now show the debugging process for the accuracy problem with the help of ``Minifer``. 
# The accuracy problem refers to the case where the outputs of backends eager and inductor are different. 
#
# For instance, we modify the example like this:

from torch._dynamo.utils import same

def foo2(x1, x2):
    a = torch.neg(x1)
    b = torch.maximum(x2, a)
    y = torch.cat([b], dim=0)
    return y

x1 = torch.randn((1, 8), dtype=torch.float32)
x2 = torch.randn((8390, 8), dtype=torch.float32)

expected_result = foo2(x1, x2)

compiled_foo2 = torch.compile(foo2)
actual_result = compiled_foo2(x1, x2)

assert same(expected_result, actual_result) == True

######################################################################
# And also modify the ``neg`` function:

def neg3(x):
    return f"decltype({x})(2 * {x})"

######################################################################
# An accuracy problem would be raised as follows:
#
# .. code-block:: sh
#
# 	torch._dynamo.utils: [ERROR] Accuracy failed: allclose not within tol=0.0001
# 	Traceback (most recent call last):
# 	  File "test_script.py", line 18, in <module>
# 	    assert same(expected_result, actual_result) == True
# 	AssertionError
#
# To debug an accuracy problem with Minifier, two environment variables are needed:
#
# .. code-block:: sh
#
#    TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4 python xx.py
#
# Which gives us logging information that demonstrates the steps of minifying:
#
# .. code-block:: sh
#
#     Started off with 6 nodes
#
#     Trying granularity 2
#     Strategy: Truncate suffix (G: 2) (6 nodes, 2 inputs)
#     SUCCESS: Went from 6 to 4 nodes
#
#     Trying granularity 4
#     Strategy: Remove unused inputs (G: 4) (4 nodes, 2 inputs)
#     SUCCESS: Went from 4 to 3 nodes
#
# After running, we get the final minified graph with the target node ``neg``:

def forward2(self, arg0_1):
    neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
    return (neg,)

######################################################################
# For more usage details about Minifier, please refer to `Troubleshooting <https://pytorch.org/docs/stable/dynamo/troubleshooting.html>`_.


######################################################################
# Performance profiling
# ---------------------
#
# Within this section, we will demonstrate the process of conducting performance analysis for a model that has been compiled using the Inductor CPU backend.
# In the example below, we benchmark a Hugging Face Transformer model ``MobileBertForQuestionAnswering`` with both the eager mode and the Inductor graph mode.
# The execution time and the speedup ratio of Inductor are printed after the benchmark.
# We use Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz and run benchmark on the first socket to demonstrate the optimization within this section.
# We set following environment variable as a best practice to benchmark on Intel(R) CPU.

#########################################################
# .. code-block:: shell
#
#     export KMP_BLOCKTIME=1
#     export KMP_SETTINGS=1
#     export KMP_AFFINITY=granularity=fine,compact,1,0
#     export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
#     export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
#     numactl -C 0-31 -m 0 python bench.py
#

# bench.py
from transformers import MobileBertForQuestionAnswering
# Initialize an eager model
model = MobileBertForQuestionAnswering.from_pretrained("csarron/mobilebert-uncased-squad-v2")
seq_length = 128
bs = 128
vocab_size = model.config.vocab_size
input = torch.randint(0, vocab_size, (bs, seq_length), dtype=torch.int64)
input_dict = {"input_ids": input}

# Initialize the inductor model
compiled_model = torch.compile(model)
with torch.no_grad():
    compiled_model(**input_dict)

NUM_ITERS=50
import timeit
with torch.no_grad():
    # warmup
    for _ in range(10):
        model(**input_dict)
    eager_t = timeit.timeit("model(**input_dict)", number=NUM_ITERS, globals=globals())

with torch.no_grad():
    # warmup
    for _ in range(10):
        compiled_model(**input_dict)
    inductor_t = timeit.timeit("compiled_model(**input_dict)", number=NUM_ITERS, globals=globals())
# print(f"eager use: {eager_t * 1000 / NUM_ITERS} ms/iter")
# print(f"inductor use: {inductor_t * 1000 / NUM_ITERS} ms/iter")
# print(f"speed up ratio: {eager_t / inductor_t}")


######################################################################
# Output:
#
# .. code-block:: shell
#
#     eager use: 802.1023553796113 ms/iter
#     inductor use: 339.95180135127157 ms/iter
#     speed up ratio: 2.359459053287382
#
# In our own testing, we find the Inductor CPU backend speed up the model by around 2.355x.
#
#
# Next, let's dive deep into the performance at the operation level to understand where the speed-up comes from.
# `Pytorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_ is a good tool to help us. 
# Inductor CPU backend has the support to report the time of the fusion kernels to the profiler with the ``enable_kernel_profile`` configuration option:

from torch._inductor import config
config.cpp.enable_kernel_profile = True

######################################################################
# Following the steps in `Pytorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_
# We are able to get the profiling table and trace files.

# bench.py
from torch.profiler import profile, schedule, ProfilerActivity
RESULT_DIR = "./prof_trace"
my_schedule = schedule(
    skip_first=10,
    wait=5,
    warmup=5,
    active=1,
    repeat=5)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
    # print(output)
    p.export_chrome_trace(f"{RESULT_DIR}/{p.step_num}.json")

for _ in range(10):
    model(**input_dict)  # compiled_model(**input_dict) to get inductor model profiling

total = 0
with profile(
    activities=[ProfilerActivity.CPU],
    schedule=my_schedule,
    on_trace_ready=trace_handler
) as p:
    for _ in range(50):
        model(**input_dict)  # compiled_model(**input_dict) to get inductor model profiling
        p.step()

######################################################################
# We get the following performance profiling table for the eager-mode model (omitting some columns):
#
# .. code-block:: shell
#
#     -------------------------  ------------  ------------  ------------
#                          Name   CPU total %     CPU total    # of Calls
#     -------------------------  ------------  ------------  ------------
#                   aten::addmm        45.73%     370.814ms           362
#                     aten::add        19.89%     161.276ms           363
#                   aten::copy_        14.97%     121.416ms           488
#                     aten::mul         9.02%      73.154ms           194
#               aten::clamp_min         8.81%      71.444ms            96
#                     aten::bmm         5.46%      44.258ms            48
#                 ProfilerStep*       100.00%     810.920ms             1
#                     aten::div         2.89%      23.447ms            24
#                aten::_softmax         1.00%       8.087ms            24
#                  aten::linear        46.48%     376.888ms           362
#                   aten::clone         2.77%      22.430ms            98
#                       aten::t         0.31%       2.502ms           362
#                    aten::view         0.14%       1.161ms           850
#               aten::transpose         0.17%       1.377ms           386
#            aten::index_select         0.12%     952.000us             3
#                  aten::expand         0.12%     986.000us           458
#                  aten::matmul         8.31%      67.420ms            48
#                     aten::cat         0.09%     703.000us             1
#              aten::as_strided         0.08%     656.000us           963
#                    aten::relu         8.86%      71.864ms            96
#     -------------------------  ------------  ------------  ------------
#     Self CPU time total: 810.920ms
#

######################################################################
#
# Similarly, we also get the table for the compiled model with Inductor (omitting some columns):
#
# .. code-block:: shell
#
#     -----------------------------------------------  ------------  ------------  ------------
#                                                Name   CPU total %     CPU total    # of Calls
#     -----------------------------------------------  ------------  ------------  ------------
#                                    mkl::_mkl_linear        68.79%     231.573ms           362
#                                           aten::bmm         8.02%      26.992ms            48
#                                       ProfilerStep*       100.00%     336.642ms             1
#       graph_0_cpp_fused_constant_pad_nd_embedding_0         0.27%     915.000us             1
#                                         aten::empty         0.27%     911.000us           362
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_151         0.27%     901.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_226         0.27%     899.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_361         0.27%     898.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_121         0.27%     895.000us             1
#       graph_0_cpp_fused__mkl_linear_add_mul_relu_31         0.27%     893.000us             1
#       graph_0_cpp_fused__mkl_linear_add_mul_relu_76         0.26%     892.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_256         0.26%     892.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_346         0.26%     892.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_241         0.26%     891.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_316         0.26%     891.000us             1
#       graph_0_cpp_fused__mkl_linear_add_mul_relu_91         0.26%     890.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_106         0.26%     890.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_211         0.26%     890.000us             1
#       graph_0_cpp_fused__mkl_linear_add_mul_relu_61         0.26%     889.000us             1
#      graph_0_cpp_fused__mkl_linear_add_mul_relu_286         0.26%     889.000us             1
#     -----------------------------------------------  ------------  ------------  ------------
#     Self CPU time total: 336.642ms 
#
# From the profiling table of the eager model, we can see the most time consumption ops are [``aten::addmm``, ``aten::add``, ``aten::copy_``, ``aten::mul``, ``aten::clamp_min``, ``aten::bmm``].
# Comparing with the inductor model profiling table, we notice an ``mkl::_mkl_linear`` entry and multiple fused kernels in the form ``graph_0_cpp_fused_*``. They are the major
# optimizations that the inductor model is doing. Let us discuss them separately.
#
# (1) Regarding ``mkl::_mkl_linear``: You may notice the number of calls to this kernel is 362, which is exactly the same as ``aten::linear`` in the eager model profiling table.
# The CPU total of ``aten::linear`` is 376.888ms, while it is 231.573ms for ``mkl::_mkl_linear``. This suggests a ~1.63x for the "linear" part.
# The speedup mainly comes from `packing the weight tensor to block memory format <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/cblas-gemm-pack-002.html>`_
# and invoking `cblas_sgemm_compute <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/cblas-gemm-compute-002.html>`_ within the Inductor CPU backend
# to have a better cache behavior during GEMM computation.
#
# (2) Regarding other memory-intensive ops: The end-to-end latency for the eager/inductor model is 802/339ms in our testing. So we can roughly infer that the speed up for the other memory-intensive ops is around 3.94x.
# Let's read the generated code to understand how the inductor achieves this impressive optimization. You can find the generated code by 
# searching ``cpp_fused__mkl_linear_add_mul_relu_151`` in ``output_code.py``
# 


cpp_fused__mkl_linear_add_mul_relu_151 = async_compile.cpp('''
#include <ATen/record_function.h>
#include "/tmp/torchinductor_root/lr/clrlgu27q4ggd472umdzwsu6qcpqxcuusjxqvx2hwitjbujiiz7z.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    RECORD_FUNCTION("graph_0_cpp_fused__mkl_linear_add_mul_relu_151", c10::ArrayRef<c10::IValue>({}));
    #pragma omp parallel num_threads(32)
    {
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(16384L); i0+=static_cast<long>(1L))
            {
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (512L*i0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(i1 + (512L*i0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr0 + static_cast<long>(i1 + (512L*i0)));
                }
            }
        }
    }
}''')

######################################################################
# From the generated code above, we can see this kernel has done a typical `Loop Fusion <https://en.wikipedia.org/wiki/Loop_fission_and_fusion>`_ on ``[add, add, mul, add]``.
# This is a memory-bound bottle neck preventing good performance. To get a more intuitive feeling about this optimization, 
# we can infer the sizes and stride of the inputs and further benchmark this ``[add, add, mul, add]`` pattern.

# bench.py
def func(arg_0, arg_1, arg_2, arg_3, arg_4):
    add_0 = arg_0 + arg_1
    add_1 = add_0 + arg_2
    mul_1 = add_1 * arg_3
    add_2 = mul_1 + arg_4
    arg_2 = add_2
    return arg_2

arg_0 = torch.rand(16384, 512)
arg_1 = torch.rand(1, 512)
arg_2 = torch.zeros(16384, 512)
arg_3 = torch.rand(1, 512)
arg_4 = torch.rand(1, 512)

input = (arg_0, arg_1, arg_2, arg_3, arg_4)
inductor_func = torch.compile(func)
with torch.no_grad():
    inductor_func(*input)

import timeit
NUM_ITERS=100
with torch.no_grad():
    # warmup
    for _ in range(10):
        func(*input)
    eager_t = timeit.timeit("func(*input)", number=NUM_ITERS, globals=globals())

with torch.no_grad():
    # warmup
    for _ in range(10):
        inductor_func(*input)
    inductor_t = timeit.timeit("inductor_func(*input)", number=NUM_ITERS, globals=globals())
# print(f"eager use: {eager_t * 1000 / NUM_ITERS} ms/iter")
# print(f"inductor use: {inductor_t * 1000 / NUM_ITERS} ms/iter")
# print(f"speed up ratio: {eager_t / inductor_t}")

######################################################################
# Output:
#
# .. code-block:: shell
#
#     eager use: 5.780875144992024 ms/iter
#     inductor use: 0.9588955780491233 ms/iter
#     speed up ratio: 6.0286805751604735
#
#
# This is just an example. The profiling table shows all element-wise op are fused within the inductor automatically in this model. You can read more kernels in
# `output_code.py`


#########################################################################
# Conclusion
# ----------
#
# The document gives an in-depth tutorial for the Inductor CPU backend.
#
# With motivating examples, we walk through the process of debugging and profiling.
# The main idea is to narrow down the problem.
#
# We demonstrate step by step the way to delve deeper the issue and find the root cause of failures, with the help of debugging logging and the tool Minifier.
# Firstly determine which component the failure occurs in and then try to generate the smallest snippet of code that can reproduce the failure.
#
# When the performance with Inductor is better than that of eager mode, we provide a solid analytical method for performance profiling.
# We show how to find the time-consuming hotspot with PyTorch Profiler and figure out the operator-level or kernel-level reason to explain the phenomenon.
