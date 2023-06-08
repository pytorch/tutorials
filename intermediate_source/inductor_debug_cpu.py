# -*- coding: utf-8 -*-

"""
Inductor CPU backend debugging and profiling
============================================

**Author**: `Liao Xuan <https://github.com/Valentine233>`_, `Zhu Haozhe <https://github.com/zhuhaozhe>`_
"""

#########################################################################
# Overview
# --------
#
# This document is intended to introduce the usage, debugging and performance profiling for ``torch.compile`` with Inductor CPU backend.
#     1. For the usage, we will show how to print debugging loggings and how to inductor an in-depth analysis with config parameters.
#     2. For debugging, we will demonstrate the process to debug a functional failure. There are usually two types of functional failure.
#        One is the error occurring during running. It prevents a model from giving the final result, such as compilation error and runtime error.
#        The other is the accuracy problem. The model gives a final result, but the value is wrong. We usually compare the result of inductor with that of eager.
#        The main idea of debugging is to narrow down the problem. We firstly determine that the failure occurs in inductor and then try to find the minimum code snippet with failure.
#     3. For the profiling, we will show what to do when the performance is not good.
#        This tutorial will walk you through the process of profiling, including how to find the time-consuming hotpot and determine the root cause. There are two typical scenarios for performance profiling.
#        One is the case where the execution time with inductor is longer than that of eager. The other is the model regression between two PyTorch versions where both FX graph and output code could change.
#     4. In the final part, we will propose several debugging tools to be implemented and upstreamt in the future.
#
# Here is a simple example to run the ``torch.compile`` with Inductor.

import torch

def fn(x):
    return torch.neg(x)

x = torch.randn((2, 4, 28))
compiled_fn = torch.compile(fn) # backend=inductor as default
result = compiled_fn(x)

#########################################################################
# Get more loggings
# ^^^^^^^^^^^^^^^^^
#
# The simple example above would not give any debugging info. If you'd want to get more useful logging, you can add a ``TORCH_COMPILE_DEBUG`` environment variable:
#
# .. code:: shell
#
# 	TORCH_COMPILE_DEBUG=1 python xx.py
#
# The time taken in each step is shown. This also does the graph visualization and prints the output code. In logging, a temporary debug tracing directory like this can be found.
#
# .. code:: shell
#
# 	torch._inductor.debug: [WARNING] model___20 debug trace: /tmp/torchinductor_root/rx/crxfi2ybd7yp5sbj2pnhw33wfhtdw7wumvrobyp5sjvdui5ktjc2.debug
#
# In this directory, the following files are saved for debugging purposes.
#
# +-------------------------+----------------------------------------------------------+
# | File                    | Description                                              |
# +-------------------------+----------------------------------------------------------+
# | fx_graph_runnable.py    | Executable FX graph, post decomps, pre pattern match     |
# +-------------------------+----------------------------------------------------------+
# | fx_graph_transformed.py | Transformed FX graph, post pattern match                 |
# +-------------------------+----------------------------------------------------------+
# | ir_post_fusion.txt      | Inductor IR before fusion                                |
# +-------------------------+----------------------------------------------------------+
# | ir_pre_fusion.txt       | Inductor IR after fusion                                 |
# +-------------------------+----------------------------------------------------------+
# | output_code.py          | Generated Python code for graph, with cpp/triton kernels |
# +-------------------------+----------------------------------------------------------+
#
# ``fx_graph_runnable.py`` and ``output_code.py`` are both runnable and editable in order to make debugging easier.
#
# Here is another way to print logging for Inductor:
#
# .. code:: shell
#
# 	TORCH_LOGS="+inductor,output_code,schedule" python xx.py
#
# +--------------+-------------------------------------------------------------+
# | Parameter    | Description                                                 |
# +--------------+-------------------------------------------------------------+
# | +inductor    | Set the logging level of Inductor to DEBUG, default is INFO |
# +--------------+-------------------------------------------------------------+
# | output_code  | Print output code with cpp/triton kernels                   |
# +--------------+-------------------------------------------------------------+
# | schedule     | Print reasons for not doing vectorization in cpp kernels    |
# +--------------+-------------------------------------------------------------+
#
# Conducting an in-depth analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Moreover, there are several config parameters helping the analysis.
#
# +--------------------------------------------------+---------------------------------------------------------------------+
# | Parameter                                        | Description                                                         |
# +--------------------------------------------------+---------------------------------------------------------------------+
# | torch._inductor.config.max_fusion_size           | Set the maximum number of nodes allowed in one fusion               |
# +--------------------------------------------------+---------------------------------------------------------------------+
# | torch._inductor.config.cpp.simdlen               | Specify the bit width for cpp vectorization                         |
# +--------------------------------------------------+---------------------------------------------------------------------+
# | torch._inductor.config.cpp.min_chunk_size        | Set the minimum number of workloads one thread should at least take |
# +--------------------------------------------------+---------------------------------------------------------------------+
# | torch._inductor.config.cpp.enable_kernel_profile | Allow cpp kernel performance profiling via profiler                 |
# +--------------------------------------------------+---------------------------------------------------------------------+


######################################################################
# Debugging
# ---------
#
# Determine component of error
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When encountering errors or an accuracy problem, a straightforward solution to find the bug is to narrow down the problem. The first thing to do is to determine the component where the error occurs. Luckily, it can be simply achieved by changing the backend of ``torch.compile``.
#
# +----------------------------------------+-----------------------------------------+
# | Code                                   | Description                             |
# +----------------------------------------+-----------------------------------------+
# | torch.compile(fn, backend="eager")     | Enable Dynamo                           |
# +----------------------------------------+-----------------------------------------+
# | torch.compile(fn, backend="aot_eager") | Enable Dynamo + AOT autograd            |
# +----------------------------------------+-----------------------------------------+
# | torch.compile(fn, backend="inductor")  | Enable Dynamo + AOT autograd + Inductor |
# +----------------------------------------+-----------------------------------------+
#
# If the model can successfully run when the backend is set to ``eager`` or ``aot_eager`` while it fails with ``inductor``, we can narrow down the failure to Inductor.
#
#
# Example
# ^^^^^^^
#
# Here is an example for the subsequent debugging:

import torch
from torch._dynamo.utils import same

def foo(x1, x2):
    a = torch.neg(x1)
    b = torch.maximum(x2, a)
    y = torch.cat([b], dim=0)
    return y

x1 = torch.randint(256, (1,), dtype=torch.uint8)
x2 = torch.randint(256, (8390,), dtype=torch.uint8)

expected_result = fn(x1, x2)

compiled_fn = torch.compile(fn)
actual_result = compiled_fn(x1, x2)

assert same(expected_result, actual_result) == True

######################################################################
# The implementation of ``neg`` in the ``cpp`` codegen is as follows:

def neg(x):
    return f"decltype({x})(-{x})"

######################################################################
# In order to demonstrate the debugging, we will modify the function to a wrong one later.
#
#
# Errors debugging
# ^^^^^^^^^^^^^^^^
#
# If a compile error occurs, the root cause is usually shown in the traceback log.
#
# For example, the ``neg`` function is modified like this:

def neg(x):
    return f"-{x}"


######################################################################
# The logging gives the following compile error with a rather clear reason. In this case, the root cause is that data types of maximum's inputs are inconsistent.
#
# .. code:: shell
#
# 	…
# 	torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
# 	CppCompileError: C++ compile error
# 	…
# 	/tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp: In function ‘void kernel(const unsigned char*, const unsigned char*, unsigned char*)’:
# 	/tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp:14:53: error: no matching function for call to ‘max_propagate_nan(unsigned char&, int&)’
# 	   14 |             auto tmp3 = max_propagate_nan(tmp0, tmp2);
# 	      |                                                     ^
# 	In file included from /tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp:2:
# 	/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note: candidate: ‘template<class scalar_t> scalar_t max_propagate_nan(scalar_t, scalar_t)’
# 	   27 | inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
# 	      |                 ^~~~~~~~~~~~~~~~~
# 	/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note:   template argument deduction/substitution failed:
# 	/tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp:14:53: note:   deduced conflicting types for parameter ‘scalar_t’ (‘unsigned char’ and ‘int’)
# 	   14 |             auto tmp3 = max_propagate_nan(tmp0, tmp2);
# 	      |                                                     ^
#
#
# Otherwise, if the model runs with other errors, we can do the model code reduction until finding the minimum code snippet with failure. Thus, the target operators and kernels are located.
#
#
# Accuracy debugging
# ^^^^^^^^^^^^^^^^^^^
#
# The accuracy problem refers the case where outputs of backends eager and inductor are different. As FX graph is generated before Inductor and output code is generated after Inductor, we can narrow down the problem by comparing their outputs.
#
# If a model has several graphs, the first step is to compare the final outputs of FX graph and output code for each graph, given the same input. The target is to find the first graph occurring error or with different outputs. Binary search is suggested to use for efficiency.
#
# When a model has only one graph or the problematic graph has been found with the above step, compare the intermediate outputs of FX graph and output code in each graph, given the same input. The idea is to continuously narrow down the problem.
#
# For example, we modify the ``neg`` function like this:

def neg(x):
    return f"decltype({x})(2 * {x})"


######################################################################
# An accuracy problem would be raised as follows.
#
# .. code:: shell
#
# 	torch._dynamo.utils: [ERROR] Accuracy failed: allclose not within tol=0.0001
# 	Traceback (most recent call last):
# 	  File "test_script.py", line 18, in <module>
# 	    assert same(expected_result, actual_result) == True
# 	AssertionError
#
#
# By comparing the intermediate outputs of FX graph and output code, it would be found that outputs are already different after doing ``torch.neg``.
#
# Here are the modifications of FX graph and output code:
#
# **Changes of teh FX graph:**

# Before
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0_1, arg1_1):
        neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
        maximum = torch.ops.aten.maximum.default(arg1_1, neg);  arg1_1 = neg = None
        clone = torch.ops.aten.clone.default(maximum);  maximum = None
        return (clone,)

# After
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, arg0_1, arg1_1):
        neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
        return (neg,)

######################################################################
# **Changes of the output code:**

# Before
cpp_fused_cat_maximum_neg_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
extern "C" void kernel(const long* in_ptr0,
                        const long* in_ptr1,
                        long* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(8390L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(i0)];
            auto tmp1 = in_ptr1[static_cast<long>(0L)];
            auto tmp2 = decltype(tmp1)(2 * tmp1);
            auto tmp3 = max_propagate_nan(tmp0, tmp2);
            out_ptr0[static_cast<long>(i0)] = tmp3;
        }
    }
}
''')

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf0 = empty_strided((8390, ), (1, ), device='cpu', dtype=torch.int64)
    cpp_fused_cat_maximum_neg_0(c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    del arg1_1
    return (buf0, )

# After
cpp_fused_cat_maximum_neg_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
extern "C" void kernel(const long* in_ptr0,
                        const long* in_ptr1,
                        long* out_ptr0)
{
    {
        auto tmp1 = in_ptr1[static_cast<long>(0L)];
        auto tmp2 = decltype(tmp1)(2 * tmp1);
        out_ptr0[static_cast<long>(0L)] = tmp2;
    }
}
''')

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf0 = empty_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    cpp_fused_cat_maximum_neg_0(c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    del arg1_1
    return (buf0, )

######################################################################
# You can use the PyTorch debugging tool called `Minifier <https://pytorch.org/docs/stable/dynamo/troubleshooting.html>`_. It helps to automatically generate a minified problematic graph.


######################################################################
# Performance profiling
# ---------------------
#
# For this part, we will describe how to analyze the inductor model performance.
# First, we choose an eager model as a baseline. We set up a benchmark to compare the end-to-end performance between the eager model and the inductor model.

from transformers import MobileBertForQuestionAnswering
import torch
# init an eager model
model = MobileBertForQuestionAnswering.from_pretrained("csarron/mobilebert-uncased-squad-v2")
seq_length = 128
bs = 128
vocab_size = model.config.vocab_size
input = torch.randint(0, vocab_size, (bs, seq_length), dtype=torch.int64)
input_dict = {"input_ids": input}

# init inductor model
inductor_model = torch.compile(model)
with torch.no_grad():
    inductor_model(**input_dict)

NUM_ITERS=100
import timeit
with torch.no_grad():
    # warmup
    for _ in range(10):
        model(**input_dict)
    eager_t = timeit.timeit("model(**input_dict)", number=NUM_ITERS, globals=globals())

with torch.no_grad():
    # warmup
    for _ in range(10):
        inductor_model(**input_dict)
    inductor_t = timeit.timeit("inductor_model(**input_dict)", number=NUM_ITERS, globals=globals())
print(f"eager use: {eager_t * 1000 / NUM_ITERS} ms/iter")
print(f"inductor use: {inductor_t * 1000 / NUM_ITERS} ms/iter")
print(f"speed up ratio: {eager_t / inductor_t}")


######################################################################
# Output:
#
# .. code-block:: shell
#
#     eager use: 802.1023553796113 ms/iter
#     inductor use: 339.95180135127157 ms/iter
#     speed up ratio: 2.359459053287382
#
# The inductor model speed-up is 2.58x.
#
#
# Secondly, we can deep dive into op-level performance to understand where is the speed-up comes from.
# `Pytorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_ is a good tool to help us.
# To enable kernel profile with inductor model, we need to set ``enable_kernel_profile`` by:

from torch._inductor import config
config.cpp.enable_kernel_profile = True

######################################################################
# Following the steps in `Pytorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_
# We are able to get the profiling table and trace files.

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
    print(output)
    p.export_chrome_trace(f"{RESULT_DIR}/{p.step_num}.json")

for _ in range(10):
    model(**input_dict)  # inductor_model(**input_dict) to get inductor model profiling

total = 0
with profile(
    activities=[ProfilerActivity.CPU],
    schedule=my_schedule,
    on_trace_ready=trace_handler
) as p:
    for _ in range(100):
        model(**input_dict)  # inductor_model(**input_dict) to get inductor model profiling
        p.step()

######################################################################
# We will get the following profile table for the eager model:
#
# .. image:: ../_static/img/eager_prof.png
#
# Similarly, get the table for the inductor model:
#
# .. image:: ../_static/img/inductor_prof.png
#
# From the profiling table of the eager model, we can see the most time consumption ops are [aten::addmm, aten::add, aten::copy_, aten::mul, aten::clamp_min, aten::bmm].
# Comparing with the inductor model profiling table, we notice there are ``mkl::_mkl_linear`` and fused kernel called ``graph_0_cpp_fused_*``. They are the major
# optimization that the inductor model is doing. Let us discuss them separately.
#
# (1) Regard to ``mkl::_mkl_linear``: You may notice the number of calls to this kernel is 362, which is exactly the same as ``aten::linear`` in the eager model profiling table.
# The CPU total of ``aten::linear`` is 376.888ms, at the mean time it is 231.573ms for ``mkl::_mkl_linear``. This suggests inductor model speed up ~1.63x for the "linear" part.
#
# (2) Regarding non-linear part: The end-to-end latency for the eager/inductor model is 802/339ms. The speed up for the non-linear part is ~3.94x.
# Let's read the generated code to understand how the inductor achieves this impressive optimization. You are able to find the generated code by 
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
}
''')

######################################################################
# From the generated code above, we can see this kernel has done a typical `Loop Fusion <https://en.wikipedia.org/wiki/Loop_fission_and_fusion>`_ on [add, add, mul, add].
# We can infer the sizes and stride of the inputs and further bench this [add, add, mul, add] pattern.

import torch
def func(x0, x1, x3, x5, x7):
    x2 = x0 + x1
    x4 = x2 + x3
    x6 = x4 * x5
    x8 = x6 + x7
    x3 = x8
    return x3

x0 = torch.rand(16384, 512)
x1 = torch.rand(1, 512)
x3 = torch.zeros(16384, 512)
x5 = torch.rand(1, 512)
x7 = torch.rand(1, 512)

input = (x0, x1, x3, x5, x7)
inductor_func = torch.compile(func)
with torch.no_grad():
    inductor_func(*input)

import timeit
NUM_ITERS=1000
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
print(f"eager use: {eager_t * 1000 / NUM_ITERS} ms/iter")
print(f"inductor use: {inductor_t * 1000 / NUM_ITERS} ms/iter")
print(f"speed up ratio: {eager_t / inductor_t}")
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


######################################################################
# Future work
# -----------
#
# Implement and upstream the debug tools
# 	1. **Graph merger**: Merge graphs of a model into a single large graph. Thus, graphs can be compared quickly between different versions of PyTorch. `#102958 <https://github.com/pytorch/pytorch/pull/102958>`_
# 	2. **Graph matching**: In order to know what each kernel does, this tool matches C++ kernel with FX graph operators and adds corresponding operators before each kernel in the ``.cpp`` output code. `#102958 <https://github.com/pytorch/pytorch/pull/102958>`_
# 	3. **Save inputs and outputs**: For the purpose of reproducing rapidly the failure of a large model, it is necessary to add serializations for the inputs and outputs among graphs and intermediate outputs in graphs.
# 	4. **Test case generation**: When a user has found the operators which are inefficient with cpp kernels, a tool is needed to automatically write a test case. Specifically, one test case can be generated for each kernel, with the corresponding small FX graph and input.
# 	5. **Minifier optimization**: Keep refining Minifier and make it adapted for more scenarios.
