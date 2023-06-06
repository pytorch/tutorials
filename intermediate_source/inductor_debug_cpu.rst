Inductor CPU backend debugging and profiling
==============================================

**Author**: `Liao Xuan <https://github.com/Valentine233>`_, `Zhu Haozhe <https://github.com/zhuhaozhe>`_

Usage
--------------

Start with an example
^^^^^^^^^^^^^^^^^^^

Here is a simple example to run the ``torch.compile`` with Inductor.

.. code-block:: python

	import torch
	
	def fn(x):
	    return torch.neg(x)
	
	x = torch.randn((2, 4, 28))
	compiled_fn = torch.compile(fn) # backend=inductor as default
	result = compiled_fn(x)

Get more loggings
^^^^^^^^^^^^^^^^^^^

However, the above code would not give any debugging info. If we want to get more useful logging, one way is to add an environment variable.

.. code:: shell

	TORCH_COMPILE_DEBUG=1 python xx.py

The time taken in each step is shown. This also does the graph visualization and prints the output code. In logging, a temperate debug tracing directory like this can be found.

.. code:: shell

	torch._inductor.debug: [WARNING] model___20 debug trace: /tmp/torchinductor_root/rx/crxfi2ybd7yp5sbj2pnhw33wfhtdw7wumvrobyp5sjvdui5ktjc2.debug

The directory saves several files for debugging.

+-------------------------+----------------------------------------------------------+
| fx_graph_readable.py    | Readable FX graph, post decomps                          |
+-------------------------+----------------------------------------------------------+
| fx_graph_runnable.py    | Executable FX graph, post decomps, pre pattern match     |
+-------------------------+----------------------------------------------------------+
| fx_graph_transformed.py | Transformed FX graph, post pattern match                 |
+-------------------------+----------------------------------------------------------+
| ir_post_fusion.txt      | Inductor IR before fusion                                |
+-------------------------+----------------------------------------------------------+
| ir_pre_fusion.txt       | Inductor IR after fusion                                 |
+-------------------------+----------------------------------------------------------+
| output_code.py          | Generated Python code for graph, with cpp/triton kernels |
+-------------------------+----------------------------------------------------------+


``fx_graph_runnable.py`` and ``output_code.py`` are both runnable and editable in order to make debugging easier.


Here is another way to print logging for Inductor.

.. code:: shell

	TORCH_LOGS="+inductor,output_code,schedule" python xx.py

+--------------+-------------------------------------------------------------+
| +inductor    | Set the logging level of Inductor to DEBUG, default is INFO |
+--------------+-------------------------------------------------------------+
| +output_code | Print output code with cpp/triton kernels                   |
+--------------+-------------------------------------------------------------+
| +schedule    | Print reasons for not doing vectorization in cpp kernels    |
+--------------+-------------------------------------------------------------+

Configs to do deeper analysis
^^^^^^^^^^^^^^^^^^^

Moreover, there are several config parameters helping the analysis.

+--------------------------------------------------+---------------------------------------------------------------------+
| torch._inductor.config.max_fusion_size           | Set the maximum number of nodes allowed in one fusion               |
+--------------------------------------------------+---------------------------------------------------------------------+
| torch._inductor.config.cpp.simdlen               | Specify the bit width for cpp vectorization                         |
+--------------------------------------------------+---------------------------------------------------------------------+
| torch._inductor.config.cpp.min_chunk_size        | Set the minimum number of workloads one thread should at least take |
+--------------------------------------------------+---------------------------------------------------------------------+
| torch._inductor.config.cpp.enable_kernel_profile | Allow cpp kernel performance profiling via profiler                 |
+--------------------------------------------------+---------------------------------------------------------------------+


Debugging
--------------

Determine component of error
^^^^^^^^^^^^^^^^^^^

When encountering errors or accuracy problem, a straightforward solution to find the bug is to narrow down the problem. The first thing to do is to determine the component where error occurs. Luckily, it can be simply achieved by changing the backend of ``torch.compile``.

+----------------------------------------+-----------------------------------------+
| torch.compile(fn, backend="eager")     | Enable Dynamo                           |
+----------------------------------------+-----------------------------------------+
| torch.compile(fn, backend="aot_eager") | Enable Dynamo + AOT autograd            |
+----------------------------------------+-----------------------------------------+
| torch.compile(fn, backend="inductor")  | Enable Dynamo + AOT autograd + Inductor |
+----------------------------------------+-----------------------------------------+

If the model can successfully run when backend is eager or aot_eager while it fails with inductor, we can narrow down the failure to Inductor.


Example
^^^^^^^^^^^^^^^^^^^

Here is an example for the subsequent debugging.

.. code-block:: python

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


The implementation of ``neg`` in cpp codegen is as follows.

.. code-block:: python

	def neg(x):
	        return f"decltype({x})(-{x})"


In order to demonstrate the debugging, we will modify the function to a wrong one later.

Errors debugging
^^^^^^^^^^^^^^^^^^^

If it occurs a compile error, the root cause is usually shown in traceback log.

For example, the ``neg`` function is modified like this.

.. code-block:: python

	def neg(x):
	        return f"-{x}"


The logging gives the following compile error with a rather clear reason. In this case, the root cause is that data types of maximum's inputs are inconsistent.

.. code:: shell

	…
	torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
	CppCompileError: C++ compile error
	…
	/tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp: In function ‘void kernel(const unsigned char*, const unsigned char*, unsigned char*)’:
	/tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp:14:53: error: no matching function for call to ‘max_propagate_nan(unsigned char&, int&)’
	   14 |             auto tmp3 = max_propagate_nan(tmp0, tmp2);
	      |                                                     ^
	In file included from /tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp:2:
	/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note: candidate: ‘template<class scalar_t> scalar_t max_propagate_nan(scalar_t, scalar_t)’
	   27 | inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
	      |                 ^~~~~~~~~~~~~~~~~
	/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note:   template argument deduction/substitution failed:
	/tmp/torchinductor_root/2x/c2xgxsooklulr4u54etfnnha7dsu6xzbwdscttvs7dkpba3uwkem.cpp:14:53: note:   deduced conflicting types for parameter ‘scalar_t’ (‘unsigned char’ and ‘int’)
	   14 |             auto tmp3 = max_propagate_nan(tmp0, tmp2);
	      |                                                     ^


Otherwise, if the model runs with other errors, we can do the model code reduction until finding the minimum code snippet with failure. Thus, the target operators and kernels are located.


Accuracy debugging
^^^^^^^^^^^^^^^^^^^

The accuracy problem refers the case where outputs of backends eager and inductor are different. As FX graph is generated before Inductor and output code is generated after Inductor, we can narrow down the problem by comparing their outputs.

If a model has several graphs, the first step is to compare the final outputs of FX graph and output code for each graph, given the same input. The target is to find the first graph occurring error or with different outputs. Binary search is suggested to use for efficiency.

When a model has only one graph or the problematic graph has been found with the above step, compare the intermediate outputs of FX graph and output code in each graph, given the same input. The idea is to continuously narrow down the problem.

For example, we modify the ``neg`` function like this.

.. code-block:: python

	def neg(x):
	        return f"decltype({x})(2 * {x})"


An accuracy problem would be raised as follows.

.. code:: shell

	torch._dynamo.utils: [ERROR] Accuracy failed: allclose not within tol=0.0001
	Traceback (most recent call last):
	  File "test_script.py", line 18, in <module>
	    assert same(expected_result, actual_result) == True
	AssertionError


By comparing the intermediate outputs of FX graph and output code, it would be found that outputs are already different after doing ``torch.neg``.

Specifically, the modifications of FX graph and output code are attached.

*Change of FX graph*

.. code-block:: python

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


*Change of output code*

.. code-block:: python

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


Note that there exists a debugging tool provided by PyTorch, called `Minifier <https://pytorch.org/docs/stable/dynamo/troubleshooting.html>`_. It helps us automatically generate a minified problematic graph.


Performance profiling
--------------

For this part, we are going to describe how to analyze torchinductor model performance.
Firsly, we choose an eager model as a baseline. We set up a benchmark to compare
the end to end performance between eager model and inductor model.

.. code-block:: python

    from transformers import T5ForConditionalGeneration
    # init an eager model
    eager_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    seq_length = 1024
    bs = 4
    vocab_size = model.config.vocab_size
    input = torch.randint(0, vocab_size, (bs, seq_length), dtype=torch.int64)
    input_dict = {"input_ids": input}
    input_dict["decoder_input_ids"] = input
    # init inductor model
    inductor_model = torch.compile(model)
    compiled(**input_dict)
    eager_t = 0
    inductor_t = 0
    for _ in range(100):
        model(**input_dict)
    for _ in range(1000):
        eager_start = time.time()
        model(**input_dict)
        eager_end = time.time()
        eager_t += eager_end - eager_start

    for _ in range(100):
        model(**input_dict)
    for _ in range(1000):
        inductor_start = time.time()
        compiled(**input_dict)
        inductor_end = time.time()
        inductor_t += inductor_end - inductor_start

    print(model.__class__)
    print("eager use:", eager_t)
    print("inductor use:", inductor_t)
    print("ratio:", eager_t / inductor_t)
        
Output:

.. code-block:: shell

    eager use: 410.12550354003906
    inductor use: 478.59081745147705
    ratio: 0.8569439458198976

We see inductor model spent more time than eager model, which does not meet our expectation.
To deep dive op-level performance, we can use `Pytorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_

To enable kernel profile in inductor, we need set ``enable_kernel_profile`` by:

.. code-block:: python

    from torch._inductor import config
    config.cpp.enable_kernel_profile = True

Following the steps in `Pytorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_
we are able to get the profiling table and trace files.

.. code-block:: python

    from torch.profiler import profile, schedule, ProfilerActivity
    my_schedule = schedule(
        skip_first=10,
        wait=5,
        warmup=5,
        active=1,
        repeat=5)

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
        print(output)
        p.export_chrome_trace(RESULT_DIR + "/" + str(p.step_num) + ".json")

    for _ in range(nwarmup):
        model(**input_dict)

    total = 0
    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=my_schedule,
        on_trace_ready=trace_handler
    ) as p:
        for _ in range(100):
            begin = time.time()
            model(**input_dict)
            end=time.time()
            total += (end - begin)
            p.step()
    print("latency: {} ms".format(1000*(total)/100))

We will get following profile tables for eager model

.. code-block:: shell

    -----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                aten::mm        33.33%     138.616ms        33.33%     138.616ms       1.429ms            97  
                aten::add_        19.38%      80.596ms        19.38%      80.596ms       4.242ms            19  
                aten::bmm        18.78%      78.104ms        18.78%      78.104ms       2.170ms            36  
            aten::_softmax        11.32%      47.082ms        11.32%      47.082ms       2.616ms            18  
                aten::copy_         3.89%      16.190ms         3.89%      16.190ms     103.121us           157  
            ProfilerStep*         3.53%      14.702ms       100.00%     415.949ms     415.949ms             1  
                aten::add         2.37%       9.849ms         2.39%       9.958ms     144.319us            69  
                aten::mul         1.13%       4.693ms         1.14%       4.726ms      65.639us            72  
            aten::clamp_min         0.85%       3.541ms         0.85%       3.541ms     295.083us            12  
        aten::index_select         0.84%       3.480ms         1.06%       4.401ms       1.100ms             4  
            aten::linear         0.63%       2.637ms        33.95%     141.194ms       1.456ms            97  
                aten::pow         0.61%       2.520ms         0.61%       2.554ms      79.812us            32  
            aten::matmul         0.50%       2.067ms        56.53%     235.132ms       1.768ms           133  
            aten::select         0.22%     900.000us         0.22%     910.000us     113.750us             8  
                aten::log         0.18%     740.000us         0.18%     740.000us     370.000us             2  
        aten::_unsafe_view         0.17%     718.000us         0.17%     718.000us       3.840us           187  
                aten::sum         0.17%     715.000us         0.20%     831.000us      25.969us            32  
            aten::transpose         0.15%     642.000us         0.18%     741.000us       3.963us           187  
            aten::reshape         0.15%     622.000us         3.66%      15.241ms      88.098us           173  
                aten::fill_         0.15%     613.000us         0.15%     613.000us      15.718us            39  
    -----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 415.949ms

And get above table for inductor model

.. code-block:: shell

    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        mkl::_mkl_linear        28.24%     133.979ms        28.39%     134.689ms       1.389ms            97  
                                                aten::bmm        15.65%      74.250ms        15.65%      74.251ms       2.063ms            36  
                            graph_0_cpp_fused__softmax_7         4.24%      20.123ms         4.24%      20.123ms      20.123ms             1  
                            graph_0_cpp_fused__softmax_42         4.17%      19.773ms         4.17%      19.773ms      19.773ms             1  
                            graph_0_cpp_fused__softmax_35         4.16%      19.751ms         4.16%      19.751ms      19.751ms             1  
                            graph_0_cpp_fused__softmax_21         4.15%      19.674ms         4.15%      19.674ms      19.674ms             1  
                            graph_0_cpp_fused__softmax_14         4.14%      19.654ms         4.14%      19.654ms      19.654ms             1  
                            graph_0_cpp_fused__softmax_28         4.13%      19.576ms         4.13%      19.576ms      19.576ms             1  
                            graph_0_cpp_fused__softmax_56         2.83%      13.404ms         2.83%      13.404ms      13.404ms             1  
                            graph_0_cpp_fused__softmax_80         2.82%      13.371ms         2.82%      13.371ms      13.371ms             1  
                            graph_0_cpp_fused__softmax_68         2.81%      13.323ms         2.81%      13.323ms      13.323ms             1  
                            graph_0_cpp_fused__softmax_92         2.80%      13.297ms         2.80%      13.297ms      13.297ms             1  
                            graph_0_cpp_fused__softmax_104         2.78%      13.208ms         2.78%      13.208ms      13.208ms             1  
                            graph_0_cpp_fused__softmax_2         2.63%      12.468ms         2.63%      12.468ms      12.468ms             1  
                                            ProfilerStep*         1.61%       7.616ms       100.00%     474.360ms     474.360ms             1  
                            graph_0_cpp_fused__softmax_73         0.49%       2.320ms         0.49%       2.320ms       2.320ms             1  
                            graph_0_cpp_fused__softmax_85         0.49%       2.309ms         0.49%       2.309ms       2.309ms             1  
                            graph_0_cpp_fused__softmax_97         0.48%       2.283ms         0.48%       2.283ms       2.283ms             1  
                            graph_0_cpp_fused__softmax_61         0.48%       2.268ms         0.48%       2.268ms       2.268ms             1  
                            graph_0_cpp_fused__softmax_49         0.48%       2.255ms         0.48%       2.255ms       2.255ms             1  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 474.360ms

We can search the most time consuming ``graph_0_cpp_fused__softmax_7`` in ``output_code.py`` to see the generated code:

.. code-block:: python

    cpp_fused__softmax_7 = async_compile.cpp('''
    #include <ATen/record_function.h>
    #include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
    extern "C" void kernel(float* in_out_ptr0,
                        const float* in_ptr1,
                        float* out_ptr0,
                        float* out_ptr1)
    {
        RECORD_FUNCTION("graph_0_cpp_fused__softmax_7", c10::ArrayRef<c10::IValue>({}));
        auto in_ptr0 = in_out_ptr0;
        #pragma omp parallel num_threads(32)
        {
            {
                #pragma omp for  collapse(2)
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
                {
                    for(long i1=static_cast<long>(0L); i1<static_cast<long>(8L); i1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long i2=static_cast<long>(0L); i2<static_cast<long>(1024L); i2+=static_cast<long>(1L))
                        {
                            {
                                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                                for(long i3=static_cast<long>(0L); i3<static_cast<long>(1024L); i3+=static_cast<long>(1L))
                                {
                                    auto tmp0 = in_ptr0[static_cast<long>(i3 + (1024L*i2) + (1048576L*i1) + (8388608L*i0))];
                                    auto tmp1 = static_cast<long>(i3 + ((-1L)*i2));
                                    auto tmp2 = static_cast<long>(0);
                                    auto tmp3 = tmp1 > tmp2;
                                    auto tmp4 = static_cast<long>(tmp3);
                                    auto tmp5 = static_cast<long>(16);
                                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                                    auto tmp7 = tmp6 + tmp2;
                                    auto tmp8 = std::abs(tmp1);
                                    auto tmp9 = static_cast<long>(8);
                                    auto tmp10 = tmp8 < tmp9;
                                    auto tmp11 = static_cast<float>(tmp8);
                                    auto tmp12 = static_cast<float>(8.0);
                                    auto tmp13 = tmp11 / tmp12;
                                    auto tmp14 = std::log(tmp13);
                                    auto tmp15 = static_cast<float>(2.772588722239781);
                                    auto tmp16 = tmp14 / tmp15;
                                    auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                                    auto tmp18 = static_cast<long>(tmp17);
                                    auto tmp19 = tmp18 + tmp9;
                                    auto tmp20 = static_cast<long>(15);
                                    auto tmp21 = min_propagate_nan(tmp19, tmp20);
                                    auto tmp22 = tmp10 ? tmp8 : tmp21;
                                    auto tmp23 = tmp7 + tmp22;
                                    auto tmp24 = in_ptr1[static_cast<long>(i1 + (8L*tmp23))];
                                    auto tmp25 = static_cast<float>(0.0);
                                    auto tmp26 = tmp24 + tmp25;
                                    auto tmp27 = tmp0 + tmp26;
                                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp27);
                                }
                                out_ptr0[static_cast<long>(i2 + (1024L*i1) + (8192L*i0))] = tmp_acc0;
                            }
                        }
                    }
                }
            }
            {
                #pragma omp for  collapse(2)
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(4L); i0+=static_cast<long>(1L))
                {
                    for(long i1=static_cast<long>(0L); i1<static_cast<long>(8L); i1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long i2=static_cast<long>(0L); i2<static_cast<long>(1024L); i2+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long i3=static_cast<long>(0L); i3<static_cast<long>(1024L); i3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(i3 + (1024L*i2) + (1048576L*i1) + (8388608L*i0))];
                                auto tmp28 = out_ptr0[static_cast<long>(i2 + (1024L*i1) + (8192L*i0))];
                                auto tmp1 = static_cast<long>(i3 + ((-1L)*i2));
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 > tmp2;
                                auto tmp4 = static_cast<long>(tmp3);
                                auto tmp5 = static_cast<long>(16);
                                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                                auto tmp7 = tmp6 + tmp2;
                                auto tmp8 = std::abs(tmp1);
                                auto tmp9 = static_cast<long>(8);
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = static_cast<float>(tmp8);
                                auto tmp12 = static_cast<float>(8.0);
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = std::log(tmp13);
                                auto tmp15 = static_cast<float>(2.772588722239781);
                                auto tmp16 = tmp14 / tmp15;
                                auto tmp17 = decltype(tmp16)(tmp16 * tmp12);
                                auto tmp18 = static_cast<long>(tmp17);
                                auto tmp19 = tmp18 + tmp9;
                                auto tmp20 = static_cast<long>(15);
                                auto tmp21 = min_propagate_nan(tmp19, tmp20);
                                auto tmp22 = tmp10 ? tmp8 : tmp21;
                                auto tmp23 = tmp7 + tmp22;
                                auto tmp24 = in_ptr1[static_cast<long>(i1 + (8L*tmp23))];
                                auto tmp25 = static_cast<float>(0.0);
                                auto tmp26 = tmp24 + tmp25;
                                auto tmp27 = tmp0 + tmp26;
                                auto tmp29 = tmp27 - tmp28;
                                in_out_ptr0[static_cast<long>(i3 + (1024L*i2) + (1048576L*i1) + (8388608L*i0))] = tmp29;
                            }
                        }
                    }
                }
            }
            {
                #pragma omp for 
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(33554432L); i0+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(i0));
                    auto tmp1 = tmp0.exp();
                    tmp1.store(in_out_ptr0 + static_cast<long>(i0));
                }
            }
            {
                #pragma omp for 
                for(long i0=static_cast<long>(0L); i0<static_cast<long>(32768L); i0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                        float tmp_acc0 = 0;
                        auto tmp_acc0_vec = at::vec::Vectorized<float>(tmp_acc0);
                        for(long i1=static_cast<long>(0L); i1<static_cast<long>(1024L); i1+=static_cast<long>(16L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(i1 + (1024L*i0)));
                            tmp_acc0_vec += tmp0;
                        }
                        tmp_acc0 += at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(i0)] = tmp_acc0;
                    }
                }
            }
        }
    }
    ''')

With the kernel name ``cpp_fused__softmax_*`` and considering the profile 
results together, we may suspect the generated code for ``softmax`` is
inefficient. We encourage you to report an issue with all you findings above.


Future work
--------------

Implement and up-stream the debug tools
	1. **Cosim**: Merge graphs of a model into a single large graph. Thus, graphs can be compared quickly between different versions of PyTorch. `#102958 <https://github.com/pytorch/pytorch/pull/102958>`_
	2. **Graph matching**: In order to know what each kernel does, this tool matches cpp kernel with FX graph operators and adds corresponding operators before each kernel in cpp output code. `#102958 <https://github.com/pytorch/pytorch/pull/102958>`_
	3. **Save inputs and outputs**: For the purpose of reproducing rapidly the failure of a large model, it is necessary to add serializations for the inputs and outputs among graphs and intermediate outputs in graphs.
	4. **Test case generation**: When a user has found the operators which are inefficient with cpp kernels, a tool is needed to automatically write a test case. Specifically, one test case can be generated for each kernel, with the corresponding small FX graph and input.
	5. **Minifier optimization**: Keep refining Minifier and make it adapted for more scenarios.
