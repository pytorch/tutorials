# -*- coding: utf-8 -*-
"""
Inductor CPU backend debugging and profiling
============================================

**Authors**: `Liao Xuan <https://github.com/Valentine233>`_, `Zhu Haozhe <https://github.com/zhuhaozhe>`_, `Gong Jiong <https://github.com/jgong5>`_, `Wang Weihan <https://github.com/EikanWang>`_

Overview
--------

PyTorch 2.0 introduced the compilation API called ``torch.compile``.
This new feature offers a significant speedup over eager mode execution
through graph-level optimization powered by the default Inductor backend.

This tutorial is intended to provide an in-depth introduction on the debugging
and performance profiling on Inductor CPU backend by delving into the
intricacies of ``torch.compile``.

Meanwhile, you may also find related tutorials about ``torch.compile``
around `basic usage <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_,
comprehensive `troubleshooting <https://pytorch.org/docs/stable/dynamo/troubleshooting.html>`_
and GPU-specific knowledge like
`GPU performance profiling <https://github.com/pytorch/pytorch/blob/main/docs/source/compile/profiling_torch_compile.rst>`_.

We will start debugging with a motivating example that triggers compilation
issues and accuracy problems by demonstrating the process of debugging to
pinpoint the problems.
By enabling logging and exploring the underlying generated code,
you can learn how to narrow down the failure step by step and finally figure
out the route cause.

Following that, we will proceed to discuss how to profile the compiled code
and, through a performance comparison with eager mode,
elaborate on the reasons why ``torch.compile`` can provide an additional
performance boost compared to its eager counterpart.
"""

######################################################################
# Debugging
# ---------
#
# Here is a simple example to run the ``torch.compile`` using Inductor and
# compare its result with eager mode:

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


##############################################################################
# The correct implementation of ``neg`` in the ``cpp`` codegen is as follows:

def neg1(x):
    return f"decltype({x})(-{x})"


###########################################################################
# In order to demonstrate the debugging, we will modify the function to a
# wrong one later.
#
# Get more logging information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# No debugging information would be provided if you run this simple example by
# default. In order to get more useful debugging and logging information, we
# usually add a ``TORCH_COMPILE_DEBUG`` environment variable like below:
#
# .. code-block:: bash
#
#    TORCH_COMPILE_DEBUG=1 python xx.py
#
# This would print more debug information in the output logs and also dump the
# intermediate IRs generated during the codegen process. You can find the
# dumped file paths in the log like below:
#
# .. code-block:: bash
#
#    torch._inductor.debug: [WARNING] model___20 debug trace: /tmp/torchinductor_root/rx/crxfi2ybd7yp5sbj2pnhw33wfhtdw7wumvrobyp5sjvdui5ktjc2.debug
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
# Note that ``fx_graph_runnable.py`` and ``output_code.py`` are both runnable
# and editable in order to make debugging easier.
# Here are the main parts of code extracted from the files and we correlate the
# C++ generated line with the FX code line.
#
# ``fx_graph_runnable``:

def forward1(self, arg0_1, arg1_1):
    neg = torch.ops.aten.neg.default(arg0_1);
    arg0_1 = None
    maximum = torch.ops.aten.maximum.default(arg1_1, neg);
    arg1_1 = neg = None
    clone = torch.ops.aten.clone.default(maximum);
    maximum = None
    return (clone,)


######################################################################
# C++ kernel in ``output_code``:

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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When encountering errors or accuracy problems, a straightforward solution to
# find the bug is to narrow down the problem. The first thing to do is to
# determine the component where the error occurs. Luckily, it can be simply
# achieved by changing the backend of ``torch.compile``.
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
# If the model can successfully run when the backend is set to ``eager`` or
# ``aot_eager`` while it fails with ``inductor``, we can narrow down the failure
# to Inductor.
#
#
# Compilation error
# ~~~~~~~~~~~~~~~~~
#
# As we know, the evolved chain of graph-level optimization is like:
#
# .. code-block:: bash
#
#    torch.neg (Python) -> torch.ops.aten.neg.default (within FX graph) -> ops.neg (within IR node) -> tmp2 = -tmp1 (within C++ kernel)
#
# If you encounter a compilation error, there is something wrong when compiling
# C++ kernels in the output code.
# This type of error indicates that bugs are introduced when lowering IR nodes
# to output code.
# The root cause of compilation error is usually shown in the traceback log.
#
# For example, the ``neg`` function is modified like this:

def neg2(x):
    return f"-{x}"
