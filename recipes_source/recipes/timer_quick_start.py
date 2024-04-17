"""
Timer quick start
=================

In this tutorial, we're going to cover the primary APIs of
`torch.utils.benchmark.Timer`. The PyTorch Timer is based on the
`timeit.Timer <https://docs.python.org/3/library/timeit.html#timeit.Timer>`__
API, with several PyTorch specific modifications. Familiarity with the
builtin `Timer` class is not required for this tutorial, however we assume
that the reader is familiar with the fundamentals of performance work.

For a more comprehensive performance tuning tutorial, see
`PyTorch Benchmark <https://pytorch.org/tutorials/recipes/recipes/benchmark.html>`__.


**Contents:**
    1. `Defining a Timer <#defining-a-timer>`__
    2. `Wall time: Timer.blocked_autorange(...) <#wall-time-timer-blocked-autorange>`__
    3. `C++ snippets <#c-snippets>`__
    4. `Instruction counts: Timer.collect_callgrind(...) <#instruction-counts-timer-collect-callgrind>`__
    5. `Instruction counts: Delving deeper <#instruction-counts-delving-deeper>`__
    6. `A/B testing with Callgrind <#a-b-testing-with-callgrind>`__
    7. `Wrapping up <#wrapping-up>`__
    8. `Footnotes <#footnotes>`__
"""


###############################################################################
# 1. Defining a Timer
# ~~~~~~~~~~~~~~~~~~~
#
# A `Timer` serves as a task definition.
#

from torch.utils.benchmark import Timer

timer = Timer(
    # The computation which will be run in a loop and timed.
    stmt="x * y",

    # `setup` will be run before calling the measurement loop, and is used to
    # populate any state which is needed by `stmt`
    setup="""
        x = torch.ones((128,))
        y = torch.ones((128,))
    """,

    # Alternatively, ``globals`` can be used to pass variables from the outer scope.
    # 
    #    globals={
    #        "x": torch.ones((128,)),
    #        "y": torch.ones((128,)),
    #    },

    # Control the number of threads that PyTorch uses. (Default: 1)
    num_threads=1,
)

###############################################################################
# 2. Wall time: ``Timer.blocked_autorange(...)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This method will handle details such as picking a suitable number if repeats,
# fixing the number of threads, and providing a convenient representation of
# the results.
#

# Measurement objects store the results of multiple repeats, and provide
# various utility features.
from torch.utils.benchmark import Measurement

m: Measurement = timer.blocked_autorange(min_run_time=1)
print(m)

###############################################################################
# .. code-block:: none
#    :caption: **Snippet wall time.**
#
#         <torch.utils.benchmark.utils.common.Measurement object at 0x7f1929a38ed0>
#         x * y
#         setup:
#           x = torch.ones((128,))
#           y = torch.ones((128,))
#
#           Median: 2.34 us
#           IQR:    0.07 us (2.31 to 2.38)
#           424 measurements, 1000 runs per measurement, 1 thread
#

###############################################################################
# 3. C++ snippets
# ~~~~~~~~~~~~~~~
#

from torch.utils.benchmark import Language

cpp_timer = Timer(
    "x * y;",
    """
        auto x = torch::ones({128});
        auto y = torch::ones({128});
    """,
    language=Language.CPP,
)

print(cpp_timer.blocked_autorange(min_run_time=1))

###############################################################################
# .. code-block:: none
#    :caption: **C++ snippet wall time.**
#
#         <torch.utils.benchmark.utils.common.Measurement object at 0x7f192b019ed0>
#         x * y;
#         setup:
#           auto x = torch::ones({128});
#           auto y = torch::ones({128});
#
#           Median: 1.21 us
#           IQR:    0.03 us (1.20 to 1.23)
#           83 measurements, 10000 runs per measurement, 1 thread
#

###############################################################################
# Unsurprisingly, the C++ snippet is both faster and has lower variation.
#

###############################################################################
# 4. Instruction counts: ``Timer.collect_callgrind(...)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For deep dive investigations, ``Timer.collect_callgrind`` wraps
# `Callgrind <https://valgrind.org/docs/manual/cl-manual.html>`__ in order to
# collect instruction counts. These are useful as they offer fine grained and
# deterministic (or very low noise in the case of Python) insights into how a
# snippet is run.
#

from torch.utils.benchmark import CallgrindStats, FunctionCounts

stats: CallgrindStats = cpp_timer.collect_callgrind()
print(stats)

###############################################################################
# .. code-block:: none
#    :caption: **C++ Callgrind stats (summary)**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.CallgrindStats object at 0x7f1929a35850>
#         x * y;
#         setup:
#           auto x = torch::ones({128});
#           auto y = torch::ones({128});
#
#                                 All          Noisy symbols removed
#             Instructions:       563600                     563600
#             Baseline:                0                          0
#         100 runs per measurement, 1 thread
#

###############################################################################
# 5. Instruction counts: Delving deeper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The string representation of ``CallgrindStats`` is similar to that of
# Measurement. `Noisy symbols` are a Python concept (removing calls in the
# CPython interpreter which are known to be noisy).
#
# For more detailed analysis, however, we will want to look at specific calls.
# ``CallgrindStats.stats()`` returns a ``FunctionCounts`` object to make this easier.
# Conceptually, ``FunctionCounts`` can be thought of as a tuple of pairs with some
# utility methods, where each pair is `(number of instructions, file path and
# function name)`.
#
# A note on paths:
#   One generally doesn't care about absolute path. For instance, the full path
#   and function name for a multiply call is something like:
#
# .. code-block:: sh
#
#    /the/prefix/to/your/pytorch/install/dir/pytorch/build/aten/src/ATen/core/TensorMethods.cpp:at::Tensor::mul(at::Tensor const&) const [/the/path/to/your/conda/install/miniconda3/envs/ab_ref/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so]
#
#   when in reality, all of the information that we're interested in can be
#   represented in:
#
# .. code-block:: sh
#
#    build/aten/src/ATen/core/TensorMethods.cpp:at::Tensor::mul(at::Tensor const&) const
#
#   ``CallgrindStats.as_standardized()`` makes a best effort to strip low signal
#   portions of the file path, as well as the shared object and is generally
#   recommended.
#

inclusive_stats = stats.as_standardized().stats(inclusive=False)
print(inclusive_stats[:10])

###############################################################################
# .. code-block:: none
#    :caption: **C++ Callgrind stats (detailed)**
#
#         torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f192a6dfd90>
#           47264  ???:_int_free
#           25963  ???:_int_malloc
#           19900  build/../aten/src/ATen/TensorIter ... (at::TensorIteratorConfig const&)
#           18000  ???:__tls_get_addr
#           13500  ???:malloc
#           11300  build/../c10/util/SmallVector.h:a ... (at::TensorIteratorConfig const&)
#           10345  ???:_int_memalign
#           10000  build/../aten/src/ATen/TensorIter ... (at::TensorIteratorConfig const&)
#            9200  ???:free
#            8000  build/../c10/util/SmallVector.h:a ... IteratorBase::get_strides() const
#
#         Total: 173472
#

###############################################################################
# That's still quite a lot to digest. Let's use the `FunctionCounts.transform`
# method to trim some of the function path, and discard the function called.
# When we do, the counts of any collisions (e.g. `foo.h:a()` and `foo.h:b()`
# will both map to `foo.h`) will be added together.
#

import os
import re

def group_by_file(fn_name: str):
    if fn_name.startswith("???"):
        fn_dir, fn_file = fn_name.split(":")[:2]
    else:
        fn_dir, fn_file = os.path.split(fn_name.split(":")[0])
        fn_dir = re.sub("^.*build/../", "", fn_dir)
        fn_dir = re.sub("^.*torch/", "torch/", fn_dir)

    return f"{fn_dir:<15} {fn_file}"

print(inclusive_stats.transform(group_by_file)[:10])

###############################################################################
# .. code-block:: none
#    :caption: **Callgrind stats (condensed)**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f192995d750>
#           118200  aten/src/ATen   TensorIterator.cpp
#            65000  c10/util        SmallVector.h
#            47264  ???             _int_free
#            25963  ???             _int_malloc
#            20900  c10/util        intrusive_ptr.h
#            18000  ???             __tls_get_addr
#            15900  c10/core        TensorImpl.h
#            15100  c10/core        CPUAllocator.cpp
#            13500  ???             malloc
#            12500  c10/core        TensorImpl.cpp
#
#         Total: 352327
#

###############################################################################
# 6. A/B testing with ``Callgrind``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# One of the most useful features of instruction counts is they allow fine
# grained comparison of computation, which is critical when analyzing
# performance.
#
# To see this in action, lets compare our multiplication of two size 128
# Tensors with a {128} x {1} multiplication, which will broadcast the second
# Tensor:
#   result = {a0 * b0, a1 * b0, ..., a127 * b0}
#

broadcasting_stats = Timer(
    "x * y;",
    """
        auto x = torch::ones({128});
        auto y = torch::ones({1});
    """,
    language=Language.CPP,
).collect_callgrind().as_standardized().stats(inclusive=False)

###############################################################################
# Often we want to A/B test two different environments. (e.g. testing a PR, or
# experimenting with compile flags.) This is quite simple, as ``CallgrindStats``,
# ``FunctionCounts``, and Measurement are all pickleable. Simply save measurements
# from each environment, and load them in a single process for analysis.
#

import pickle

# Let's round trip `broadcasting_stats` just to show that we can.
broadcasting_stats = pickle.loads(pickle.dumps(broadcasting_stats))


# And now to diff the two tasks:
delta = broadcasting_stats - inclusive_stats

def extract_fn_name(fn: str):
    """Trim everything except the function name."""
    fn = ":".join(fn.split(":")[1:])
    return re.sub(r"\(.+\)", "(...)", fn)

# We use `.transform` to make the diff readable:
print(delta.transform(extract_fn_name))


###############################################################################
# .. code-block:: none
#    :caption: **Instruction count delta**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f192995d750>
#             17600  at::TensorIteratorBase::compute_strides(...)
#             12700  at::TensorIteratorBase::allocate_or_resize_outputs()
#             10200  c10::SmallVectorImpl<long>::operator=(...)
#              7400  at::infer_size(...)
#              6200  at::TensorIteratorBase::invert_perm(...) const
#              6064  _int_free
#              5100  at::TensorIteratorBase::reorder_dimensions()
#              4300  malloc
#              4300  at::TensorIteratorBase::compatible_stride(...) const
#               ...
#               -28  _int_memalign
#              -100  c10::impl::check_tensor_options_and_extract_memory_format(...)
#              -300  __memcmp_avx2_movbe
#              -400  at::detail::empty_cpu(...)
#             -1100  at::TensorIteratorBase::numel() const
#             -1300  void at::native::(...)
#             -2400  c10::TensorImpl::is_contiguous(...) const
#             -6100  at::TensorIteratorBase::compute_fast_setup_type(...)
#            -22600  at::TensorIteratorBase::fast_set_up(...)
#
#         Total: 58091
#

###############################################################################
# So the broadcasting version takes an extra 580 instructions per call (recall
# that we're collecting 100 runs per sample), or about 10%. There are quite a
# few ``TensorIterator`` calls, so lets drill down to those. ``FunctionCounts.filter``
# makes this easy.
#

print(delta.transform(extract_fn_name).filter(lambda fn: "TensorIterator" in fn))

###############################################################################
# .. code-block:: none
#    :caption: **Instruction count delta (filter)**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f19299544d0>
#             17600  at::TensorIteratorBase::compute_strides(...)
#             12700  at::TensorIteratorBase::allocate_or_resize_outputs()
#              6200  at::TensorIteratorBase::invert_perm(...) const
#              5100  at::TensorIteratorBase::reorder_dimensions()
#              4300  at::TensorIteratorBase::compatible_stride(...) const
#              4000  at::TensorIteratorBase::compute_shape(...)
#              2300  at::TensorIteratorBase::coalesce_dimensions()
#              1600  at::TensorIteratorBase::build(...)
#             -1100  at::TensorIteratorBase::numel() const
#             -6100  at::TensorIteratorBase::compute_fast_setup_type(...)
#            -22600  at::TensorIteratorBase::fast_set_up(...)
#
#         Total: 24000
#

###############################################################################
# This makes plain what is going on: there is a fast path in ``TensorIterator``
# setup, but in the {128} x {1} case we miss it and have to do a more general
# analysis which is more expensive. The most prominent call omitted by the
# filter is `c10::SmallVectorImpl<long>::operator=(...)`, which is also part
# of the more general setup.
#

###############################################################################
# 7. Wrapping up
# ~~~~~~~~~~~~~~
#
# In summary, use `Timer.blocked_autorange` to collect wall times. If timing
# variation is too high, increase `min_run_time`, or move to C++ snippets if
# convenient.
#
# For fine grained analysis, use `Timer.collect_callgrind` to measure
# instruction counts and `FunctionCounts.(__add__ / __sub__ / transform / filter)`
# to slice-and-dice them.
#

###############################################################################
# 8. Footnotes
# ~~~~~~~~~~~~
#
#   - Implied ``import torch``
#       If `globals` does not contain "torch", Timer will automatically
#       populate it. This means that ``Timer("torch.empty(())")`` will work.
#       (Though other imports should be placed in `setup`,
#       e.g. ``Timer("np.zeros(())", "import numpy as np")``)
#
#   - ``REL_WITH_DEB_INFO``
#       In order to provide full information about the PyTorch internals which
#       are executed, ``Callgrind`` needs access to C++ debug symbols. This is
#       accomplished by setting ``REL_WITH_DEB_INFO=1`` when building PyTorch.
#       Otherwise function calls will be opaque. (The resultant ``CallgrindStats``
#       will warn if debug symbols are missing.)
