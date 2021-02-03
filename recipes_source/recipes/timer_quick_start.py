"""
Timer quick start
=================

In this tutorial, we're going to run through a simple example and demonstrate
the API of `torch.utils.benchmark.Timer`. The code below assumes that you are
familiar with the fundamentals of performance work, but need a refresher on
`Timer`. A more comprehensive performace tuning tutorial is available at:

    https://pytorch.org/tutorials/recipes/recipes/benchmark.html


Case study: When less work is surprisingly more expensive.
----------------------------------------------------------
Consider two tasks:
  - Pointwise multiplication two size `n` Tensors
      vs.
  - Pointwise multiplication of a size `n` Tensor with a size one Tensor

Clearly the second is conceptually less work, as it is simply a degenerate
case when a single value is broadcast to a vector dim:
  {a0 * b0, a1 * b1, ..., a127 * b127}
    vs.
  {a0 * b0, a0 * b1, ..., a0 * b127}

Let's see what happens when we measure it.
"""

import torch
from torch.utils.benchmark import Language, Timer

def measure(b_size):
    return Timer(
        "x * y",
        f"""
        x = torch.ones((128,))
        y = torch.ones(({b_size},))
        """,
    ).blocked_autorange(min_run_time=1)

print(measure(b_size=128), "\n")
print(measure(b_size=1), "\n")

###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     (The top half of the __repr__ has been trimmed for convenience.)
#
#     Case A: {128} x {128}
#       Median: 2.59 us
#       IQR:    0.09 us (2.55 to 2.64)
#       381 measurements, 1000 runs per measurement, 1 thread
#
#     Case B: {128} x {1}
#       Median: 2.69 us
#       IQR:    0.08 us (2.65 to 2.74)
#       366 measurements, 1000 runs per measurement, 1 thread
#


###############################################################################
# From the look of it, the second (simpler) case is slower but it's hard to
# tell if it's real or within the noise. For lower variance measurements we
# can rerun the benchmark using the C++ frontend:
#

def measure_cpp(b_size):
    return Timer(
        "x * y;",
        f"""
        auto x = torch::ones({{128}});
        auto y = torch::ones({{{b_size}}});
        """,
        language=Language.CPP,
    ).blocked_autorange(min_run_time=1)

print(measure_cpp(b_size=128), "\n")
print(measure_cpp(b_size=1), "\n")

###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     Case A: {128} x {128}
#       Median: 1.47 us
#       IQR:    0.02 us (1.46 to 1.48)
#       68 measurements, 10000 runs per measurement, 1 thread
#
#     Case B: {128} x {1}
#       Median: 1.60 us
#       IQR:    0.04 us (1.58 to 1.62)
#       60 measurements, 10000 runs per measurement, 1 thread
#

###############################################################################
# The C++ wall times confirm that what we're seeing isn't just noise, but they
# do nothing to illuminate what is happening. For that, let's count
# instructions using Callgrind.

def measure_cpp_callgrind(b_size):
    return Timer(
        "x * y;",
        f"""
        auto x = torch::ones({{128}});
        auto y = torch::ones({{{b_size}}});
        """,
        language=Language.CPP,
    ).collect_callgrind()

for _ in range(3):
    print(measure_cpp_callgrind(128).counts())

print()
for _ in range(3):
    print(measure_cpp_callgrind(1).counts())

###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     587972
#     587972
#     587972
#
#     646262
#     646262
#     646262
#

###############################################################################
# As you can see, one of the selling points of Callgrind is that it is 100%
# deterministic. This is very useful because it lets us A/B test very subtle
# changes. Let's take a look at the CallgrindStats object that Timer returns:

from torch.utils.benchmark import CallgrindStats

s0: CallgrindStats = Timer(
    "x * y;",
    f"""
    auto x = torch::ones({{128}});
    auto y = torch::ones({{128}});
    """,
    language=Language.CPP,
).collect_callgrind()

print(s0)


###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.CallgrindStats object at 0x7f40c0e15a90>
#     x * y;
#     setup:
#       auto x = torch::ones({128});
#       auto y = torch::ones({128});
#
#                               All          Noisy symbols removed
#         Instructions:       587972                     587972
#         Baseline:                0                          0
#     100 runs per measurement, 1 thread
#


###############################################################################
# For C++, total count is pretty simple: WYSIWYG.
# The story is more complicated for Python. The CPython interpreter is
# non-deterministic, so there is some jitter when collecting callgrind counts.
# To compensate, methods are provided to "denoise" by removing known noisy
# Python symbols:
#   Timer(...).collect_callgrind().counts(denoise=True)
#   Timer(...).collect_callgrind().stats().denoise()
#
# Now, back to the C++ frontend!

# Lots of long C++ templates, so we will want to increase the print width.
torch.set_printoptions(linewidth=200)

# By default, show inclusive counts. Set `inclusive=False` for exlusive counts.
print(s0.stats())

###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f40c0ebadd0>
#       41137  ???:_int_free [/usr/lib64/libc-2.28.so]
#       26975  ???:_int_malloc [/usr/lib64/libc-2.28.so]
#       20400  ???:__tls_get_addr [/usr/lib64/ld-2.28.so]
#       19800  /data/users/taylorrobie/repos/pytorch/build/../aten/src/ATen/TensorIterator.cpp:at::TensorIte ... ers/taylorrobie/miniconda3/envs/ab_ref/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so]
#       13500  ???:malloc [/usr/lib64/libc-2.28.so]
#       11300  /data/users/taylorrobie/repos/pytorch/build/../c10/util/SmallVector.h:at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#       10000  /data/users/taylorrobie/repos/pytorch/build/../aten/src/ATen/TensorIterator.cpp:at::TensorIte ... ers/taylorrobie/miniconda3/envs/ab_ref/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so]
#        9200  ???:free [/usr/lib64/libc-2.28.so]
#        9200  /data/users/taylorrobie/repos/pytorch/build/../aten/src/ATen/record_function.cpp:at::shouldRu ... ers/taylorrobie/miniconda3/envs/ab_ref/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so]
#         ...
#         100  /data/users/taylorrobie/repos/pytorch/build/../c10/util/FunctionRef.h:at::native::(anonymous namespace)::mul_kernel(at::TensorIterator&)
#         100  /data/users/taylorrobie/repos/pytorch/build/../c10/util/FunctionRef.h:at::TensorIteratorBase::for_each(c10::function_ref<void (char**, long const*, long)>, long)
#         100  /data/users/taylorrobie/repos/pytorch/build/../c10/util/ArrayRef.h:at::TensorIteratorBase::get_data_ptrs(c10::ArrayRef<char*>, c10::ArrayRef<long>) const
#         100  /data/users/taylorrobie/repos/pytorch/build/../c10/core/impl/LocalDispatchKeySet.h:c10::impl::ExcludeDispatchKeyGuard::ExcludeDispatchKeyGuard(c10::DispatchKeySet)
#         100  /data/users/taylorrobie/repos/pytorch/build/../c10/core/Storage.h:c10::TensorImpl::release_resources()
#         100  /data/users/taylorrobie/repos/pytorch/build/../c10/core/ScalarType.h:c10::computeDispatchKey(c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>)
#         100  /data/users/taylorrobie/repos/pytorch/build/../aten/src/ATen/core/dispatch/DispatchKeyExtract ... (at::Tensor const&, at::Tensor const&)> const&, at::Tensor const&, at::Tensor const&) const'2
#         100  /data/users/taylorrobie/repos/pytorch/build/../aten/src/ATen/core/dispatch/DispatchKeyExtract ... r (at::Tensor const&, at::Tensor const&)> const&, at::Tensor const&, at::Tensor const&) const
#         100  /data/users/taylorrobie/repos/pytorch/build/../aten/src/ATen/core/boxing/impl/WrapFunctionInt ... const&, at::Tensor const&)>::call(c10::OperatorKernel*, at::Tensor const&, at::Tensor const&)
#
#     Total: 587972
#

###############################################################################
# There's some useful information here. The most expensive sections of the code
# are malloc, free, and various PyTorch utilities. However there is also a lot
# of extraneous path junk cluttering things up. For that, you can call the
# `as_standardized` method which will make a best effort to strip extraneous
# parts of the symbol name:

print(s0.as_standardized().stats())

###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f40c0ec7c50>
#       41137  ???:_int_free
#       26975  ???:_int_malloc
#       20400  ???:__tls_get_addr
#       19800  build/../aten/src/ATen/TensorIterator.cpp:at::TensorIteratorBase::compute_types(at::TensorIteratorConfig const&)
#       13500  ???:malloc
#       11300  build/../c10/util/SmallVector.h:at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#       10000  build/../aten/src/ATen/TensorIterator.cpp:at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#       9200  build/../aten/src/ATen/record_function.cpp:at::shouldRunRecordFunction(bool*)
#       9200  ???:free
#         ...
#         100  build/../c10/core/ScalarType.h:c10::computeDispatchKey(c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>)
#         100  build/../aten/src/ATen/core/dispatch/DispatchKeyExtractor.h:at::Tensor c10::Dispatcher::call< ... (at::Tensor const&, at::Tensor const&)> const&, at::Tensor const&, at::Tensor const&) const'2
#         100  build/../aten/src/ATen/core/dispatch/DispatchKeyExtractor.h:at::Tensor c10::Dispatcher::call< ... r (at::Tensor const&, at::Tensor const&)> const&, at::Tensor const&, at::Tensor const&) const
#         100  build/../aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:c10::impl::wrap_kernel_func ... const&, at::Tensor const&)>::call(c10::OperatorKernel*, at::Tensor const&, at::Tensor const&)
#         100  /usr/include/c++/8/bits/stl_function.h:at::detail::empty_cpu(c10::ArrayRef<long>, c10::option ... ::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>)
#         100  /usr/include/c++/8/bits/atomic_base.h:void at::native::DispatchStub<void (*)(at::TensorIterator&), at::native::mul_stub>::operator()<at::TensorIterator&>(c10::DeviceType, at::TensorIterator&)
#         100  /usr/include/c++/8/bits/atomic_base.h:c10::TensorImpl::~TensorImpl()
#         100  /usr/include/c++/8/bits/atomic_base.h:at::native::mul(at::Tensor const&, at::Tensor const&)
#         100  /home/taylorrobie/local/miniconda3/envs/ab_ref/lib/python3.7/site-packages/torch/include/c10/util/intrusive_ptr.h:c10::intrusive_ptr_target::release_resources()
#
#     Total: 587972
#

###############################################################################
# `CallgrindStats.stats()` produces a `FunctionCounts` object, which is really
# just a glorified tuple. Both CallgrindStats and FunctionCounts are
# pickleable. A common workflow when measuring changes is to collect a set
# of measurements, dump them to a file, change environments, repeat, and
# finally load everything into an analysis script or notebook for spelunking.

import pickle
from torch.utils.benchmark import FunctionCounts

s0_stats: FunctionCounts = s0.stats()

# Round trip `s0` just to show that we can.
s0_revived: CallgrindStats = pickle.loads(pickle.dumps(s0))
s0_revived_stats: FunctionCounts = s0_revived.stats()

assert len(s0_stats) == len(s0_revived_stats)
for i, j in zip(s0_stats, s0_revived_stats):
    assert i == j


###############################################################################
# We can also diff stats, and here standardization becomes important. If we
# are comparing runs from different environments, one may have a symbol:
#   /some/common/prefix/env_a/foo.cpp:at::bar()
#
# while the other has:
#   /some/common/prefix/env_b/foo.cpp:at::bar()
#
# If that section of the code is unchanged, diffing the two would yield:
#   -123456 /some/common/prefix/env_a/foo.cpp:at::bar()
#    123456 /some/common/prefix/env_b/foo.cpp:at::bar()
#
# Instead, they should just cancel out. However in order to do so we have to
# indicate that the install path is not important. Let's try it with our A/B
# test:

s1 = Timer(
    "x * y;",
    f"""
    auto x = torch::ones({{128}});
    auto y = torch::ones({{1}});
    """,
    language=Language.CPP,
).collect_callgrind()

delta = s1.as_standardized().stats() - s0.as_standardized().stats()
print(delta)

###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f40c0ebadd0>
#         9300  build/../aten/src/ATen/TensorIterator.cpp:at::TensorIteratorBase::compute_strides(at::TensorIteratorConfig const&)
#         8900  build/../c10/util/SmallVector.h:c10::SmallVectorImpl<long>::operator=(c10::SmallVectorImpl<long>&&)
#         8200  build/../aten/src/ATen/TensorIterator.cpp:at::TensorIteratorBase::allocate_or_resize_outputs()
#         6000  ???:_int_free
#         4800  build/../c10/util/SmallVector.h:at::TensorIteratorBase::compute_strides(at::TensorIteratorConfig const&)
#         4600  build/../aten/src/ATen/ExpandUtils.cpp:at::infer_size(c10::ArrayRef<long>, c10::ArrayRef<long>)
#         4300  ???:malloc
#         3900  build/../c10/util/SmallVector.h:at::TensorIteratorBase::allocate_or_resize_outputs()
#         3900  ???:__memcpy_avx_unaligned_erms
#           ...
#         -1000  build/../c10/core/TensorImpl.h:at::TensorIteratorBase::compute_fast_setup_type(at::TensorIteratorConfig const&)
#         -1100  build/../aten/src/ATen/TensorIterator.cpp:at::TensorIteratorBase::numel() const
#         -1300  build/../c10/util/typeid.h:at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#         -1500  /usr/lib/gcc/x86_64-redhat-linux/8/include/avxintrin.h:void at::native::(anonymous namespace)::vectorized_loop<at::native::(anonymous namespace)::mul_kernel(at::TensorIterator&)::{lambda()
#         -2100  build/aten/src/ATen/core/TensorBody.h:at::TensorIteratorBase::compute_fast_setup_type(at::TensorIteratorConfig const&)
#         -2400  build/../c10/core/TensorImpl.cpp:c10::TensorImpl::is_contiguous(c10::MemoryFormat) const
#         -3000  build/../aten/src/ATen/TensorIterator.cpp:at::TensorIteratorBase::compute_fast_setup_type(at::TensorIteratorConfig const&)
#         -7800  build/../aten/src/ATen/TensorIterator.cpp:at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#       -11300  build/../c10/util/SmallVector.h:at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#
#     Total: 58290
#

###############################################################################
# Now we're getting closer. Note that 90% of the calls were indistinguishable,
# and we can focus on the remaining 10%. Two functions that can help us narrow
# down even more are `.transform` and `.filter`. Each takes a callable that is
# passed the function name (including path), and it either returns a new value
# (transform) or a boolean of whether to keep the entry (filter).

import os
import re

def trim_path(fn: str):
    """Trim everything before the filename, and align function names."""
    fpath = fn.split(":")[0]
    if "/" in fpath:
        prefix, fpath = os.path.split(fpath)
        fn = fn[len(prefix) + 1:]

    return f"{fn[:len(fpath)]:<29} {fn[len(fpath) + 1:]}"

pretty_delta = delta.transform(trim_path)
print(pretty_delta)

###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f40c0ec7cd0>
#         9300  TensorIterator.cpp            at::TensorIteratorBase::compute_strides(at::TensorIteratorConfig const&)
#         8900  SmallVector.h                 c10::SmallVectorImpl<long>::operator=(c10::SmallVectorImpl<long>&&)
#         8200  TensorIterator.cpp            at::TensorIteratorBase::allocate_or_resize_outputs()
#         6000  ???                           _int_free
#         4800  SmallVector.h                 at::TensorIteratorBase::compute_strides(at::TensorIteratorConfig const&)
#         4600  ExpandUtils.cpp               at::infer_size(c10::ArrayRef<long>, c10::ArrayRef<long>)
#         4300  ???                           malloc
#         3900  SmallVector.h                 at::TensorIteratorBase::allocate_or_resize_outputs()
#         3900  ???                           __memcpy_avx_unaligned_erms
#          ...
#        -1000  TensorImpl.h                  at::TensorIteratorBase::compute_fast_setup_type(at::TensorIteratorConfig const&)
#        -1100  TensorIterator.cpp            at::TensorIteratorBase::numel() const
#        -1300  typeid.h                      at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#        -1500  avxintrin.h                   void at::native::(anonymous namespace)::vectorized_loop<at::native::(anonymous namespace)::mul_kernel(at::TensorIterator&)::{lambda()
#        -2100  TensorBody.h                  at::TensorIteratorBase::compute_fast_setup_type(at::TensorIteratorConfig const&)
#        -2400  TensorImpl.cpp                c10::TensorImpl::is_contiguous(c10::MemoryFormat) const
#        -3000  TensorIterator.cpp            at::TensorIteratorBase::compute_fast_setup_type(at::TensorIteratorConfig const&)
#        -7800  TensorIterator.cpp            at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#       -11300  SmallVector.h                 at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#
#     Total: 58290
#

###############################################################################
# Now there are a lot of TensorIterator calls. Let's use `filter` to drill down

tensor_iterator_delta = pretty_delta.filter(lambda fn: "TensorIterator" in fn)
print(tensor_iterator_delta)

###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f40c0ec7f50>
#         9300  TensorIterator.cpp            at::TensorIteratorBase::compute_strides(at::TensorIteratorConfig const&)
#         8200  TensorIterator.cpp            at::TensorIteratorBase::allocate_or_resize_outputs()
#         4800  SmallVector.h                 at::TensorIteratorBase::compute_strides(at::TensorIteratorConfig const&)
#         3900  SmallVector.h                 at::TensorIteratorBase::allocate_or_resize_outputs()
#         3100  SmallVector.h                 at::TensorIteratorBase::reorder_dimensions()
#         2800  TensorIterator.cpp            at::TensorIteratorBase::invert_perm(c10::ArrayRef<long>) const
#         2600  SmallVector.h                 at::TensorIteratorBase::invert_perm(c10::ArrayRef<long>) const
#         2300  TensorIterator.cpp            at::TensorIteratorBase::compatible_stride(int) const
#         2100  TensorBody.h                  at::TensorIteratorBase::compute_strides(at::TensorIteratorConfig const&)
#           ...
#         -800  TensorImpl.h                  at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#        -1000  TensorImpl.h                  at::TensorIteratorBase::compute_fast_setup_type(at::TensorIteratorConfig const&)
#        -1100  TensorIterator.cpp            at::TensorIteratorBase::numel() const
#        -1300  typeid.h                      at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#        -1500  avxintrin.h                   void at::native::(anonymous namespace)::vectorized_loop<at::native::(anonymous namespace)::mul_kernel(at::TensorIterator&)::{lambda()
#        -2100  TensorBody.h                  at::TensorIteratorBase::compute_fast_setup_type(at::TensorIteratorConfig const&)
#        -3000  TensorIterator.cpp            at::TensorIteratorBase::compute_fast_setup_type(at::TensorIteratorConfig const&)
#        -7800  TensorIterator.cpp            at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#       -11300  SmallVector.h                 at::TensorIteratorBase::fast_set_up(at::TensorIteratorConfig const&)
#
#     Total: 23400
#

###############################################################################
# Aha! So about 40% of the difference is that the {128} x {128} gets to take a
# fast path during setup (at::TensorIteratorBase::fast_set_up) whereas the
# {128} x {1} code does a full strided broadcasting calculation. If we look
# at the remaining calls it's some more SmallVector bookkeeping associated with
# size and stride calculations, as well as subtle differences in the memory
# allocation of the two variants.

print(pretty_delta - tensor_iterator_delta)

###############################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f40c0ec7bd0>
#         8900  SmallVector.h                 c10::SmallVectorImpl<long>::operator=(c10::SmallVectorImpl<long>&&)
#         6000  ???                           _int_free
#         4600  ExpandUtils.cpp               at::infer_size(c10::ArrayRef<long>, c10::ArrayRef<long>)
#         4300  ???                           malloc
#         3900  ???                           __memcpy_avx_unaligned_erms
#         2300  ???                           free
#         1200  stl_vector.h                  at::infer_size(c10::ArrayRef<long>, c10::ArrayRef<long>)
#         1200  ???                           operator new(unsigned long)
#         1100  stl_algobase.h                c10::SmallVectorImpl<long>::operator=(c10::SmallVectorImpl<long>&&)
#          ...
#          200  ???                           0x000000000c77b220
#          100  hacky_wrapper_for_legacy_signatures.h c10::impl::check_tensor_options_and_extract_memory_format(c10::TensorOptions const&, c10::optional<c10::MemoryFormat>)
#          100  Optional.h                    c10::impl::check_tensor_options_and_extract_memory_format(c10::TensorOptions const&, c10::optional<c10::MemoryFormat>)
#           72  ???                           _int_malloc
#           18  ???                           unlink_chunk.isra.2
#         -200  Utils.cpp                     at::detail::empty_cpu(c10::ArrayRef<long>, c10::optional<c10:: ... ::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>)
#         -200  TensorOptions.h               c10::impl::check_tensor_options_and_extract_memory_format(c10::TensorOptions const&, c10::optional<c10::MemoryFormat>)
#         -300  ???                           __memcmp_avx2_movbe
#        -2400  TensorImpl.cpp                c10::TensorImpl::is_contiguous(c10::MemoryFormat) const
#
#     Total: 34890
#

###############################################################################
# Given that the difference is an O(1) overhead rather than an O(n) compute
# difference (except perhaps the malloc / free deltas), we expect that at some
# point the {n} x {1} broadcasting computation will overtake the {n} x {n}
# case. We'll leave it as an excercise to the user to verify; on my machine the
# crossover happens at n~=2048.
#
# Editorial note:
#   This guide was formulated as a way to show how one could run down a failure
#   to use AVX instructions. (And indeed, I would encourage the reader to
#   investigate ({128} x {128}) vs. ({127} x {127}) to see such an effect.)
#   However the actual culprit was entirety different, and even though we had
#   a very strong prior (due to contriving an example to show it...)
#   instruction counts were able to bring the true story to light.
