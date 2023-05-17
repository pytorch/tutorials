"""
PyTorch Benchmark
====================================
This recipe provides a quick-start guide to using PyTorch
``benchmark`` module to measure and compare code performance.

Introduction
------------
Benchmarking is an important step in writing code. It helps
us validate that our code meets performance expectations,
compare different approaches to solving the same problem and
prevent performance regressions.

There are many options when it comes to benchmarking PyTorch code
including the Python builtin ``timeit`` module. However, benchmarking
PyTorch code has many caveats that can be easily overlooked such as
managing the number of threads and synchronizing CUDA devices. Moreover,
generating Tensor inputs for benchmarking can be quite tedious.

This recipe demonstrates how to use PyTorch ``benchmark`` module to avoid
common mistakes while making it easier to compare performance of
different code, generate input for benchmarking and more.

Setup
-----
Before we begin, install ``torch`` if it isn’t already available.

::

   pip install torch

"""


######################################################################
# Steps
# -----
#
# 1. Defining functions to benchmark
# 2. Benchmarking with ``timeit.Timer``
# 3. Benchmarking with ``torch.utils.benchmark.Timer``
# 4. Benchmarking with ``Blocked Autorange``
# 5. Comparing benchmark results
# 6. Saving/Loading benchmark results
# 7. Generating inputs with ``Fuzzed Parameters``
# 8. Collecting instruction counts with ``Callgrind``
#
# 1. Defining functions to benchmark
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As of the time of this writing, `torch.dot <https://pytorch.org/docs/stable/generated/torch.dot.html?highlight=dot#torch.dot>`__
# does not support batched mode, so we will compare two approaches to
# implementing it using existing ``torch`` operators: one approach uses a
# combination of ``mul`` and ``sum`` while the other reduces the problem to ``bmm``.
#

import torch


def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to ``bmm``'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


# Input for benchmarking
x = torch.randn(10000, 64)

# Ensure that both functions compute the same output
assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))


######################################################################
# 2. Benchmarking with ``timeit.Timer``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, let's benchmark the code using Python's builtin ``timeit`` module.
# We keep the benchmark code simple here so we can compare the defaults
# of ``timeit`` and ``torch.utils.benchmark``.
#

import timeit

t0 = timeit.Timer(
    stmt='batched_dot_mul_sum(x, x)', 
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = timeit.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     mul_sum(x, x):  111.6 us
#     bmm(x, x):       70.0 us
#


######################################################################
# 3. Benchmarking with ``torch.utils.benchmark.Timer``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PyTorch ``benchmark`` module was designed to be familiar to those who
# have used the ``timeit`` module before. However, its defaults make it
# easier and safer to use for benchmarking PyTorch code. Let's first
# compare the same basic API as above.
#

import torch.utils.benchmark as benchmark

t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)', 
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = benchmark.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

print(t0.timeit(100))
print(t1.timeit(100))

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d0f0>
#     batched_dot_mul_sum(x, x)
#     setup: from __main__ import batched_dot_mul_sum
#       379.29 us
#       1 measurement, 100 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb103d67048>
#     batched_dot_bmm(x, x)
#     setup: from __main__ import batched_dot_bmm
#       716.42 us
#       1 measurement, 100 runs , 1 thread
#

######################################################################
# Even though the APIs are the same for the basic functionality, there
# are some important differences. ``benchmark.Timer.timeit()`` returns the
# time per run as opposed to the total runtime like ``timeit.Timer.timeit()``
# does. PyTorch ``benchmark`` module also provides formatted string
# representations for printing the results.
#
# Another important difference, and the reason why the results diverge
# is that PyTorch benchmark module runs in a single thread by default.
# We can change the number of threads with the ``num_threads`` argument.
#
# ``torch.utils.benchmark.Timer`` takes several additional arguments
# including: ``label``, ``sub_label``, ``description`` and ``env`` which change
# the __repr__ of the measurement object returned and are used for
# grouping the results (more on this later).
#

num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')

t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)', 
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using mul and sum')

t1 = benchmark.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using bmm')

print(t0.timeit(100))
print(t1.timeit(100))

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     Benchmarking on 40 threads
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb103d54080>
#     Multithreaded batch dot: Implemented using mul and sum
#     setup: from __main__ import batched_dot_mul_sum
#       118.47 us
#       1 measurement, 100 runs , 40 threads
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb16935d2e8>
#     Multithreaded batch dot: Implemented using bmm
#     setup: from __main__ import batched_dot_bmm
#       68.21 us
#       1 measurement, 100 runs , 40 threads

######################################################################
# Running ``benchmark`` with all threads available gives similar results
# as the ``timeit`` module. More importantly, which version is faster
# depends on how many threads we run the code with. This is why it's
# important to benchmark the code with thread settings that are
# representative of real use cases. Another important thing to remember
# is to synchronize CPU and CUDA when benchmarking on the GPU. Let's run
# the above benchmarks again on a CUDA tensor and see what happens.
#

x = torch.randn(10000, 1024, device='cuda')

t0 = timeit.Timer(
    stmt='batched_dot_mul_sum(x, x)', 
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = timeit.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

# Ran each twice to show difference before/after warm-up
print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     mul_sum(x, x):   27.6 us
#     mul_sum(x, x):   25.3 us
#     bmm(x, x):      2775.5 us
#     bmm(x, x):       22.4 us
#

t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)', 
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = benchmark.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

# Run only once since benchmark module does warm-up for us
print(t0.timeit(100))
print(t1.timeit(100))

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d080>
#     batched_dot_mul_sum(x, x)
#     setup: from __main__ import batched_dot_mul_sum
#       232.93 us
#       1 measurement, 100 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d0f0>
#     batched_dot_bmm(x, x)
#     setup: from __main__ import batched_dot_bmm
#       181.04 us
#       1 measurement, 100 runs , 1 thread
#

######################################################################
# The results reveal something interesting. The first run of the ``bmm``
# version using the ``timeit`` module takes much longer than the second
# run. This is because ``bmm`` calls into `cuBLAS` which needs to be
# loaded the first time it's called which takes some time. This is why
# it's important to do a warm-up run before benchmarking, luckily for
# us, PyTorch's ``benchmark`` module takes care of that.
#
# The difference in the results between ``timeit`` and ``benchmark`` modules
# is because the `timeit` module is not synchronizing CUDA and is thus only
# timing the time to launch the kernel. PyTorch's ``benchmark`` module does
# the synchronization for us.


######################################################################
# 4. Benchmarking with `Blocked Autorange`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# While ``timeit.Timer.autorange`` takes a single continuous measurement
# of at least 0.2 seconds, `torch.utils.benchmark.blocked_autorange`
# takes many measurements whose times total at least 0.2 seconds (which
# can be changed by the `min_run_time` parameter) subject to the constraint
# that timing overhead is a small fraction of the overall measurement.
# This is accomplished by first running with an increasing number of runs
# per loop until the runtime is much larger than measurement overhead
# (which also serves as a warm up), and then taking measurements until
# the target time is reached. This has the useful properties that it wastes
# less data and allows us to compute statistics to estimate the reliability
# of the measurements.
#

m0 = t0.blocked_autorange()
m1 = t1.blocked_autorange()

print(m0)
print(m1)

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d0f0>
#     batched_dot_mul_sum(x, x)
#     setup: from __main__ import batched_dot_mul_sum
#       231.79 us
#       1 measurement, 1000 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d080>
#     batched_dot_bmm(x, x)
#     setup: from __main__ import batched_dot_bmm
#       Median: 162.08 us
#       2 measurements, 1000 runs per measurement, 1 thread
#

######################################################################
# We can also inspect the individual statistics from the returned
# measurements object.

print(f"Mean:   {m0.mean * 1e6:6.2f} us")
print(f"Median: {m0.median * 1e6:6.2f} us")

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     Mean:   231.79 us
#     Median: 231.79 us
#

######################################################################
# 5. Comparing benchmark results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# So far we've been comparing our two versions of batched dot against a
# single input. In practice, we want to try a combination of inputs as
# well as different number of threads. The ``Compare`` class helps display
# the results of many measurements in a formatted table. It uses the
# annotations described above (`label`, `sub_label`, `num_threads`, etc.) as
# well as `description` to group and organize the table. Let's use
# ``Compare`` to see how our functions perform for different input sizes
# and number of threads.
#

from itertools import product

# Compare takes a list of measurements which we'll save in results.
results = []

sizes = [1, 64, 1024, 10000]
for b, n in product(sizes, sizes):
    # label and sub_label are the rows
    # description is the column
    label = 'Batched dot'
    sub_label = f'[{b}, {n}]'
    x = torch.ones((b, n))
    for num_threads in [1, 4, 16, 32]:
        results.append(benchmark.Timer(
            stmt='batched_dot_mul_sum(x, x)',
            setup='from __main__ import batched_dot_mul_sum',
            globals={'x': x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='mul/sum',
        ).blocked_autorange(min_run_time=1))
        results.append(benchmark.Timer(
            stmt='batched_dot_bmm(x, x)',
            setup='from __main__ import batched_dot_bmm',
            globals={'x': x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='bmm',
        ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.print()

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     [--------------- Batched dot ----------------]
#                           |  mul/sum   |    bmm   
#     1 threads: -----------------------------------
#           [1, 1]          |       5.9  |      11.2
#           [1, 64]         |       6.4  |      11.4
#           [1, 1024]       |       6.7  |      14.2
#           [1, 10000]      |      10.2  |      23.7
#           [64, 1]         |       6.3  |      11.5
#           [64, 64]        |       8.6  |      15.4
#           [64, 1024]      |      39.4  |     204.4
#           [64, 10000]     |     274.9  |     748.5
#           [1024, 1]       |       7.7  |      17.8
#           [1024, 64]      |      40.3  |      76.4
#           [1024, 1024]    |     432.4  |    2795.9
#           [1024, 10000]   |   22657.3  |   11899.5
#           [10000, 1]      |      16.9  |      74.8
#           [10000, 64]     |     300.3  |     609.4
#           [10000, 1024]   |   23098.6  |   27246.1
#           [10000, 10000]  |  267073.7  |  118823.7
#     4 threads: -----------------------------------
#           [1, 1]          |       6.0  |      11.5
#           [1, 64]         |       6.2  |      11.2
#           [1, 1024]       |       6.8  |      14.3
#           [1, 10000]      |      10.2  |      23.7
#           [64, 1]         |       6.3  |      16.2
#           [64, 64]        |       8.8  |      18.2
#           [64, 1024]      |      41.5  |     189.1
#           [64, 10000]     |      91.7  |     849.1
#           [1024, 1]       |       7.6  |      17.4
#           [1024, 64]      |      43.5  |      33.5
#           [1024, 1024]    |     135.4  |    2782.3
#           [1024, 10000]   |    7471.1  |   11874.0
#           [10000, 1]      |      16.8  |      33.9
#           [10000, 64]     |     118.7  |     173.2
#           [10000, 1024]   |    7264.6  |   27824.7
#           [10000, 10000]  |  100060.9  |  121499.0
#     16 threads: ----------------------------------
#           [1, 1]          |       6.0  |      11.3
#           [1, 64]         |       6.2  |      11.2
#           [1, 1024]       |       6.9  |      14.2
#           [1, 10000]      |      10.3  |      23.8
#           [64, 1]         |       6.4  |      24.1
#           [64, 64]        |       9.0  |      23.8
#           [64, 1024]      |      54.1  |     188.5
#           [64, 10000]     |      49.9  |     748.0
#           [1024, 1]       |       7.6  |      23.4
#           [1024, 64]      |      55.5  |      28.2
#           [1024, 1024]    |      66.9  |    2773.9
#           [1024, 10000]   |    6111.5  |   12833.7
#           [10000, 1]      |      16.9  |      27.5
#           [10000, 64]     |      59.5  |      73.7
#           [10000, 1024]   |    6295.9  |   27062.0
#           [10000, 10000]  |   71804.5  |  120365.8
#     32 threads: ----------------------------------
#           [1, 1]          |       5.9  |      11.3
#           [1, 64]         |       6.2  |      11.3
#           [1, 1024]       |       6.7  |      14.2
#           [1, 10000]      |      10.5  |      23.8
#           [64, 1]         |       6.3  |      31.7
#           [64, 64]        |       9.1  |      30.4
#           [64, 1024]      |      72.0  |     190.4
#           [64, 10000]     |     103.1  |     746.9
#           [1024, 1]       |       7.6  |      28.4
#           [1024, 64]      |      70.5  |      31.9
#           [1024, 1024]    |      65.6  |    2804.6
#           [1024, 10000]   |    6764.0  |   11871.4
#           [10000, 1]      |      17.8  |      31.8
#           [10000, 64]     |     110.3  |      56.0
#           [10000, 1024]   |    6640.2  |   27592.2
#           [10000, 10000]  |   73003.4  |  120083.2
#
#     Times are in microseconds (us).
#

######################################################################
# The results above indicate that the version which reduces to ``bmm``
# is better for larger tensors running on multiple threads, while for
# smaller and/or single thread code, the other version is better.
#
# ``Compare`` also provides functions for changing the table format
#

compare.trim_significant_figures()
compare.colorize()
compare.print()


######################################################################
# 6. Saving/Loading benchmark results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# `Measurements` (and ``CallgrindStats`` which are described in section 8)
# can be serialized by the ``pickle`` module. This makes A/B testing easy, as you can collect
# measurements from two separate environments, pickle them, and then
# load both in a single environment. Timer even takes an `env`
# constructor argument so that such A/B testing works seamlessly.
#
# Let's imagine that rather than two Python functions, the add/sum
# and ``bmm`` approaches were in two different builds of PyTorch.
# The example below demonstrates how one might A/B test them. For
# simplicity, we only use a subset of shapes, and simply round trip
# results through pickle rather than actually using multiple environments
# and writing results to disk.
#

import pickle

ab_test_results = []
for env in ('environment A: mul/sum', 'environment B: bmm'):
    for b, n in ((1, 1), (1024, 10000), (10000, 1)):
        x = torch.ones((b, n))
        dot_fn = (batched_dot_mul_sum if env == 'environment A: mul/sum' else batched_dot_bmm)
        m = benchmark.Timer(
            stmt='batched_dot(x, x)',
            globals={'x': x, 'batched_dot': dot_fn},
            num_threads=1,
            label='Batched dot',
            description=f'[{b}, {n}]',
            env=env,
        ).blocked_autorange(min_run_time=1)
        ab_test_results.append(pickle.dumps(m))

ab_results = [pickle.loads(i) for i in ab_test_results]
compare = benchmark.Compare(ab_results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     [------------------------------------- Batched dot -------------------------------------]
#                                                    |  [1, 1]  |  [1024, 10000]  |  [10000, 1]
#     1 threads: ------------------------------------------------------------------------------
#       (environment A: mul/sum)  batched_dot(x, x)  |     7    |      36000      |      21
#       (environment B: bmm)      batched_dot(x, x)  |    14    |      40000      |      85
#
#     Times are in microseconds (us).
#

# And just to show that we can round trip all of the results from earlier:
round_tripped_results = pickle.loads(pickle.dumps(results))
assert(str(benchmark.Compare(results)) == str(benchmark.Compare(round_tripped_results)))


######################################################################
# 7. Generating inputs with `Fuzzed Parameters`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As we've seen in the previous section, there can be some stark
# performance differences depending on the input tensors. Hence, it
# is a good idea to run benchmarks on a number of different inputs.
# However, creating all these input tensors can be tedious which is
# where ``torch.utils.benchmark.Fuzzer`` and related classes come in.
# Let's take a look at how we can use the ``Fuzzer`` to create some test
# cases for the benchmark.
#

from torch.utils.benchmark import Fuzzer, FuzzedParameter, FuzzedTensor, ParameterAlias

# Generates random tensors with 128 to 10000000 elements and sizes k0 and k1 chosen from a
# ``loguniform`` distribution in [1, 10000], 40% of which will be discontiguous on average.
example_fuzzer = Fuzzer(
    parameters = [
        FuzzedParameter('k0', minval=1, maxval=10000, distribution='loguniform'),
        FuzzedParameter('k1', minval=1, maxval=10000, distribution='loguniform'),
    ],
    tensors = [
        FuzzedTensor('x', size=('k0', 'k1'), min_elements=128, max_elements=10000000, probability_contiguous=0.6)
    ],
    seed=0,
)

results = []
for tensors, tensor_params, params in example_fuzzer.take(10):
    # description is the column label
    sub_label=f"{params['k0']:<6} x {params['k1']:<4} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
    results.append(benchmark.Timer(
        stmt='batched_dot_mul_sum(x, x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='mul/sum',
    ).blocked_autorange(min_run_time=1))
    results.append(benchmark.Timer(
        stmt='batched_dot_bmm(x, x)',
        setup='from __main__ import batched_dot_bmm',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='bmm',
    ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.print()

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     [--------------------- Batched dot ---------------------]
#                                          |  mul/sum  |   bmm 
#     1 threads: ----------------------------------------------
#           725    x 257                   |      87   |    180
#           49     x 383                   |      15   |     30
#           34     x 1468                  |      30   |    118
#           187    x 5039                  |     400   |   1200
#           2140   x 1296 (discontiguous)  |    2000   |  41000
#           78     x 1598                  |      74   |    310
#           519    x 763                   |     190   |   1500
#           141    x 1082                  |      87   |    500
#           78     x 5    (discontiguous)  |       9   |     20
#           187    x 1                     |      12   |     10
#
#     Times are in microseconds (us). 
#

######################################################################
# There is a lot of flexibility for defining your own ``fuzzers`` which
# is great for creating a powerful set of inputs to benchmark. But to
# make things even simpler, PyTorch benchmark module comes with some
# built-in ``fuzzers`` for common benchmarking needs. Let's take a look at
# how we can use one of these built-in ``fuzzers``.
#

from torch.utils.benchmark.op_fuzzers import binary

results = []
for tensors, tensor_params, params in binary.BinaryOpFuzzer(seed=0).take(10):
    sub_label=f"{params['k0']:<6} x {params['k1']:<4} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
    results.append(benchmark.Timer(
        stmt='batched_dot_mul_sum(x, x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='mul/sum',
    ).blocked_autorange(min_run_time=1))
    results.append(benchmark.Timer(
        stmt='batched_dot_bmm(x, x)',
        setup='from __main__ import batched_dot_bmm',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='bmm',
    ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize(rowwise=True)
compare.print()

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     [----------------------- Batched dot ------------------------]
#                                              |  mul/sum  |   bmm  
#     1 threads: ---------------------------------------------------
#           64     x 473  (discontiguous)      |    10000  |   40000
#           16384  x 12642115 (discontiguous)  |       31  |      78
#           8192   x 892                       |     4800  |   20400
#           512    x 64   (discontiguous)      |   110000  |  400000
#           493    x 27   (discontiguous)      |     1100  |    2440
#           118    x 32   (discontiguous)      |      870  |    2030
#           16     x 495  (discontiguous)      |    23600  |   24000
#           488    x 62374                     |    90000  |  100000
#           240372 x 69                        |    40000  |   16000
#           40156  x 32   (discontiguous)      |     2670  |    5000
#    
#     Times are in microseconds (us).
#

######################################################################
# 8. Collecting instruction counts with ``Callgrind``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# One of the challenges of optimizing code is the variation and opacity of
# wall time. There are many sources of non-determinism, from adaptive clock
# speeds to resource contention with other processes. Furthermore, end-to-end
# time gives no insight into where time is being spent, which is really what
# we're interested in when optimizing code.
#
# A complementary approach is to also collect instruction counts. These counts
# are a proxy metric and do not capture all aspects of performance
# (e.g. memory or I/O bound tasks), however they do have several useful
# properties. Instruction counts are reproducible, insensitive to environmental
# variation, and offer fine grained insight into where a program is spending
# cycles.
#
# To see the utility of instruction counts, let us look at how we might
# reduce the overhead of `batched_dot_mul_sum`. The obvious solution is to
# move it to C++, so we avoid going between Python and C++ multiple times.
#
# Fortunately, the source is nearly identical. One question that we have to ask
# in C++ is whether we should take arguments by value or reference.
#

batched_dot_src = """\
/* ---- Python ---- */
// def batched_dot_mul_sum(a, b):
//     return a.mul(b).sum(-1)

torch::Tensor batched_dot_mul_sum_v0(
    const torch::Tensor a,
    const torch::Tensor b) {
  return a.mul(b).sum(-1);
}

torch::Tensor batched_dot_mul_sum_v1(
    const torch::Tensor& a,
    const torch::Tensor& b) {
  return a.mul(b).sum(-1);
}
"""


# PyTorch makes it easy to test our C++ implementations by providing a utility
# to JIT compile C++ source into Python extensions:
import os
from torch.utils import cpp_extension
cpp_lib = cpp_extension.load_inline(
    name='cpp_lib',
    cpp_sources=batched_dot_src,
    extra_cflags=['-O3'],
    extra_include_paths=[
        # `load_inline` needs to know where to find ``pybind11`` headers.
        os.path.join(os.getenv('CONDA_PREFIX'), 'include')
    ],
    functions=['batched_dot_mul_sum_v0', 'batched_dot_mul_sum_v1']
)

# `load_inline` will create a shared object that is loaded into Python. When we collect
# instruction counts Timer will create a subprocess, so we need to re-import it. The
# import process is slightly more complicated for C extensions, but that's all we're
# doing here.
module_import_str = f"""\
# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
import importlib.util
spec = importlib.util.spec_from_file_location("cpp_lib", {repr(cpp_lib.__file__)})
cpp_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp_lib)"""

import textwrap
def pretty_print(result):
    """Import machinery for ``cpp_lib.so`` can get repetitive to look at."""
    print(repr(result).replace(textwrap.indent(module_import_str, "  "), "  import cpp_lib"))


t_baseline = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='''\
from __main__ import batched_dot_mul_sum
x = torch.randn(2, 2)''')

t0 = benchmark.Timer(
    stmt='cpp_lib.batched_dot_mul_sum_v0(x, x)',
    setup=f'''\
{module_import_str}
x = torch.randn(2, 2)''')

t1 = benchmark.Timer(
    stmt='cpp_lib.batched_dot_mul_sum_v1(x, x)',
    setup=f'''\
{module_import_str}
x = torch.randn(2, 2)''')

# Moving to C++ did indeed reduce overhead, but it's hard to tell which
# calling convention is more efficient. v1 (call with references) seems to
# be a bit faster, but it's within measurement error.
pretty_print(t_baseline.blocked_autorange())
pretty_print(t0.blocked_autorange())
pretty_print(t1.blocked_autorange())

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb16935d2e8>
#     batched_dot_mul_sum(x, x)
#     setup:
#       from __main__ import batched_dot_mul_sum
#       x = torch.randn(2, 2)
#    
#       6.92 us
#       1 measurement, 100000 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb16935d2e8>
#     cpp_lib.batched_dot_mul_sum_v0(x, x)
#     setup:
#       import cpp_lib
#       x = torch.randn(2, 2)
#    
#       5.29 us
#       1 measurement, 100000 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb16935d2e8>
#     cpp_lib.batched_dot_mul_sum_v1(x, x)
#     setup:
#       import cpp_lib
#       x = torch.randn(2, 2)
#    
#       5.22 us
#       1 measurement, 100000 runs , 1 thread
#

# Let's use ``Callgrind`` to determine which is better.
stats_v0 = t0.collect_callgrind()
stats_v1 = t1.collect_callgrind()

pretty_print(stats_v0)
pretty_print(stats_v1)

# `.as_standardized` removes file names and some path prefixes, and makes
# it easier to read the function symbols.
stats_v0 = stats_v0.as_standardized()
stats_v1 = stats_v1.as_standardized()

# `.delta` diffs the instruction counts, and `.denoise` removes several
# functions in the Python interpreter that are known to have significant
# jitter.
delta = stats_v1.delta(stats_v0).denoise()

# `.transform` is a convenience API for transforming function names. It is
# useful for increasing cancelation when ``diff-ing`` instructions, as well as
# just generally improving readability.
replacements = (
    ("???:void pybind11", "pybind11"),
    ("batched_dot_mul_sum_v0", "batched_dot_mul_sum_v1"),
    ("at::Tensor, at::Tensor", "..."),
    ("at::Tensor const&, at::Tensor const&", "..."),
    ("auto torch::detail::wrap_pybind_function_impl_", "wrap_pybind_function_impl_"),
)
for before, after in replacements:
    delta = delta.transform(lambda l: l.replace(before, after))

# We can use print options to control how much of the function to display.
torch.set_printoptions(linewidth=160)

# Once parsed, the instruction counts make clear that passing `a` and `b`
# by reference is more efficient as it skips some ``c10::TensorImpl`` bookkeeping
# for the intermediate Tensors, and is also works better with ``pybind11``. This
# is consistent with our noisy wall time observations.
print(delta)

######################################################################
# .. code-block::
#
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.CallgrindStats object at 0x7fb0f06e7630>
#     cpp_lib.batched_dot_mul_sum_v0(x, x)
#     setup:
#       import cpp_lib
#       x = torch.randn(2, 2)
#                                All          Noisy symbols removed
#         Instructions:      2392671                    2392671
#         Baseline:             4367                       4367
#     100 runs per measurement, 1 thread
#     Warning: PyTorch was not built with debug symbols.
#              Source information may be limited. Rebuild with
#              REL_WITH_DEB_INFO=1 for more detailed results.
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.CallgrindStats object at 0x7fb10400d208>
#     cpp_lib.batched_dot_mul_sum_v1(x, x)
#     setup:
#       import cpp_lib
#       x = torch.randn(2, 2)
#                                All          Noisy symbols removed
#         Instructions:      2378978                    2378978
#         Baseline:             4367                       4367
#         100 runs per measurement, 1 thread
#         Warning: PyTorch was not built with debug symbols.
#                  Source information may be limited. Rebuild with
#                  REL_WITH_DEB_INFO=1 for more detailed results.
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7fb1000ab358>
#               86  ???:0x000000000020d9e0
#           56  ???:0x000000000020db10
#        -1100  pybind11::cpp_function::initialize<wrap_pybind_function_impl_<at::Tensor ... r (&)(...), std::integer_sequence<unsigned long, 0ul, 1ul>)::{lambda(...)
#        -1600  ???:wrap_pybind_function_impl_<at::Tensor (&)(...), 0ul, 1ul>(at::Tensor (&)(...), std::integer_sequence<unsigned long, 0ul, 1ul>)::{lambda(...)
#        -5200  ???:c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reset_()
#        -5935  ???:0x000000000022c0e0
#     Total: -13693
#


######################################################################
# Learn More
# ----------
#
# Take a look at these other recipes to continue your learning:
#
# -  `PyTorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler.html>`_
#
