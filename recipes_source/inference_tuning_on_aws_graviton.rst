(Beta) PyTorch Inference Performance Tuning on AWS Graviton Processors
======================================================================

**Author**: `Sunita Nadampalli <https://github.com/snadampal>`_

`AWS Graviton <https://aws.amazon.com/ec2/graviton/>`_ is a series of ARM-based processors designed by AWS. AWS Graviton3 processors are optimized for Machine Learning (ML) workloads, including support for ``bfloat16``, Scalable Vector Extension (SVE) and twice the Single Instruction Multiple Data (SIMD) bandwidth compared to Graviton2.

PyTorch provides native reference ATen kernels for the machine learning operators like convolutions, matmul, relu, etc. These operators can be accelerated with platform specific kernel implementations from Basic Linear Algebra (BLAS) libraries. On AWS Graviton CPUs, MKLDNN with Arm Compute Library (`ACL <https://github.com/ARM-software/ComputeLibrary>`_) and `OpenBLAS <https://github.com/OpenMathLib/OpenBLAS>`_ libraries provide optimized implementations for a subset of the operators. Both these libraries are integrated into PyTorch with PyTorch 2.0 version.

In this tutorial we will cover how to achieve the best inference performance for linear layer neural network on AWS Graviton3 CPUs (`AWS c7g instance <https://aws.amazon.com/ec2/instance-types/c7g/>`_) with ``bfloa16`` kernels and with the right backend selection.

Contents
--------
1. Basic Usage
2. Speed up inference with Bfloat16 fast math kernels
3. Improve inference performance with OpenBLAS for smaller batch dimensions
4. Optimize memory allocation overhead with Linux Transparent huge pages
5. Conclusion

.. note::
   To successfully run this tutorial and reproduce the speedup numbers shown below, you need an instance from the Graviton3 family (``c7g/r7g/m7g``) of hardware. For this tutorial, we used the `c7g.xl (4vcpu) instance <https://aws.amazon.com/ec2/instance-types/c7g/>`_ .

Basic Usage
---------------

PyTorch natively supports AWS Graviton3 optimizations starting with PyTorch 2.0 version.
Please refer to this `blog <https://pytorch.org/blog/optimized-pytorch-w-graviton/>`_ for more details on the optimizations.

1. Install PyTorch by running the following command:

   .. code-block::

      python3 -m pip install torch

2. We will start by importing the required dependencies and defining the device will run on:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.profiler import profile, record_function, ProfilerActivity

    # AWS Graviton3 cpu
    device = ("cpu")
    print(f"Using {device} device")


3. Given linear layers are at the heart of several neural networks, including transformers, we take a linear layer for this demo. We define our neural network by subclassing ``nn.Module``, and initializing the layers in ``__init__``. We construct the network with a typical large language model parameters to match the real world scenario:

.. code-block:: python

  class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 11008),
            nn.ReLU(),
            nn.Linear(11008, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

4. Let's create an instance of ``MyNeuralNetwork``, and move it to the device:

.. code-block:: python

    model = MyNeuralNetwork().to(device)
    print(model)

Next, let's get the prediction probabilities by passing them through an instance of the ``nn.Softmax`` module:

.. code-block:: python

    X = torch.rand(1, 64, 64, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

output:

.. code-block::

    Predicted class: tensor([2])

Our network functionality is verified. Next, we will profile the performance. Lets' check two different scenarios: small and large batch dimensions.

**Scenario 1:** A larger batch dimension, for example 256:

.. code-block:: python

    # warm it up first and loop over multiple times to have enough execution time

    X = torch.rand(256, 64, 64, device=device)

    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #Warmup
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


Following is the profiler output with the default PyTorch configuration:

.. table::
   :widths: auto

   ======================  ============   ===========  =============  ===========  ============  ============
                  Name      Self CPU %      Self CPU    CPU total %    CPU total   CPU time avg    # of Calls
   ======================  ============   ===========  =============  ===========  ============  ============
           aten::addmm        97.61%         15.813s        98.61%       15.977s      53.255ms           300
       aten::clamp_min         1.09%       177.032ms         1.09%     177.032ms     885.160us           200
            aten::copy         1.00%       162.054ms         1.00%     162.054ms     540.180us           300
     mymodel_inference         0.22%        35.738ms       100.00%       16.201s       16.201s             1
          aten::linear         0.02%         2.955ms        98.66%       15.985s      53.282ms           300
               aten::t         0.01%         2.421ms         0.03%       5.043ms      16.810us           300
            aten::relu         0.01%         2.356ms         1.11%     179.388ms     896.940us           200
   ======================  ============   ===========  =============  ===========  ============  ============

**Self CPU time total:** 16.201s


Speed up Inference with ``bfloat16`` Fast Math Kernels
----------------------------------------------------------

AWS Graviton3 processors support `bfloat16 MMLA instructions <https://developer.arm.com/documentation/ddi0596/2020-12/SVE-Instructions/BFMMLA--BFloat16-floating-point-matrix-multiply-accumulate->`_. Arm Compute Library (`ACL <https://github.com/ARM-software/ComputeLibrary>`_) provides optimized ``bfloat16`` General Matrix Multiplication (GEMM) kernels for AWS Graviton processors, and are integrated into PyTorch via MKLDNN backend starting with PyTorch 2.0.  The inference performance can be optimized with the fast math GEMM kernels. The fast math mode is not enabled by default because these kernels perform GEMM in ``bfloat16`` precision instead of ``float``, and hence results in a slight drop in the model inference accuracy. However, the accuracy drop is within the ``cosine similarity`` threshold defined for ``bfloat16`` backend in ``torchbench`` test suite, and hence acceptable for majority of the applications. To enable the fast math GEMM kernels, set the following environment variable:

.. code-block:: bash

    $ export DNNL_DEFAULT_FPMATH_MODE=BF16


When you run the above inference script, you should see the following profiler output with the MKLDNN fast math mode enabled:

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  ============  ============
                  Name      Self CPU %     Self CPU    CPU total %     CPU total   CPU time avg    # of Calls
   ======================  ============  ============  ============  ============  ============  ============
           aten::addmm        95.61%        6.943s        97.10%        7.052s      23.507ms           300
       aten::clamp_min         2.31%     167.653ms         2.31%     167.653ms     838.265us           200
            aten::copy         1.48%     107.593ms         1.48%     107.593ms     358.643us           300
     mymodel_inference         0.43%      31.167ms       100.00%        7.262s        7.262s             1
          aten::linear         0.04%       2.911ms        97.21%        7.060s      23.533ms           300
               aten::t         0.03%       2.414ms         0.07%       4.892ms      16.307us           300
            aten::relu         0.03%       2.281ms         2.34%     169.934ms     849.670us           200
   ======================  ============  ============  ============  ============  ============  ============

**Self CPU time total:** 7.262s


This is around ``2x (7.262s vs 16.201s)`` performance improvement with the ``bfloat16`` fastmath kernels. Next, let’s look at the smaller batch dimension scenario.

**Scenario 2:** A smaller batch dimension, for example, 32:

.. code-block:: python

    X = torch.rand(32, 64, 64, device=device)
    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #Warmup
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


You should see the following profiler output when the above script is run with the PyTorch default configuration:

.. table::
   :widths: auto

   ======================  =============  ============  ============  ============  ============  ============
                     Name    Self CPU %      Self CPU   CPU total %     CPU total   CPU time avg    # of Calls
   ======================  =============  ============  ============  ============  ============  ============
           aten::addmm        95.51%         5.821s        97.04%        5.914s      19.713ms           300
       aten::clamp_min         2.33%      142.244ms         2.33%     142.244ms     711.220us           200
            aten::copy         1.51%       92.322ms         1.51%      92.322ms     307.740us           300
     mymodel_inference         0.45%       27.713ms       100.00%        6.094s        6.094s             1
          aten::linear         0.04%        2.495ms        97.16%        5.921s      19.736ms           300
               aten::t         0.03%        2.131ms         0.07%       4.441ms      14.803us           300
            aten::relu         0.03%        1.942ms         2.37%     144.186ms     720.930us           200
   ======================  =============  ============  ============  ============  ============  ============

**Self CPU time total:** 6.094s


The following output is the profiler output when run with the MKLDNN fast math mode enabled:

.. code-block:: bash

   $ export DNNL_DEFAULT_FPMATH_MODE=BF16

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  ============   =============
                   Name     Self CPU %      Self CPU    CPU total %   CPU total    CPU time avg    # of Calls
   ======================  ============  ============  ============  ============  ============   =============
           aten::addmm        93.31%        3.848s        95.66%        3.944s      13.148ms           300
       aten::clamp_min         3.43%     141.309ms         3.43%     141.309ms     706.545us           200
            aten::copy         2.33%      95.916ms         2.33%      95.916ms     319.720us           300
     mymodel_inference         0.67%      27.431ms       100.00%        4.123s        4.123s             1
          aten::linear         0.06%       2.471ms        95.83%        3.951s      13.170ms           300
               aten::t         0.05%       2.027ms         0.10%       4.243ms      14.143us           300
            aten::relu         0.05%       1.928ms         3.47%     143.237ms     716.185us           200
   ======================  ============  ============  ============  ============  ============   =============

**Self CPU time total:** 4.123s

The MKLDNN fast math mode yields approximately a **1.47x  (4.123s vs 6.094s)**  performance improvement for smaller batch dimensions. Although this improvement is noteworthy, the overall performance still leaves room for improvement. This is because of the runtime overhead (weights reorders and kernel launch time) from oneDNN and ACL backend outweighing the compute benefits from the ACL GEMM kernels for the smaller batch compute.


Improve Inference Performance with OpenBLAS for Smaller Batch Dimensions
------------------------------------------------------------------------

The inference performance for smaller batch dimensions can be improved by offloading the smaller shapes from MKLDNN to OpenBLAS backend. We are working on making the backend selection automatic, with robust heuristics, for the future releases. Till the heuristics are implemented, the smaller shapes can be offloaded to OpenBLAS by increasing the threshold for MKLDNN backend selection. In the following example, we use ``64`` as the threshold, so that input with ``batch dimension of 32`` is not dispatched to MKLDNN. Instead, it is dispatched to OpenBLAS.

.. code-block:: bash

   $ export TORCH_MKLDNN_MATMUL_MIN_DIM=64

Here is the profiler output with OpenBLAS backend:

.. table::
   :widths: auto

   ======================  ============  ============  ============  =============  ============  =============
                     Name    Self CPU %      Self CPU   CPU total %     CPU total   CPU time avg    # of Calls
   ======================  ============  ============  ============  =============  ============  =============
           aten::addmm        96.25%        1.958s        97.51%        1.984s        6.612ms           300
       aten::clamp_min         1.28%      26.124ms         1.28%      26.124ms      130.620us           200
            aten::copy         1.23%      24.951ms         1.23%      24.951ms       83.170us           300
     mymodel_inference         0.86%      17.423ms       100.00%        2.034s         2.034s             1
          aten::linear         0.08%       1.691ms        97.74%        1.988s        6.628ms           300
               aten::t         0.07%       1.520ms         0.14%       2.945ms        9.817us           300
            aten::relu         0.06%       1.258ms         1.35%      27.382ms      136.910us           200
   ======================  ============  ============  ============  =============  ============  =============

**Self CPU time total:** 2.034s


As you can see above, switching to OpenBLAS doubled the performance **(2.034s vs 4.123s)** compared to the default MKLDNN backend configuration. This becomes significant for even smaller batch dimensions, for example, for a batch dimension of 10:

.. code-block:: python

    X = torch.rand(10, 64, 64, device=device)
    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #Warmup
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


The following is the profiler output with MKLDNN fast math mode:

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  =============  =============
                     Name    Self CPU %      Self CPU   CPU total %     CPU total   CPU time avg    # of Calls
   ======================  ============  ============  ============  ============  =============  =============
           aten::addmm        87.81%        3.613s        91.90%        3.781s      12.604ms           300
       aten::clamp_min         7.18%     295.437ms         7.18%     295.437ms       1.477ms           200
            aten::copy         4.07%     167.516ms         4.07%     167.516ms     558.387us           300
     mymodel_inference         0.67%      27.708ms       100.00%        4.115s        4.115s             1
          aten::linear         0.06%       2.499ms        92.06%        3.788s      12.627ms           300
               aten::t         0.05%       1.982ms         0.11%       4.385ms      14.617us           300
            aten::relu         0.05%       1.932ms         7.23%     297.369ms       1.487ms           200
   ======================  ============  ============  ============  ============  =============  =============

**Self CPU time total:** 4.115s


and the following is the profiler output with the OpenBLAS backend:

.. code-block:: bash

   $ export TORCH_MKLDNN_MATMUL_MIN_DIM=64

.. table::
   :widths: auto

   ======================  =============  ============  ============  ============  =============  ============
                   Name     Self CPU %      Self CPU     CPU total %   CPU total    CPU time avg    # of Calls
   ======================  =============  ============  ============  ============  =============  ============
           aten::addmm        92.66%        1.179s        95.23%        1.211s         4.038ms           300
       aten::clamp_min         2.83%      36.060ms         2.83%      36.060ms       180.300us           200
            aten::copy         2.52%      32.013ms         2.52%      32.013ms       106.710us           300
     mymodel_inference         1.38%      17.521ms       100.00%        1.272s          1.272s             1
          aten::linear         0.14%       1.750ms        95.60%        1.216s         4.054ms           300
               aten::t         0.12%       1.475ms         0.24%       3.033ms        10.110us           300
            aten::relu         0.10%       1.285ms         2.94%      37.345ms       186.725us           200
   ======================  =============  ============  ============  ============  =============  ============

**Self CPU time total:** 1.272s


Here we observed **3.2x (1.272s vs 4.115s)** performance improvement by tuning the backend thresholds appropriately.


Optimize Memory Allocation Overhead with Linux Transparent Huge Pages (THP)
---------------------------------------------------------------------------

We also observed that for these larger networks, tensor memory allocations take significant portion of the inference latency. This can be optimized by enabling Linux transparent huge page allocations from PyTorch C10 memory allocator. Currently the feature is not enabled by default because it will increase the memory footprint marginally. Set the following environment variable to enable it:

.. code-block:: bash

    $ export THP_MEM_ALLOC_ENABLE=1

For the batch dimension of 256 and with MKLDNN fast math mode:

.. code-block:: python

    X = torch.rand(256, 64, 64, device=device)
    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #Warmup
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


The following is the profiler output with THP memory allocations enabled:

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  ==============  ============
                     Name   Self CPU %    Self CPU     CPU total %    CPU total     CPU time avg    # of Calls
   ======================  ============  ============  ============  ============  ==============  ============
           aten::addmm        91.31%        6.115s        94.39%        6.321s      21.069ms           300
       aten::clamp_min         4.82%     322.568ms         4.82%     322.568ms       1.613ms           200
            aten::copy         3.06%     204.602ms         3.06%     204.602ms     682.007us           300
     mymodel_inference         0.61%      40.777ms       100.00%        6.697s        6.697s             1
          aten::linear         0.05%       3.082ms        94.51%        6.329s      21.097ms           300
            aten::relu         0.04%       2.547ms         4.85%     325.115ms       1.626ms           200
   ======================  ============  ============  ============  ============  ==============  ============

**Self CPU time total:** 6.697s

This is an additional **1.08x or 8% (6.697s vs 7.262s)** improvement on top of the already optimized MKLDNN fast math mode measured above.


Conclusion
------------

In this tutorial, we covered PyTorch inference on AWS Graviton3 instances by covering the basic usage, demonstrating speedups with fast math kernels, comparing different backends for different batch dimensions, and how to optimize tensor memory allocation latencies with Linux transparent huge pages. The recommendation is to use MKLDNN backend with Bfloat16 fastmath mode and THP memory allocations for larger tensor shapes and to use OpenBLAS backend for smaller tensor shapes. We hope that you will give it a try!
