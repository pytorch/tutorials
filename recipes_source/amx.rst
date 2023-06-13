==============================================
Leverage Intel® Advanced Matrix Extensions
==============================================

Introduction
============

Advanced Matrix Extensions (AMX), also known as Intel® Advanced Matrix Extensions (Intel® AMX), is an x86 extension,
which introduce two new components: a 2-dimensional register file called 'tiles' and an accelerator of Tile Matrix Multiplication (TMUL) that is able to operate on those tiles.
AMX is designed to work on matrices to accelerate deep-learning training and inference on the CPU and is ideal for workloads like natural-language processing, recommendation systems and image recognition.

Intel advances AI capabilities with 4th Gen Intel® Xeon® Scalable processors and Intel® AMX, delivering 3x to 10x higher inference and training performance versus the previous generation, see `Accelerate AI Workloads with Intel® AMX`_.
Compared to 3rd Gen Intel Xeon Scalable processors running Intel® Advanced Vector Extensions 512 Neural Network Instructions (Intel® AVX-512 VNNI),
4th Gen Intel Xeon Scalable processors running Intel AMX can perform 2,048 INT8 operations per cycle, rather than 256 INT8 operations per cycle. They can also perform 1,024 BF16 operations per cycle, as compared to 64 FP32 operations per cycle, see page 4 of `Accelerate AI Workloads with Intel® AMX`_.
For more detailed information of AMX, see `Intel® AMX Overview`_.


AMX in PyTorch
==============

PyTorch leverages AMX for computing intensive operators with BFloat16 and quantization with INT8 by its backend oneDNN
to get higher performance out-of-box on x86 CPUs with AMX support.
For more detailed information of oneDNN, see `oneDNN`_.

The operation is fully handled by oneDNN according to the execution code path generated. For example, when a supported operation gets executed into oneDNN implementation on a hardware platform with AMX support, AMX instructions will be invoked automatically inside oneDNN.
Since oneDNN is the default acceleration library for PyTorch CPU, no manual operations are required to enable the AMX support.

Guidelines of leveraging AMX with workloads
-------------------------------------------

This section provides guidelines on how to leverage AMX with various workloads.

- BFloat16 data type: 

  - Using ``torch.cpu.amp`` or ``torch.autocast("cpu")`` would utilize AMX acceleration for supported operators.

   ::

      model = model.to(memory_format=torch.channels_last)
      with torch.cpu.amp.autocast():
         output = model(input)

.. note:: Use ``torch.channels_last`` memory format to get better performance. 

- Quantization:

  - Applying quantization would utilize AMX acceleration for supported operators.

- torch.compile:

  - When the generated graph model runs into oneDNN implementations with the supported operators, AMX accelerations will be activated.

.. note:: When using PyTorch on CPUs that support AMX, the framework will automatically enable AMX usage by default. This means that PyTorch will attempt to leverage the AMX feature whenever possible to speed up matrix multiplication operations. However, it's important to note that the decision to dispatch to the AMX kernel ultimately depends on the internal optimization strategy of the oneDNN library and the quantization backend, which PyTorch relies on for performance enhancements. The specific details of how AMX utilization is handled internally by PyTorch and the oneDNN library may be subject to change with updates and improvements to the framework.


CPU operators that can leverage AMX:
------------------------------------

BF16 CPU ops that can leverage AMX:

- ``conv1d``
- ``conv2d``
- ``conv3d``
- ``conv_transpose1d``
- ``conv_transpose2d``
- ``conv_transpose3d``
- ``bmm``
- ``mm``
- ``baddbmm``
- ``addmm``
- ``addbmm``
- ``linear``
- ``matmul``

Quantization CPU ops that can leverage AMX:

- ``conv1d``
- ``conv2d``
- ``conv3d``
- ``conv_transpose1d``
- ``conv_transpose2d``
- ``conv_transpose3d``
- ``linear``



Confirm AMX is being utilized
------------------------------

Set environment variable ``export ONEDNN_VERBOSE=1``, or use ``torch.backends.mkldnn.verbose`` to enable oneDNN to dump verbose messages.

::

   with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
       with torch.cpu.amp.autocast():
           model(input)

For example, get oneDNN verbose:

::

   onednn_verbose,info,oneDNN v2.7.3 (commit 6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e)
   onednn_verbose,info,cpu,runtime:OpenMP,nthr:128
   onednn_verbose,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with bfloat16 and 8-bit integer support
   onednn_verbose,info,gpu,runtime:none
   onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
   onednn_verbose,exec,cpu,reorder,simple:any,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,attr-scratchpad:user ,,2,5.2561
   ...
   onednn_verbose,exec,cpu,convolution,jit:avx512_core_amx_bf16,forward_training,src_bf16::blocked:acdb:f0 wei_bf16:p:blocked:ABcd16b16a2b:f0 bia_f32::blocked:a:f0 dst_bf16::blocked:acdb:f0,attr-scratchpad:user ,alg:convolution_direct,mb7_ic2oc1_ih224oh111kh3sh2dh1ph1_iw224ow111kw3sw2dw1pw1,0.628906
   ...
   onednn_verbose,exec,cpu,matmul,brg:avx512_core_amx_int8,undef,src_s8::blocked:ab:f0 wei_s8:p:blocked:BA16a64b4a:f0 dst_s8::blocked:ab:f0,attr-scratchpad:user ,,1x30522:30522x768:1x768,7.66382
   ...

If you get the verbose of ``avx512_core_amx_bf16`` for BFloat16 or ``avx512_core_amx_int8`` for quantization with INT8, it indicates that AMX is activated.


Conclusion
----------


In this tutorial, we briefly introduced AMX, how to utilize AMX in PyTorch to accelerate workloads, and how to confirm that AMX is being utilized.

With the improvements and updates of PyTorch and oneDNN, the utilization of AMX may be subject to change accordingly.

As always, if you run into any problems or have any questions, you can use
`forum <https://discuss.pytorch.org/>`_ or `GitHub issues
<https://github.com/pytorch/pytorch/issues>`_ to get in touch. 


.. _Accelerate AI Workloads with Intel® AMX: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/ai-solution-brief.html

.. _Intel® AMX Overview: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html

.. _oneDNN: https://oneapi-src.github.io/oneDNN/index.html
