==============================================
Leverage Advanced Matrix Extensions
==============================================

Introduction
============

Advanced Matrix Extensions (AMX), also known as Intel® Advanced Matrix Extensions (Intel® AMX), is an extension to the x86 instruction set architecture (ISA).
Intel advances AI capabilities with 4th Gen Intel® Xeon® Scalable processors and Intel® AMX, delivering 3x to 10x higher inference and training performance versus the previous generation, see `Accelerate AI Workloads with Intel® AMX`_.
AMX supports two data types, INT8 and BFloat16, compared to AVX512 FP32, it can achieve up to 32x and 16x acceleration, respectively, see figure 6 of `Accelerate AI Workloads with Intel® AMX`_.
For more detailed information of AMX, see `Intel® AMX Overview`_.


AMX in PyTorch
==============

PyTorch leverages AMX for computing intensive operators with BFloat16 and quantization with INT8 by its backend oneDNN
to get higher performance out-of-box on x86 CPUs with AMX support.
For more detailed information of oneDNN, see `oneDNN`_.

The operation is fully handled by oneDNN according to the execution code path generated. I.e. when a supported operation gets executed into oneDNN implementation on a hardware platform with AMX support, AMX instructions will be invoked automatically inside oneDNN.
Since oneDNN is the default acceleration library for CPU, no manual operations are required to enable the AMX support.

- BF16 CPU ops that can leverage AMX:

``conv1d``,
``conv2d``,
``conv3d``,
``conv_transpose1d``,
``conv_transpose2d``,
``conv_transpose3d``,
``bmm``,
``mm``,
``baddbmm``,
``addmm``,
``addbmm``,
``linear``,
``matmul``,
``_convolution``

- Quantization CPU ops that can leverage AMX:

``conv1d``,
``conv2d``,
``conv3d``,
``conv1d``,
``conv2d``,
``conv3d``,
``conv_transpose1d``,
``conv_transpose2d``,
``conv_transpose3d``,
``linear``

Note: For quantized linear, whether to leverage AMX depends on the policy of the quantization backend.

Guidelines of leveraging AMX with workloads
--------------------------------------------------

- BFloat16 data type: 

Using ``torch.cpu.amp`` or ``torch.autocast("cpu")`` would utilize AMX acceleration.

::

   model = model.to(memory_format=torch.channels_last)
   with torch.cpu.amp.autocast():
       output = model(input)

Note: Use channels last format to get better performance. 

- quantization:

Applying quantization would utilize AMX acceleration.

- torch.compile:

When the generated graph model runs into oneDNN implementations with the supported operators mentioned in lists above, AMX accelerations will be activated.


Confirm AMX is being utilized
------------------------------

Set environment variable ``export ONEDNN_VERBOSE=1`` to get oneDNN verbose at runtime.

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

If we get the verbose of ``avx512_core_amx_bf16`` for BFloat16 or ``avx512_core_amx_int8`` for quantization with INT8, it indicates that AMX is activated.

.. _Accelerate AI Workloads with Intel® AMX: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/ai-solution-brief.html

.. _Intel® AMX Overview: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html

.. _oneDNN: https://oneapi-src.github.io/oneDNN/index.html
