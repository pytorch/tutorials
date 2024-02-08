"""
(prototype) GPU Quantization with TorchAO
======================================================

**Author**: `HDCharles <https://github.com/HDCharles>`_

In this tutorial, we will walk you through the quantization and optimization
of the popular `segment anything model <https://github.com/facebookresearch/segment-anything>`_. These
steps will mimic some of those taken to develop the
`segment-anything-fast <https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/modeling/image_encoder.py#L15>`_
repo. This step-by-step guide demonstrates how you can
apply these techniques to speed up your own models, especially those
that use transformers. To that end, we will focus on widely applicable
techniques, such as optimizing performance with ``torch.compile`` and
quantization and measure their impact.

"""


######################################################################
# Set up Your Environment
# --------------------------------
#
# First, let's configure your environment. This guide was written for CUDA 12.1.
# We have run this tutorial on an A100-PG509-200 power limited to 330.00 W. If you
# are using a different hardware, you might see different performance numbers.
#
#
# .. code-block:: bash
#
#    > conda create -n myenv python=3.10
#    > pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
#    > pip install git+https://github.com/facebookresearch/segment-anything.git
#    > pip install git+https://github.com/pytorch-labs/ao.git
#
# Segment Anything Model checkpoint setup:
#
# 1. Go to the `segment-anything repo <checkpoint https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints>`_ and download the ``vit_h`` checkpoint. Alternatively, you can just use ``wget``: `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth --directory-prefix=<path>
# 2. Pass in that directory by editing the code below to say:
#
# .. code-block::
#
# {sam_checkpoint_base_path}=<path>
#
# This was run on an A100-PG509-200 power limited to 330.00 W
#

import torch
from torchao.quantization import change_linear_weights_to_int8_dqtensors
from segment_anything import sam_model_registry
from torch.utils.benchmark import Timer

sam_checkpoint_base_path = "data"
model_type = 'vit_h'
model_name = 'sam_vit_h_4b8939.pth'
checkpoint_path = f"{sam_checkpoint_base_path}/{model_name}"
batchsize = 16
only_one_block = True


@torch.no_grad()
def benchmark(f, *args, **kwargs):
    for _ in range(3):
        f(*args, **kwargs)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    res = t0.adaptive_autorange(.03, min_run_time=.2, max_run_time=20)
    return {'time':res.median * 1e3, 'memory': torch.cuda.max_memory_allocated()/1e9}

def get_sam_model(only_one_block=False, batchsize=1):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).cuda()
    model = sam.image_encoder.eval()
    image = torch.randn(batchsize, 3, 1024, 1024, device='cuda')

    # code to use just a single block of the model
    if only_one_block:
        model = model.blocks[0]
        image = torch.randn(batchsize, 64, 64, 1280, device='cuda')
    return model, image


######################################################################
# In this tutorial, we focus on quantizing the ``image_encoder`` because the
# inputs to it are statically sized while the prompt encoder and mask
# decoder have variable sizes which makes them harder to quantize.
#
# We’ll focus on just a single block at first to make the analysis easier.
#
# Let's start by measuring the baseline runtime.

try:
    model, image = get_sam_model(only_one_block, batchsize)
    fp32_res = benchmark(model, image)
    print(f"base fp32 runtime of the model is {fp32_res['time']:0.2f}ms and peak memory {fp32_res['memory']:0.2f}GB")
    # base fp32 runtime of the model is 186.16ms and peak memory 6.33GB
except Exception as e:
    print("unable to run fp32 model: ", e)



######################################################################
# We can achieve an instant performance boost by converting the model to bfloat16.
# The reason we opt for bfloat16 over fp16 is due to its dynamic range, which is comparable to
# that of fp32. Both bfloat16 and fp32 possess 8 exponential bits, whereas fp16 only has 4. This
# larger dynamic range helps protect us from overflow errors and other issues that can arise
# when scaling and rescaling tensors due to quantization.
#

model, image = get_sam_model(only_one_block, batchsize)
model = model.to(torch.bfloat16)
image = image.to(torch.bfloat16)
bf16_res = benchmark(model, image)
print(f"bf16 runtime of the block is {bf16_res['time']:0.2f}ms and peak memory {bf16_res['memory']: 0.2f}GB")
# bf16 runtime of the block is 25.43ms and peak memory  3.17GB


######################################################################
# Just this quick change improves runtime by a factor of ~7x in the tests we have
# conducted (186.16ms to 25.43ms).
#
# Next, let's use ``torch.compile`` with our model to see how much the performance
# improves.
#

model_c = torch.compile(model, mode='max-autotune')
comp_res = benchmark(model_c, image)
print(f"bf16 compiled runtime of the block is {comp_res['time']:0.2f}ms and peak memory {comp_res['memory']: 0.2f}GB")
# bf16 compiled runtime of the block is 19.95ms and peak memory  2.24GB


######################################################################
# The first time this is run, you should see a sequence of ``AUTOTUNE``
# outputs which occurs when inductor compares the performance between
# various kernel parameters for a kernel. This only happens once (unless
# you delete your cache) so if you run the cell again you should just get
# the benchmark output.
#
# ``torch.compile`` yields about another 27% improvement. This brings the
# model to a reasonable baseline where we now have to work a bit harder
# for improvements.
#
# Next, let's apply quantization. Quantization for GPUs comes in three main forms
# in `torchao <https://github.com/pytorch-labs/ao>`_ which is just native
# pytorch+python code. This includes:
#
# * int8 dynamic quantization
# * int8 weight-only quantization
# * int4 weight-only quantization
#
# Different models, or sometimes different layers in a model can require different techniques.
# For models which are heavily compute bound, dynamic quantization tends
# to work the best since it swaps the normal expensive floating point
# matmul ops with integer versions. Weight-only quantization works better
# in memory bound situations where the benefit comes from loading less
# weight data, rather than doing less computation. The torchao APIs:
#
# ``change_linear_weights_to_int8_dqtensors``,
# ``change_linear_weights_to_int8_woqtensors`` or
# ``change_linear_weights_to_int4_woqtensors``
#
# can be used to easily apply the desired quantization technique and then
# once the model is compiled with ``torch.compile`` with ``max-autotune``, quantization is
# complete and we can see our speedup.
#
# .. note::
#    You might experience issues with these on older versions of PyTorch. If you run
#    into an issue, you can use ``apply_dynamic_quant`` and
#    ``apply_weight_only_int8_quant`` instead as drop in replacement for the two
#    above (no replacement for int4).
#
#  The difference between the two APIs is that ``change_linear_weights`` API
# alters the weight tensor of the linear module so instead of doing a
# normal linear, it does a quantized operation. This is helpful when you
# have non-standard linear ops that do more than one thing. The ``apply``
# APIs directly swap the linear modules for a quantized module which
# works on older versions but doesn’t work with non-standard linear
# modules.
#
# In this case Segment Anything is compute-bound so we’ll use dynamic quantization:
#

del model_c, model, image
model, image = get_sam_model(only_one_block, batchsize)
model = model.to(torch.bfloat16)
image = image.to(torch.bfloat16)
change_linear_weights_to_int8_dqtensors(model)
model_c = torch.compile(model, mode='max-autotune')
quant_res = benchmark(model_c, image)
print(f"bf16 compiled runtime of the quantized block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB")
# bf16 compiled runtime of the quantized block is 19.04ms and peak memory  3.58GB


######################################################################
# With quantization, we have improved performance a bit more but memory usage increased
# significantly.
#
# This is for two reasons:
#
# 1) Quantization adds overhead to the model
#    since we need to quantize and dequantize the input and output. For small
#    batch sizes this overhead can actually make the model go slower.
# 2) Even though we are doing a quantized matmul, such as ``int8 x int8``,
#    the result of the multiplication gets stored in an int32 tensor
#    which is twice the size of the result from the non-quantized model.
#    If we can avoid creating this int32 tensor, our memory usage will improve a lot.
#
# We can fix #2 by fusing the integer matmul with the subsequent rescale
# operation since the final output will be bf16, if we immediately convert
# the int32 tensor to bf16 and instead store that we’ll get better
# performance in terms of both runtime and memory.
#
# The way to do this, is to enable the option
# ``force_fuse_int_mm_with_mul`` in the inductor config.
#

del model_c, model, image
model, image = get_sam_model(only_one_block, batchsize)
model = model.to(torch.bfloat16)
image = image.to(torch.bfloat16)
torch._inductor.config.force_fuse_int_mm_with_mul = True
change_linear_weights_to_int8_dqtensors(model)
model_c = torch.compile(model, mode='max-autotune')
quant_res = benchmark(model_c, image)
print(f"bf16 compiled runtime of the fused quantized block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB")
# bf16 compiled runtime of the fused quantized block is 18.78ms and peak memory  2.37GB


######################################################################
# The fusion improves performance by another small bit (about 6% over the
# baseline in total) and removes almost all the memory increase, the
# remaining amount (2.37GB quantized vs 2.24GB unquantized) is due to
# quantization overhead which cannot be helped.
#
# We’re still not done though, we can apply a few general purpose
# optimizations to get our final best-case performance.
#
# 1) We can sometimes improve performance by disabling epilogue fusion
#    since the autotuning process can be confused by fusions and choose
#    bad kernel parameters.
# 2) We can apply coordinate descent tuning in all directions to enlarge
#    the search area for kernel parameters.
#

del model_c, model, image
model, image = get_sam_model(only_one_block, batchsize)
model = model.to(torch.bfloat16)
image = image.to(torch.bfloat16)
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
change_linear_weights_to_int8_dqtensors(model)
model_c = torch.compile(model, mode='max-autotune')
quant_res = benchmark(model_c, image)
print(f"bf16 compiled runtime of the final quantized block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB")
# bf16 compiled runtime of the final quantized block is 18.16ms and peak memory  2.39GB


######################################################################
# As you can see, we’ve squeezed another small improvement from the model,
# taking our total improvement to over 10x compared to our original. To
# get a final estimate of the impact of quantization lets do an apples to
# apples comparison on the full model since the actual improvement will
# differ block by block depending on the shapes involved.
#

try:
    del model_c, model, image
    model, image = get_sam_model(False, batchsize)
    model = model.to(torch.bfloat16)
    image = image.to(torch.bfloat16)
    model_c = torch.compile(model, mode='max-autotune')
    quant_res = benchmark(model_c, image)
    print(f"bf16 compiled runtime of the compiled full model is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB")
    # bf16 compiled runtime of the compiled full model is 729.65ms and peak memory  23.96GB

    del model_c, model, image
    model, image = get_sam_model(False, batchsize)
    model = model.to(torch.bfloat16)
    image = image.to(torch.bfloat16)
    change_linear_weights_to_int8_dqtensors(model)
    model_c = torch.compile(model, mode='max-autotune')
    quant_res = benchmark(model_c, image)
    print(f"bf16 compiled runtime of the quantized full model is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB")
    # bf16 compiled runtime of the quantized full model is 677.28ms and peak memory  24.93GB
except Exception as e:
    print("unable to run full model: ", e)



######################################################################
# Conclusion
# -----------------
# In this tutorial, we have learned about the quantization and optimization techniques
# on the example of the segment anything model.

# In the end, we achieved a full-model apples to apples quantization speedup
# of about 7.7% on batch size 16 (677.28ms to 729.65ms). We can push this a
# bit further by increasing the batch size and optimizing other parts of
# the model. For example, this can be done with some form of flash attention.
#
# For more information visit
# `torchao <https://github.com/pytorch-labs/ao>`_ and try it on your own
# models.
#
