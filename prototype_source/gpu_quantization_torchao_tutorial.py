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
that use transformers. To that end we will focus on widely applicable
techniques, such as optimizing performance with ``torch.compile`` and
quantization and measure their impact.

"""


######################################################################
# env setup (assumes cuda 12.1)
#
# ::
#
#    > conda create -n myenv python=3.10
#    > pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
#    > pip install git+https://github.com/facebookresearch/segment-anything.git
#    > pip install git+https://github.com/pytorch-labs/ao.git
#
#    SAM checkpoint setup:
#    1. go here and download the vit_h checkpoint https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints
#    or just `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth --directory-prefix=<path>
#    2. pass in that directory via {sam_checkpoint_base_path}=<path>
#
# This was run on an A100-PG509-200 power limited to 330.00 W
#

import torch
from torchao.quantization import change_linear_weights_to_int8_dqtensors
from segment_anything import sam_model_registry

sam_checkpoint_base_path = "~/local/models"
model_type = 'vit_h'
model_name = 'sam_vit_h_4b8939.pth'
checkpoint_path = f"{sam_checkpoint_base_path}/{model_name}"
batchsize = 16
only_one_block = True

from torch.utils.benchmark import Timer

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
# In this tutorial we focus on quantizing the image_encoder because the
# inputs to it are statically sized while the prompt encoder and mask
# decoder have variable sizes which makes them harder to quantize
#
# We’ll focus on just a single block at first to make the analysis easier.
#
# Lets start by measuring the baseline runtime.
#

model, image = get_sam_model(only_one_block, batchsize)
fp32_res = benchmark(model, image)
print(f"base fp32 runtime of the model is {fp32_res['time']:0.2f}ms and peak memory {fp32_res['memory']:0.2f}GB")
# base fp32 runtime of the model is 186.16ms and peak memory 6.33GB


######################################################################
# We can obtain an immediate speedup simply by converting the model to
# bfloat16. We use bfloat16 rather than fp16 because bfloat16 has a
# dynamic range similar to fp32 because they both have 8 exponential bits
# while fp16 only has 4 which tends to insulate us from overflow errors
# and other problems that occur when scaling and rescaling tensors due to
# quantization.
#

model = model.to(torch.bfloat16)
image = image.to(torch.bfloat16)
bf16_res = benchmark(model, image)
print(f"bf16 runtime of the block is {bf16_res['time']:0.2f}ms and peak memory {bf16_res['memory']: 0.2f}GB")
# bf16 runtime of the block is 25.43ms and peak memory  3.17GB


######################################################################
# Just this quick change improves runtime by a factor of ~7x in my
# experience (186.16ms to 25.43ms).
#
# Next we torch.compile the model and see how much the performance
# improves
#

model_c = torch.compile(model, mode='max-autotune')
comp_res = benchmark(model_c, image)
print(f"bf16 compiled runtime of the block is {comp_res['time']:0.2f}ms and peak memory {comp_res['memory']: 0.2f}GB")
# bf16 compiled runtime of the block is 19.95ms and peak memory  2.24GB


######################################################################
# The first time this is run, you should see a sequence of AUTOTUNE
# outputs which occurs when inductor compares the performance between
# various kernel parameters for a kernel. This only happens once (unless
# you delete your cache) so if you run the cell again you should just get
# the benchmark output.
#
# torch.compile yields about another 27% improvement. This brings the
# model to a reasonable baseline where we now have to work a bit harder
# for improvements.
#
# Now lets apply quantization. Quantization for GPUs comes in 3 main forms
# in `torchao <https://github.com/pytorch-labs/ao>`_ which is just native
# pytorch+python code. We have int8 dynamic quantization, int8 weight-only
# quantization and int4 weight-only quantization. Different models, or
# sometimes different layers in a model can require different techniques.
# For models which are heavily compute bound, dynamic quantization tends
# to work the best since it swaps the normal expensive floating point
# matmul ops with integer versions. Weight-only quantization works better
# in memory bound situations where the benefit comes from loading less
# weight data, rather than doing less computation. The torchao api’s:
#
# ``change_linear_weights_to_int8_dqtensors``,
# ``change_linear_weights_to_int8_woqtensors`` or
# ``change_linear_weights_to_int4_woqtensors``
#
# can be used to easily apply the desired quantization technique and then
# once the model is torch.compile-d with max-autotune, quantization is
# complete and we can see our speedup.
#
# Note: These api’s may be buggy on older versions of pytorch, if you run
# into that issue, you can instead use
#
# ``apply_dynamic_quant`` and ``apply_weight_only_int8_quant``
#
# As drop in replacement for the 2 above (no replacement for int4). The
# difference between the two apis is that ‘change_linear_weights’ api’s
# alter the weight tensor of the linear module so instead of doing a
# normal linear, it does a quantized operation. This is helpful when users
# have non-standard linear ops that do more than 1 thing. The ‘apply’
# api’s directly swap the linear modules for a quantized module which
# works on older versions but doesn’t work with non-standard linear
# modules.
#
# In this case SAM is compute bound so we’ll use dynamic quantization
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
# With quantization we improve perf a bit more but memory usage shot way
# up.
#
# This is for two reasons:
#
# 1) Quantization adds overhead to the model
#    since we need to quantize and dequantize the input and output. For small
#    batchsizes this overhead can actually make the model go slower.
# 2) Even though we are doing a quantized matmul, i.e. int8 x int8,
#    the result of the multiplication gets stored in an int32 tensor
#    which is twice the size of the result from the non-quantized model.
#    If we can avoid creating this int32 tensor our memory usage will improve a lot.
#
# We can fix #2 by fusing the integer matmul with the subsequent rescale
# operation since the final output will be bf16, if we immediately convert
# the int32 tensor to bf16 and instead store that we’ll get getter
# performance in terms of both runtime and memory.
#
# The way to do this is to enable the option
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
# 1) We can sometimes improve performance be disabling epilogue fusion
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



######################################################################
# In the end, we achieved a full-model apples to apples quantization speedup
# of about 7.7% on batch size 16 (677.28ms to 729.65ms). We can push this a
# bit further by increasing the batch size and optimizing other parts of
# the model. For example, this can be done with some form of flash attention. 
#
# For more information visit
# `torchao <https://github.com/pytorch-labs/ao>`_ and try it on your own
# models.
#
