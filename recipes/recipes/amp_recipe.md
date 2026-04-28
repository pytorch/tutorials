Note

Go to the end
to download the full example code.

# Automatic Mixed Precision

**Author**: [Michael Carilli](https://github.com/mcarilli)

[torch.cuda.amp](https://pytorch.org/docs/stable/amp.html) provides convenience methods for mixed precision,
where some operations use the `torch.float32` (`float`) datatype and other operations
use `torch.float16` (`half`). Some ops, like linear layers and convolutions,
are much faster in `float16` or `bfloat16`. Other ops, like reductions, often require the dynamic
range of `float32`. Mixed precision tries to match each op to its appropriate datatype,
which can reduce your network's runtime and memory footprint.

Ordinarily, "automatic mixed precision training" uses [torch.autocast](https://pytorch.org/docs/stable/amp.html#torch.autocast) and
[torch.cuda.amp.GradScaler](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler) together.

This recipe measures the performance of a simple network in default precision,
then walks through adding `autocast` and `GradScaler` to run the same network in
mixed precision with improved performance.

You may download and run this recipe as a standalone Python script.
The only requirements are PyTorch 1.6 or later and a CUDA-capable GPU.

Mixed precision primarily benefits Tensor Core-enabled architectures (Volta, Turing, Ampere).
This recipe should show significant (2-3X) speedup on those architectures.
On earlier architectures (Kepler, Maxwell, Pascal), you may observe a modest speedup.
Run `nvidia-smi` to display your GPU's architecture.

```
# Timing utilities
```

## A simple network

The following sequence of linear layers and ReLUs should show a speedup with mixed precision.

`batch_size`, `in_size`, `out_size`, and `num_layers` are chosen to be large enough to saturate the GPU with work.
Typically, mixed precision provides the greatest speedup when the GPU is saturated.
Small networks may be CPU bound, in which case mixed precision won't improve performance.
Sizes are also chosen such that linear layers' participating dimensions are multiples of 8,
to permit Tensor Core usage on Tensor Core-capable GPUs (see Troubleshooting below).

Exercise: Vary participating sizes and see how the mixed precision speedup changes.

```
# Creates data in default precision.
# The same data is used for both default and mixed precision trials below.
# You don't need to manually change inputs' ``dtype`` when enabling mixed precision.
```

## Default Precision

Without `torch.cuda.amp`, the following simple network executes all ops in default precision (`torch.float32`):

## Adding `torch.autocast`

Instances of [torch.autocast](https://pytorch.org/docs/stable/amp.html#autocasting)
serve as context managers that allow regions of your script to run in mixed precision.

In these regions, CUDA ops run in a `dtype` chosen by `autocast`
to improve performance while maintaining accuracy.
See the [Autocast Op Reference](https://pytorch.org/docs/stable/amp.html#autocast-op-reference)
for details on what precision `autocast` chooses for each op, and under what circumstances.

## Adding `GradScaler`

[Gradient scaling](https://pytorch.org/docs/stable/amp.html#gradient-scaling)
helps prevent gradients with small magnitudes from flushing to zero
("underflowing") when training with mixed precision.

[torch.cuda.amp.GradScaler](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler)
performs the steps of gradient scaling conveniently.

```
# Constructs a ``scaler`` once, at the beginning of the convergence run, using default arguments.
# If your network fails to converge with default ``GradScaler`` arguments, please file an issue.
# The same ``GradScaler`` instance should be used for the entire convergence run.
# If you perform multiple convergence runs in the same script, each run should use
# a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
```

## All together: "Automatic Mixed Precision"

(The following also demonstrates `enabled`, an optional convenience argument to `autocast` and `GradScaler`.
If False, `autocast` and `GradScaler`'s calls become no-ops.
This allows switching between default precision and mixed precision without if/else statements.)

## Inspecting/modifying gradients (e.g., clipping)

All gradients produced by `scaler.scale(loss).backward()` are scaled. If you wish to modify or inspect
the parameters' `.grad` attributes between `backward()` and `scaler.step(optimizer)`, you should
unscale them first using [scaler.unscale_(optimizer)](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.unscale_).

## Saving/Resuming

To save/resume Amp-enabled runs with bitwise accuracy, use
[scaler.state_dict](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.state_dict) and
[scaler.load_state_dict](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.load_state_dict).

When saving, save the `scaler` state dict alongside the usual model and optimizer state `dicts`.
Do this either at the beginning of an iteration before any forward passes, or at the end of
an iteration after `scaler.update()`.

```
# Write checkpoint as desired, e.g.,
# torch.save(checkpoint, "filename")
```

When resuming, load the `scaler` state dict alongside the model and optimizer state `dicts`.
Read checkpoint as desired, for example:

```
dev = torch.cuda.current_device()
checkpoint = torch.load("filename",
 map_location = lambda storage, loc: storage.cuda(dev))
```

If a checkpoint was created from a run *without* Amp, and you want to resume training *with* Amp,
load model and optimizer states from the checkpoint as usual. The checkpoint won't contain a saved `scaler` state, so
use a fresh instance of `GradScaler`.

If a checkpoint was created from a run *with* Amp and you want to resume training *without* `Amp`,
load model and optimizer states from the checkpoint as usual, and ignore the saved `scaler` state.

## Inference/Evaluation

`autocast` may be used by itself to wrap inference or evaluation forward passes. `GradScaler` is not necessary.

## Advanced topics

See the [Automatic Mixed Precision Examples](https://pytorch.org/docs/stable/notes/amp_examples.html) for advanced use cases including:

- Gradient accumulation
- Gradient penalty/double backward
- Networks with multiple models, optimizers, or losses
- Multiple GPUs (`torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`)
- Custom autograd functions (subclasses of `torch.autograd.Function`)

If you perform multiple convergence runs in the same script, each run should use
a dedicated fresh `GradScaler` instance. `GradScaler` instances are lightweight.

If you're registering a custom C++ op with the dispatcher, see the
[autocast section](https://pytorch.org/tutorials/advanced/dispatcher.html#autocast)
of the dispatcher tutorial.

## Troubleshooting

### Speedup with Amp is minor

1. Your network may fail to saturate the GPU(s) with work, and is therefore CPU bound. Amp's effect on GPU performance
won't matter.

- A rough rule of thumb to saturate the GPU is to increase batch and/or network size(s)
as much as you can without running OOM.
- Try to avoid excessive CPU-GPU synchronization (`.item()` calls, or printing values from CUDA tensors).
- Try to avoid sequences of many small CUDA ops (coalesce these into a few large CUDA ops if you can).
2. Your network may be GPU compute bound (lots of `matmuls`/convolutions) but your GPU does not have Tensor Cores.
In this case a reduced speedup is expected.
3. The `matmul` dimensions are not Tensor Core-friendly. Make sure `matmuls` participating sizes are multiples of 8.
(For NLP models with encoders/decoders, this can be subtle. Also, convolutions used to have similar size constraints
for Tensor Core use, but for CuDNN versions 7.3 and later, no such constraints exist. See
[here](https://github.com/NVIDIA/apex/issues/221#issuecomment-478084841) for guidance.)

### Loss is inf/NaN

First, check if your network fits an advanced use case.
See also [Prefer binary_cross_entropy_with_logits over binary_cross_entropy](https://pytorch.org/docs/stable/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy).

If you're confident your Amp usage is correct, you may need to file an issue, but before doing so, it's helpful to gather the following information:

1. Disable `autocast` or `GradScaler` individually (by passing `enabled=False` to their constructor) and see if `infs`/`NaNs` persist.
2. If you suspect part of your network (e.g., a complicated loss function) overflows , run that forward region in `float32`
and see if `infs`/NaN``s persist.
`The autocast docstring <https://pytorch.org/docs/stable/amp.html#torch.autocast>`_'s last code snippet
shows forcing a subregion to run in ``float32 (by locally disabling `autocast` and casting the subregion's inputs).

### Type mismatch error (may manifest as `CUDNN_STATUS_BAD_PARAM`)

`Autocast` tries to cover all ops that benefit from or require casting.
[Ops that receive explicit coverage](https://pytorch.org/docs/stable/amp.html#autocast-op-reference)
are chosen based on numerical properties, but also on experience.
If you see a type mismatch error in an `autocast` enabled forward region or a backward pass following that region,
it's possible `autocast` missed an op.

Please file an issue with the error backtrace. `export TORCH_SHOW_CPP_STACKTRACES=1` before running your script to provide
fine-grained information on which backend op is failing.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: amp_recipe.ipynb`](../../_downloads/13cdb386a4b0dc48c626f32e6cf8681d/amp_recipe.ipynb)

[`Download Python source code: amp_recipe.py`](../../_downloads/cadb3a57e7a6d7c149b5ae377caf36a8/amp_recipe.py)

[`Download zipped: amp_recipe.zip`](../../_downloads/cabe7a4e5a2617bc2c00a1066bf03f4f/amp_recipe.zip)