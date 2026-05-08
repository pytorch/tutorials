Note

Go to the end
to download the full example code.

# (beta) Running the compiled optimizer with an LR Scheduler

**Author:** [Michael Lazos](https://github.com/mlazos)

The optimizer is a key algorithm for training any deep learning model.
In this example, we will show how to pair the optimizer, which has been compiled using `torch.compile`,
with the LR schedulers to accelerate training convergence.

Note

This tutorial requires PyTorch 2.3.0 or later.

## Model Setup

For this example, we'll use a simple sequence of linear layers.

```
# Create simple model

# run forward pass

# run backward to populate the grads for our optimizer below
```

## Setting up and running the compiled optimizer with LR Scheduler

In this section, we'll use the Adam optimizer with LinearLR Scheduler
and create a helper function to wrap the `step()` call for each of them
in `torch.compile()`.

Note

`torch.compile` is only supported on CUDA devices that have a compute capability of 7.0 or higher.

```
# exit cleanly if we are on a device that doesn't support ``torch.compile``

# !!! IMPORTANT !!! Wrap the lr in a Tensor if we are pairing the
# the optimizer with an LR Scheduler.
# Without this, torch.compile will recompile as the value of the LR
# changes.

# Warmup runs to compile the function
```

## Extension: What happens with a non-tensor LR?

For the curious, we will show how to peek into what happens with `torch.compile` when we don't wrap the
LR in a tensor.

```
# No longer wrap the LR in a tensor here

# Setup logging to view recompiles

# Warmup runs to compile the function
# We will now recompile on each iteration
# as the value of the lr is mutated.
```

With this example, we can see that we recompile the optimizer a few times
due to the guard failure on the `lr` in `param_groups[0]`.

## Conclusion

In this tutorial we showed how to pair the optimizer compiled with `torch.compile`
with an LR Scheduler to accelerate training convergence. We used a model consisting
of a simple sequence of linear layers with the Adam optimizer paired
with a LinearLR scheduler to demonstrate the LR changing across iterations.

See also:

- [Compiled optimizer tutorial](https://pytorch.org/tutorials/recipes/compiling_optimizer.html) - an intro into the compiled optimizer.
- [Compiling the optimizer with PT2](https://dev-discuss.pytorch.org/t/compiling-the-optimizer-with-pt2/1669) - deeper technical details on the compiled optimizer.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: compiling_optimizer_lr_scheduler.ipynb`](../_downloads/a96dbf475cc2f41befeebc6b79a1dbbe/compiling_optimizer_lr_scheduler.ipynb)

[`Download Python source code: compiling_optimizer_lr_scheduler.py`](../_downloads/cba164d7ed0d3cf63f210d336236ec14/compiling_optimizer_lr_scheduler.py)

[`Download zipped: compiling_optimizer_lr_scheduler.zip`](../_downloads/ca3f7d220c5dcf522653cfde03534974/compiling_optimizer_lr_scheduler.zip)