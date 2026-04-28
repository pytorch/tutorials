Note

Go to the end
to download the full example code.

# Fusing Convolution and Batch Norm using Custom Function

Fusing adjacent convolution and batch norm layers together is typically an
inference-time optimization to improve run-time. It is usually achieved
by eliminating the batch norm layer entirely and updating the weight
and bias of the preceding convolution [0]. However, this technique is not
applicable for training models.

In this tutorial, we will show a different technique to fuse the two layers
that can be applied during training. Rather than improved runtime, the
objective of this optimization is to reduce memory usage.

The idea behind this optimization is to see that both convolution and
batch norm (as well as many other ops) need to save a copy of their input
during forward for the backward pass. For large
batch sizes, these saved inputs are responsible for most of your memory usage,
so being able to avoid allocating another input tensor for every
convolution batch norm pair can be a significant reduction.

In this tutorial, we avoid this extra allocation by combining convolution
and batch norm into a single layer (as a custom function). In the forward
of this combined layer, we perform normal convolution and batch norm as-is,
with the only difference being that we will only save the inputs to the convolution.
To obtain the input of batch norm, which is necessary to backward through
it, we recompute convolution forward again during the backward pass.

It is important to note that the usage of this optimization is situational.
Though (by avoiding one buffer saved) we always reduce the memory allocated at
the end of the forward pass, there are cases when the *peak* memory allocated
may not actually be reduced. See the final section for more details.

For simplicity, in this tutorial we hardcode bias=False, stride=1, padding=0, dilation=1,
and groups=1 for Conv2D. For BatchNorm2D, we hardcode eps=1e-3, momentum=0.1,
affine=False, and track_running_statistics=False. Another small difference
is that we add epsilon in the denominator outside of the square root in the computation
of batch norm.

[0] [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/](https://nenadmarkus.com/p/fusing-batchnorm-and-conv/)

## Backward Formula Implementation for Convolution

Implementing a custom function requires us to implement the backward
ourselves. In this case, we need both the backward formulas for Conv2D
and BatchNorm2D. Eventually we'd chain them together in our unified
backward function, but below we first implement them as their own
custom functions so we can validate their correctness individually

When testing with `gradcheck`, it is important to use double precision

## Backward Formula Implementation for Batch Norm

Batch Norm has two modes: training and `eval` mode. In training mode
the sample statistics are a function of the inputs. In `eval` mode,
we use the saved running statistics, which are not a function of the inputs.
This makes non-training mode's backward significantly simpler. Below
we implement and test only the training mode case.

Testing with `gradcheck`

## Fusing Convolution and BatchNorm

Now that the bulk of the work has been done, we can combine
them together. Note that in (1) we only save a single buffer
for backward, but this also means we recompute convolution forward
in (5). Also see that in (2), (3), (4), and (6), it's the same
exact code as the examples above.

The next step is to wrap our functional variant in a stateful
nn.Module

Use `gradcheck` to validate the correctness of our backward formula

## Testing out our new Layer

Use `FusedConvBN` to train a basic network
The code below is after some light modifications to the example here:
[pytorch/examples](https://github.com/pytorch/examples/tree/master/mnist)

```
# Record memory allocated at the end of the forward pass
```

## A Comparison of Memory Usage

If CUDA is enabled, print out memory usage for both fused=True and fused=False
For an example run on NVIDIA GeForce RTX 3070, NVIDIA CUDA® Deep Neural Network library (cuDNN) 8.0.5: fused peak memory: 1.56GB,
unfused peak memory: 2.68GB

It is important to note that the *peak* memory usage for this model may vary depending
the specific cuDNN convolution algorithm used. For shallower models, it
may be possible for the peak memory allocated of the fused model to exceed
that of the unfused model! This is because the memory allocated to compute
certain cuDNN convolution algorithms can be high enough to "hide" the typical peak
you would expect to be near the start of the backward pass.

For this reason, we also record and display the memory allocated at the end
of the forward pass as an approximation, and to demonstrate that we indeed
allocate one fewer buffer per fused `conv-bn` pair.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: custom_function_conv_bn_tutorial.ipynb`](../_downloads/e42651bf8aa9a118fc1867c909799393/custom_function_conv_bn_tutorial.ipynb)

[`Download Python source code: custom_function_conv_bn_tutorial.py`](../_downloads/187aea79daf1552dd05cdde1f4b4e34d/custom_function_conv_bn_tutorial.py)

[`Download zipped: custom_function_conv_bn_tutorial.zip`](../_downloads/ece2140454444fcab89059884b672a81/custom_function_conv_bn_tutorial.zip)