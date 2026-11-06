Note

Go to the end
to download the full example code.

# Visualizing Gradients

**Author:** [Justin Silver](https://github.com/j-silv)

This tutorial explains how to extract and visualize gradients at any
layer in a neural network. By inspecting how information flows from the
end of the network to the parameters we want to optimize, we can debug
issues such as [vanishing or exploding
gradients](https://arxiv.org/abs/1211.5063) that occur during
training.

Before starting, make sure you understand [tensors and how to manipulate
them](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).
A basic knowledge of [how autograd
works](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
would also be useful.

## Setup

First, make sure [PyTorch is
installed](https://pytorch.org/get-started/locally/) and then import
the necessary libraries.

Next, we'll be creating a network intended for the MNIST dataset,
similar to the architecture described by the [batch normalization
paper](https://arxiv.org/abs/1502.03167).

To illustrate the importance of gradient visualization, we will
instantiate one version of the network with batch normalization
(BatchNorm), and one without it. Batch normalization is an extremely
effective technique to resolve [vanishing/exploding
gradients](https://arxiv.org/abs/1211.5063), and we will be verifying
that experimentally.

The model we use has a configurable number of repeating fully-connected
layers which alternate between `nn.Linear`, `norm_layer`, and
`nn.Sigmoid`. If batch normalization is enabled, then `norm_layer`
will use
[BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html),
otherwise it will use the
[Identity](https://docs.pytorch.org/docs/stable/generated/torch.nn.Identity.html)
transformation.

Next we set up some dummy data, instantiate two versions of the model,
and initialize the optimizers.

```
# set up dummy data

# init model
```

We can verify that batch normalization is only being applied to one of
the models by probing one of the internal layers:

## Registering hooks

Because we wrapped up the logic and state of our model in a
`nn.Module`, we need another method to access the intermediate
gradients if we want to avoid modifying the module code directly. This
is done by [registering a
hook](https://docs.pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution).

Warning

Using backward pass hooks attached to output tensors is preferred over using `retain_grad()` on the tensors themselves. An alternative method is to directly attach module hooks (e.g. `register_full_backward_hook()`) so long as the `nn.Module` instance does not do perform any in-place operations. For more information, please refer to [this issue](https://github.com/pytorch/pytorch/issues/61519).

The following code defines our hooks and gathers descriptive names for
the network's layers.

```
# note that wrapper functions are used for Python closure
# so that we can pass arguments.

# register hooks
```

## Training and visualization

Let's now train the models for a few epochs:

After running the forward and backward pass, the gradients for all the
intermediate tensors should be present in `grads_bn` and
`grads_nobn`. We compute the mean absolute value of each gradient
matrix so that we can compare the two models.

With the average gradients computed, we can now plot them and see how
the values change as a function of the network depth. Notice that when
we don't apply batch normalization, the gradient values in the
intermediate layers fall to zero very quickly. The batch normalization
model, however, maintains non-zero gradients in its intermediate layers.

## Conclusion

In this tutorial, we demonstrated how to visualize the gradient flow
through a neural network wrapped in a `nn.Module` class. We
qualitatively showed how batch normalization helps to alleviate the
vanishing gradient issue which occurs with deep neural networks.

If you would like to learn more about how PyTorch's autograd system
works, please visit the references below. If you have
any feedback for this tutorial (improvements, typo fixes, etc.) then
please use the [PyTorch Forums](https://discuss.pytorch.org/) and/or
the [issue tracker](https://github.com/pytorch/tutorials/issues) to
reach out.

## (Optional) Additional exercises

- Try increasing the number of layers (`num_layers`) in our model and
see what effect this has on the gradient flow graph
- How would you adapt the code to visualize average activations instead
of average gradients? (*Hint: in the hook_forward() function we have
access to the raw tensor output*)
- What are some other methods to deal with vanishing and exploding
gradients?

## References

- [A Gentle Introduction to
torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Automatic Differentiation with
torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial)
- [Autograd
mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [On the difficulty of training Recurrent Neural
Networks](https://arxiv.org/abs/1211.5063)

%%%%%%RUNNABLE_CODE_REMOVED%%%%%%

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: visualizing_gradients_tutorial.ipynb`](../_downloads/ee0bd22c8fd862ec4f59f792d8694771/visualizing_gradients_tutorial.ipynb)

[`Download Python source code: visualizing_gradients_tutorial.py`](../_downloads/b0daeef258d2e426aeb59acc0d09e0ef/visualizing_gradients_tutorial.py)

[`Download zipped: visualizing_gradients_tutorial.zip`](../_downloads/5dd22895367e30899a24ff182c869b3a/visualizing_gradients_tutorial.zip)