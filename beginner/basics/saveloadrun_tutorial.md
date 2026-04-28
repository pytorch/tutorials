Note

Go to the end
to download the full example code.

[Learn the Basics](intro.html) ||
[Quickstart](quickstart_tutorial.html) ||
[Tensors](tensorqs_tutorial.html) ||
[Datasets & DataLoaders](data_tutorial.html) ||
[Transforms](transforms_tutorial.html) ||
[Build Model](buildmodel_tutorial.html) ||
[Autograd](autogradqs_tutorial.html) ||
[Optimization](optimization_tutorial.html) ||
**Save & Load Model**

# Save and Load the Model

In this section we will look at how to persist model state with saving, loading and running model predictions.

## Saving and Loading Model Weights

PyTorch models store the learned parameters in an internal
state dictionary, called `state_dict`. These can be persisted via the `torch.save`
method:

To load model weights, you need to create an instance of the same model first, and then load the parameters
using `load_state_dict()` method.

In the code below, we set `weights_only=True` to limit the
functions executed during unpickling to only those necessary for
loading weights. Using `weights_only=True` is considered
a best practice when loading weights.

Note

be sure to call `model.eval()` method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.

## Saving and Loading Models with Shapes

When loading model weights, we needed to instantiate the model class first, because the class
defines the structure of a network. We might want to save the structure of this class together with
the model, in which case we can pass `model` (and not `model.state_dict()`) to the saving function:

We can then load the model as demonstrated below.

As described in [Saving and loading torch.nn.Modules](https://pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules),
saving `state_dict` is considered the best practice. However,
below we use `weights_only=False` because this involves loading the
model, which is a legacy use case for `torch.save`.

Note

This approach uses Python [pickle](https://docs.python.org/3/library/pickle.html) module when serializing the model, thus it relies on the actual class definition to be available when loading the model.

## Related Tutorials

- [Saving and Loading a General Checkpoint in PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [Tips for loading an nn.Module from a checkpoint](https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html?highlight=loading%20nn%20module%20from%20checkpoint)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: saveloadrun_tutorial.ipynb`](../../_downloads/11f1adacb7d237f2041ce267ac38abb6/saveloadrun_tutorial.ipynb)

[`Download Python source code: saveloadrun_tutorial.py`](../../_downloads/3648b0dccaebca71b234070fe2124770/saveloadrun_tutorial.py)

[`Download zipped: saveloadrun_tutorial.zip`](../../_downloads/0a63ed31b0b1f27896bbfba4038b8718/saveloadrun_tutorial.zip)