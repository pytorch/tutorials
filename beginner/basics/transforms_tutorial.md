Note

Go to the end
to download the full example code.

[Learn the Basics](intro.html) ||
[Quickstart](quickstart_tutorial.html) ||
[Tensors](tensorqs_tutorial.html) ||
[Datasets & DataLoaders](data_tutorial.html) ||
**Transforms** ||
[Build Model](buildmodel_tutorial.html) ||
[Autograd](autogradqs_tutorial.html) ||
[Optimization](optimization_tutorial.html) ||
[Save & Load Model](saveloadrun_tutorial.html)

# Transforms

Data does not always come in its final processed form that is required for
training machine learning algorithms. We use **transforms** to perform some
manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters -`transform` to modify the features and
`target_transform` to modify the labels - that accept callables containing the transformation logic.
The [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) module offers
several commonly-used transforms out of the box.

The FashionMNIST features are in PIL Image format, and the labels are integers.
For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors.
To make these transformations, we use the `torchvision.transforms.v2` API along with `torch.nn.functional.one_hot`.

## ToImage() and ToDtype()

The `torchvision.transforms.v2` API replaces the legacy `ToTensor` transform with a two-step pipeline.
[v2.ToImage](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToImage.html)
converts a PIL image or NumPy `ndarray` into a `torchvision.tv_tensors.Image` tensor, and
[v2.ToDtype](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html)
with `scale=True` casts it to `float32` and scales the pixel intensity values to the range [0., 1.].

## Lambda Transforms

Lambda transforms apply any user-defined lambda function. Here, we use
[torch.nn.functional.one_hot](https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html)
to turn the integer label into a one-hot encoded tensor of size 10 (the number of labels in our dataset),
then cast it to `float` to match the expected dtype.

---

### Further Reading

- [Getting started with transforms v2](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html)
- [torchvision.transforms.v2 API](https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: transforms_tutorial.ipynb`](../../_downloads/9bdb71ef4a637dc36fb461904ccb7056/transforms_tutorial.ipynb)

[`Download Python source code: transforms_tutorial.py`](../../_downloads/2f1ec3031a7101e25403c5d53a40a401/transforms_tutorial.py)

[`Download zipped: transforms_tutorial.zip`](../../_downloads/f65fa134d1dbd7b77ef50ad2846ed92b/transforms_tutorial.zip)