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

```
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2

ds = datasets.FashionMNIST(
 root="data",
 train=True,
 download=True,
 transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
 target_transform=v2.Lambda(
 lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
 ),
)
```

```
0%| | 0.00/26.4M [00:00<?, ?B/s]
 0%| | 65.5k/26.4M [00:00<01:10, 375kB/s]
 1%| | 197k/26.4M [00:00<00:44, 595kB/s]
 3%|▎ | 786k/26.4M [00:00<00:13, 1.86MB/s]
 12%|█▏ | 3.21M/26.4M [00:00<00:03, 6.58MB/s]
 35%|███▍ | 9.18M/26.4M [00:00<00:01, 16.5MB/s]
 57%|█████▋ | 15.2M/26.4M [00:01<00:00, 22.5MB/s]
 80%|████████ | 21.2M/26.4M [00:01<00:00, 26.4MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 20.0MB/s]

 0%| | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 339kB/s]

 0%| | 0.00/4.42M [00:00<?, ?B/s]
 1%|▏ | 65.5k/4.42M [00:00<00:11, 373kB/s]
 5%|▌ | 229k/4.42M [00:00<00:05, 702kB/s]
 20%|██ | 885k/4.42M [00:00<00:01, 2.09MB/s]
 81%|████████ | 3.57M/4.42M [00:00<00:00, 7.29MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.27MB/s]

 0%| | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 51.3MB/s]
```

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

```
target_transform = v2.Lambda(
 lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
)
```

---

### Further Reading

- [Getting started with transforms v2](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html)
- [torchvision.transforms.v2 API](https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended)

**Total running time of the script:** (0 minutes 4.264 seconds)

[`Download Jupyter notebook: transforms_tutorial.ipynb`](../../_downloads/9bdb71ef4a637dc36fb461904ccb7056/transforms_tutorial.ipynb)

[`Download Python source code: transforms_tutorial.py`](../../_downloads/2f1ec3031a7101e25403c5d53a40a401/transforms_tutorial.py)

[`Download zipped: transforms_tutorial.zip`](../../_downloads/f65fa134d1dbd7b77ef50ad2846ed92b/transforms_tutorial.zip)