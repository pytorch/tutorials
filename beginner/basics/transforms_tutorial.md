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
To make these transformations, we use `ToTensor` and `Lambda`.

```
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
 root="data",
 train=True,
 download=True,
 transform=ToTensor(),
 target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

```
0%| | 0.00/26.4M [00:00<?, ?B/s]
 0%| | 65.5k/26.4M [00:00<01:10, 376kB/s]
 1%| | 229k/26.4M [00:00<00:37, 705kB/s]
 3%|▎ | 918k/26.4M [00:00<00:11, 2.18MB/s]
 14%|█▍ | 3.67M/26.4M [00:00<00:03, 7.52MB/s]
 36%|███▋ | 9.63M/26.4M [00:00<00:00, 17.1MB/s]
 56%|█████▌ | 14.7M/26.4M [00:01<00:00, 21.2MB/s]
 76%|███████▋ | 20.2M/26.4M [00:01<00:00, 24.3MB/s]
 96%|█████████▋| 25.5M/26.4M [00:01<00:00, 26.2MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 18.8MB/s]

 0%| | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 338kB/s]

 0%| | 0.00/4.42M [00:00<?, ?B/s]
 1%|▏ | 65.5k/4.42M [00:00<00:11, 373kB/s]
 4%|▍ | 197k/4.42M [00:00<00:07, 595kB/s]
 19%|█▉ | 852k/4.42M [00:00<00:01, 2.03MB/s]
 76%|███████▌ | 3.34M/4.42M [00:00<00:00, 6.84MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.29MB/s]

 0%| | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 29.5MB/s]
```

## ToTensor()

[ToTensor](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor)
converts a PIL image or NumPy `ndarray` into a `FloatTensor`. and scales
the image's pixel intensity values in the range [0., 1.]

## Lambda Transforms

Lambda transforms apply any user-defined lambda function. Here, we define a function
to turn the integer into a one-hot encoded tensor.
It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls
[scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html) which assigns a
`value=1` on the index as given by the label `y`.

```
target_transform = Lambda(lambda y: torch.zeros(
 10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

---

### Further Reading

- [torchvision.transforms API](https://pytorch.org/vision/stable/transforms.html)

**Total running time of the script:** (0 minutes 4.346 seconds)

[`Download Jupyter notebook: transforms_tutorial.ipynb`](../../_downloads/9bdb71ef4a637dc36fb461904ccb7056/transforms_tutorial.ipynb)

[`Download Python source code: transforms_tutorial.py`](../../_downloads/2f1ec3031a7101e25403c5d53a40a401/transforms_tutorial.py)

[`Download zipped: transforms_tutorial.zip`](../../_downloads/f65fa134d1dbd7b77ef50ad2846ed92b/transforms_tutorial.zip)