Note

Go to the end
to download the full example code.

[Learn the Basics](intro.html) ||
[Quickstart](quickstart_tutorial.html) ||
[Tensors](tensorqs_tutorial.html) ||
**Datasets & DataLoaders** ||
[Transforms](transforms_tutorial.html) ||
[Build Model](buildmodel_tutorial.html) ||
[Autograd](autogradqs_tutorial.html) ||
[Optimization](optimization_tutorial.html) ||
[Save & Load Model](saveloadrun_tutorial.html)

# Datasets & DataLoaders

Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code
to be decoupled from our model training code for better readability and modularity.
PyTorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`
that allow you to use pre-loaded datasets as well as your own data.
`Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around
the `Dataset` to enable easy access to the samples.

PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that
subclass `torch.utils.data.Dataset` and implement functions specific to the particular data.
They can be used to prototype and benchmark your model. You can find them
here: [Image Datasets](https://pytorch.org/vision/stable/datasets.html),
[Text Datasets](https://pytorch.org/text/stable/datasets.html), and
[Audio Datasets](https://pytorch.org/audio/stable/datasets.html)

## Loading a Dataset

Here is an example of how to load the [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) dataset from TorchVision.
Fashion-MNIST is a dataset of Zalando's article images consisting of 60,000 training examples and 10,000 test examples.
Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

We load the [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist) with the following parameters:

- `root` is the path where the train/test data is stored,
- `train` specifies training or test dataset,
- `download=True` downloads the data from the internet if it's not available at `root`.
- `transform` and `target_transform` specify the feature and label transformations

## Iterating and Visualizing the Dataset

We can index `Datasets` manually like a list: `training_data[index]`.
We use `matplotlib` to visualize some samples in our training data.

---

## Creating a Custom Dataset for your files

A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
Take a look at this implementation; the FashionMNIST images are stored
in a directory `img_dir`, and their labels are stored separately in a CSV file `annotations_file`.

In the next sections, we'll break down what's happening in each of these functions.

### `__init__`

The __init__ function is run once when instantiating the Dataset object. We initialize
the directory containing the images, the annotations file, and both transforms (covered
in more detail in the next section).

The labels.csv file looks like:

```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

### `__len__`

The __len__ function returns the number of samples in our dataset.

Example:

### `__getitem__`

The __getitem__ function loads and returns a sample from the dataset at the given index `idx`.
Based on the index, it identifies the image's location on disk, converts that to a tensor using `decode_image`, retrieves the
corresponding label from the csv data in `self.img_labels`, calls the transform functions on them (if applicable), and returns the
tensor image and corresponding label in a tuple.

---

## Preparing your data for training with DataLoaders

The `Dataset` retrieves our dataset's features and labels one sample at a time. While training a model, we typically want to
pass samples in "minibatches", reshuffle the data at every epoch to reduce model overfitting, and use Python's `multiprocessing` to
speed up data retrieval.

`DataLoader` is an iterable that abstracts this complexity for us in an easy API.

## Iterate through the DataLoader

We have loaded that dataset into the `DataLoader` and can iterate through the dataset as needed.
Each iteration below returns a batch of `train_features` and `train_labels` (containing `batch_size=64` features and labels respectively).
Because we specified `shuffle=True`, after we iterate over all batches the data is shuffled (for finer-grained control over
the data loading order, take a look at [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)).

```
# Display image and label.
```

---

## Further Reading

- [torch.utils.data API](https://pytorch.org/docs/stable/data.html)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: data_tutorial.ipynb`](../../_downloads/36608d2d57f623ba3a623e0c947a8c3e/data_tutorial.ipynb)

[`Download Python source code: data_tutorial.py`](../../_downloads/56e3f440fc204e02856f8889c226d2d1/data_tutorial.py)

[`Download zipped: data_tutorial.zip`](../../_downloads/89855d8fec84a240291d4492f4ece548/data_tutorial.zip)