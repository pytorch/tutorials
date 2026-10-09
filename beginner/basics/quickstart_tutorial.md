Note

Go to the end
to download the full example code.

[Learn the Basics](intro.html) ||
**Quickstart** ||
[Tensors](tensorqs_tutorial.html) ||
[Datasets & DataLoaders](data_tutorial.html) ||
[Transforms](transforms_tutorial.html) ||
[Build Model](buildmodel_tutorial.html) ||
[Autograd](autogradqs_tutorial.html) ||
[Optimization](optimization_tutorial.html) ||
[Save & Load Model](saveloadrun_tutorial.html)

# Quickstart

This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.

## Working with data

PyTorch has two [primitives to work with data](https://pytorch.org/docs/stable/data.html):
`torch.utils.data.DataLoader` and `torch.utils.data.Dataset`.
`Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around
the `Dataset`.

```
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
```

PyTorch offers domain-specific libraries such as [TorchText](https://pytorch.org/text/stable/index.html),
[TorchVision](https://pytorch.org/vision/stable/index.html), and [TorchAudio](https://pytorch.org/audio/stable/index.html),
all of which include datasets. For this tutorial, we will be using a TorchVision dataset.

The `torchvision.datasets` module contains `Dataset` objects for many real-world vision data like
CIFAR, COCO ([full list here](https://pytorch.org/vision/stable/datasets.html)). In this tutorial, we
use the FashionMNIST dataset. Every TorchVision `Dataset` includes two arguments: `transform` and
`target_transform` to modify the samples and labels respectively.

```
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
 root="data",
 train=True,
 download=True,
 transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
 root="data",
 train=False,
 download=True,
 transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
```

```
0%| | 0.00/26.4M [00:00<?, ?B/s]
 0%| | 65.5k/26.4M [00:00<01:09, 378kB/s]
 1%| | 197k/26.4M [00:00<00:43, 600kB/s]
 3%|▎ | 754k/26.4M [00:00<00:14, 1.79MB/s]
 11%|█▏ | 3.01M/26.4M [00:00<00:03, 6.22MB/s]
 32%|███▏ | 8.49M/26.4M [00:00<00:01, 15.3MB/s]
 55%|█████▌ | 14.5M/26.4M [00:01<00:00, 22.0MB/s]
 79%|███████▊ | 20.8M/26.4M [00:01<00:00, 26.5MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 20.1MB/s]

 0%| | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 342kB/s]

 0%| | 0.00/4.42M [00:00<?, ?B/s]
 1%|▏ | 65.5k/4.42M [00:00<00:11, 376kB/s]
 5%|▌ | 229k/4.42M [00:00<00:05, 708kB/s]
 21%|██ | 918k/4.42M [00:00<00:01, 2.19MB/s]
 82%|████████▏ | 3.64M/4.42M [00:00<00:00, 7.48MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.33MB/s]

 0%| | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 33.4MB/s]
```

We pass the `Dataset` as an argument to `DataLoader`. This wraps an iterable over our dataset, and supports
automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
in the dataloader iterable will return a batch of 64 features and labels.

```
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
 print(f"Shape of X [N, C, H, W]: {X.shape}")
 print(f"Shape of y: {y.shape} {y.dtype}")
 break
```

```
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```

Read more about [loading data in PyTorch](data_tutorial.html).

---

## Creating Models

To define a neural network in PyTorch, we create a class that inherits
from [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We define the layers of the network
in the `__init__` function and specify how data will pass through the network in the `forward` function. To accelerate
operations in the neural network, we move it to the [accelerator](https://pytorch.org/docs/stable/torch.html#accelerators)
such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

```
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
 def __init__(self):
 super().__init__()
 self.flatten = nn.Flatten()
 self.linear_relu_stack = nn.Sequential(
 nn.Linear(28*28, 512),
 nn.ReLU(),
 nn.Linear(512, 512),
 nn.ReLU(),
 nn.Linear(512, 10)
 )

 def forward(self, x):
 x = self.flatten(x)
 logits = self.linear_relu_stack(x)
 return logits

model = NeuralNetwork().to(device)
print(model)
```

```
Using cuda device
NeuralNetwork(
 (flatten): Flatten(start_dim=1, end_dim=-1)
 (linear_relu_stack): Sequential(
 (0): Linear(in_features=784, out_features=512, bias=True)
 (1): ReLU()
 (2): Linear(in_features=512, out_features=512, bias=True)
 (3): ReLU()
 (4): Linear(in_features=512, out_features=10, bias=True)
 )
)
```

Read more about [building neural networks in PyTorch](buildmodel_tutorial.html).

---

## Optimizing the Model Parameters

To train a model, we need a [loss function](https://pytorch.org/docs/stable/nn.html#loss-functions)
and an [optimizer](https://pytorch.org/docs/stable/optim.html).

```
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
backpropagates the prediction error to adjust the model's parameters.

```
def train(dataloader, model, loss_fn, optimizer):
 size = len(dataloader.dataset)
 model.train()
 for batch, (X, y) in enumerate(dataloader):
 X, y = X.to(device), y.to(device)

 # Compute prediction error
 pred = model(X)
 loss = loss_fn(pred, y)

 # Backpropagation
 loss.backward()
 optimizer.step()
 optimizer.zero_grad()

 if batch % 100 == 0:
 loss, current = loss.item(), (batch + 1) * len(X)
 print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
```

We also check the model's performance against the test dataset to ensure it is learning.

```
def test(dataloader, model, loss_fn):
 size = len(dataloader.dataset)
 num_batches = len(dataloader)
 model.eval()
 test_loss, correct = 0, 0
 with torch.no_grad():
 for X, y in dataloader:
 X, y = X.to(device), y.to(device)
 pred = model(X)
 test_loss += loss_fn(pred, y).item()
 correct += (pred.argmax(1) == y).type(torch.float).sum().item()
 test_loss /= num_batches
 correct /= size
 print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

The training process is conducted over several iterations (*epochs*). During each epoch, the model learns
parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
accuracy increase and the loss decrease with every epoch.

```
epochs = 5
for t in range(epochs):
 print(f"Epoch {t+1}\n-------------------------------")
 train(train_dataloader, model, loss_fn, optimizer)
 test(test_dataloader, model, loss_fn)
print("Done!")
```

```
Epoch 1
-------------------------------
loss: 2.301718 [ 64/60000]
loss: 2.289845 [ 6464/60000]
loss: 2.275136 [12864/60000]
loss: 2.270360 [19264/60000]
loss: 2.244479 [25664/60000]
loss: 2.229858 [32064/60000]
loss: 2.232226 [38464/60000]
loss: 2.209149 [44864/60000]
loss: 2.202223 [51264/60000]
loss: 2.169456 [57664/60000]
Test Error:
 Accuracy: 50.5%, Avg loss: 2.165119

Epoch 2
-------------------------------
loss: 2.178059 [ 64/60000]
loss: 2.163512 [ 6464/60000]
loss: 2.116070 [12864/60000]
loss: 2.124641 [19264/60000]
loss: 2.065725 [25664/60000]
loss: 2.022393 [32064/60000]
loss: 2.043396 [38464/60000]
loss: 1.978659 [44864/60000]
loss: 1.971365 [51264/60000]
loss: 1.895216 [57664/60000]
Test Error:
 Accuracy: 56.0%, Avg loss: 1.895718

Epoch 3
-------------------------------
loss: 1.934251 [ 64/60000]
loss: 1.896154 [ 6464/60000]
loss: 1.787186 [12864/60000]
loss: 1.817235 [19264/60000]
loss: 1.711022 [25664/60000]
loss: 1.662655 [32064/60000]
loss: 1.686336 [38464/60000]
loss: 1.594773 [44864/60000]
loss: 1.606853 [51264/60000]
loss: 1.503928 [57664/60000]
Test Error:
 Accuracy: 59.9%, Avg loss: 1.521603

Epoch 4
-------------------------------
loss: 1.591889 [ 64/60000]
loss: 1.553264 [ 6464/60000]
loss: 1.405535 [12864/60000]
loss: 1.474097 [19264/60000]
loss: 1.365304 [25664/60000]
loss: 1.350387 [32064/60000]
loss: 1.375024 [38464/60000]
loss: 1.301211 [44864/60000]
loss: 1.328254 [51264/60000]
loss: 1.237443 [57664/60000]
Test Error:
 Accuracy: 63.1%, Avg loss: 1.259337

Epoch 5
-------------------------------
loss: 1.338156 [ 64/60000]
loss: 1.319268 [ 6464/60000]
loss: 1.152916 [12864/60000]
loss: 1.258151 [19264/60000]
loss: 1.140380 [25664/60000]
loss: 1.152286 [32064/60000]
loss: 1.185781 [38464/60000]
loss: 1.123737 [44864/60000]
loss: 1.156528 [51264/60000]
loss: 1.080697 [57664/60000]
Test Error:
 Accuracy: 64.8%, Avg loss: 1.097443

Done!
```

Read more about [Training your model](optimization_tutorial.html).

---

## Saving Models

A common way to save a model is to serialize the internal state dictionary (containing the model parameters).

```
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

```
Saved PyTorch Model State to model.pth
```

## Loading Models

The process for loading a model includes re-creating the model structure and loading
the state dictionary into it.

```
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

```
<All keys matched successfully>
```

This model can now be used to make predictions.

```
classes = [
 "T-shirt/top",
 "Trouser",
 "Pullover",
 "Dress",
 "Coat",
 "Sandal",
 "Shirt",
 "Sneaker",
 "Bag",
 "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
 x = x.to(device)
 pred = model(x)
 predicted, actual = classes[pred[0].argmax(0)], classes[y]
 print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

```
Predicted: "Ankle boot", Actual: "Ankle boot"
```

Read more about [Saving & Loading your model](saveloadrun_tutorial.html).

**Total running time of the script:** (0 minutes 56.006 seconds)

[`Download Jupyter notebook: quickstart_tutorial.ipynb`](../../_downloads/af0caf6d7af0dda755f4c9d7af9ccc2c/quickstart_tutorial.ipynb)

[`Download Python source code: quickstart_tutorial.py`](../../_downloads/51f1e1167acc0fda8f9d8fd8597ee626/quickstart_tutorial.py)

[`Download zipped: quickstart_tutorial.zip`](../../_downloads/b52a0c6f52468d6fc6aa7623ebc1f99c/quickstart_tutorial.zip)