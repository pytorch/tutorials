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

```
import torch
import torchvision.models as models
```

## Saving and Loading Model Weights

PyTorch models store the learned parameters in an internal
state dictionary, called `state_dict`. These can be persisted via the `torch.save`
method:

```
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

```
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /var/lib/ci-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth

 0%| | 0.00/528M [00:00<?, ?B/s]
 7%|▋ | 35.1M/528M [00:00<00:01, 368MB/s]
 16%|█▌ | 85.2M/528M [00:00<00:01, 460MB/s]
 24%|██▍ | 129M/528M [00:00<00:00, 440MB/s]
 34%|███▍ | 178M/528M [00:00<00:00, 469MB/s]
 43%|████▎ | 229M/528M [00:00<00:00, 489MB/s]
 52%|█████▏ | 276M/528M [00:00<00:00, 435MB/s]
 62%|██████▏ | 325M/528M [00:00<00:00, 458MB/s]
 71%|███████ | 374M/528M [00:00<00:00, 477MB/s]
 80%|████████ | 424M/528M [00:00<00:00, 490MB/s]
 90%|████████▉ | 473M/528M [00:01<00:00, 498MB/s]
 99%|█████████▉| 523M/528M [00:01<00:00, 504MB/s]
100%|██████████| 528M/528M [00:01<00:00, 478MB/s]
```

To load model weights, you need to create an instance of the same model first, and then load the parameters
using `load_state_dict()` method.

In the code below, we set `weights_only=True` to limit the
functions executed during unpickling to only those necessary for
loading weights. Using `weights_only=True` is considered
a best practice when loading weights.

```
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
```

```
VGG(
 (features): Sequential(
 (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (1): ReLU(inplace=True)
 (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (3): ReLU(inplace=True)
 (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (6): ReLU(inplace=True)
 (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (8): ReLU(inplace=True)
 (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (11): ReLU(inplace=True)
 (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (13): ReLU(inplace=True)
 (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (15): ReLU(inplace=True)
 (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (18): ReLU(inplace=True)
 (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (20): ReLU(inplace=True)
 (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (22): ReLU(inplace=True)
 (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (25): ReLU(inplace=True)
 (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (27): ReLU(inplace=True)
 (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (29): ReLU(inplace=True)
 (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 )
 (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
 (classifier): Sequential(
 (0): Linear(in_features=25088, out_features=4096, bias=True)
 (1): ReLU(inplace=True)
 (2): Dropout(p=0.5, inplace=False)
 (3): Linear(in_features=4096, out_features=4096, bias=True)
 (4): ReLU(inplace=True)
 (5): Dropout(p=0.5, inplace=False)
 (6): Linear(in_features=4096, out_features=1000, bias=True)
 )
)
```

Note

be sure to call `model.eval()` method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.

## Saving and Loading Models with Shapes

When loading model weights, we needed to instantiate the model class first, because the class
defines the structure of a network. We might want to save the structure of this class together with
the model, in which case we can pass `model` (and not `model.state_dict()`) to the saving function:

```
torch.save(model, 'model.pth')
```

We can then load the model as demonstrated below.

As described in [Saving and loading torch.nn.Modules](https://pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules),
saving `state_dict` is considered the best practice. However,
below we use `weights_only=False` because this involves loading the
model, which is a legacy use case for `torch.save`.

```
model = torch.load('model.pth', weights_only=False)
```

Note

This approach uses Python [pickle](https://docs.python.org/3/library/pickle.html) module when serializing the model, thus it relies on the actual class definition to be available when loading the model.

## Related Tutorials

- [Saving and Loading a General Checkpoint in PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [Tips for loading an nn.Module from a checkpoint](https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html?highlight=loading%20nn%20module%20from%20checkpoint)

**Total running time of the script:** (0 minutes 4.288 seconds)

[`Download Jupyter notebook: saveloadrun_tutorial.ipynb`](../../_downloads/11f1adacb7d237f2041ce267ac38abb6/saveloadrun_tutorial.ipynb)

[`Download Python source code: saveloadrun_tutorial.py`](../../_downloads/3648b0dccaebca71b234070fe2124770/saveloadrun_tutorial.py)

[`Download zipped: saveloadrun_tutorial.zip`](../../_downloads/0a63ed31b0b1f27896bbfba4038b8718/saveloadrun_tutorial.zip)