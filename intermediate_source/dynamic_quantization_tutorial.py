"""
Dynamic quantization
====================

This tutorial shows how to use PyTorch's dynamic quantization
capabilities, as well as

- Explain what quantization is and why you would want to use it, including
  the difference between:
  - Dynamic quantization
  - Post-training quantization
- Use `torch.quantization.quantize_dynamic` to perform Dynamic Quantization of
  models after training

"""

######################################################################
# Dynamic Quantization
# --------------------
# Dynamic quantization simply involves converting weights from float32
# to int8. Computations with these weights will thus be performed
# using efficient int8 kernels, which, along with the reduced memory
# bandwidth for weights, results in faster compute. Dynamic
# quantization is less helpful with architectures primarily consisting
# of convolution operations, where the number of operations relative
# to the number of weights is large, than with architectures with
# mostly matrix multiplications, such as Transformers.

# imports
import time
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization.quantize import quantize_dynamic

from torchvision import datasets, transforms

# set device and cude
use_cuda = False
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

# set seed
seed = 42
torch.manual_seed(seed)

# Floating point LeNet model
class FloatLeNetModel(nn.Module):
    def __init__(self):
        super(FloatLeNetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_pretrained_model(saved_model_file: str,
                          model_module: nn.Module):
    inference_model=model_module().to(device)
    inference_model.load_state_dict(torch.load(saved_model_file))
    return inference_model


def test(model: nn.Module,
         device: torch.device,
         test_loader: torch.utils.data.DataLoader,
         num_batches: int) -> None:
    '''
    Helper function to evaluate
    '''
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if (batch_idx > num_batches):
                break

    print('\nTest set accuracy: {}/{} ({:.0f}%)\n'.format(
          correct,
          test_loader.batch_size*(num_batches),
          100. * correct /(test_loader.batch_size*(num_batches))))


def run_model(inference_model, batch_size=1000, num_batches=10):

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, download=True,transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test(inference_model, device, test_loader, num_batches)

# Note - change to '.' in final tutorial
mnist_root_dir = '.'

mnist_data = datasets.MNIST(mnist_root_dir, train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))
saved_model_dir = 'data/'

######################################################################
# Note: the line of code below depends on `new_mnist_cnn.pt` having the
# same architecture as `FloatLeNetModel`

my_model = load_pretrained_model(saved_model_dir + 'dq_mnist_cnn.pt', FloatLeNetModel)


def print_size_of(obj):
    pickle.dump(obj, open( "temp.p", "wb" ) )
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print("Size of original (unquantized) model")
print_size_of(my_model)

######################################################################
# Dynamic quantization is really simple: A one line API that replaces
# modules with dynamic quantized versions. We currently support
# dynamic quantization only for ``nn.Linear`` modules, with more being
# planned.

quantized_model = quantize_dynamic(my_model)

print("Size of quantized model")
print_size_of(quantized_model)

print("Starting to run original (unquantized) model")
start = time.time()
run_model(my_model, batch_size = 1000, num_batches = 10)
end = time.time()
print('Time to run original (unquantized) model',end-start)

print("Starting to run quantized model")
start = time.time()
run_model(quantized_model, batch_size = 1000, num_batches = 10)
end = time.time()
print('Time to run quantized model',end-start)

