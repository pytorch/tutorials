Intel® Extension for PyTorch* Backend
=====================================

To work better with `torch.compile`, Intel® Extension for PyTorch* implements a backend ``ipex``. 
It targets to improve hardware resource usage efficiency on Intel platforms for better performance.
The `ipex` backend is implemented with further customizations designed in Intel® Extension for
PyTorch* for the model compilation.

Usage Example
~~~~~~~~~~~~~

Train FP32
----------

Check the example below to learn how to utilize the `ipex` backend with `torch.compile` for model training with FP32 data type.

.. code:: python

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = 'datasets/cifar10/'

   transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   train_dataset = torchvision.datasets.CIFAR10(
     root=DATA,
     train=True,
     transform=transform,
     download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(
     dataset=train_dataset,
     batch_size=128
   )

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
   model.train()

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex

   # Invoke the following API optionally, to apply frontend optimizations
   model, optimizer = ipex.optimize(model, optimizer=optimizer)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   for batch_idx, (data, target) in enumerate(train_loader):
       optimizer.zero_grad()
       output = compile_model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()


Train BF16
----------

Check the example below to learn how to utilize the `ipex` backend with `torch.compile` for model training with BFloat16 data type.

.. code:: python

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = 'datasets/cifar10/'

   transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   train_dataset = torchvision.datasets.CIFAR10(
     root=DATA,
     train=True,
     transform=transform,
     download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(
     dataset=train_dataset,
     batch_size=128
   )

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
   model.train()

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex

   # Invoke the following API optionally, to apply frontend optimizations
   model, optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   with torch.cpu.amp.autocast():
       for batch_idx, (data, target) in enumerate(train_loader):
           optimizer.zero_grad()
           output = compile_model(data)
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()


Inference FP32
--------------

Check the example below to learn how to utilize the `ipex` backend with `torch.compile` for model inference with FP32 data type.

.. code:: python

   import torch
   import torchvision.models as models

   model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex

   # Invoke the following API optionally, to apply frontend optimizations
   model = ipex.optimize(model, weights_prepack=False)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   with torch.no_grad():
       compile_model(data)


Inference BF16
--------------

Check the example below to learn how to utilize the `ipex` backend with `torch.compile` for model inference with BFloat16 data type.

.. code:: python

   import torch
   import torchvision.models as models

   model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex

   # Invoke the following API optionally, to apply frontend optimizations
   model = ipex.optimize(model, dtype=torch.bfloat16, weights_prepack=False)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
       compile_model(data)
