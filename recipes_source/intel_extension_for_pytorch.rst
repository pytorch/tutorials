Intel® Extension for PyTorch*
=============================

Intel® Extension for PyTorch* extends PyTorch* with up-to-date features
optimizations for an extra performance boost on Intel hardware. Optimizations
take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and
Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel
X\ :sup:`e`\  Matrix Extensions (XMX) AI engines on Intel discrete GPUs.
Moreover, through PyTorch* `xpu` device, Intel® Extension for PyTorch* provides
easy GPU acceleration for Intel discrete GPUs with PyTorch*.

Intel® Extension for PyTorch* has been released as an open–source project
at `Github <https://github.com/intel/intel-extension-for-pytorch>`_.

- Source code for CPU is available at `master branch <https://github.com/intel/intel-extension-for-pytorch/tree/master>`_.
- Source code for GPU is available at `xpu-master branch <https://github.com/intel/intel-extension-for-pytorch/tree/xpu-master>`_.

Features
--------

Intel® Extension for PyTorch* shares most of features for CPU and GPU.

- **Ease-of-use Python API:** Intel® Extension for PyTorch* provides simple
  frontend Python APIs and utilities for users to get performance optimizations
  such as graph optimization and operator optimization with minor code changes.
  Typically, only 2 to 3 clauses are required to be added to the original code.
- **Channels Last:** Comparing to the default NCHW memory format, channels_last
  (NHWC) memory format could further accelerate convolutional neural networks.
  In Intel® Extension for PyTorch*, NHWC memory format has been enabled for
  most key CPU operators, though not all of them have been merged to PyTorch
  master branch yet. They are expected to be fully landed in PyTorch upstream
  soon.
- **Auto Mixed Precision (AMP):** Low precision data type BFloat16 has been
  natively supported on the 3rd Generation Xeon scalable Servers (aka Cooper
  Lake) with AVX512 instruction set and will be supported on the next
  generation of Intel® Xeon® Scalable Processors with Intel® Advanced Matrix
  Extensions (Intel® AMX) instruction set with further boosted performance. The
  support of Auto Mixed Precision (AMP) with BFloat16 for CPU and BFloat16
  optimization of operators have been massively enabled in Intel® Extension
  for PyTorch*, and partially upstreamed to PyTorch master branch. Most of
  these optimizations will be landed in PyTorch master through PRs that are
  being submitted and reviewed. Auto Mixed Precision (AMP) with both BFloat16
  and Float16 have been enabled for Intel discrete GPUs.
- **Graph Optimization:** To optimize performance further with torchscript,
  Intel® Extension for PyTorch* supports fusion of frequently used operator
  patterns, like Conv2D+ReLU, Linear+ReLU, etc. The benefit of the fusions are
  delivered to users in a transparent fashion. Detailed fusion patterns
  supported can be found `here <https://github.com/intel/intel-extension-for-pytorch>`_.
  The graph optimization will be up-streamed to PyTorch with the introduction
  of oneDNN Graph API.
- **Operator Optimization:** Intel® Extension for PyTorch* also optimizes
  operators and implements several customized operators for performance. A few
  ATen operators are replaced by their optimized counterparts in Intel®
  Extension for PyTorch* via ATen registration mechanism. Moreover, some
  customized operators are implemented for several popular topologies. For
  instance, ROIAlign and NMS are defined in Mask R-CNN. To improve performance
  of these topologies, Intel® Extension for PyTorch* also optimized these
  customized operators.

Getting Started
---------------

Minor code changes are required for users to get start with Intel® Extension
for PyTorch*. Both PyTorch imperative mode and TorchScript mode are
supported. This section introduces usage of Intel® Extension for PyTorch* API
functions for both imperative mode and TorchScript mode, covering data type
Float32 and BFloat16. C++ usage will also be introduced at the end.

You just need to import Intel® Extension for PyTorch* package and apply its
optimize function against the model object. If it is a training workload, the
optimize function also needs to be applied against the optimizer object.

For training and inference with BFloat16 data type, `torch.cpu.amp` has been
enabled in PyTorch upstream to support mixed precision with convenience.
BFloat16 datatype has been enabled excessively for CPU operators in PyTorch
upstream and Intel® Extension for PyTorch*. Meanwhile `torch.xpu.amp`,
registered by Intel® Extension for PyTorch*, enables easy usage of BFloat16
and Float16 data types on Intel discrete GPUs. Either `torch.cpu.amp` or
`torch.xpu.amp` matches each operator to its appropriate datatype automatically
and returns the best possible performance.

Examples -- CPU
---------------

This section shows examples of training and inference on CPU with Intel®
Extension for PyTorch*

The code changes that are required for Intel® Extension for PyTorch* are
highlighted.

Training
~~~~~~~~

Float32
^^^^^^^

.. code:: python3

   import torch
   import torchvision
   import intel_extension_for_pytorch as ipex

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
   model, optimizer = ipex.optimize(model, optimizer=optimizer)

   for batch_idx, (data, target) in enumerate(train_loader):
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       print(batch_idx)
   torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pth')

BFloat16
^^^^^^^^

.. code:: python3

   import torch
   import torchvision
   import intel_extension_for_pytorch as ipex

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
   model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

   for batch_idx, (data, target) in enumerate(train_loader):
       optimizer.zero_grad()
       with torch.cpu.amp.autocast():
           output = model(data)
           loss = criterion(output, target)
           loss.backward()
       optimizer.step()
       print(batch_idx)
   torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pth')

Inference - Imperative Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Float32
^^^^^^^

.. code:: python3

   import torch
   import torchvision.models as models

   model = models.resnet50(pretrained=True)
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex
   model = ipex.optimize(model)
   ######################################################

   with torch.no_grad():
     model(data)

BFloat16
^^^^^^^^

.. code:: python3

   import torch
   from transformers import BertModel

   model = BertModel.from_pretrained(args.model_name)
   model.eval()

   vocab_size = model.config.vocab_size
   batch_size = 1
   seq_length = 512
   data = torch.randint(vocab_size, size=[batch_size, seq_length])

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex
   model = ipex.optimize(model, dtype=torch.bfloat16)
   ######################################################

   with torch.no_grad():
     with torch.cpu.amp.autocast():
       model(data)

Inference - TorchScript Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TorchScript mode makes graph optimization possible, hence improves
performance for some topologies. Intel® Extension for PyTorch* enables most
commonly used operator pattern fusion, and users can get the performance
benefit without additional code changes.

Float32
^^^^^^^

.. code:: python3

   import torch
   import torchvision.models as models

   model = models.resnet50(pretrained=True)
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex
   model = ipex.optimize(model)
   ######################################################

   with torch.no_grad():
     d = torch.rand(1, 3, 224, 224)
     model = torch.jit.trace(model, d)
     model = torch.jit.freeze(model)

     model(data)

BFloat16
^^^^^^^^

.. code:: python3

   import torch
   from transformers import BertModel

   model = BertModel.from_pretrained(args.model_name)
   model.eval()

   vocab_size = model.config.vocab_size
   batch_size = 1
   seq_length = 512
   data = torch.randint(vocab_size, size=[batch_size, seq_length])

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex
   model = ipex.optimize(model, dtype=torch.bfloat16)
   ######################################################

   with torch.no_grad():
     with torch.cpu.amp.autocast():
       d = torch.randint(vocab_size, size=[batch_size, seq_length])
       model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
       model = torch.jit.freeze(model)

       model(data)

Examples -- GPU
---------------

This section shows examples of training and inference on GPU with Intel®
Extension for PyTorch*

The code changes that are required for Intel® Extension for PyTorch* are
highlighted with comments in a line above.

Training
~~~~~~~~

Float32
^^^^^^^

.. code:: python3

   import torch
   import torchvision
   ############# code changes ###############
   import intel_extension_for_pytorch as ipex
   ############# code changes ###############

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
   #################################### code changes ################################
   model = model.to("xpu")
   model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
   #################################### code changes ################################

   for batch_idx, (data, target) in enumerate(train_loader):
       ########## code changes ##########
       data = data.to("xpu")
       target = target.to("xpu")
       ########## code changes ##########
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       print(batch_idx)
   torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pth')

BFloat16
^^^^^^^^

.. code:: python3

   import torch
   import torchvision
   ############# code changes ###############
   import intel_extension_for_pytorch as ipex
   ############# code changes ###############

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
   ##################################### code changes ################################
   model = model.to("xpu")
   model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
   ##################################### code changes ################################

   for batch_idx, (data, target) in enumerate(train_loader):
       optimizer.zero_grad()
       ######################### code changes #########################
       data = data.to("xpu")
       target = target.to("xpu")
       with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
       ######################### code changes #########################
           output = model(data)
           loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       print(batch_idx)
   torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pth')

Inference - Imperative Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Float32
^^^^^^^

.. code:: python3

   import torch
   import torchvision.models as models
   ############# code changes ###############
   import intel_extension_for_pytorch as ipex
   ############# code changes ###############

   model = models.resnet50(pretrained=True)
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   model = model.to(memory_format=torch.channels_last)
   data = data.to(memory_format=torch.channels_last)

   #################### code changes ################
   model = model.to("xpu")
   data = data.to("xpu")
   model = ipex.optimize(model, dtype=torch.float32)
   #################### code changes ################

   with torch.no_grad():
     model(data)

BFloat16
^^^^^^^^

.. code:: python3

   import torch
   import torchvision.models as models
   ############# code changes ###############
   import intel_extension_for_pytorch as ipex
   ############# code changes ###############

   model = models.resnet50(pretrained=True)
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   model = model.to(memory_format=torch.channels_last)
   data = data.to(memory_format=torch.channels_last)

   #################### code changes #################
   model = model.to("xpu")
   data = data.to("xpu")
   model = ipex.optimize(model, dtype=torch.bfloat16)
   #################### code changes #################

   with torch.no_grad():
     ################################# code changes ######################################
     with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=False):
     ################################# code changes ######################################
       model(data)

Float16
^^^^^^^

.. code:: python3

   import torch
   import torchvision.models as models
   ############# code changes ###############
   import intel_extension_for_pytorch as ipex
   ############# code changes ###############

   model = models.resnet50(pretrained=True)
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   model = model.to(memory_format=torch.channels_last)
   data = data.to(memory_format=torch.channels_last)

   #################### code changes ################
   model = model.to("xpu")
   data = data.to("xpu")
   model = ipex.optimize(model, dtype=torch.float16)
   #################### code changes ################

   with torch.no_grad():
     ################################# code changes ######################################
     with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=False):
     ################################# code changes ######################################
       model(data)

Inference - TorchScript Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TorchScript mode makes graph optimization possible, hence improves
performance for some topologies. Intel® Extension for PyTorch* enables most
commonly used operator pattern fusion, and users can get the performance
benefit without additional code changes.

Float32
^^^^^^^

.. code:: python3

   import torch
   from transformers import BertModel
   ############# code changes ###############
   import intel_extension_for_pytorch as ipex
   ############# code changes ###############

   model = BertModel.from_pretrained(args.model_name)
   model.eval()

   vocab_size = model.config.vocab_size
   batch_size = 1
   seq_length = 512
   data = torch.randint(vocab_size, size=[batch_size, seq_length])

   #################### code changes ################
   model = model.to("xpu")
   data = data.to("xpu")
   model = ipex.optimize(model, dtype=torch.float32)
   #################### code changes ################

   with torch.no_grad():
     d = torch.randint(vocab_size, size=[batch_size, seq_length])
     ##### code changes #####
     d = d.to("xpu")
     ##### code changes #####
     model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
     model = torch.jit.freeze(model)

     model(data)

BFloat16
^^^^^^^^

.. code:: python3

   import torch
   from transformers import BertModel
   ############# code changes ###############
   import intel_extension_for_pytorch as ipex
   ############# code changes ###############

   model = BertModel.from_pretrained(args.model_name)
   model.eval()

   vocab_size = model.config.vocab_size
   batch_size = 1
   seq_length = 512
   data = torch.randint(vocab_size, size=[batch_size, seq_length])

   #################### code changes #################
   model = model.to("xpu")
   data = data.to("xpu")
   model = ipex.optimize(model, dtype=torch.bfloat16)
   #################### code changes #################

   with torch.no_grad():
     d = torch.randint(vocab_size, size=[batch_size, seq_length])
     ################################# code changes ######################################
     d = d.to("xpu")
     with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=False):
     ################################# code changes ######################################
       model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
       model = torch.jit.freeze(model)

       model(data)

Float16
^^^^^^^

.. code:: python3

   import torch
   from transformers import BertModel
   ############# code changes ###############
   import intel_extension_for_pytorch as ipex
   ############# code changes ###############

   model = BertModel.from_pretrained(args.model_name)
   model.eval()

   vocab_size = model.config.vocab_size
   batch_size = 1
   seq_length = 512
   data = torch.randint(vocab_size, size=[batch_size, seq_length])

   #################### code changes ################
   model = model.to("xpu")
   data = data.to("xpu")
   model = ipex.optimize(model, dtype=torch.float16)
   #################### code changes ################

   with torch.no_grad():
     d = torch.randint(vocab_size, size=[batch_size, seq_length])
     ################################# code changes ######################################
     d = d.to("xpu")
     with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=False):
     ################################# code changes ######################################
       model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
       model = torch.jit.freeze(model)

       model(data)

C++ (CPU only)
~~~~~~~~~~~~~~

To work with libtorch, C++ library of PyTorch, Intel® Extension for PyTorch*
provides its C++ dynamic library as well. The C++ library is supposed to handle
inference workload only, such as service deployment. For regular development,
please use Python interface. Comparing to usage of libtorch, no specific code
changes are required, except for converting input data into channels last data
format. Compilation follows the recommended methodology with CMake. Detailed
instructions can be found in `PyTorch tutorial <https://pytorch.org/tutorials/advanced/cpp_export.html#depending-on-libtorch-and-building-the-application>`_.
During compilation, Intel optimizations will be activated automatically
once C++ dynamic library of Intel® Extension for PyTorch* is linked.

**example-app.cpp**

.. code:: cpp

   #include <torch/script.h>
   #include <iostream>
   #include <memory>

   int main(int argc, const char* argv[]) {
       torch::jit::script::Module module;
       try {
           module = torch::jit::load(argv[1]);
       }
       catch (const c10::Error& e) {
           std::cerr << "error loading the model\n";
           return -1;
       }
       std::vector<torch::jit::IValue> inputs;
       // make sure input data are converted to channels last format
       inputs.push_back(torch::ones({1, 3, 224, 224}).to(c10::MemoryFormat::ChannelsLast));

       at::Tensor output = module.forward(inputs).toTensor();

       return 0;
   }

**CMakeLists.txt**

::

   cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
   project(example-app)

   find_package(intel_ext_pt_cpu REQUIRED)

   add_executable(example-app example-app.cpp)
   target_link_libraries(example-app "${TORCH_LIBRARIES}")

   set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

**Command for compilation**

::

   $ cmake -DCMAKE_PREFIX_PATH=<LIBPYTORCH_PATH> ..
   $ make

If `Found INTEL_EXT_PT_CPU` is shown as `TRUE`, the extension had been linked
into the binary. This can be verified with the Linux command `ldd`.

::

   $ cmake -DCMAKE_PREFIX_PATH=/workspace/libtorch ..
   -- The C compiler identification is GNU 9.3.0
   -- The CXX compiler identification is GNU 9.3.0
   -- Check for working C compiler: /usr/bin/cc
   -- Check for working C compiler: /usr/bin/cc -- works
   -- Detecting C compiler ABI info
   -- Detecting C compiler ABI info - done
   -- Detecting C compile features
   -- Detecting C compile features - done
   -- Check for working CXX compiler: /usr/bin/c++
   -- Check for working CXX compiler: /usr/bin/c++ -- works
   -- Detecting CXX compiler ABI info
   -- Detecting CXX compiler ABI info - done
   -- Detecting CXX compile features
   -- Detecting CXX compile features - done
   -- Looking for pthread.h
   -- Looking for pthread.h - found
   -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
   -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
   -- Looking for pthread_create in pthreads
   -- Looking for pthread_create in pthreads - not found
   -- Looking for pthread_create in pthread
   -- Looking for pthread_create in pthread - found
   -- Found Threads: TRUE
   -- Found Torch: /workspace/libtorch/lib/libtorch.so
   -- Found INTEL_EXT_PT_CPU: TRUE
   -- Configuring done
   -- Generating done
   -- Build files have been written to: /workspace/build

   $ ldd example-app
           ...
           libtorch.so => /workspace/libtorch/lib/libtorch.so (0x00007f3cf98e0000)
           libc10.so => /workspace/libtorch/lib/libc10.so (0x00007f3cf985a000)
           libintel-ext-pt-cpu.so => /workspace/libtorch/lib/libintel-ext-pt-cpu.so (0x00007f3cf70fc000)
           libtorch_cpu.so => /workspace/libtorch/lib/libtorch_cpu.so (0x00007f3ce16ac000)
           ...
           libdnnl_graph.so.0 => /workspace/libtorch/lib/libdnnl_graph.so.0 (0x00007f3cde954000)
           ...

Model Zoo (CPU only)
--------------------

Use cases that had already been optimized by Intel engineers are available at
`Model Zoo for Intel® Architecture <https://github.com/IntelAI/models/>`_ (with
the branch name in format of `pytorch-r<version>-models`). Many PyTorch use
cases for benchmarking are also available on the GitHub page. You can get
performance benefits out-of-the-box by simply running scripts in the Model Zoo.

Tutorials
---------

More detailed tutorials are available in the official Intel® Extension
for PyTorch* Documentation:

- `CPU <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/>`_
- `GPU <https://intel.github.io/intel-extension-for-pytorch/xpu/latest/>`_
