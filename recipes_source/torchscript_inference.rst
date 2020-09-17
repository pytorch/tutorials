TorchScript for Deployment
==========================

In this recipe, you will learn:

-  What TorchScript is
-  How to export your trained model in TorchScript format
-  How to load your TorchScript model in C++ and do inference

Requirements
------------

-  PyTorch 1.5
-  TorchVision 0.6.0
-  libtorch 1.5
-  C++ compiler

The instructions for installing the three PyTorch components are
available at `pytorch.org`_. The C++ compiler will depend on your
platform.

What is TorchScript?
--------------------

**TorchScript** is an intermediate representation of a PyTorch model
(subclass of ``nn.Module``) that can then be run in a high-performance
environment like C++. It’s a high-performance subset of Python that is
meant to be consumed by the **PyTorch JIT Compiler,** which performs
run-time optimization on your model’s computation. TorchScript is the
recommended model format for doing scaled inference with PyTorch models.
For more information, see the PyTorch `Introduction to TorchScript
tutorial`_, the `Loading A TorchScript Model in C++ tutorial`_, and the
`full TorchScript documentation`_, all of which are available on
`pytorch.org`_.

How to Export Your Model
------------------------

As an example, let’s take a pretrained vision model. All of the
pretrained models in TorchVision are compatible with TorchScript.

Run the following Python 3 code, either in a script or from the REPL:

.. code:: python3

   import torch
   import torch.nn.functional as F
   import torchvision.models as models

   r18 = models.resnet18(pretrained=True)       # We now have an instance of the pretrained model
   r18_scripted = torch.jit.script(r18)         # *** This is the TorchScript export
   dummy_input = torch.rand(1, 3, 224, 224)     # We should run a quick test

Let’s do a sanity check on the equivalence of the two models:

::

   unscripted_output = r18(dummy_input)         # Get the unscripted model's prediction...
   scripted_output = r18_scripted(dummy_input)  # ...and do the same for the scripted version

   unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
   scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

   print('Python model top 5 results:\n  {}'.format(unscripted_top5))
   print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))

You should see that both versions of the model give the same results:

::

   Python model top 5 results:
     tensor([[463, 600, 731, 899, 898]])
   TorchScript model top 5 results:
     tensor([[463, 600, 731, 899, 898]])

With that check confirmed, go ahead and save the model:

::

   r18_scripted.save('r18_scripted.pt')

Loading TorchScript Models in C++
---------------------------------

Create the following C++ file and name it ``ts-infer.cpp``:

.. code:: cpp

   #include <torch/script.h>
   #include <torch/nn/functional/activation.h>


   int main(int argc, const char* argv[]) {
       if (argc != 2) {
           std::cerr << "usage: ts-infer <path-to-exported-model>\n";
           return -1;
       }

       std::cout << "Loading model...\n";

       // deserialize ScriptModule
       torch::jit::script::Module module;
       try {
           module = torch::jit::load(argv[1]);
       } catch (const c10::Error& e) {
           std::cerr << "Error loading model\n";
           std::cerr << e.msg_without_backtrace();
           return -1;
       }

       std::cout << "Model loaded successfully\n";

       torch::NoGradGuard no_grad; // ensures that autograd is off
       module.eval(); // turn off dropout and other training-time layers/functions

       // create an input "image"
       std::vector<torch::jit::IValue> inputs;
       inputs.push_back(torch::rand({1, 3, 224, 224}));

       // execute model and package output as tensor
       at::Tensor output = module.forward(inputs).toTensor();

       namespace F = torch::nn::functional;
       at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
       std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
       at::Tensor top5 = std::get<1>(top5_tensor);

       std::cout << top5[0] << "\n";

       std::cout << "\nDONE\n";
       return 0;
   }

This program:

-  Loads the model you specify on the command line
- Creates a dummy “image” input tensor
- Performs inference on the input

Also, notice that there is no dependency on TorchVision in this code.
The saved version of your TorchScript model has your learning weights
*and* your computation graph - nothing else is needed.

Building and Running Your C++ Inference Engine
----------------------------------------------

Create the following ``CMakeLists.txt`` file:

::

   cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
   project(custom_ops)

   find_package(Torch REQUIRED)

   add_executable(ts-infer ts-infer.cpp)
   target_link_libraries(ts-infer "${TORCH_LIBRARIES}")
   set_property(TARGET ts-infer PROPERTY CXX_STANDARD 11)

Make the program:

::

   cmake -DCMAKE_PREFIX_PATH=<path to your libtorch installation>
   make

Now, we can run inference in C++, and verify that we get a result:

::

   $ ./ts-infer r18_scripted.pt
   Loading model...
   Model loaded successfully
    418
    845
    111
    892
    644
   [ CPULongType{5} ]

   DONE

Important Resources
-------------------

-  `pytorch.org`_ for installation instructions, and more documentation
   and tutorials.
-  `Introduction to TorchScript tutorial`_ for a deeper initial
   exposition of TorchScript
-  `Full TorchScript documentation`_ for complete TorchScript language
   and API reference

.. _pytorch.org: https://pytorch.org/
.. _Introduction to TorchScript tutorial: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
.. _Full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
.. _Loading A TorchScript Model in C++ tutorial: https://pytorch.org/tutorials/advanced/cpp_export.html
.. _full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
