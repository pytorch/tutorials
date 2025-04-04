:orphan:

Extension
=========

This section provides insights into extending PyTorch's capabilities.
It covers custom operations, frontend APIs, and advanced topics like
C++ extensions and dispatcher usage.

.. raw:: html

    <div id="tutorial-cards-container">

    <nav class="navbar navbar-expand-lg navbar-light tutorials-nav col-12">
        <div class="tutorial-tags-container">
            <div id="dropdown-filter-tags">
                <div class="tutorial-filter-menu">
                    <div class="tutorial-filter filter-btn all-tag-selected" data-tag="all">All</div>
                </div>
            </div>
        </div>
    </nav>

    <hr class="tutorials-hr">

    <div class="row">

    <div id="tutorial-cards">
    <div class="list">

.. Add tutorial cards below this line

.. customcarditem::
   :header: PyTorch Custom Operators Landing Page
   :card_description: This is the landing page for all things related to custom operators in PyTorch.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/custom_ops_landing_page.html
   :tags: Extending-PyTorch,Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Custom Python Operators
   :card_description: Create Custom Operators in Python. Useful for black-boxing a Python function for use with torch.compile.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/python_custom_ops.html
   :tags: Extending-PyTorch,Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Custom C++ and CUDA Operators
   :card_description: How to extend PyTorch with custom C++ and CUDA operators.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/cpp_custom_ops.html
   :tags: Extending-PyTorch,Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Custom Function Tutorial: Double Backward
   :card_description: Learn how to write a custom autograd Function that supports double backward.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/custom_function_double_backward_tutorial.html
   :tags: Extending-PyTorch,Frontend-APIs

.. customcarditem::
   :header: Custom Function Tutorial: Fusing Convolution and Batch Norm
   :card_description: Learn how to create a custom autograd Function that fuses batch norm into a convolution to improve memory usage.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/custom_function_conv_bn_tutorial.html
   :tags: Extending-PyTorch,Frontend-APIs

.. customcarditem::
   :header: Custom C++ and CUDA Extensions
   :card_description: Create a neural network layer with no parameters using numpy. Then use scipy to create a neural network layer that has learnable weights.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/cpp_extension.html
   :tags: Extending-PyTorch,Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Operators
   :card_description: Implement a custom TorchScript operator in C++, how to build it into a shared library, how to use it in Python to define TorchScript models and lastly how to load it into a C++ application for inference workloads.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Operators.png
   :link: advanced/torch_script_custom_ops.html
   :tags: Extending-PyTorch,Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Classes
   :card_description: This is a continuation of the custom operator tutorial, and introduces the API we’ve built for binding C++ classes into TorchScript and Python simultaneously.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Classes.png
   :link: advanced/torch_script_custom_classes.html
   :tags: Extending-PyTorch,Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Registering a Dispatched Operator in C++
   :card_description: The dispatcher is an internal component of PyTorch which is responsible for figuring out what code should actually get run when you call a function like torch::add.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/dispatcher.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Extending Dispatcher For a New Backend in C++
   :card_description: Learn how to extend the dispatcher to add a new device living outside of the pytorch/pytorch repo and maintain it to keep in sync with native PyTorch devices.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/extend_dispatcher.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Facilitating New Backend Integration by PrivateUse1
   :card_description: Learn how to integrate a new backend living outside of the pytorch/pytorch repo and maintain it to keep in sync with the native PyTorch backend.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/privateuseone.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. End of tutorial card section

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Extending PyTorch

   advanced/custom_ops_landing_page
   advanced/python_custom_ops
   advanced/cpp_custom_ops
   intermediate/custom_function_double_backward_tutorial
   intermediate/custom_function_conv_bn_tutorial
   advanced/cpp_extension
   advanced/torch_script_custom_ops
   advanced/torch_script_custom_classes
   advanced/dispatcher
   advanced/extend_dispatcher
   advanced/privateuseone
