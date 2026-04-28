# Extension

This section provides insights into extending PyTorch's capabilities.
It covers custom operations, frontend APIs, and advanced topics like
C++ extensions and dispatcher usage.

---

[#### PyTorch Custom Operators Landing Page

This is the landing page for all things related to custom operators in PyTorch.

Extending-PyTorch,Frontend-APIs,C++,CUDA

![](_static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png)](advanced/custom_ops_landing_page.html)

[#### Custom Python Operators

Create Custom Operators in Python. Useful for black-boxing a Python function for use with torch.compile.

Extending-PyTorch,Frontend-APIs,C++,CUDA

![](_static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png)](advanced/python_custom_ops.html)

[#### Custom C++ and CUDA Operators

How to extend PyTorch with custom C++ and CUDA operators.

Extending-PyTorch,Frontend-APIs,C++,CUDA

![](_static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png)](advanced/cpp_custom_ops.html)

[#### Custom Function Tutorial: Double Backward

Learn how to write a custom autograd Function that supports double backward.

Extending-PyTorch,Frontend-APIs

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/custom_function_double_backward_tutorial.html)

[#### Custom Function Tutorial: Fusing Convolution and Batch Norm

Learn how to create a custom autograd Function that fuses batch norm into a convolution to improve memory usage.

Extending-PyTorch,Frontend-APIs

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/custom_function_conv_bn_tutorial.html)

[#### Registering a Dispatched Operator in C++

The dispatcher is an internal component of PyTorch which is responsible for figuring out what code should actually get run when you call a function like torch::add.

Extending-PyTorch,Frontend-APIs,C++

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](advanced/dispatcher.html)

[#### Extending Dispatcher For a New Backend in C++

Learn how to extend the dispatcher to add a new device living outside of the pytorch/pytorch repo and maintain it to keep in sync with native PyTorch devices.

Extending-PyTorch,Frontend-APIs,C++

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](advanced/extend_dispatcher.html)

[#### Facilitating New Backend Integration by PrivateUse1

Learn how to integrate a new backend living outside of the pytorch/pytorch repo and maintain it to keep in sync with the native PyTorch backend.

Extending-PyTorch,Frontend-APIs,C++

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](advanced/privateuseone.html)