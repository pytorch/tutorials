# Compilers

Explore PyTorch compilers to optimize and deploy models efficiently.
Learn about APIs like `torch.compile` and `torch.export`
that let you enhance model performance and streamline deployment
processes.
Explore advanced topics such as compiled autograd, dynamic compilation
control, as well as third-party backend solutions.

Warning

TorchScript is no longer in active development.

---

[#### torch.compile Tutorial

Speed up your models with minimal code changes using torch.compile, the latest PyTorch compiler solution.

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/torch_compile_tutorial.html)

[#### torch.compile End-to-End Tutorial

An example of applying torch.compile to a real model, demonstrating speedups.

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/torch_compile_full_example.html)

[#### Compiled Autograd: Capturing a larger backward graph for torch.compile

Learn how to use compiled autograd to capture a larger backward graph.

Model-Optimization,CUDA,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/compiled_autograd_tutorial.html)

[#### Inductor CPU Backend Debugging and Profiling

Learn the usage, debugging and performance profiling for ``torch.compile`` with Inductor CPU backend.

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/inductor_debug_cpu.html)

[#### Dynamic Compilation Control with torch.compiler.set_stance

Learn how to use torch.compiler.set_stance

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](recipes/torch_compiler_set_stance_tutorial.html)

[#### Demonstration of torch.export flow, common challenges and the solutions to address them

Learn how to export models for popular usecases

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](recipes/torch_export_challenges_solutions.html)

[#### (beta) Compiling the Optimizer with torch.compile

Speed up the optimizer using torch.compile

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](recipes/compiling_optimizer.html)

[#### (beta) Running the compiled optimizer with an LR Scheduler

Speed up training with LRScheduler and torch.compiled optimizer

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](recipes/compiling_optimizer_lr_scheduler.html)

[#### Using Variable Length Attention with ``torch.compile``

Speed up training with torch.compiled variable length attention

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/variable_length_attention_tutorial.html)

[#### Using User-Defined Triton Kernels with ``torch.compile``

Learn how to use user-defined kernels with ``torch.compile``

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](recipes/torch_compile_user_defined_triton_kernel_tutorial.html)

[#### Compile Time Caching in ``torch.compile``

Learn how to use compile time caching in ``torch.compile``

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](recipes/torch_compile_caching_tutorial.html)

[#### Compile Time Caching Configurations

Learn how to configure compile time caching in ``torch.compile``

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](recipes/torch_compile_caching_configuration_tutorial.html)

[#### Reducing torch.compile cold start compilation time with regional compilation

Learn how to use regional compilation to control cold start compile time

Model-Optimization,torch.compile

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](recipes/regional_compilation.html)

[#### torch.export AOTInductor Tutorial for Python runtime

Learn an end-to-end example of how to use AOTInductor for python runtime.

Basics,torch.export

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](recipes/torch_export_aoti_python.html)

[#### Deep dive into torch.export

Learn how to use torch.export to export PyTorch models into standardized model representations.

Basics,torch.export

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](intermediate/torch_export_tutorial.html)

[#### (optional) Exporting a PyTorch model to ONNX using TorchDynamo backend and Running it using ONNX Runtime

Build a image classifier model in PyTorch and convert it to ONNX before deploying it with ONNX Runtime.

Production,ONNX,Backends

![](_static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png)](beginner/onnx/export_simple_model_to_onnx_tutorial.html)

[#### Extending the ONNX exporter operator support

Demonstrate end-to-end how to address unsupported operators in ONNX.

Production,ONNX,Backends

![](_static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png)](beginner/onnx/onnx_registry_tutorial.html)

[#### Exporting a model with control flow to ONNX

Demonstrate how to handle control flow logic while exporting a PyTorch model to ONNX.

Production,ONNX,Backends

![](_static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png)](beginner/onnx/export_control_flow_model_to_onnx_tutorial.html)

[#### Building a Convolution/Batch Norm fuser in FX

Build a simple FX pass that fuses batch norm into convolution to improve performance during inference.

FX

![](_static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png)](intermediate/torch_compile_conv_bn_fuser.html)

[#### Building a Simple Performance Profiler with FX

Build a simple FX interpreter to record the runtime of op, module, and function calls and report statistics

FX

![](_static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png)](intermediate/fx_profiling_tutorial.html)