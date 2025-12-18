.. _glossary:

================
PyTorch Glossary
================

This glossary provides definitions for terms commonly used in PyTorch documentation.

.. glossary::
   :sorted:

   ATen
      Short for "A Tensor Library". The foundational tensor and mathematical
      operation library on which all else is built.

   attention mechanism
      A technique used in deep learning models, particularly transformer architectures,
      to selectively focus on certain input elements or tokens when computing output
      representations, improving performance and interpretability.

   backward pass
      The backward pass is part of the backpropagation algorithm where the error
      gradients are computed and propagated backwards through the network, adjusting
      the weights and biases to minimize the loss.

   backpropagation
      An essential algorithm in training neural networks. It calculates the gradient
      of the loss function with respect to the model's parameters, allowing the
      network to learn from its mistakes and improve over time.

   CNN
      Convolutional Neural Network: A type of neural network designed for image and
      video processing, using convolutional and pooling layers to extract features.

   Compound Kernel
      Opposed to :term:`Device Kernel`, Compound kernels are usually
      device-agnostic and belong to :term:`Compound Operation`.

   Compound Operation
      A Compound Operation is composed of other operations. Its kernel is usually
      device-agnostic. Normally it doesn't have its own derivative functions defined.
      Instead, AutoGrad automatically computes its derivative based on operations it
      uses.

   Composite Operation
      Same as :term:`Compound Operation`.

   Convolutional Neural Network
      A type of neural network designed for image and video processing, using
      convolutional and pooling layers to extract features. Also known as CNN.

   CUDA
      Compute Unified Device Architecture: A parallel computing platform developed
      by NVIDIA that allows developers to use GPUs for general-purpose computing,
      including machine learning and deep learning applications.

   Custom Operation
      An Operation that is defined by users and is usually a :term:`Compound Operation`.
      For example, this `tutorial <https://pytorch.org/docs/stable/notes/extending.html>`_
      details how to create Custom Operations.

   Device Kernel
      Device-specific kernel of a :term:`Leaf Operation`.

   embedding
      A way to represent categorical variables as dense vectors, often used in
      natural language processing and recommender systems.

   epoch
      An epoch is a unit of measurement in machine learning that represents one
      complete pass through the entire training dataset. During each epoch, the
      model's weights are updated based on the loss calculated from the predictions
      made on the training data.

   forward pass
      The forward pass is the process of passing input data through a neural network
      to obtain an output prediction. It's the first step in training a model,
      followed by the backward pass and optimization.

   GPU
      Graphics Processing Unit: A specialized electronic circuit designed to quickly
      manipulate and alter memory to accelerate computations. In the context of AI
      and machine learning, GPUs are used to accelerate computationally intensive
      tasks like training neural networks.

   gradient
      In machine learning, the gradient represents the rate of change of the loss
      function with respect to the model's parameters. It's used in backpropagation
      to update the weights and biases during training.

   Inductor
      A PyTorch component that enables just-in-time (JIT) compilation of PyTorch
      models, allowing for faster inference times and better performance on CPUs
      and GPUs. It is the default backend for torch.compile.

   inference
      The process of making predictions or drawing conclusions from a trained AI
      model, typically involving the application of the learned relationships to
      new, unseen data.

   JIT
      Just-In-Time Compilation: A compilation technique where code is compiled into
      machine code at runtime, just before it is executed.

   Kernel
      Implementation of a PyTorch operation, specifying what should be done when an
      operation executes.

   Leaf Operation
      An operation that's considered a basic operation, as opposed to a :term:`Compound
      Operation`. Leaf Operation always has dispatch functions defined, usually has a
      derivative function defined as well.

   loss function
      A loss function, also known as a cost function, is a mathematical function
      used to evaluate the performance of a machine learning model during training,
      providing a measure of how well the model is doing.

   LSTM
      Long Short-Term Memory Network: A type of recurrent neural network (RNN)
      designed to handle sequential data with long-term dependencies. LSTMs use
      memory cells and gates to selectively retain information over time.

   Native Operation
      An operation that comes natively with PyTorch ATen, for example ``aten::matmul``.

   Non-Leaf Operation
      Same as :term:`Compound Operation`.

   Operation
      A unit of work. For example, the work of matrix multiplication is an operation
      called ``aten::matmul``.

   optimizer
      An algorithm used to update the weights and biases of a neural network during
      training to minimize the loss function. Common optimizers include SGD, Adam,
      and RMSprop.

   quantization
      A technique used to reduce the precision of numerical values in a deep learning
      model, often to reduce memory usage, improve performance, and enable deployment
      on resource-constrained devices.

   RNN
      Recurrent Neural Network: A type of neural network designed for sequential data,
      using recurrent connections to capture temporal dependencies.

   Scripting
      Using ``torch.jit.script`` on a function to inspect source code and compile it as
      :term:`TorchScript` code.

   tensor
      Tensors are a specialized data structure that are very similar to arrays and
      matrices. In PyTorch, tensors are used to encode the inputs and outputs of a
      model, as well as the model's parameters.

   torch.compile
      A PyTorch function that compiles PyTorch code into an optimized form, allowing
      for faster execution and better performance. It is the main entry point for
      PyTorch 2.x optimizations.

   TorchScript
      Deprecated. Use :term:`torch.compile` instead.

   Tracing
      Using ``torch.jit.trace`` on a function to get an executable that can be optimized
      using just-in-time compilation.

   transformer
      A type of neural network architecture introduced in the paper "Attention is All
      You Need" (Vaswani et al., 2017), which relies entirely on self-attention
      mechanisms to process sequential data, such as text or images.
