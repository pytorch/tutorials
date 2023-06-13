Using CUDA Graphs in PyTorch C++ API
====================================

.. note::
   |edit| View and edit this tutorial in `GitHub <https://github.com/pytorch/tutorials/blob/main/advanced_source/cpp_cuda_graphs.rst>`__. The full source code is available on `GitHub <https://github.com/pytorch/tutorials/blob/main/advanced_source/cpp_cuda_graphs>`__.

Prerequisites:

-  `Using the PyTorch C++ Frontend <../advanced_source/cpp_frontend.html>`__
-  `CUDA semantics <https://pytorch.org/docs/master/notes/cuda.html>`__
-  Pytorch 2.0 or later
-  CUDA 11 or later

NVIDIA’s CUDA Graphs have been a part of CUDA Toolkit library since the
release of `version 10 <https://developer.nvidia.com/blog/cuda-graphs/>`_.
They are capable of greatly reducing the CPU overhead increasing the
performance of applications.

In this tutorial, we will be focusing on using CUDA Graphs for `C++
frontend of PyTorch <https://pytorch.org/tutorials/advanced/cpp_frontend.html>`_.
The C++ frontend is mostly utilized in production and deployment applications which
are important parts of PyTorch use cases. Since `the first appearance
<https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>`_
the CUDA Graphs won users’ and developer’s hearts for being a very performant
and at the same time simple-to-use tool. In fact, CUDA Graphs are used by default
in ``torch.compile`` of PyTorch 2.0 to boost the productivity of training and inference.

We would like to demonstrate CUDA Graphs usage on PyTorch’s `MNIST
example <https://github.com/pytorch/examples/tree/main/cpp/mnist>`_.
The usage of CUDA Graphs in LibTorch (C++ Frontend) is very similar to its
`Python counterpart <https://pytorch.org/docs/main/notes/cuda.html#cuda-graphs>`_
but with some differences in syntax and functionality.

Getting Started
---------------

The main training loop consists of the several steps and depicted in the
following code chunk:

.. code-block:: cpp

  for (auto& batch : data_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    loss.backward();
    optimizer.step();
  }

The example above includes a forward pass, a backward pass, and weight updates.

In this tutorial, we will be applying CUDA Graph on all the compute steps through the whole-network
graph capture. But before doing so, we need to slightly modify the source code. What we need
to do is preallocate tensors for reusing them in the main training loop. Here is an example
implementation:

.. code-block:: cpp

  torch::TensorOptions FloatCUDA =
      torch::TensorOptions(device).dtype(torch::kFloat);
  torch::TensorOptions LongCUDA =
      torch::TensorOptions(device).dtype(torch::kLong);

  torch::Tensor data = torch::zeros({kTrainBatchSize, 1, 28, 28}, FloatCUDA);
  torch::Tensor targets = torch::zeros({kTrainBatchSize}, LongCUDA);
  torch::Tensor output = torch::zeros({1}, FloatCUDA);
  torch::Tensor loss = torch::zeros({1}, FloatCUDA);

  for (auto& batch : data_loader) {
    data.copy_(batch.data);
    targets.copy_(batch.target);
    training_step(model, optimizer, data, targets, output, loss);
  }

Where ``training_step`` simply consists of forward and backward passes with corresponding optimizer calls:

.. code-block:: cpp

  void training_step(
      Net& model,
      torch::optim::Optimizer& optimizer,
      torch::Tensor& data,
      torch::Tensor& targets,
      torch::Tensor& output,
      torch::Tensor& loss) {
    optimizer.zero_grad();
    output = model.forward(data);
    loss = torch::nll_loss(output, targets);
    loss.backward();
    optimizer.step();
  }

PyTorch’s CUDA Graphs API is relying on Stream Capture which in our case would be used like this:

.. code-block:: cpp

  at::cuda::CUDAGraph graph;
  at::cuda::CUDAStream captureStream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(captureStream);

  graph.capture_begin();
  training_step(model, optimizer, data, targets, output, loss);
  graph.capture_end();

Before the actual graph capture, it is important to run several warm-up iterations on side stream to
prepare CUDA cache as well as CUDA libraries (like CUBLAS and CUDNN) that will be used during
the training:

.. code-block:: cpp

  at::cuda::CUDAStream warmupStream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(warmupStream);
  for (int iter = 0; iter < num_warmup_iters; iter++) {
    training_step(model, optimizer, data, targets, output, loss);
  }

After the successful graph capture, we can replace ``training_step(model, optimizer, data, targets, output, loss);``
call via ``graph.replay();`` to do the training step.

Training Results
----------------

Taking the code for a spin we can see the following output from ordinary non-graphed training:

.. code-block:: shell

  $ time ./mnist
  Train Epoch: 1 [59584/60000] Loss: 0.3921
  Test set: Average loss: 0.2051 | Accuracy: 0.938
  Train Epoch: 2 [59584/60000] Loss: 0.1826
  Test set: Average loss: 0.1273 | Accuracy: 0.960
  Train Epoch: 3 [59584/60000] Loss: 0.1796
  Test set: Average loss: 0.1012 | Accuracy: 0.968
  Train Epoch: 4 [59584/60000] Loss: 0.1603
  Test set: Average loss: 0.0869 | Accuracy: 0.973
  Train Epoch: 5 [59584/60000] Loss: 0.2315
  Test set: Average loss: 0.0736 | Accuracy: 0.978
  Train Epoch: 6 [59584/60000] Loss: 0.0511
  Test set: Average loss: 0.0704 | Accuracy: 0.977
  Train Epoch: 7 [59584/60000] Loss: 0.0802
  Test set: Average loss: 0.0654 | Accuracy: 0.979
  Train Epoch: 8 [59584/60000] Loss: 0.0774
  Test set: Average loss: 0.0604 | Accuracy: 0.980
  Train Epoch: 9 [59584/60000] Loss: 0.0669
  Test set: Average loss: 0.0544 | Accuracy: 0.984
  Train Epoch: 10 [59584/60000] Loss: 0.0219
  Test set: Average loss: 0.0517 | Accuracy: 0.983

  real    0m44.287s
  user    0m44.018s
  sys    0m1.116s

While the training with the CUDA Graph produces the following output:

.. code-block:: shell

  $ time ./mnist --use-train-graph
  Train Epoch: 1 [59584/60000] Loss: 0.4092
  Test set: Average loss: 0.2037 | Accuracy: 0.938
  Train Epoch: 2 [59584/60000] Loss: 0.2039
  Test set: Average loss: 0.1274 | Accuracy: 0.961
  Train Epoch: 3 [59584/60000] Loss: 0.1779
  Test set: Average loss: 0.1017 | Accuracy: 0.968
  Train Epoch: 4 [59584/60000] Loss: 0.1559
  Test set: Average loss: 0.0871 | Accuracy: 0.972
  Train Epoch: 5 [59584/60000] Loss: 0.2240
  Test set: Average loss: 0.0735 | Accuracy: 0.977
  Train Epoch: 6 [59584/60000] Loss: 0.0520
  Test set: Average loss: 0.0710 | Accuracy: 0.978
  Train Epoch: 7 [59584/60000] Loss: 0.0935
  Test set: Average loss: 0.0666 | Accuracy: 0.979
  Train Epoch: 8 [59584/60000] Loss: 0.0744
  Test set: Average loss: 0.0603 | Accuracy: 0.981
  Train Epoch: 9 [59584/60000] Loss: 0.0762
  Test set: Average loss: 0.0547 | Accuracy: 0.983
  Train Epoch: 10 [59584/60000] Loss: 0.0207
  Test set: Average loss: 0.0525 | Accuracy: 0.983

  real    0m6.952s
  user    0m7.048s
  sys    0m0.619s

Conclusion
----------

As we can see, just by applying a CUDA Graph on the `MNIST example
<https://github.com/pytorch/examples/tree/main/cpp/mnist>`_ we were able to gain the performance
by more than six times for training. This kind of large performance improvement was achievable due to
the small model size. In case of larger models with heavy GPU usage, the CPU overhead is less impactful
so the improvement will be smaller. Nevertheless, it is always advantageous to use CUDA Graphs to
gain the performance of GPUs.
