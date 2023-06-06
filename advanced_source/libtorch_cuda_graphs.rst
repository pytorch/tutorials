Using CUDA Graphs in PyTorch C++ API
====================================

NVIDIA’s CUDA Graph is a powerful tool. CUDA Graphs are capable of greatly reducing
the CPU overhead increasing the performance.

In this example we would like to demonstrate CUDA Graphs usage on PyTorch’s `MNIST
example <https://github.com/pytorch/examples/tree/main/cpp/mnist>`_.
The usage of CUDA Graphs in LibTorch is very similar to its `Python counterpart
<https://pytorch.org/docs/main/notes/cuda.html#cuda-graphs>`_ with only major
difference in language syntax.


The main training training loop consists of the several steps and depicted in the
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

In this tutorial we will be applying CUDA Graph on all the compute steps via whole-network
graph capture. Before doing so, we need to slightly modify the source code by preallocating
tensors and reusing them in all the work that is being done on GPU:

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

Where training step simply consists of forward and backward passes with corresponding optimizer calls:

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

PyTorch’s CUDA Graphs API is relying on Stream Capture which in general looks like the following:

.. code-block:: cpp

  at::cuda::CUDAGraph test_graph;
  at::cuda::CUDAStream captureStream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(captureStream);

  graph.capture_begin();
  training_step(model, optimizer, data, targets, output, loss);
  graph.capture_end();

Before actual graph capture it is important to run several warm-up iterations on side stream to
prepare CUDA cache as well as CUDA libraries (like CUBLAS and CUDNN) that will be used during
the training:

.. code-block:: cpp

  at::cuda::CUDAStream warmupStream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(warmupStream);
  for (int iter = 0; iter < num_warmup_iters; iter++) {
    training_step(model, optimizer, data, targets, output, loss);
  }

After successful graph capturing we can replace `training_step(model, optimizer, data, targets, output, loss);`
call via `graph.replay();`.

The full source code is available in GitHub.

The ordinary eager-mode produces the following output:

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

While the CUDA Graph output is the following:

.. code-block:: shell

$ time ./mnist --use-train-graph
CUDA is available! Training on GPU.
Using CUDA Graph for training.
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

As we can see, just applying a CUDA Graph for the training step we were able to gain the performance by more than 6 times.
