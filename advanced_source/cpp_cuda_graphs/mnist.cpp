#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

// Model that we will be training
struct Net : torch::nn::Module {
  Net()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Dropout2d conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

void stream_sync(
    at::cuda::CUDAStream& dependency,
    at::cuda::CUDAStream& dependent) {
  at::cuda::CUDAEvent cuda_ev;
  cuda_ev.record(dependency);
  cuda_ev.block(dependent);
}

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

void capture_train_graph(
    Net& model,
    torch::optim::Optimizer& optimizer,
    torch::Tensor& data,
    torch::Tensor& targets,
    torch::Tensor& output,
    torch::Tensor& loss,
    at::cuda::CUDAGraph& graph,
    const short num_warmup_iters = 7) {
  model.train();

  auto warmupStream = at::cuda::getStreamFromPool();
  auto captureStream = at::cuda::getStreamFromPool();
  auto legacyStream = at::cuda::getCurrentCUDAStream();

  at::cuda::setCurrentCUDAStream(warmupStream);

  stream_sync(legacyStream, warmupStream);

  for (C10_UNUSED const auto iter : c10::irange(num_warmup_iters)) {
    training_step(model, optimizer, data, targets, output, loss);
  }

  stream_sync(warmupStream, captureStream);
  at::cuda::setCurrentCUDAStream(captureStream);

  graph.capture_begin();
  training_step(model, optimizer, data, targets, output, loss);
  graph.capture_end();

  stream_sync(captureStream, legacyStream);
  at::cuda::setCurrentCUDAStream(legacyStream);
}

template <typename DataLoader>
void train(
    size_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size,
    torch::Tensor& data,
    torch::Tensor& targets,
    torch::Tensor& output,
    torch::Tensor& loss,
    at::cuda::CUDAGraph& graph,
    bool use_graph) {
  model.train();

  size_t batch_idx = 0;

  for (const auto& batch : data_loader) {
    if (batch.data.size(0) != kTrainBatchSize ||
        batch.target.size(0) != kTrainBatchSize) {
      continue;
    }

    data.copy_(batch.data);
    targets.copy_(batch.target);

    if (use_graph) {
      graph.replay();
    } else {
      training_step(model, optimizer, data, targets, output, loss);
    }

    if (batch_idx++ % kLogInterval == 0) {
      float train_loss = loss.item<float>();
      std::cout << "\rTrain Epoch:" << epoch << " ["
                << batch_idx * batch.data.size(0) << "/" << dataset_size
                << "] Loss: " << train_loss;
    }
  }
}

void test_step(
    Net& model,
    torch::Tensor& data,
    torch::Tensor& targets,
    torch::Tensor& output,
    torch::Tensor& loss) {
  output = model.forward(data);
  loss = torch::nll_loss(output, targets, {}, torch::Reduction::Sum);
}

void capture_test_graph(
    Net& model,
    torch::Tensor& data,
    torch::Tensor& targets,
    torch::Tensor& output,
    torch::Tensor& loss,
    torch::Tensor& total_loss,
    torch::Tensor& total_correct,
    at::cuda::CUDAGraph& graph,
    const int num_warmup_iters = 7) {
  torch::NoGradGuard no_grad;
  model.eval();

  auto warmupStream = at::cuda::getStreamFromPool();
  auto captureStream = at::cuda::getStreamFromPool();
  auto legacyStream = at::cuda::getCurrentCUDAStream();

  at::cuda::setCurrentCUDAStream(warmupStream);
  stream_sync(captureStream, legacyStream);

  for (C10_UNUSED const auto iter : c10::irange(num_warmup_iters)) {
    test_step(model, data, targets, output, loss);
    total_loss += loss;
    total_correct += output.argmax(1).eq(targets).sum();
  }

  stream_sync(warmupStream, captureStream);
  at::cuda::setCurrentCUDAStream(captureStream);

  graph.capture_begin();
  test_step(model, data, targets, output, loss);
  graph.capture_end();

  stream_sync(captureStream, legacyStream);
  at::cuda::setCurrentCUDAStream(legacyStream);
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size,
    torch::Tensor& data,
    torch::Tensor& targets,
    torch::Tensor& output,
    torch::Tensor& loss,
    torch::Tensor& total_loss,
    torch::Tensor& total_correct,
    at::cuda::CUDAGraph& graph,
    bool use_graph) {
  torch::NoGradGuard no_grad;

  model.eval();
  loss.zero_();
  total_loss.zero_();
  total_correct.zero_();

  for (const auto& batch : data_loader) {
    if (batch.data.size(0) != kTestBatchSize ||
        batch.target.size(0) != kTestBatchSize) {
      continue;
    }
    data.copy_(batch.data);
    targets.copy_(batch.target);

    if (use_graph) {
      graph.replay();
    } else {
      test_step(model, data, targets, output, loss);
    }
    total_loss += loss;
    total_correct += output.argmax(1).eq(targets).sum();
  }

  float test_loss = total_loss.item<float>() / dataset_size;
  float test_accuracy =
      static_cast<float>(total_correct.item<int64_t>()) / dataset_size;

  std::cout << std::endl
            << "Test set: Average loss: " << test_loss
            << " | Accuracy: " << test_accuracy << std::endl;
}

int main(int argc, char* argv[]) {
  if (!torch::cuda::is_available()) {
    std::cout << "CUDA is not available!" << std::endl;
    return -1;
  }

  bool use_train_graph = false;
  bool use_test_graph = false;

  std::vector<std::string> arguments(argv + 1, argv + argc);
  for (std::string& arg : arguments) {
    if (arg == "--use-train-graph") {
      std::cout << "Using CUDA Graph for training." << std::endl;
      use_train_graph = true;
    }
    if (arg == "--use-test-graph") {
      std::cout << "Using CUDA Graph for testing." << std::endl;
      use_test_graph = true;
    }
  }

  torch::manual_seed(1);
  torch::cuda::manual_seed(1);
  torch::Device device(torch::kCUDA);

  Net model;
  model.to(device);

  auto train_dataset =
      torch::data::datasets::MNIST(kDataRoot)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset =
      torch::data::datasets::MNIST(
          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  torch::TensorOptions FloatCUDA =
      torch::TensorOptions(device).dtype(torch::kFloat);
  torch::TensorOptions LongCUDA =
      torch::TensorOptions(device).dtype(torch::kLong);

  torch::Tensor train_data =
      torch::zeros({kTrainBatchSize, 1, 28, 28}, FloatCUDA);
  torch::Tensor train_targets = torch::zeros({kTrainBatchSize}, LongCUDA);
  torch::Tensor train_output = torch::zeros({1}, FloatCUDA);
  torch::Tensor train_loss = torch::zeros({1}, FloatCUDA);

  torch::Tensor test_data =
      torch::zeros({kTestBatchSize, 1, 28, 28}, FloatCUDA);
  torch::Tensor test_targets = torch::zeros({kTestBatchSize}, LongCUDA);
  torch::Tensor test_output = torch::zeros({1}, FloatCUDA);
  torch::Tensor test_loss = torch::zeros({1}, FloatCUDA);
  torch::Tensor test_total_loss = torch::zeros({1}, FloatCUDA);
  torch::Tensor test_total_correct = torch::zeros({1}, LongCUDA);

  at::cuda::CUDAGraph train_graph;
  at::cuda::CUDAGraph test_graph;

  capture_train_graph(
      model,
      optimizer,
      train_data,
      train_targets,
      train_output,
      train_loss,
      train_graph);

  capture_test_graph(
      model,
      test_data,
      test_targets,
      test_output,
      test_loss,
      test_total_loss,
      test_total_correct,
      test_graph);

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(
        epoch,
        model,
        device,
        *train_loader,
        optimizer,
        train_dataset_size,
        train_data,
        train_targets,
        train_output,
        train_loss,
        train_graph,
        use_train_graph);
    test(
        model,
        device,
        *test_loader,
        test_dataset_size,
        test_data,
        test_targets,
        test_output,
        test_loss,
        test_total_loss,
        test_total_correct,
        test_graph,
        use_test_graph);
  }

  std::cout << " Training/testing complete" << std::endl;
  return 0;
}
