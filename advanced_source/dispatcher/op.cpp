#include <torch/torch.h>
#include <torch/script.h>

#include <ATen/NamedTensorUtils.h>

using torch::Tensor;
using torch::DeviceType;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;

// BEGIN myadd
Tensor myadd(const Tensor& self, const Tensor& other) {
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("myops::myadd", "")
    .typed<decltype(myadd)>();
  return op.call(self, other);
}
// END myadd

// BEGIN TORCH_LIBRARY
TORCH_LIBRARY(myops, m) {
  m.def("myadd(Tensor self, Tensor other) -> Tensor");
}
// END TORCH_LIBRARY

// BEGIN myadd_cpu
Tensor myadd_cpu(const Tensor& self_, const Tensor& other_) {
  TORCH_CHECK(self_.sizes() == other_.sizes());
  TORCH_INTERNAL_ASSERT(self_.device().type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(other_.device().type() == DeviceType::CPU);
  Tensor self = self_.contiguous();
  Tensor other = other_.contiguous();
  Tensor result = torch::empty(self.sizes(), self.options());
  const float* self_ptr = self.data_ptr<float>();
  const float* other_ptr = other.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = self_ptr[i] + other_ptr[i];
  }
  return result;
}
// END myadd_cpu

// BEGIN TORCH_LIBRARY_IMPL CPU
TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("myadd", myadd_cpu);
}
// END TORCH_LIBRARY_IMPL CPU

Tensor myadd_cuda(const Tensor& self, const Tensor& other) {
  // Insert your CUDA implementation here
  TORCH_CHECK(0, "CUDA not yet implemented");
}

// BEGIN TORCH_LIBRARY_IMPL CUDA
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("myadd", myadd_cuda);
}
// END TORCH_LIBRARY_IMPL CUDA

// BEGIN myadd_autograd
class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
 public:
  static Tensor forward(
      AutogradContext *ctx, torch::Tensor self, torch::Tensor other) {
    at::AutoNonVariableTypeMode g;
    return myadd(self, other);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto grad_output = grad_outputs[0];
    return {grad_output, grad_output};
  }
};

Tensor myadd_autograd(const Tensor& self, const Tensor& other) {
  return MyAddFunction::apply(self, other)[0];
}
// END myadd_autograd

// BEGIN TORCH_LIBRARY_IMPL Autograd
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl("myadd", myadd_autograd);
}
// END TORCH_LIBRARY_IMPL Autograd

#if 0
// BEGIN TORCH_LIBRARY_IMPL Named
Tensor myadd_named(const Tensor& self, const Tensor& other) {
  // TODO: shouldn't need to do size check here
  TORCH_CHECK(self.sizes() == other.sizes());
  auto maybe_outnames = at::unify_from_right(self.names(), other.names());
  auto result = ([&]() {
    at::NoNamesGuard guard;
    return myadd(self, other);
  })();
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

TORCH_LIBRARY_IMPL(myops, Named, m) {
  m.impl("myadd", myadd_named);
}
// END TORCH_LIBRARY_IMPL Named
#endif
