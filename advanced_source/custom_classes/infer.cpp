#include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  torch::jit::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("foo.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<c10::IValue> inputs = {"foobarbaz"};
  auto output = module.forward(inputs).toString();
  std::cout << output->string() << std::endl;
}
