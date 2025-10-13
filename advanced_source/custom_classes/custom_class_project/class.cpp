// BEGIN class
// This header is all you need to do the C++ portions of this
// tutorial
#include <torch/script.h>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <torch/custom_class.h>

#include <string>
#include <vector>

template <class T>
struct MyStackClass : torch::CustomClassHolder {
  std::vector<T> stack_;
  MyStackClass(std::vector<T> init) : stack_(init.begin(), init.end()) {}

  void push(T x) {
    stack_.push_back(x);
  }
  T pop() {
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }

  c10::intrusive_ptr<MyStackClass> clone() const {
    return c10::make_intrusive<MyStackClass>(stack_);
  }

  void merge(const c10::intrusive_ptr<MyStackClass>& c) {
    for (auto& elem : c->stack_) {
      push(elem);
    }
  }
};
// END class

// BEGIN free_function
c10::intrusive_ptr<MyStackClass<std::string>> manipulate_instance(const c10::intrusive_ptr<MyStackClass<std::string>>& instance) {
  instance->pop();
  return instance;
}
// END free_function

// BEGIN binding
// Notice a few things:
// - We pass the class to be registered as a template parameter to
//   `torch::class_`. In this instance, we've passed the
//   specialization of the MyStackClass class ``MyStackClass<std::string>``.
//   In general, you cannot register a non-specialized template
//   class. For non-templated classes, you can just pass the
//   class name directly as the template parameter.
// - The arguments passed to the constructor make up the "qualified name"
//   of the class. In this case, the registered class will appear in
//   Python and C++ as `torch.classes.my_classes.MyStackClass`. We call
//   the first argument the "namespace" and the second argument the
//   actual class name.
TORCH_LIBRARY(my_classes, m) {
  m.class_<MyStackClass<std::string>>("MyStackClass")
    // The following line registers the contructor of our MyStackClass
    // class that takes a single `std::vector<std::string>` argument,
    // i.e. it exposes the C++ method `MyStackClass(std::vector<T> init)`.
    // Currently, we do not support registering overloaded
    // constructors, so for now you can only `def()` one instance of
    // `torch::init`.
    .def(torch::init<std::vector<std::string>>())
    // The next line registers a stateless (i.e. no captures) C++ lambda
    // function as a method. Note that a lambda function must take a
    // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
    // as the first argument. Other arguments can be whatever you want.
    .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
      return self->stack_.back();
    })
    // The following four lines expose methods of the MyStackClass<std::string>
    // class as-is. `torch::class_` will automatically examine the
    // argument and return types of the passed-in method pointers and
    // expose these to Python and TorchScript accordingly. Finally, notice
    // that we must take the *address* of the fully-qualified method name,
    // i.e. use the unary `&` operator, due to C++ typing rules.
    .def("push", &MyStackClass<std::string>::push)
    .def("pop", &MyStackClass<std::string>::pop)
    .def("clone", &MyStackClass<std::string>::clone)
    .def("merge", &MyStackClass<std::string>::merge)
// END binding
#ifndef NO_PICKLE
// BEGIN def_pickle
    // class_<>::def_pickle allows you to define the serialization
    // and deserialization methods for your C++ class.
    // Currently, we only support passing stateless lambda functions
    // as arguments to def_pickle
    .def_pickle(
          // __getstate__
          // This function defines what data structure should be produced
          // when we serialize an instance of this class. The function
          // must take a single `self` argument, which is an intrusive_ptr
          // to the instance of the object. The function can return
          // any type that is supported as a return value of the TorchScript
          // custom operator API. In this instance, we've chosen to return
          // a std::vector<std::string> as the salient data to preserve
          // from the class.
          [](const c10::intrusive_ptr<MyStackClass<std::string>>& self)
              -> std::vector<std::string> {
            return self->stack_;
          },
          // __setstate__
          // This function defines how to create a new instance of the C++
          // class when we are deserializing. The function must take a
          // single argument of the same type as the return value of
          // `__getstate__`. The function must return an intrusive_ptr
          // to a new instance of the C++ class, initialized however
          // you would like given the serialized state.
          [](std::vector<std::string> state)
              -> c10::intrusive_ptr<MyStackClass<std::string>> {
            // A convenient way to instantiate an object and get an
            // intrusive_ptr to it is via `make_intrusive`. We use
            // that here to allocate an instance of MyStackClass<std::string>
            // and call the single-argument std::vector<std::string>
            // constructor with the serialized state.
            return c10::make_intrusive<MyStackClass<std::string>>(std::move(state));
          });
// END def_pickle
#endif // NO_PICKLE

// BEGIN def_free
    m.def(
      "manipulate_instance(__torch__.torch.classes.my_classes.MyStackClass x) -> __torch__.torch.classes.my_classes.MyStackClass Y",
      manipulate_instance
    );
// END def_free
}
