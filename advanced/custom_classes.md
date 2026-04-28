# Extending PyTorch with Custom C++ Classes

This tutorial introduces an API for binding C++ classes into PyTorch.
The API is very similar to
[pybind11](https://github.com/pybind/pybind11), and most of the concepts will transfer
over if you're familiar with that system.

## Implementing and Binding the Class in C++

For this tutorial, we are going to define a simple C++ class that maintains persistent
state in a member variable.

```
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
```

There are several things to note:

- `torch/custom_class.h` is the header you need to include to extend PyTorch
with your custom class.
- Notice that whenever we are working with instances of the custom
class, we do it via instances of `c10::intrusive_ptr<>`. Think of `intrusive_ptr`
as a smart pointer like `std::shared_ptr`, but the reference count is stored
directly in the object, as opposed to a separate metadata block (as is done in
`std::shared_ptr`. `torch::Tensor` internally uses the same pointer type;
and custom classes have to also use this pointer type so that we can
consistently manage different object types.
- The second thing to notice is that the user-defined class must inherit from
`torch::CustomClassHolder`. This ensures that the custom class has space to
store the reference count.

Now let's take a look at how we will make this class visible to PyTorch, a process called
*binding* the class:

```
// Notice a few things:
// - We pass the class to be registered as a template parameter to
// `torch::class_`. In this instance, we've passed the
// specialization of the MyStackClass class ``MyStackClass<std::string>``.
// In general, you cannot register a non-specialized template
// class. For non-templated classes, you can just pass the
// class name directly as the template parameter.
// - The arguments passed to the constructor make up the "qualified name"
// of the class. In this case, the registered class will appear in
// Python and C++ as `torch.classes.my_classes.MyStackClass`. We call
// the first argument the "namespace" and the second argument the
// actual class name.
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
 ;
}
```

## Building the Example as a C++ Project With CMake

Now, we're going to build the above C++ code with the [CMake](https://cmake.org) build system. First, take all the C++ code
we've covered so far and place it in a file called `class.cpp`.
Then, write a simple `CMakeLists.txt` file and place it in the
same directory. Here is what `CMakeLists.txt` should look like:

```
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)

find_package(Torch REQUIRED)

# Define our library target
add_library(custom_class SHARED class.cpp)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(custom_class "${TORCH_LIBRARIES}")
```

Also, create a `build` directory. Your file tree should look like this:

```
custom_class_project/
 class.cpp
 CMakeLists.txt
 build/
```

Go ahead and invoke cmake and then make to build the project:

```
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
 -- The C compiler identification is GNU 7.3.1
 -- The CXX compiler identification is GNU 7.3.1
 -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc
 -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc -- works
 -- Detecting C compiler ABI info
 -- Detecting C compiler ABI info - done
 -- Detecting C compile features
 -- Detecting C compile features - done
 -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++
 -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++ -- works
 -- Detecting CXX compiler ABI info
 -- Detecting CXX compiler ABI info - done
 -- Detecting CXX compile features
 -- Detecting CXX compile features - done
 -- Looking for pthread.h
 -- Looking for pthread.h - found
 -- Looking for pthread_create
 -- Looking for pthread_create - not found
 -- Looking for pthread_create in pthreads
 -- Looking for pthread_create in pthreads - not found
 -- Looking for pthread_create in pthread
 -- Looking for pthread_create in pthread - found
 -- Found Threads: TRUE
 -- Found torch: /torchbind_tutorial/libtorch/lib/libtorch.so
 -- Configuring done
 -- Generating done
 -- Build files have been written to: /torchbind_tutorial/build
$ make -j
 Scanning dependencies of target custom_class
 [ 50%] Building CXX object CMakeFiles/custom_class.dir/class.cpp.o
 [100%] Linking CXX shared library libcustom_class.so
 [100%] Built target custom_class
```

What you'll find is there is now (among other things) a dynamic library
file present in the build directory. On Linux, this is probably named
`libcustom_class.so`. So the file tree should look like:

```
custom_class_project/
 class.cpp
 CMakeLists.txt
 build/
 libcustom_class.so
```

## Using the C++ Class from Python

Now that we have our class and its registration compiled into an `.so` file,
we can load that .so into Python and try it out. Here's a script that
demonstrates that:

```
# `torch.classes.load_library()` allows you to pass the path to your .so file
# to load it in and make the custom C++ classes available to both Python and
# TorchScript

# You can query the loaded libraries like this:

# prints {'/custom_class_project/build/libcustom_class.so'}

# We can find and instantiate our custom C++ class in python by using the
# `torch.classes` namespace:
#
# This instantiation will invoke the MyStackClass(std::vector<T> init)
# constructor we registered earlier

# We can call methods in Python

# Test custom operator

# Returning and passing instances of custom classes works as you'd expect

# We can also use the class in TorchScript
# For now, we need to assign the class's type to a local in order to
# annotate the type on the TorchScript function. This may change
# in the future.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

## Defining Serialization/Deserialization Methods for Custom C++ Classes

If you try to save a `ScriptModule` with a custom-bound C++ class as
an attribute, you'll get the following error:

```
# export_attr.py

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

```
$ python export_attr.py
RuntimeError: Cannot serialize custom bound C++ class __torch__.torch.classes.my_classes.MyStackClass. Please define serialization methods via def_pickle for this class. (pushIValueImpl at ../torch/csrc/jit/pickler.cpp:128)
```

This is because PyTorch cannot automatically figure out what information
save from your C++ class. You must specify that manually. The way to do that
is to define `__getstate__` and `__setstate__` methods on the class using
the special `def_pickle` method on `class_`.

Note

The semantics of `__getstate__` and `__setstate__` are
equivalent to that of the Python pickle module. You can
[read more](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md#getstate-and-setstate)
about how we use these methods.

Here is an example of the `def_pickle` call we can add to the registration of
`MyStackClass` to include serialization methods:

```
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
```

Note

We take a different approach from pybind11 in the pickle API. Whereas pybind11
as a special function `pybind11::pickle()` which you pass into `class_::def()`,
we have a separate method `def_pickle` for this purpose. This is because the
name `torch::jit::pickle` was already taken, and we didn't want to cause confusion.

Once we have defined the (de)serialization behavior in this way, our script can
now run successfully:

```
$ python ../export_attr.py
testing
```

## Defining Custom Operators that Take or Return Bound C++ Classes

Once you've defined a custom C++ class, you can also use that class
as an argument or return from a custom operator (i.e. free functions). Suppose
you have the following free function:

```
c10::intrusive_ptr<MyStackClass<std::string>> manipulate_instance(const c10::intrusive_ptr<MyStackClass<std::string>>& instance) {
 instance->pop();
 return instance;
}
```

You can register it running the following code inside your `TORCH_LIBRARY`
block:

```
m.def(
 "manipulate_instance(__torch__.torch.classes.my_classes.MyStackClass x) -> __torch__.torch.classes.my_classes.MyStackClass Y",
 manipulate_instance
 );
```

Once this is done, you can use the op like the following example:

```
class TryCustomOp(torch.nn.Module):
 def __init__(self):
 super(TryCustomOp, self).__init__()
 self.f = torch.classes.my_classes.MyStackClass(["foo", "bar"])

 def forward(self):
 return torch.ops.my_classes.manipulate_instance(self.f)
```

Note

Registration of an operator that takes a C++ class as an argument requires that
the custom class has already been registered. You can enforce this by
making sure the custom class registration and your free function definitions
are in the same `TORCH_LIBRARY` block, and that the custom class
registration comes first. In the future, we may relax this requirement,
so that these can be registered in any order.

## Conclusion

This tutorial walked you through how to expose a C++ class to PyTorch, how to
register its methods, how to use that class from Python, and how to save and
load code using the class and run that code in a standalone C++ process. You
are now ready to extend your PyTorch models with C++ classes that interface
with third party C++ libraries or implement any other use case that requires
the lines between Python and C++ to blend smoothly.

As always, if you run into any problems or have questions, you can use our
[forum](https://discuss.pytorch.org/) or [GitHub issues](https://github.com/pytorch/pytorch/issues) to get in touch. Also, our
[frequently asked questions (FAQ) page](https://pytorch.org/cppdocs/notes/faq.html) may have helpful information.