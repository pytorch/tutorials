Extending TorchScript with Custom C++ Classes
===============================================

This tutorial is a follow-on to the
`custom operator <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
tutorial, and introduces the API we've built for binding C++ classes into TorchScript
and Python simultaneously. The API is very similar to
`pybind11 <https://github.com/pybind/pybind11>`_, and most of the concepts will transfer
over if you're familiar with that system.

Implementing and Binding the Class in C++
-----------------------------------------

For this tutorial, we are going to define a simple C++ class that maintains persistent
state in a member variable.

.. code-block:: cpp

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
  struct Stack : torch::jit::CustomClassHolder {
    std::vector<T> stack_;
    Stack(std::vector<T> init) : stack_(init.begin(), init.end()) {}

    void push(T x) {
      stack_.push_back(x);
    }
    T pop() {
      auto val = stack_.back();
      stack_.pop_back();
      return val;
    }

    c10::intrusive_ptr<Stack> clone() const {
      return c10::make_intrusive<Stack>(stack_);
    }

    void merge(const c10::intrusive_ptr<Stack>& c) {
      for (auto& elem : c->stack_) {
        push(elem);
      }
    }
  };

There are several things to note:

- ``torch/custom_class.h`` is the header you need to include to extend TorchScript
  with your custom class.
- Notice that whenever we are working with instances of the custom
  class, we do it via instances of ``c10::intrusive_ptr<>``. Think of ``intrusive_ptr``
  as a smart pointer like ``std::shared_ptr``. The reason for using this smart pointer
  is to ensure consistent lifetime management of the object instances between languages
  (C++, Python and TorchScript).
- The second thing to notice is that the user-defined class must inherit from
  ``torch::jit::CustomClassHolder``. This ensures that everything is set up to handle
  the lifetime management system previously mentioned.

Now let's take a look at how we will make this class visible to TorchScript, a process called
*binding* the class:

.. code-block:: cpp

  // Notice a few things:
  // - We pass the class to be registered as a template parameter to
  //   `torch::jit::class_`. In this instance, we've passed the
  //   specialization of the Stack class ``Stack<std::string>``.
  //   In general, you cannot register a non-specialized template
  //   class. For non-templated classes, you can just pass the
  //   class name directly as the template parameter.
  // - The single parameter to ``torch::jit::class_()`` is a
  //   string indicating the name of the class. This is the name
  //   the class will appear as in both Python and TorchScript.
  //   For example, our Stack class would appear as ``torch.classes.Stack``.
  static auto testStack =
    torch::jit::class_<Stack<std::string>>("Stack")
        // The following line registers the contructor of our Stack
        // class that takes a single `std::vector<std::string>` argument,
        // i.e. it exposes the C++ method `Stack(std::vector<T> init)`.
        // Currently, we do not support registering overloaded
        // constructors, so for now you can only `def()` one instance of
        // `torch::jit::init`.
        .def(torch::jit::init<std::vector<std::string>>())
        // The next line registers a stateless (i.e. no captures) C++ lambda
        // function as a method. Note that a lambda function must take a
        // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
        // as the first argument. Other arguments can be whatever you want.
        .def("top", [](const c10::intrusive_ptr<Stack<std::string>>& self) {
          return self->stack_.back();
        })
        // The following four lines expose methods of the Stack<std::string>
        // class as-is. `torch::jit::class_` will automatically examine the
        // argument and return types of the passed-in method pointers and
        // expose these to Python and TorchScript accordingly. Finally, notice
        // that we must take the *address* of the fully-qualified method name,
        // i.e. use the unary `&` operator, due to C++ typing rules.
        .def("push", &Stack<std::string>::push)
        .def("pop", &Stack<std::string>::pop)
        .def("clone", &Stack<std::string>::clone)
        .def("merge", &Stack<std::string>::merge);



Building the Example as a C++ Project With CMake
------------------------------------------------

Now, we're going to build the above C++ code with the `CMake
<https://cmake.org>`_ build system. First, take all the C++ code
we've covered so far and place it in a file called ``class.cpp``.
Then, write a simple ``CMakeLists.txt`` file and place it in the
same directory. Here is what ``CMakeLists.txt`` should look like:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
  project(custom_class)

  find_package(Torch REQUIRED)

  # Define our library target
  add_library(custom_class SHARED class.cpp)
  set(CMAKE_CXX_STANDARD 14)
  # Link against LibTorch
  target_link_libraries(custom_class "${TORCH_LIBRARIES}")

Also, create a ``build`` directory. Your file tree should look like this::

  custom_class_project/
    class.cpp
    CMakeLists.txt
    build/

Now, to build the project, go ahead and download the appropriate libtorch
binary from the `PyTorch website <https://pytorch.org/>`_. Extract the
zip archive somewhere (within the project directory might be convenient)
and note the path you've extracted it to. Next, go ahead and invoke cmake and
then make to build the project:

.. code-block:: shell

  $ cd build
  $ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
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

What you'll find is there is now (among other things) a dynamic library
file present in the build directory. On Linux, this is probably named
``libcustom_class.so``. So the file tree should look like::

  custom_class_project/
    class.cpp
    CMakeLists.txt
    build/
      libcustom_class.so

Using the C++ Class from Python and TorchScript
-----------------------------------------------

Now that we have our class and its registration compiled into an ``.so`` file,
we can load that `.so` into Python and try it out. Here's a script that
demonstrates that:

.. code-block:: python

  import torch

  # `torch.classes.load_library()` allows you to pass the path to your .so file
  # to load it in and make the custom C++ classes available to both Python and
  # TorchScript
  torch.classes.load_library("libcustom_class.so")
  # You can query the loaded libraries like this:
  print(torch.classes.loaded_libraries)
  # prints {'/custom_class_project/build/libcustom_class.so'}

  # We can find and instantiate our custom C++ class in python by using the
  # `torch.classes` namespace:
  #
  # This instantiation will invoke the Stack(std::vector<T> init) constructor
  # we registered earlier
  s = torch.classes.Stack(["foo", "bar"])

  # We can call methods in Python
  s.push("pushed")
  assert s.pop() == "pushed"

  # Returning and passing instances of custom classes works as you'd expect
  s2 = s.clone()
  s.merge(s2)
  for expected in ["bar", "foo", "bar", "foo"]:
      assert s.pop() == expected

  # We can also use the class in TorchScript
  # For now, we need to assign the class's type to a local in order to
  # annotate the type on the TorchScript function. This may change
  # in the future.
  Stack = torch.classes.Stack

  @torch.jit.script
  def do_stacks(s : Stack): # We can pass a custom class instance to TorchScript
      s2 = torch.classes.Stack(["hi", "mom"]) # We can instantiate the class
      s2.merge(s) # We can call a method on the class
      return s2.clone(), s2.top()  # We can also return instances of the class
                                   # from TorchScript function/methods

  stack, top = do_stacks(torch.classes.Stack(["wow"]))
  assert top == "wow"
  for expected in ["wow", "mom", "hi"]:
      assert stack.pop() == expected

Saving, Loading, and Running TorchScript Code Using Custom Classes
------------------------------------------------------------------

We can also use custom-registered C++ classes in a C++ process using
libtorch. As an example, let's define a simple ``nn.Module`` that
instantiates and calls a method on our Stack class:

.. code-block:: python

  import torch

  torch.classes.load_library('libcustom_class.so')

  class Foo(torch.nn.Module):
      def __init__(self):
          super().__init__()

      def forward(self, s : str) -> str:
          stack = torch.classes.Stack(["hi", "mom"])
          return stack.pop() + s

  scripted_foo = torch.jit.script(Foo())
  print(scripted_foo.graph)

  scripted_foo.save('foo.pt')

``foo.pt`` in our filesystem now contains the serialized TorchScript
program we've just defined.

Now, we're going to define a new CMake project to show how you can load
this model and its required .so file. For a full treatment of how to do this,
please have a look at the `Loading a TorchScript Model in C++ Tutorial <https://pytorch.org/tutorials/advanced/cpp_export.html>`_.

Similarly to before, let's create a file structure containing the following::

  cpp_inference_example/
    infer.cpp
    CMakeLists.txt
    foo.pt
    build/
    custom_class_project/
      class.cpp
      CMakeLists.txt
      build/

Notice we've copied over the serialized ``foo.pt`` file, as well as the source
tree from the ``custom_class_project`` above. We will be adding the
``custom_class_project`` as a dependency to this C++ project so that we can
build the custom class into the binary.

Let's populate ``infer.cpp`` with the following:

.. code-block:: cpp

  #include <torch/script.h>

  #include <iostream>
  #include <memory>

  int main(int argc, const char* argv[]) {
    torch::jit::script::Module module;
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

And similarly let's define our CMakeLists.txt file:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
  project(infer)

  find_package(Torch REQUIRED)

  add_subdirectory(custom_class_project)

  # Define our library target
  add_executable(infer infer.cpp)
  set(CMAKE_CXX_STANDARD 14)
  # Link against LibTorch
  target_link_libraries(infer "${TORCH_LIBRARIES}")
  # This is where we link in our libcustom_class code, making our
  # custom class available in our binary.
  target_link_libraries(infer -Wl,--no-as-needed custom_class)

You know the drill: ``cd build``, ``cmake``, and ``make``:

.. code-block:: shell

  $ cd build
  $ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
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
    -- Found torch: /local/miniconda3/lib/python3.7/site-packages/torch/lib/libtorch.so
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /cpp_inference_example/build
  $ make -j
    Scanning dependencies of target custom_class
    [ 25%] Building CXX object custom_class_project/CMakeFiles/custom_class.dir/class.cpp.o
    [ 50%] Linking CXX shared library libcustom_class.so
    [ 50%] Built target custom_class
    Scanning dependencies of target infer
    [ 75%] Building CXX object CMakeFiles/infer.dir/infer.cpp.o
    [100%] Linking CXX executable infer
    [100%] Built target infer

And now we can run our exciting C++ binary:

.. code-block:: shell

  $ ./infer
    momfoobarbaz

Incredible!

Defining Serialization/Deserialization Methods for Custom C++ Classes
---------------------------------------------------------------------

If you try to save a ``ScriptModule`` with a custom-bound C++ class as
an attribute, you'll get the following error:

.. code-block:: python

  # export_attr.py
  import torch

  torch.classes.load_library('libcustom_class.so')

  class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = torch.classes.Stack(["just", "testing"])

    def forward(self, s : str) -> str:
        return self.stack.pop() + s

  scripted_foo = torch.jit.script(Foo())

  scripted_foo.save('foo.pt')

.. code-block:: shell

  $ python export_attr.py
  RuntimeError: Cannot serialize custom bound C++ class __torch__.torch.classes.Stack. Please define serialization methods via torch::jit::pickle_ for this class. (pushIValueImpl at ../torch/csrc/jit/pickler.cpp:128)

This is because TorchScript cannot automatically figure out what information
save from your C++ class. You must specify that manually. The way to do that
is to define ``__getstate__`` and ``__setstate__`` methods on the class using
the special ``def_pickle`` method on ``class_``.

.. note::
  The semantics of ``__getstate__`` and ``__setstate__`` in TorchScript are
  equivalent to that of the Python pickle module. You can
  `read more <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md#getstate-and-setstate>`_
  about how we use these methods.

Here is an example of how we can update the registration code for our
``Stack`` class to include serialization methods:

.. code-block:: cpp

  static auto testStack =
    torch::jit::class_<Stack<std::string>>("Stack")
        .def(torch::jit::init<std::vector<std::string>>())
        .def("top", [](const c10::intrusive_ptr<Stack<std::string>>& self) {
          return self->stack_.back();
        })
        .def("push", &Stack<std::string>::push)
        .def("pop", &Stack<std::string>::pop)
        .def("clone", &Stack<std::string>::clone)
        .def("merge", &Stack<std::string>::merge)
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
              [](const c10::intrusive_ptr<Stack<std::string>>& self)
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
                  -> c10::intrusive_ptr<Stack<std::string>> {
                // A convenient way to instantiate an object and get an
                // intrusive_ptr to it is via `make_intrusive`. We use
                // that here to allocate an instance of Stack<std::string>
                // and call the single-argument std::vector<std::string>
                // constructor with the serialized state.
                return c10::make_intrusive<Stack<std::string>>(std::move(state));
              });

.. note::
  We take a different approach from pybind11 in the pickle API. Whereas pybind11
  as a special function ``pybind11::pickle()`` which you pass into ``class_::def()``,
  we have a separate method ``def_pickle`` for this purpose. This is because the
  name ``torch::jit::pickle`` was already taken, and we didn't want to cause confusion.

Once we have defined the (de)serialization behavior in this way, our script can
now run successfully:

.. code-block:: python

  import torch

  torch.classes.load_library('libcustom_class.so')

  class Foo(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.stack = torch.classes.Stack(["just", "testing"])

      def forward(self, s : str) -> str:
          return self.stack.pop() + s

  scripted_foo = torch.jit.script(Foo())

  scripted_foo.save('foo.pt')
  loaded = torch.jit.load('foo.pt')

  print(loaded.stack.pop())

.. code-block:: shell

  $ python ../export_attr.py
  testing

Conclusion
----------

This tutorial walked you through how to expose a C++ class to TorchScript
(and by extension Python), how to register its methods, how to use that
class from Python and TorchScript, and how to save and load code using
the class and run that code in a standalone C++ process. You are now ready
to extend your TorchScript models with C++ classes that interface with
third party C++ libraries or implement any other use case that requires the
lines between Python, TorchScript and C++ to blend smoothly.

As always, if you run into any problems or have questions, you can use our
`forum <https://discuss.pytorch.org/>`_ or `GitHub issues
<https://github.com/pytorch/pytorch/issues>`_ to get in touch. Also, our
`frequently asked questions (FAQ) page
<https://pytorch.org/cppdocs/notes/faq.html>`_ may have helpful information.
