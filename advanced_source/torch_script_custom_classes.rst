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

There are several things to note:

- ``torch/custom_class.h`` is the header you need to include to extend TorchScript
  with your custom class.
- Notice that whenever we are working with instances of the custom
  class, we do it via instances of ``c10::intrusive_ptr<>``. Think of ``intrusive_ptr``
  as a smart pointer like ``std::shared_ptr``. The reason for using this smart pointer
  is to ensure consistent lifetime management of the object instances between languages
  (C++, Python and TorchScript).
- The second thing to notice is that the user-defined class must inherit from
  ``torch::CustomClassHolder``. This ensures that everything is set up to handle
  the lifetime management system previously mentioned.

Now let's take a look at how we will make this class visible to TorchScript, a process called
*binding* the class:

.. code-block:: cpp

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
  static auto testStack =
    torch::class_<MyStackClass<std::string>>("my_classes", "MyStackClass")
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
        .def("merge", &MyStackClass<std::string>::merge);



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
  # This instantiation will invoke the MyStackClass(std::vector<T> init) constructor
  # we registered earlier
  s = torch.classes.my_classes.MyStackClass(["foo", "bar"])

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
  MyStackClass = torch.classes.my_classes.MyStackClass

  @torch.jit.script
  def do_stacks(s : MyStackClass): # We can pass a custom class instance to TorchScript
      s2 = torch.classes.my_classes.MyStackClass(["hi", "mom"]) # We can instantiate the class
      s2.merge(s) # We can call a method on the class
      return s2.clone(), s2.top()  # We can also return instances of the class
                                   # from TorchScript function/methods

  stack, top = do_stacks(torch.classes.my_classes.MyStackClass(["wow"]))
  assert top == "wow"
  for expected in ["wow", "mom", "hi"]:
      assert stack.pop() == expected

Saving, Loading, and Running TorchScript Code Using Custom Classes
------------------------------------------------------------------

We can also use custom-registered C++ classes in a C++ process using
libtorch. As an example, let's define a simple ``nn.Module`` that
instantiates and calls a method on our MyStackClass class:

.. code-block:: python

  import torch

  torch.classes.load_library('libcustom_class.so')

  class Foo(torch.nn.Module):
      def __init__(self):
          super().__init__()

      def forward(self, s : str) -> str:
          stack = torch.classes.my_classes.MyStackClass(["hi", "mom"])
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
    torch::script::Module module;
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

Moving Custom Classes To/From IValues
-------------------------------------

It's also possible that you may need to move custom classes into or out of
``IValue``s, such as when you take or return ``IValue``s from TorchScript methods
or you want to instantiate a custom class attribute in C++. For creating an
``IValue`` from a custom C++ class instance:

- ``torch::make_custom_class<T>()`` provides an API similar to c10::intrusive_ptr<T>
  in that it will take whatever set of arguments you provide to it, call the constructor
  of T that matches that set of arguments, and wrap that instance up and return it.
  However, instead of returning just a pointer to a custom class object, it returns
  an ``IValue`` wrapping the object. You can then pass this ``IValue`` directly to
  TorchScript.
- In the event that you already have an ``intrusive_ptr`` pointing to your class, you
  can directly construct an IValue from it using the constructor ``IValue(intrusive_ptr<T>)``.

For converting ``IValue``s back to custom classes:

- ``IValue::toCustomClass<T>()`` will return an ``intrusive_ptr<T>`` pointing to the
  custom class that the ``IValue`` contains. Internally, this function is checking
  that ``T`` is registered as a custom class and that the ``IValue`` does in fact contain
  a custom class. You can check whether the ``IValue`` contains a custom class manually by
  calling ``isCustomClass()``.

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
        self.stack = torch.classes.my_classes.MyStackClass(["just", "testing"])

    def forward(self, s : str) -> str:
        return self.stack.pop() + s

  scripted_foo = torch.jit.script(Foo())

  scripted_foo.save('foo.pt')

.. code-block:: shell

  $ python export_attr.py
  RuntimeError: Cannot serialize custom bound C++ class __torch__.torch.classes.my_classes.MyStackClass. Please define serialization methods via def_pickle for this class. (pushIValueImpl at ../torch/csrc/jit/pickler.cpp:128)

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
``MyStackClass`` class to include serialization methods:

.. code-block:: cpp

  static auto testStack =
    torch::class_<MyStackClass<std::string>>("my_classes", "MyStackClass")
        .def(torch::init<std::vector<std::string>>())
        .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
          return self->stack_.back();
        })
        .def("push", &MyStackClass<std::string>::push)
        .def("pop", &MyStackClass<std::string>::pop)
        .def("clone", &MyStackClass<std::string>::clone)
        .def("merge", &MyStackClass<std::string>::merge)
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
          self.stack = torch.classes.my_classes.MyStackClass(["just", "testing"])

      def forward(self, s : str) -> str:
          return self.stack.pop() + s

  scripted_foo = torch.jit.script(Foo())

  scripted_foo.save('foo.pt')
  loaded = torch.jit.load('foo.pt')

  print(loaded.stack.pop())

.. code-block:: shell

  $ python ../export_attr.py
  testing

Defining Custom Operators that Take or Return Bound C++ Classes
---------------------------------------------------------------

Once you've defined a custom C++ class, you can also use that class
as an argument or return from a custom operator (i.e. free functions). Here's an
example of how to do that:

.. code-block:: cpp

  c10::intrusive_ptr<MyStackClass<std::string>> manipulate_instance(const c10::intrusive_ptr<MyStackClass<std::string>>& instance) {
    instance->pop();
    return instance;
  }

  static auto instance_registry = torch::RegisterOperators().op(
  torch::RegisterOperators::options()
      .schema(
          "foo::manipulate_instance(__torch__.torch.classes.my_classes.MyStackClass x) -> __torch__.torch.classes.my_classes.MyStackClass Y")
      .catchAllKernel<decltype(manipulate_instance), &manipulate_instance>());

Refer to the `custom op tutorial <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
for more details on the registration API.

Once this is done, you can use the op like the following example:

.. code-block:: python

  class TryCustomOp(torch.nn.Module):
      def __init__(self):
          super(TryCustomOp, self).__init__()
          self.f = torch.classes.my_classes.MyStackClass(["foo", "bar"])

      def forward(self):
          return torch.ops.foo.manipulate_instance(self.f)

.. note::

  Registration of an operator that takes a C++ class as an argument requires that
  the custom class has already been registered. This is fine if your op is
  registered after your class in a single compilation unit, however, if your
  class is registered in a separate compilation unit from the op you will need
  to enforce that dependency. One way to do this is to wrap the class registration
  in a `Meyer's singleton <https://stackoverflow.com/q/1661529>`_, which can be
  called from the compilation unit that does the operator registration.

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
