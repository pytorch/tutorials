Extending PyTorch with Custom C++ Classes
===============================================


This tutorial introduces an API for binding C++ classes into PyTorch.
The API is very similar to
`pybind11 <https://github.com/pybind/pybind11>`_, and most of the concepts will transfer
over if you're familiar with that system.

Implementing and Binding the Class in C++
-----------------------------------------

For this tutorial, we are going to define a simple C++ class that maintains persistent
state in a member variable.

.. literalinclude:: ../advanced_source/custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN class
  :end-before: END class

There are several things to note:

- ``torch/custom_class.h`` is the header you need to include to extend PyTorch
  with your custom class.
- Notice that whenever we are working with instances of the custom
  class, we do it via instances of ``c10::intrusive_ptr<>``. Think of ``intrusive_ptr``
  as a smart pointer like ``std::shared_ptr``, but the reference count is stored
  directly in the object, as opposed to a separate metadata block (as is done in
  ``std::shared_ptr``.  ``torch::Tensor`` internally uses the same pointer type;
  and custom classes have to also use this pointer type so that we can
  consistently manage different object types.
- The second thing to notice is that the user-defined class must inherit from
  ``torch::CustomClassHolder``. This ensures that the custom class has space to
  store the reference count.

Now let's take a look at how we will make this class visible to PyTorch, a process called
*binding* the class:

.. literalinclude:: ../advanced_source/custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN binding
  :end-before: END binding
  :append:
      ;
    }



Building the Example as a C++ Project With CMake
------------------------------------------------

Now, we're going to build the above C++ code with the `CMake
<https://cmake.org>`_ build system. First, take all the C++ code
we've covered so far and place it in a file called ``class.cpp``.
Then, write a simple ``CMakeLists.txt`` file and place it in the
same directory. Here is what ``CMakeLists.txt`` should look like:

.. literalinclude:: ../advanced_source/custom_classes/custom_class_project/CMakeLists.txt
  :language: cmake

Also, create a ``build`` directory. Your file tree should look like this::

  custom_class_project/
    class.cpp
    CMakeLists.txt
    build/

Go ahead and invoke cmake and then make to build the project:

.. code-block:: shell

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

What you'll find is there is now (among other things) a dynamic library
file present in the build directory. On Linux, this is probably named
``libcustom_class.so``. So the file tree should look like::

  custom_class_project/
    class.cpp
    CMakeLists.txt
    build/
      libcustom_class.so

Using the C++ Class from Python
-----------------------------------------------

Now that we have our class and its registration compiled into an ``.so`` file,
we can load that `.so` into Python and try it out. Here's a script that
demonstrates that:

.. literalinclude:: ../advanced_source/custom_classes/custom_class_project/custom_test.py
  :language: python


Defining Serialization/Deserialization Methods for Custom C++ Classes
---------------------------------------------------------------------

If you try to save a ``ScriptModule`` with a custom-bound C++ class as
an attribute, you'll get the following error:

.. literalinclude:: ../advanced_source/custom_classes/custom_class_project/export_attr.py
  :language: python

.. code-block:: shell

  $ python export_attr.py
  RuntimeError: Cannot serialize custom bound C++ class __torch__.torch.classes.my_classes.MyStackClass. Please define serialization methods via def_pickle for this class. (pushIValueImpl at ../torch/csrc/jit/pickler.cpp:128)

This is because PyTorch cannot automatically figure out what information
save from your C++ class. You must specify that manually. The way to do that
is to define ``__getstate__`` and ``__setstate__`` methods on the class using
the special ``def_pickle`` method on ``class_``.

.. note::
  The semantics of ``__getstate__`` and ``__setstate__`` are
  equivalent to that of the Python pickle module. You can
  `read more <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md#getstate-and-setstate>`_
  about how we use these methods.

Here is an example of the ``def_pickle`` call we can add to the registration of
``MyStackClass`` to include serialization methods:

.. literalinclude:: ../advanced_source/custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN def_pickle
  :end-before: END def_pickle

.. note::
  We take a different approach from pybind11 in the pickle API. Whereas pybind11
  as a special function ``pybind11::pickle()`` which you pass into ``class_::def()``,
  we have a separate method ``def_pickle`` for this purpose. This is because the
  name ``torch::jit::pickle`` was already taken, and we didn't want to cause confusion.

Once we have defined the (de)serialization behavior in this way, our script can
now run successfully:

.. code-block:: shell

  $ python ../export_attr.py
  testing

Defining Custom Operators that Take or Return Bound C++ Classes
---------------------------------------------------------------

Once you've defined a custom C++ class, you can also use that class
as an argument or return from a custom operator (i.e. free functions). Suppose
you have the following free function:

.. literalinclude:: ../advanced_source/custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN free_function
  :end-before: END free_function

You can register it running the following code inside your ``TORCH_LIBRARY``
block:

.. literalinclude:: ../advanced_source/custom_classes/custom_class_project/class.cpp
  :language: cpp
  :start-after: BEGIN def_free
  :end-before: END def_free

Once this is done, you can use the op like the following example:

.. code-block:: python

  class TryCustomOp(torch.nn.Module):
      def __init__(self):
          super(TryCustomOp, self).__init__()
          self.f = torch.classes.my_classes.MyStackClass(["foo", "bar"])

      def forward(self):
          return torch.ops.my_classes.manipulate_instance(self.f)

.. note::

  Registration of an operator that takes a C++ class as an argument requires that
  the custom class has already been registered.  You can enforce this by
  making sure the custom class registration and your free function definitions
  are in the same ``TORCH_LIBRARY`` block, and that the custom class
  registration comes first.  In the future, we may relax this requirement,
  so that these can be registered in any order.


Conclusion
----------

This tutorial walked you through how to expose a C++ class to PyTorch, how to
register its methods, how to use that class from Python, and how to save and
load code using the class and run that code in a standalone C++ process. You
are now ready to extend your PyTorch models with C++ classes that interface
with third party C++ libraries or implement any other use case that requires
the lines between Python and C++ to blend smoothly.

As always, if you run into any problems or have questions, you can use our
`forum <https://discuss.pytorch.org/>`_ or `GitHub issues
<https://github.com/pytorch/pytorch/issues>`_ to get in touch. Also, our
`frequently asked questions (FAQ) page
<https://pytorch.org/cppdocs/notes/faq.html>`_ may have helpful information.
