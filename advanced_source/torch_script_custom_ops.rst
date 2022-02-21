Extending TorchScript with Custom C++ Operators
===============================================

The PyTorch 1.0 release introduced a new programming model to PyTorch called
`TorchScript <https://pytorch.org/docs/master/jit.html>`_. TorchScript is a
subset of the Python programming language which can be parsed, compiled and
optimized by the TorchScript compiler. Further, compiled TorchScript models have
the option of being serialized into an on-disk file format, which you can
subsequently load and run from pure C++ (as well as Python) for inference.

TorchScript supports a large subset of operations provided by the ``torch``
package, allowing you to express many kinds of complex models purely as a series
of tensor operations from PyTorch's "standard library". Nevertheless, there may
be times where you find yourself in need of extending TorchScript with a custom
C++ or CUDA function. While we recommend that you only resort to this option if
your idea cannot be expressed (efficiently enough) as a simple Python function,
we do provide a very friendly and simple interface for defining custom C++ and
CUDA kernels using `ATen <https://pytorch.org/cppdocs/#aten>`_, PyTorch's high
performance C++ tensor library. Once bound into TorchScript, you can embed these
custom kernels (or "ops") into your TorchScript model and execute them both in
Python and in their serialized form directly in C++.

The following paragraphs give an example of writing a TorchScript custom op to
call into `OpenCV <https://www.opencv.org>`_, a computer vision library written
in C++. We will discuss how to work with tensors in C++, how to efficiently
convert them to third party tensor formats (in this case, OpenCV ``Mat``), how
to register your operator with the TorchScript runtime and finally how to
compile the operator and use it in Python and C++.

Implementing the Custom Operator in C++
---------------------------------------

For this tutorial, we'll be exposing the `warpPerspective
<https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective>`_
function, which applies a perspective transformation to an image, from OpenCV to
TorchScript as a custom operator. The first step is to write the implementation
of our custom operator in C++. Let's call the file for this implementation
``op.cpp`` and make it look like this:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN warp_perspective
  :end-before: END warp_perspective

The code for this operator is quite short. At the top of the file, we include
the OpenCV header file, ``opencv2/opencv.hpp``, alongside the ``torch/script.h``
header which exposes all the necessary goodies from PyTorch's C++ API that we
need to write custom TorchScript operators. Our function ``warp_perspective``
takes two arguments: an input ``image`` and the ``warp`` transformation matrix
we wish to apply to the image. The type of these inputs is ``torch::Tensor``,
PyTorch's tensor type in C++ (which is also the underlying type of all tensors
in Python). The return type of our ``warp_perspective`` function will also be a
``torch::Tensor``.

.. tip::

  See `this note <https://pytorch.org/cppdocs/notes/tensor_basics.html>`_ for
  more information about ATen, the library that provides the ``Tensor`` class to
  PyTorch. Further, `this tutorial
  <https://pytorch.org/cppdocs/notes/tensor_creation.html>`_ describes how to
  allocate and initialize new tensor objects in C++ (not required for this
  operator).

.. attention::

  The TorchScript compiler understands a fixed number of types. Only these types
  can be used as arguments to your custom operator. Currently these types are:
  ``torch::Tensor``, ``torch::Scalar``, ``double``, ``int64_t`` and
  ``std::vector`` s of these types. Note that *only* ``double`` and *not*
  ``float``, and *only* ``int64_t`` and *not* other integral types such as
  ``int``, ``short`` or ``long`` are supported.

Inside of our function, the first thing we need to do is convert our PyTorch
tensors to OpenCV matrices, as OpenCV's ``warpPerspective`` expects ``cv::Mat``
objects as inputs. Fortunately, there is a way to do this **without copying
any** data. In the first few lines,

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN image_mat
  :end-before: END image_mat

we are calling `this constructor
<https://docs.opencv.org/trunk/d3/d63/classcv_1_1Mat.html#a922de793eabcec705b3579c5f95a643e>`_
of the OpenCV ``Mat`` class to convert our tensor to a ``Mat`` object. We pass
it the number of rows and columns of the original ``image`` tensor, the datatype
(which we'll fix as ``float32`` for this example), and finally a raw pointer to
the underlying data -- a ``float*``. What is special about this constructor of
the ``Mat`` class is that it does not copy the input data. Instead, it will
simply reference this memory for all operations performed on the ``Mat``. If an
in-place operation is performed on the ``image_mat``, this will be reflected in
the original ``image`` tensor (and vice-versa). This allows us to call
subsequent OpenCV routines with the library's native matrix type, even though
we're actually storing the data in a PyTorch tensor. We repeat this procedure to
convert the ``warp`` PyTorch tensor to the ``warp_mat`` OpenCV matrix:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN warp_mat
  :end-before: END warp_mat

Next, we are ready to call the OpenCV function we were so eager to use in
TorchScript: ``warpPerspective``. For this, we pass the OpenCV function the
``image_mat`` and ``warp_mat`` matrices, as well as an empty output matrix
called ``output_mat``. We also specify the size ``dsize`` we want the output
matrix (image) to be. It is hardcoded to ``8 x 8`` for this example:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN output_mat
  :end-before: END output_mat

The final step in our custom operator implementation is to convert the
``output_mat`` back into a PyTorch tensor, so that we can further use it in
PyTorch. This is strikingly similar to what we did earlier to convert in the
other direction. In this case, PyTorch provides a ``torch::from_blob`` method. A
*blob* in this case is intended to mean some opaque, flat pointer to memory that
we want to interpret as a PyTorch tensor. The call to ``torch::from_blob`` looks
like this:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN output_tensor
  :end-before: END output_tensor

We use the ``.ptr<float>()`` method on the OpenCV ``Mat`` class to get a raw
pointer to the underlying data (just like ``.data_ptr<float>()`` for the PyTorch
tensor earlier). We also specify the output shape of the tensor, which we
hardcoded as ``8 x 8``. The output of ``torch::from_blob`` is then a
``torch::Tensor``, pointing to the memory owned by the OpenCV matrix.

Before returning this tensor from our operator implementation, we must call
``.clone()`` on the tensor to perform a memory copy of the underlying data. The
reason for this is that ``torch::from_blob`` returns a tensor that does not own
its data. At that point, the data is still owned by the OpenCV matrix. However,
this OpenCV matrix will go out of scope and be deallocated at the end of the
function. If we returned the ``output`` tensor as-is, it would point to invalid
memory by the time we use it outside the function. Calling ``.clone()`` returns
a new tensor with a copy of the original data that the new tensor owns itself.
It is thus safe to return to the outside world.

Registering the Custom Operator with TorchScript
------------------------------------------------

Now that have implemented our custom operator in C++, we need to *register* it
with the TorchScript runtime and compiler. This will allow the TorchScript
compiler to resolve references to our custom operator in TorchScript code.
If you have ever used the pybind11 library, our syntax for registration
resembles the pybind11 syntax very closely.  To register a single function,
we write:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/op.cpp
  :language: cpp
  :start-after: BEGIN registry
  :end-before: END registry

somewhere at the top level of our ``op.cpp`` file.  The ``TORCH_LIBRARY`` macro
creates a function that will be called when your program starts.  The name
of your library (``my_ops``) is given as the first argument (it should not
be in quotes).  The second argument (``m``) defines a variable of type
``torch::Library`` which is the main interface to register your operators.
The method ``Library::def`` actually creates an operator named ``warp_perspective``,
exposing it to both Python and TorchScript.  You can define as many operators
as you like by making multiple calls to ``def``.

Behinds the scenes, the ``def`` function is actually doing quite a bit of work:
it is using template metaprogramming to inspect the type signature of your
function and translate it into an operator schema which specifies the operators
type within TorchScript's type system.

Building the Custom Operator
----------------------------

Now that we have implemented our custom operator in C++ and written its
registration code, it is time to build the operator into a (shared) library that
we can load into Python for research and experimentation, or into C++ for
inference in a no-Python environment. There exist multiple ways to build our
operator, using either pure CMake, or Python alternatives like ``setuptools``.
For brevity, the paragraphs below only discuss the CMake approach. The appendix
of this tutorial dives into other alternatives.

Environment setup
*****************

We need an installation of PyTorch and OpenCV.  The easiest and most platform
independent way to get both is to via Conda::

  conda install -c pytorch pytorch
  conda install opencv

Building with CMake
*******************

To build our custom operator into a shared library using the `CMake
<https://cmake.org>`_ build system, we need to write a short ``CMakeLists.txt``
file and place it with our previous ``op.cpp`` file. For this, let's agree on a
a directory structure that looks like this::

  warp-perspective/
    op.cpp
    CMakeLists.txt

The contents of our ``CMakeLists.txt`` file should then be the following:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/CMakeLists.txt
  :language: cpp

To now build our operator, we can run the following commands from our
``warp_perspective`` folder:

.. code-block:: shell

  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
  -- The C compiler identification is GNU 5.4.0
  -- The CXX compiler identification is GNU 5.4.0
  -- Check for working C compiler: /usr/bin/cc
  -- Check for working C compiler: /usr/bin/cc -- works
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Check for working CXX compiler: /usr/bin/c++
  -- Check for working CXX compiler: /usr/bin/c++ -- works
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
  -- Found torch: /libtorch/lib/libtorch.so
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /warp_perspective/build
  $ make -j
  Scanning dependencies of target warp_perspective
  [ 50%] Building CXX object CMakeFiles/warp_perspective.dir/op.cpp.o
  [100%] Linking CXX shared library libwarp_perspective.so
  [100%] Built target warp_perspective

which will place a ``libwarp_perspective.so`` shared library file in the
``build`` folder. In the ``cmake`` command above, we use the helper
variable ``torch.utils.cmake_prefix_path`` to conveniently tell us where
the cmake files for our PyTorch install are.

We will explore how to use and call our operator in detail further below, but to
get an early sensation of success, we can try running the following code in
Python:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/smoke_test.py
  :language: python

If all goes well, this should print something like::

  <built-in method my_ops::warp_perspective of PyCapsule object at 0x7f618fc6fa50>

which is the Python function we will later use to invoke our custom operator.

Using the TorchScript Custom Operator in Python
-----------------------------------------------

Once our custom operator is built into a shared library  we are ready to use
this operator in our TorchScript models in Python. There are two parts to this:
first loading the operator into Python, and second using the operator in
TorchScript code.

You already saw how to import your operator into Python:
``torch.ops.load_library()``. This function takes the path to a shared library
containing custom operators, and loads it into the current process. Loading the
shared library will also execute the ``TORCH_LIBRARY`` block. This will register
our custom operator with the TorchScript compiler and allow us to use that
operator in TorchScript code.

You can refer to your loaded operator as ``torch.ops.<namespace>.<function>``,
where ``<namespace>`` is the namespace part of your operator name, and
``<function>`` the function name of your operator. For the operator we wrote
above, the namespace was ``my_ops`` and the function name ``warp_perspective``,
which means our operator is available as ``torch.ops.my_ops.warp_perspective``.
While this function can be used in scripted or traced TorchScript modules, we
can also just use it in vanilla eager PyTorch and pass it regular PyTorch
tensors:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :prepend: import torch
  :start-after: BEGIN preamble
  :end-before: END preamble

producing:

.. code-block:: python

  tensor([[0.0000, 0.3218, 0.4611,  ..., 0.4636, 0.4636, 0.4636],
        [0.3746, 0.0978, 0.5005,  ..., 0.4636, 0.4636, 0.4636],
        [0.3245, 0.0169, 0.0000,  ..., 0.4458, 0.4458, 0.4458],
        ...,
        [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
        [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
        [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000]])


.. note::

    What happens behind the scenes is that the first time you access
    ``torch.ops.namespace.function`` in Python, the TorchScript compiler (in C++
    land) will see if a function ``namespace::function`` has been registered, and
    if so, return a Python handle to this function that we can subsequently use to
    call into our C++ operator implementation from Python. This is one noteworthy
    difference between TorchScript custom operators and C++ extensions: C++
    extensions are bound manually using pybind11, while TorchScript custom ops are
    bound on the fly by PyTorch itself. Pybind11 gives you more flexibility with
    regards to what types and classes you can bind into Python and is thus
    recommended for purely eager code, but it is not supported for TorchScript
    ops.

From here on, you can use your custom operator in scripted or traced code just
as you would other functions from the ``torch`` package. In fact, "standard
library" functions like ``torch.matmul`` go through largely the same
registration path as custom operators, which makes custom operators really
first-class citizens when it comes to how and where they can be used in
TorchScript.  (One difference, however, is that standard library functions
have custom written Python argument parsing logic that differs from
``torch.ops`` argument parsing.)

Using the Custom Operator with Tracing
**************************************

Let's start by embedding our operator in a traced function. Recall that for
tracing, we start with some vanilla Pytorch code:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN compute
  :end-before: END compute

and then call ``torch.jit.trace`` on it. We further pass ``torch.jit.trace``
some example inputs, which it will forward to our implementation to record the
sequence of operations that occur as the inputs flow through it. The result of
this is effectively a "frozen" version of the eager PyTorch program, which the
TorchScript compiler can further analyze, optimize and serialize:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN trace
  :end-before: END trace

Producing::

    graph(%x : Float(4:8, 8:1),
          %y : Float(8:5, 5:1),
          %z : Float(4:5, 5:1)):
      %3 : Float(4:5, 5:1) = aten::matmul(%x, %y) # test.py:10:0
      %4 : Float(4:5, 5:1) = aten::relu(%z) # test.py:10:0
      %5 : int = prim::Constant[value=1]() # test.py:10:0
      %6 : Float(4:5, 5:1) = aten::add(%3, %4, %5) # test.py:10:0
      return (%6)

Now, the exciting revelation is that we can simply drop our custom operator into
our PyTorch trace as if it were ``torch.relu`` or any other ``torch`` function:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN compute2
  :end-before: END compute2

and then trace it as before:

.. literalinclude:: ../advanced_source/torch_script_custom_ops/test.py
  :language: python
  :start-after: BEGIN trace2
  :end-before: END trace2

Producing::

    graph(%x.1 : Float(4:8, 8:1),
          %y : Float(8:5, 5:1),
          %z : Float(8:5, 5:1)):
      %3 : int = prim::Constant[value=3]() # test.py:25:0
      %4 : int = prim::Constant[value=6]() # test.py:25:0
      %5 : int = prim::Constant[value=0]() # test.py:25:0
      %6 : Device = prim::Constant[value="cpu"]() # test.py:25:0
      %7 : bool = prim::Constant[value=0]() # test.py:25:0
      %8 : Float(3:3, 3:1) = aten::eye(%3, %4, %5, %6, %7) # test.py:25:0
      %x : Float(8:8, 8:1) = my_ops::warp_perspective(%x.1, %8) # test.py:25:0
      %10 : Float(8:5, 5:1) = aten::matmul(%x, %y) # test.py:26:0
      %11 : Float(8:5, 5:1) = aten::relu(%z) # test.py:26:0
      %12 : int = prim::Constant[value=1]() # test.py:26:0
      %13 : Float(8:5, 5:1) = aten::add(%10, %11, %12) # test.py:26:0
      return (%13)

Integrating TorchScript custom ops into traced PyTorch code is as easy as this!

Using the Custom Operator with Script
*************************************

Besides tracing, another way to arrive at a TorchScript representation of a
PyTorch program is to directly write your code *in* TorchScript. TorchScript is
largely a subset of the Python language, with some restrictions that make it
easier for the TorchScript compiler to reason about programs. You turn your
regular PyTorch code into TorchScript by annotating it with
``@torch.jit.script`` for free functions and ``@torch.jit.script_method`` for
methods in a class (which must also derive from ``torch.jit.ScriptModule``). See
`here <https://pytorch.org/docs/master/jit.html>`_ for more details on
TorchScript annotations.

One particular reason to use TorchScript instead of tracing is that tracing is
unable to capture control flow in PyTorch code. As such, let us consider this
function which does use control flow:

.. code-block:: python

  def compute(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    return x.matmul(y) + z

To convert this function from vanilla PyTorch to TorchScript, we annotate it
with ``@torch.jit.script``:

.. code-block:: python

  @torch.jit.script
  def compute(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    return x.matmul(y) + z

This will just-in-time compile the ``compute`` function into a graph
representation, which we can inspect in the ``compute.graph`` property:

.. code-block:: python

  >>> compute.graph
  graph(%x : Dynamic
      %y : Dynamic) {
    %14 : int = prim::Constant[value=1]()
    %2 : int = prim::Constant[value=0]()
    %7 : int = prim::Constant[value=42]()
    %z.1 : int = prim::Constant[value=5]()
    %z.2 : int = prim::Constant[value=10]()
    %4 : Dynamic = aten::select(%x, %2, %2)
    %6 : Dynamic = aten::select(%4, %2, %2)
    %8 : Dynamic = aten::eq(%6, %7)
    %9 : bool = prim::TensorToBool(%8)
    %z : int = prim::If(%9)
      block0() {
        -> (%z.1)
      }
      block1() {
        -> (%z.2)
      }
    %13 : Dynamic = aten::matmul(%x, %y)
    %15 : Dynamic = aten::add(%13, %z, %14)
    return (%15);
  }

And now, just like before, we can use our custom operator like any other
function inside of our script code:

.. code-block:: python

  torch.ops.load_library("libwarp_perspective.so")

  @torch.jit.script
  def compute(x, y):
    if bool(x[0] == 42):
        z = 5
    else:
        z = 10
    x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
    return x.matmul(y) + z

When the TorchScript compiler sees the reference to
``torch.ops.my_ops.warp_perspective``, it will find the implementation we
registered via the ``TORCH_LIBRARY`` function in C++, and compile it into its
graph representation:

.. code-block:: python

  >>> compute.graph
  graph(%x.1 : Dynamic
      %y : Dynamic) {
      %20 : int = prim::Constant[value=1]()
      %16 : int[] = prim::Constant[value=[0, -1]]()
      %14 : int = prim::Constant[value=6]()
      %2 : int = prim::Constant[value=0]()
      %7 : int = prim::Constant[value=42]()
      %z.1 : int = prim::Constant[value=5]()
      %z.2 : int = prim::Constant[value=10]()
      %13 : int = prim::Constant[value=3]()
      %4 : Dynamic = aten::select(%x.1, %2, %2)
      %6 : Dynamic = aten::select(%4, %2, %2)
      %8 : Dynamic = aten::eq(%6, %7)
      %9 : bool = prim::TensorToBool(%8)
      %z : int = prim::If(%9)
        block0() {
          -> (%z.1)
        }
        block1() {
          -> (%z.2)
        }
      %17 : Dynamic = aten::eye(%13, %14, %2, %16)
      %x : Dynamic = my_ops::warp_perspective(%x.1, %17)
      %19 : Dynamic = aten::matmul(%x, %y)
      %21 : Dynamic = aten::add(%19, %z, %20)
      return (%21);
    }

Notice in particular the reference to ``my_ops::warp_perspective`` at the end of
the graph.

.. attention::

	The TorchScript graph representation is still subject to change. Do not rely
	on it looking like this.

And that's really it when it comes to using our custom operator in Python. In
short, you import the library containing your operator(s) using
``torch.ops.load_library``, and call your custom op like any other ``torch``
operator from your traced or scripted TorchScript code.

Using the TorchScript Custom Operator in C++
--------------------------------------------

One useful feature of TorchScript is the ability to serialize a model into an
on-disk file. This file can be sent over the wire, stored in a file system or,
more importantly, be dynamically deserialized and executed without needing to
keep the original source code around. This is possible in Python, but also in
C++. For this, PyTorch provides `a pure C++ API <https://pytorch.org/cppdocs/>`_
for deserializing as well as executing TorchScript models. If you haven't yet,
please read `the tutorial on loading and running serialized TorchScript models
in C++ <https://pytorch.org/tutorials/advanced/cpp_export.html>`_, on which the
next few paragraphs will build.

In short, custom operators can be executed just like regular ``torch`` operators
even when deserialized from a file and run in C++. The only requirement for this
is to link the custom operator shared library we built earlier with the C++
application in which we execute the model. In Python, this worked simply calling
``torch.ops.load_library``. In C++, you need to link the shared library with
your main application in whatever build system you are using. The following
example will showcase this using CMake.

.. note::

	Technically, you can also dynamically load the shared library into your C++
	application at runtime in much the same way we did it in Python. On Linux,
	`you can do this with dlopen
	<https://tldp.org/HOWTO/Program-Library-HOWTO/dl-libraries.html>`_. There exist
	equivalents on other platforms.

Building on the C++ execution tutorial linked above, let's start with a minimal
C++ application in one file, ``main.cpp`` in a different folder from our
custom operator, that loads and executes a serialized TorchScript model:

.. code-block:: cpp

  #include <torch/script.h> // One-stop header.

  #include <iostream>
  #include <memory>


  int main(int argc, const char* argv[]) {
    if (argc != 2) {
      std::cerr << "usage: example-app <path-to-exported-script-module>\n";
      return -1;
    }

    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module = torch::jit::load(argv[1]);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({4, 8}));
    inputs.push_back(torch::randn({8, 5}));

    torch::Tensor output = module.forward(std::move(inputs)).toTensor();

    std::cout << output << std::endl;
  }

Along with a small ``CMakeLists.txt`` file:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
  project(example_app)

  find_package(Torch REQUIRED)

  add_executable(example_app main.cpp)
  target_link_libraries(example_app "${TORCH_LIBRARIES}")
  target_compile_features(example_app PRIVATE cxx_range_for)

At this point, we should be able to build the application:

.. code-block:: shell

  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
  -- The C compiler identification is GNU 5.4.0
  -- The CXX compiler identification is GNU 5.4.0
  -- Check for working C compiler: /usr/bin/cc
  -- Check for working C compiler: /usr/bin/cc -- works
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Check for working CXX compiler: /usr/bin/c++
  -- Check for working CXX compiler: /usr/bin/c++ -- works
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
  -- Found torch: /libtorch/lib/libtorch.so
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /example_app/build
  $ make -j
  Scanning dependencies of target example_app
  [ 50%] Building CXX object CMakeFiles/example_app.dir/main.cpp.o
  [100%] Linking CXX executable example_app
  [100%] Built target example_app

And run it without passing a model just yet:

.. code-block:: shell

  $ ./example_app
  usage: example_app <path-to-exported-script-module>

Next, let's serialize the script function we wrote earlier that uses our custom
operator:

.. code-block:: python

  torch.ops.load_library("libwarp_perspective.so")

  @torch.jit.script
  def compute(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
    return x.matmul(y) + z

  compute.save("example.pt")

The last line will serialize the script function into a file called
"example.pt". If we then pass this serialized model to our C++ application, we
can run it straight away:

.. code-block:: shell

  $ ./example_app example.pt
  terminate called after throwing an instance of 'torch::jit::script::ErrorReport'
  what():
  Schema not found for node. File a bug report.
  Node: %16 : Dynamic = my_ops::warp_perspective(%0, %19)

Or maybe not. Maybe not just yet. Of course! We haven't linked the custom
operator library with our application yet. Let's do this right now, and to do it
properly let's update our file organization slightly, to look like this::

  example_app/
    CMakeLists.txt
    main.cpp
    warp_perspective/
      CMakeLists.txt
      op.cpp

This will allow us to add the ``warp_perspective`` library CMake target as a
subdirectory of our application target. The top level ``CMakeLists.txt`` in the
``example_app`` folder should look like this:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
  project(example_app)

  find_package(Torch REQUIRED)

  add_subdirectory(warp_perspective)

  add_executable(example_app main.cpp)
  target_link_libraries(example_app "${TORCH_LIBRARIES}")
  target_link_libraries(example_app -Wl,--no-as-needed warp_perspective)
  target_compile_features(example_app PRIVATE cxx_range_for)

This basic CMake configuration looks much like before, except that we add the
``warp_perspective`` CMake build as a subdirectory. Once its CMake code runs, we
link our ``example_app`` application with the ``warp_perspective`` shared
library.

.. attention::

  There is one crucial detail embedded in the above example: The
  ``-Wl,--no-as-needed`` prefix to the ``warp_perspective`` link line. This is
  required because we will not actually be calling any function from the
  ``warp_perspective`` shared library in our application code. We only need the
  ``TORCH_LIBRARY`` function to run. Inconveniently, this
  confuses the linker and makes it think it can just skip linking against the
  library altogether. On Linux, the ``-Wl,--no-as-needed`` flag forces the link
  to happen (NB: this flag is specific to Linux!). There are other workarounds
  for this. The simplest is to define *some function* in the operator library
  that you need to call from the main application. This could be as simple as a
  function ``void init();`` declared in some header, which is then defined as
  ``void init() { }`` in the operator library. Calling this ``init()`` function
  in the main application will give the linker the impression that this is a
  library worth linking against. Unfortunately, this is outside of our control,
  and we would rather let you know the reason and the simple workaround for this
  than handing you some opaque macro to plop in your code.

Now, since we find the ``Torch`` package at the top level now, the
``CMakeLists.txt`` file in the  ``warp_perspective`` subdirectory can be
shortened a bit. It should look like this:

.. code-block:: cmake

  find_package(OpenCV REQUIRED)
  add_library(warp_perspective SHARED op.cpp)
  target_compile_features(warp_perspective PRIVATE cxx_range_for)
  target_link_libraries(warp_perspective PRIVATE "${TORCH_LIBRARIES}")
  target_link_libraries(warp_perspective PRIVATE opencv_core opencv_photo)

Let's re-build our example app, which will also link with the custom operator
library. In the top level ``example_app`` directory:

.. code-block:: shell

  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
  -- The C compiler identification is GNU 5.4.0
  -- The CXX compiler identification is GNU 5.4.0
  -- Check for working C compiler: /usr/bin/cc
  -- Check for working C compiler: /usr/bin/cc -- works
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Check for working CXX compiler: /usr/bin/c++
  -- Check for working CXX compiler: /usr/bin/c++ -- works
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
  -- Found torch: /libtorch/lib/libtorch.so
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /warp_perspective/example_app/build
  $ make -j
  Scanning dependencies of target warp_perspective
  [ 25%] Building CXX object warp_perspective/CMakeFiles/warp_perspective.dir/op.cpp.o
  [ 50%] Linking CXX shared library libwarp_perspective.so
  [ 50%] Built target warp_perspective
  Scanning dependencies of target example_app
  [ 75%] Building CXX object CMakeFiles/example_app.dir/main.cpp.o
  [100%] Linking CXX executable example_app
  [100%] Built target example_app

If we now run the ``example_app`` binary and hand it our serialized model, we
should arrive at a happy ending:

.. code-block:: shell

  $ ./example_app example.pt
  11.4125   5.8262   9.5345   8.6111  12.3997
   7.4683  13.5969   9.0850  11.0698   9.4008
   7.4597  15.0926  12.5727   8.9319   9.0666
   9.4834  11.1747   9.0162  10.9521   8.6269
  10.0000  10.0000  10.0000  10.0000  10.0000
  10.0000  10.0000  10.0000  10.0000  10.0000
  10.0000  10.0000  10.0000  10.0000  10.0000
  10.0000  10.0000  10.0000  10.0000  10.0000
  [ Variable[CPUFloatType]{8,5} ]

Success! You are now ready to inference away.

Conclusion
----------

This tutorial walked you throw how to implement a custom TorchScript operator in
C++, how to build it into a shared library, how to use it in Python to define
TorchScript models and lastly how to load it into a C++ application for
inference workloads. You are now ready to extend your TorchScript models with
C++ operators that interface with third party C++ libraries, write custom high
performance CUDA kernels, or implement any other use case that requires the
lines between Python, TorchScript and C++ to blend smoothly.

As always, if you run into any problems or have questions, you can use our
`forum <https://discuss.pytorch.org/>`_ or `GitHub issues
<https://github.com/pytorch/pytorch/issues>`_ to get in touch. Also, our
`frequently asked questions (FAQ) page
<https://pytorch.org/cppdocs/notes/faq.html>`_ may have helpful information.

Appendix A: More Ways of Building Custom Operators
--------------------------------------------------

The section "Building the Custom Operator" explained how to build a custom
operator into a shared library using CMake. This appendix outlines two further
approaches for compilation. Both of them use Python as the "driver" or
"interface" to the compilation process. Also, both re-use the `existing
infrastructure <https://pytorch.org/docs/stable/cpp_extension.html>`_ PyTorch
provides for `*C++ extensions*
<https://pytorch.org/tutorials/advanced/cpp_extension.html>`_, which are the
vanilla (eager) PyTorch equivalent of TorchScript custom operators that rely on
`pybind11 <https://github.com/pybind/pybind11>`_ for "explicit" binding of
functions from C++ into Python.

The first approach uses C++ extensions' `convenient just-in-time (JIT)
compilation interface
<https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load>`_
to compile your code in the background of your PyTorch script the first time you
run it. The second approach relies on the venerable ``setuptools`` package and
involves writing a separate ``setup.py`` file. This allows more advanced
configuration as well as integration with other ``setuptools``-based projects.
We will explore both approaches in detail below.

Building with JIT compilation
*****************************

The JIT compilation feature provided by the PyTorch C++ extension toolkit allows
embedding the compilation of your custom operator directly into your Python
code, e.g. at the top of your training script.

.. note::

	"JIT compilation" here has nothing to do with the JIT compilation taking place
	in the TorchScript compiler to optimize your program. It simply means that
	your custom operator C++ code will be compiled in a folder under your system's
	`/tmp` directory the first time you import it, as if you had compiled it
	yourself beforehand.

This JIT compilation feature comes in two flavors. In the first, you still keep
your operator implementation in a separate file (``op.cpp``), and then use
``torch.utils.cpp_extension.load()`` to compile your extension. Usually, this
function will return the Python module exposing your C++ extension. However,
since we are not compiling our custom operator into its own Python module, we
only want to compile a plain shared library . Fortunately,
``torch.utils.cpp_extension.load()`` has an argument ``is_python_module`` which
we can set to ``False`` to indicate that we are only interested in building a
shared library and not a Python module. ``torch.utils.cpp_extension.load()``
will then compile and also load the shared library into the current process,
just like ``torch.ops.load_library`` did before:

.. code-block:: python

  import torch.utils.cpp_extension

  torch.utils.cpp_extension.load(
      name="warp_perspective",
      sources=["op.cpp"],
      extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
      is_python_module=False,
      verbose=True
  )

  print(torch.ops.my_ops.warp_perspective)

This should approximately print:

.. code-block:: python

  <built-in method my_ops::warp_perspective of PyCapsule object at 0x7f3e0f840b10>

The second flavor of JIT compilation allows you to pass the source code for your
custom TorchScript operator as a string. For this, use
``torch.utils.cpp_extension.load_inline``:

.. code-block:: python

  import torch
  import torch.utils.cpp_extension

  op_source = """
  #include <opencv2/opencv.hpp>
  #include <torch/script.h>

  torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
    cv::Mat image_mat(/*rows=*/image.size(0),
                      /*cols=*/image.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/image.data<float>());
    cv::Mat warp_mat(/*rows=*/warp.size(0),
                     /*cols=*/warp.size(1),
                     /*type=*/CV_32FC1,
                     /*data=*/warp.data<float>());

    cv::Mat output_mat;
    cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{64, 64});

    torch::Tensor output =
      torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{64, 64});
    return output.clone();
  }

  TORCH_LIBRARY(my_ops, m) {
    m.def("warp_perspective", &warp_perspective);
  }
  """

  torch.utils.cpp_extension.load_inline(
      name="warp_perspective",
      cpp_sources=op_source,
      extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
      is_python_module=False,
      verbose=True,
  )

  print(torch.ops.my_ops.warp_perspective)

Naturally, it is best practice to only use
``torch.utils.cpp_extension.load_inline`` if your source code is reasonably
short.

Note that if you're using this in a Jupyter Notebook, you should not execute
the cell with the registration multiple times because each execution registers
a new library and re-registers the custom operator. If you need to re-execute it,
please restart the Python kernel of your notebook beforehand.

Building with Setuptools
************************

The second approach to building our custom operator exclusively from Python is
to use ``setuptools``. This has the advantage that ``setuptools`` has a quite
powerful and extensive interface for building Python modules written in C++.
However, since ``setuptools`` is really intended for building Python modules and
not plain shared libraries (which do not have the necessary entry points Python
expects from a module), this route can be slightly quirky. That said, all you
need is a ``setup.py`` file in place of the ``CMakeLists.txt`` which looks like
this:

.. code-block:: python

  from setuptools import setup
  from torch.utils.cpp_extension import BuildExtension, CppExtension

  setup(
      name="warp_perspective",
      ext_modules=[
          CppExtension(
              "warp_perspective",
              ["example_app/warp_perspective/op.cpp"],
              libraries=["opencv_core", "opencv_imgproc"],
          )
      ],
      cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
  )


Notice that we enabled the ``no_python_abi_suffix`` option in the
``BuildExtension`` at the bottom. This instructs ``setuptools`` to omit any
Python-3 specific ABI suffixes in the name of the produced shared library.
Otherwise, on Python 3.7 for example, the library may be called
``warp_perspective.cpython-37m-x86_64-linux-gnu.so`` where
``cpython-37m-x86_64-linux-gnu`` is the ABI tag, but we really just want it to
be called ``warp_perspective.so``

If we now run ``python setup.py build develop`` in a terminal from within the
folder in which ``setup.py`` is situated, we should see something like:

.. code-block:: shell

  $ python setup.py build develop
  running build
  running build_ext
  building 'warp_perspective' extension
  creating build
  creating build/temp.linux-x86_64-3.7
  gcc -pthread -B /root/local/miniconda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include/torch/csrc/api/include -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include/TH -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include/THC -I/root/local/miniconda/include/python3.7m -c op.cpp -o build/temp.linux-x86_64-3.7/op.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=warp_perspective -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
  cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
  creating build/lib.linux-x86_64-3.7
  g++ -pthread -shared -B /root/local/miniconda/compiler_compat -L/root/local/miniconda/lib -Wl,-rpath=/root/local/miniconda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/op.o -lopencv_core -lopencv_imgproc -o build/lib.linux-x86_64-3.7/warp_perspective.so
  running develop
  running egg_info
  creating warp_perspective.egg-info
  writing warp_perspective.egg-info/PKG-INFO
  writing dependency_links to warp_perspective.egg-info/dependency_links.txt
  writing top-level names to warp_perspective.egg-info/top_level.txt
  writing manifest file 'warp_perspective.egg-info/SOURCES.txt'
  reading manifest file 'warp_perspective.egg-info/SOURCES.txt'
  writing manifest file 'warp_perspective.egg-info/SOURCES.txt'
  running build_ext
  copying build/lib.linux-x86_64-3.7/warp_perspective.so ->
  Creating /root/local/miniconda/lib/python3.7/site-packages/warp-perspective.egg-link (link to .)
  Adding warp-perspective 0.0.0 to easy-install.pth file

  Installed /warp_perspective
  Processing dependencies for warp-perspective==0.0.0
  Finished processing dependencies for warp-perspective==0.0.0

This will produce a shared library called ``warp_perspective.so``, which we can
pass to ``torch.ops.load_library`` as we did earlier to make our operator
visible to TorchScript:

.. code-block:: python

  >>> import torch
  >>> torch.ops.load_library("warp_perspective.so")
  >>> print(torch.ops.my_ops.warp_perspective)
  <built-in method custom::warp_perspective of PyCapsule object at 0x7ff51c5b7bd0>
