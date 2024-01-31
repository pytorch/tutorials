Registering a Dispatched Operator in C++
========================================

The dispatcher is an internal component of PyTorch which is responsible for
figuring out what code should actually get run when you call a function like
``torch::add``.  This can be nontrivial, because PyTorch operations need
to handle a lot of cross-cutting concerns that are "layered" on top of one
of another.  Here is a sampling of some of the things it handles:

* Switching between the CPU and CUDA implementations of an operator, depending
  on the devices of the input tensors.
* Switching between the autograd and backend implementations of an operator,
  depending on whether or not autograd handling is necessary.
* Applying autocasting when necessary for automatic mixed precision.
* Applying batching rules when an operator is run under a ``vmap`` call.
* Tracing execution of operations, if you are tracing a model for export.

If in your `custom operator code <torch_script_custom_ops>`_ you find yourself
manually writing if statements to handle these cases, the dispatcher APIs can
help organize your code.  (Conversely, if your custom operator is very simple
and is only for CPU inference, you probably don't need to use the dispatcher,
just use the basic API.)

In this tutorial, we will describe how to structure a custom operator
registration to use the dispatcher to organize various components.  We'll
assume that you are familiar with how to
`register an operator <torch_script_custom_ops>`_ and how to write
a `custom autograd function <cpp_autograd>`_.

Defining schema and backend implementations
-------------------------------------------

The general principle behind the dispatcher is that it divides the
implementation of an operator into multiple kernels, each of which implements
functionality for a specific *dispatch key*, e.g. CPU, CUDA.  The dispatcher
determines what the highest priority dispatch key is at the time
you call an operator (this is done by looking at both the tensor arguments as
well as some thread local state), and transfers control to the kernel for that
dispatch key.  The end effect is that when you call an operator, we first
execute the Autograd kernel, and then we redispatch to the backend kernel
depending on the device types of the passed in tensors.

Let's take a look at the various parts involved in making this
happen.  First, we must define the schema for the operator in question.
Unlike simple pybind11-style operator registration, we don't actually
provide an implementation of our operator at this point; we just
provide a schema string specifying the type signature of the operator
that all of our other kernels will abide by:

.. literalinclude:: ../advanced_source/dispatcher/op.cpp
  :language: cpp
  :start-after: BEGIN TORCH_LIBRARY
  :end-before: END TORCH_LIBRARY

Next, we need to actually provide some implementations of this operator.
For concreteness, here is a really simple implementation of addition on CPU:

.. literalinclude:: ../advanced_source/dispatcher/op.cpp
  :language: cpp
  :start-after: BEGIN myadd_cpu
  :end-before: END myadd_cpu

We'd like to register this function as an implementation of ``myops::myadd``.
However, the simple way of registering it (``def("myadd", myadd_cpu)``) would
register the kernel to run in all cases, even if the tensor is not a CPU
tensor!  (Internally, we refer to these as "catch-all" kernels, since they
catch all cases.)  To ensure that ``myadd_cpu`` is only run for
CPU tensors, we can use the ``TORCH_LIBRARY_IMPL`` macro:

.. literalinclude:: ../advanced_source/dispatcher/op.cpp
  :language: cpp
  :start-after: BEGIN TORCH_LIBRARY_IMPL CPU
  :end-before: END TORCH_LIBRARY_IMPL CPU

The ``TORCH_LIBRARY_IMPL`` lets us register implementations for operators on
a specific dispatch key (in this case, CPU).  Each call to ``impl``
associates a CPU kernel with the corresponding operator (which we previously
defined in the ``TORCH_LIBRARY`` block).  If we also have a CUDA implementation ``myadd_cuda``,
we can register it in a separate ``TORCH_LIBRARY_IMPL`` block:

.. literalinclude:: ../advanced_source/dispatcher/op.cpp
  :language: cpp
  :start-after: BEGIN TORCH_LIBRARY_IMPL CUDA
  :end-before: END TORCH_LIBRARY_IMPL CUDA

These registrations can be split across files or even across library boundaries; so
for example, you could have these two ``TORCH_LIBRARY_IMPL`` blocks compiled
into a separate ``myops_cpu`` and ``myops_cuda`` dynamic libraries.  Generally,
speaking, the structure of your registrations will look like this:

1. A single ``TORCH_LIBRARY`` that lists every custom operator in your namespace
   in a centralized place.
2. A ``TORCH_LIBRARY_IMPL`` per dispatch key that registers implementations for
   that key (e.g., CPU or CUDA).  If you like, you can further subdivide
   ``TORCH_LIBRARY_IMPL`` blocks into a block per operator. This is convenient
   if you have a separate file per operator implementation, but don't want to
   expose the operators in a header; you can just put the registration in the
   cpp file that defines your operator.

.. note::

    Did you know that you can also write ``TORCH_LIBRARY_IMPL`` blocks for existing
    core operators in PyTorch?  This is how XLA support for PyTorch is
    implemented: the ``torch_xla`` library contains a ``TORCH_LIBRARY_IMPL``
    that provides implementations for all basic operators on the XLA dispatch
    key.


For operators that do not need autograd
---------------------------------------

Note: This section only applies to versions of PyTorch ``>= 1.10``.

In the next section, we will discuss how to add autograd support to an operator.
But for the ops that do not need autograd support, the following kernel should be
registered improve useability and make your op behave like PyTorch's built-in
operators.

.. code-block:: cpp

  TORCH_LIBRARY_IMPL(myops, Autograd, m) {
    m.impl(op, autogradNotImplementedFallback());
  }

The above lines registers an ``Autograd`` kernel that appends a dummy
``NotImplemented`` node on forward (preserving the ``require_grad``-ness of the inputs).
On backward, the ``NotImplemented`` node raises an error. This can be helpful
for debugging in larger models where previously it can be hard to pin-point
exactly where the ``requires_grad``-ness is lost during the forward pass.

In-place or view ops
^^^^^^^^^^^^^^^^^^^^

To ensure correctness and best possible performance, if your op mutates an input
in-place or returns a tensor that aliases with one of the inputs, two additional
steps should be taken:

1. Register an ``ADInplaceOrView`` kernel in addition to the ``Autograd`` kernel
   above. This kernel handles the necessary bookkeeping to ensure the correctness
   of in-place or view operations. It is important to note that this ADInplaceOrView
   kernel should only be used with ``autogradNotImplementedFallback``.

.. code-block:: cpp

  TORCH_LIBRARY_IMPL(myops, Autograd, m) {
    m.impl(op, autogradNotImplementedFallback());
  }
  TORCH_LIBRARY_IMPL(myops, ADInplaceOrView, m) {
    m.impl(op, autogradNotImplementedInplaceOrViewFallback());
  }

2. The ``Autograd`` or ``ADInplaceOrView`` boxed kernels registered above
   rely on operator schema information in their logi. If your op mutates an input
   in-place or returns a tensor that aliases with one of the inputs it is important to
   ensure that your schema properly reflects this. See
   `here <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/README.md>`_
   for more information on how to annotate the schema.

.. _autograd-support:

Adding autograd support
-----------------------

At this point, we have an operator with both CPU and CUDA implementations.  How
can we add autograd support to it?  As you might guess, we will register an
autograd kernel (similar to what's described in the `custom autograd function <cpp_autograd>`_ tutorial)!
However, there is a twist: unlike the CPU and CUDA kernels, the autograd kernel
needs to *redispatch*: it needs to call back into the dispatcher to get to
the inference kernels, e.g. CPU or CUDA implementations.

Thus, before we write the autograd kernel, let's write a *dispatching function*
which calls into the dispatcher to find the right kernel for your operator.
This function constitutes the public C++ API for your operators--in fact, all of
the tensor functions in PyTorch's C++ API all call the dispatcher in the same
way under the hood.  Here's what the dispatching function looks like:

.. literalinclude:: ../advanced_source/dispatcher/op.cpp
  :language: cpp
  :start-after: BEGIN myadd
  :end-before: END myadd

Let's break it down:

* In the first line, we look up a typed operator handle from the dispatcher
  corresponding to the operator that we are going to dispatch to.
  ``findSchemaOrThrow`` takes two arguments: the (namespace qualified) name
  of the operator, and the overload name of the operator (typically just
  the empty string).  ``typed`` casts the dynamically typed handle into
  a statically typed handle (doing a runtime test to make sure you've given
  the correct C++ type), so that we can do a normal C++ call on it.  We
  pass it ``decltype(myadd)`` since the type of the dispatching function is
  the same as the type of the underlying kernels registered to the dispatcher.

  For performance, this computation is done in a static variable, so that
  we only need to do the (slow) lookup once.  If you typoed the name of the
  operator you want to call, this lookup will error the first time you call this
  function.

* In the second line, we simply ``call`` the operator handle with all of the
  arguments passed into the dispatching function.  This will actually invoke
  the dispatcher and in the end control will be transferred to whatever kernel
  is appropriate for this call.

With the dispatch function in hand, we can now write the autograd kernel:

.. literalinclude:: ../advanced_source/dispatcher/op.cpp
  :language: cpp
  :start-after: BEGIN myadd_autograd
  :end-before: END myadd_autograd

The autograd function is written as normal using ``torch::autograd::Function``,
except that instead of directly writing the implementation in ``forward()``,
we:

1. Turn off autograd handling with the ``at::AutoNonVariableTypeMode`` RAII
   guard, and then
2. Call the dispatch function ``myadd`` to call back into the dispatcher.

Without (1), your calls will infinite loop (and stack overflow), because
``myadd`` will send you back to this function (as the highest priority dispatch
key would still be autograd.) With (1),
autograd is excluded from the set of dispatch keys under consideration, and
we will go to the next handlers, which will either be CPU and CUDA.

We can now register this function in the same way we registered the CPU/CUDA
functions:

.. literalinclude:: ../advanced_source/dispatcher/op.cpp
  :language: cpp
  :start-after: BEGIN TORCH_LIBRARY_IMPL Autograd
  :end-before: END TORCH_LIBRARY_IMPL Autograd


.. note::

    In this example we register the kernel to ``Autograd``, which installs it as the
    autograd kernel for all backends. You can also register optimized kernels for specific
    backends by using the corresponding backend-specific dispatch key - for example,
    ``AutogradCPU`` or ``AutogradCUDA``. To explore these and other dispatch key
    options in more detail, check out the ``PythonDispatcher`` tool provided in
    `torch/_python_dispatcher.py <https://github.com/pytorch/pytorch/blob/master/torch/_python_dispatcher.py>`_.


Going beyond autograd
---------------------

In some sense, the dispatcher isn't doing all that much: all it does is
implement a glorified if-statement, along the lines of this:

.. code-block:: cpp

    class MyAddFunction : ... {
    public:
      static Tensor forward(
        AutogradContext *ctx, torch::Tensor self, torch::Tensor other) {

        if (self.device().type() == DeviceType::CPU) {
          return add_cpu(self, other);
        } else if (self.device().type() == DeviceType::CUDA) {
          return add_cuda(self, other);
        } else {
          TORCH_CHECK(0, "Unsupported device ", self.device().type());
        }
      }
      ...
    }

So why use the dispatcher?  There are a few reasons:

1. It is decentralized.  You can assemble all of the pieces of an operator
   (CPU, CUDA, Autograd) without having to write a single, centralized
   if statement that refers to all of them.  Importantly, third parties can
   register extra implementations for other aspects without having to patch the
   original definition of an operator.  We'll talk more about extending the
   dispatcher in `extending dispatcher for a new backend <extend_dispatcher>`_.

2. It supports more dispatch keys than CPU, CUDA and Autograd.  You can
   see a full list of dispatch keys that are currently implemented
   in PyTorch in ``c10/core/DispatchKey.h``.  These dispatch keys
   implement a variety of optional functionality for operators, and if you
   decide you want your custom operator to support this functionality,
   all you have to register a kernel for the appropriate key.

3. The dispatcher implements support for boxed fallback functions, which
   are functions that can be implemented once and apply to all operators
   in the system.  Boxed fallbacks can be used to provide default behavior
   for a dispatch key; if you use the dispatcher to implement your operator,
   you also opt into the fallbacks for all of these operations.

Here are some particular dispatch keys which you may need to define an operator
for.

Autocast
^^^^^^^^

The Autocast dispatch key implements support for
`automatic mixed precision (AMP) <https://pytorch.org/docs/stable/amp.html>`_.
An autocast wrapper kernel typically casts incoming ``float16`` or ``float32`` CUDA tensors
to some preferred precision before running the op.
For example, matmuls and convolutions on floating-point CUDA tensors usually run faster
and use less memory in ``float16`` without impairing convergence.
Autocast wrappers only have an effect in
`autocast-enabled contexts <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast>`_.

Here's an autocast wrapper for a hypothetical custom matmul, along with its registration:

.. code-block:: cpp

    // Autocast-specific helper functions
    #include <ATen/autocast_mode.h>

    Tensor mymatmul_autocast(const Tensor& self, const Tensor& other) {
      c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
      return mymatmul(at::autocast::cached_cast(at::kHalf, self),
                      at::autocast::cached_cast(at::kHalf, other));
    }

    TORCH_LIBRARY_IMPL(myops, Autocast, m) {
      m.impl("mymatmul", mymatmul_autocast);
    }

``cached_cast(kHalf, tensor)`` casts ``tensor`` to ``float16`` if ``tensor`` is CUDA and ``float32``,
otherwise, it leaves ``tensor`` unchanged (c.f. the
`eligibility policy <https://pytorch.org/docs/stable/amp.html#op-eligibility>`_ for natively autocasted ops).
This ensures if the network calls ``mymatmul`` on any mixture of ``float16`` and ``float32`` CUDA tensors,
``mymatmul`` runs in ``float16``.  Meanwhile, calls to ``mymatmul`` with non-CUDA, integer-type, or ``float64``
inputs are unaffected.  Using ``cached_cast`` to follow the native eligibility policy in your own autocast wrapper
is recommended, but not required.  For example, if you wanted to force ``float16`` execution for all input types,
you could ``return mymatmul(self.half(), other.half());`` instead of using ``cached_cast``.

Notice that, like our autograd kernels, we exclude the ``Autocast`` key from
dispatch before redispatching.

By default, if no autocast wrapper is provided,
we fallthrough directly to the regular operator implementation (no
autocasting occurs).  (We didn't use ``myadd`` for this example, since pointwise
addition doesn't need autocasting and should just fall through.)

When should an autocast wrapper be registered? Unfortunately, there aren't
cut-and-dried rules for an op's preferred precision.  You can
get a sense for some native ops' preferred precisions by looking at the
`cast lists <https://pytorch.org/docs/master/amp.html#op-specific-behavior>`_.
General guidance:

* Ops that do reductions should probably execute in ``float32``,
* Any op that does a convolution or gemm under the hood should
  probably execute in ``float16``, and
* Other ops with multiple floating-point tensor inputs should standardize
  them to a common precision (unless the implementation supports inputs with different precisions).

If your custom op falls into the third category, the ``promote_type`` template
helps figure out the widest floating-point type present among input tensors, which is
the safest choice for the execution type:

.. code-block:: cpp

    #include <ATen/autocast_mode.h>

    Tensor my_multiple_input_op_autocast(const Tensor& t0, const Tensor& t1) {
      c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
      // The required at::kHalf argument is an optimistic initial guess.
      auto exec_type = at::autocast::promote_type(at::kHalf, t0, t1);
      return my_multiple_input_op(at::autocast::cached_cast(exec_type, t0),
                                  at::autocast::cached_cast(exec_type, t1));
    }

If your custom op is :ref:`autograd-enabled<autograd-support>`, you only need to write and register
an autocast wrapper for the same name onto which the autograd wrapper is registered.
For example, if you wanted an autocast wrapper for the ``myadd`` function shown
in the autograd section, all you'd need is

.. code-block:: cpp

    Tensor myadd_autocast(const Tensor& self, const Tensor& other) {
      c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
      return myadd(at::autocast::cached_cast(<desired dtype>, self),
                   at::autocast::cached_cast(<desired dtype>, other));
    }

    TORCH_LIBRARY_IMPL(myops, Autocast, m) {
      m.impl("myadd", myadd_autocast);
    }

There are no separate gymnastics to make the backward method autocast compatible.
However, the backward method defined in your custom autograd function will run in the same
dtype as autocast sets for the forward method, so you should choose a ``<desired dtype>``
suitable for both your forward and backward methods.

Batched
^^^^^^^

Batched tensors allow you to write your code in a per-example manner, and then
have them be automatically batched when run under a ``vmap`` invocation.  The
API for writing batching rules is currently under development, but once it is
stabilized, you can add support for ``vmap`` for your operators by registering
a kernel at the Batched dispatch key.

Tracer
^^^^^^

The Tracer dispatch key implements support for recording invocations of operators
into a trace when you run ``torch.jit.trace``.  We intend to provide a
boxed fallback that will implement tracing for arbitrary operations,
see `issue #41478 <https://github.com/pytorch/pytorch/issues/41478>`_ to track
progress.
