Extending dispatcher for a new backend in C++
=============================================

In this tutorial we will walk through all necessary steps to extend the dispatcher to
add a new device living outside ``pytorch/pytorch`` repo and maintain it to keep in
sync with native PyTorch devices.  Here we'll assume that you're familiar with how
to `register a dispatched operator in C++ <dispatcher>`_ and how to write a
`custom autograd function <cpp_autograd>`_.


.. note::

   This tutorial touches a lot of internal components inside PyTorch which are being actively improved,
   please expect changes to APIs if you decide to follow this tutorial.  We'll keep this tutorial
   up to date with the latest APIs.

What's a new backend?
---------------------

Adding a new backend to PyTorch requires a lot of developement and maintainence from backend extenders.
Before adding a new backend, let's first consider a few common use cases and recommended solutions for them:

* If you have new algorithms for an existing PyTorch operator, send a PR to PyTorch.
* If you want to propose a new operator, send a feature request/PR to PyTorch.
* If you want to add support for a new device/hardware like Google TPU and customized chips, which often requires using
  hardware-specific API to write kernels, follow this tutorial and add a out-of-tree backend to PyTorch.
* If you want to add support for existing operators but with a different Tensor layout/representation
  like sparse and quantized, which enforces your kernels to be written in a way that's more efficient
  given the layout/representation limitation, follow this tutorial and add a out-of-tree backend to PyTorch.

In this tutorial we'll mainly focus on adding a new out-of-tree device below.  Adding out-of-tree support
for a different tensor layout might share many common steps with devices, but we haven't seen an example of
such integrations yet so it might require addtional work from PyTorch to support it.

Get a dispatch key for your backend
-----------------------------------

PyTorch operators are implemented in C++ and made available in Python frontend through Python bindings.
The PyTorch dispatcher divides the implementation of an operator into multiple kernels, each of which is
associated with a specific dispatch key.  Supporting a new backend in PyTorch essentially means writing
a kernel for each PyTorch operator in C++ and then registering them to a dispatch key representing your
customized backend in the dispatcher.

Dispatch key is your identifier in the dispatcher system. The dispatcher looks at the dispatch keys carried on
input tensors and calls the right kernel accordingly.  PyTorch provides three reserved dispatch keys
(and their corresponding Autograd keys) for prototyping out-of-tree backend extensions:

* PrivateUse1/AutogradPrivateUse1
* PrivateUse2/AutogradPrivateUse2
* PrivateUse3/AutogradPrivateUse3

You can choose any of keys above to prototype your customized backend.
To create a Tensor on ``PrivateUse1`` backend, you need to set dispatch key in ``TensorImpl`` constructor.

.. code-block:: cpp

  /* Example TensorImpl constructor */
  TensorImpl(
      Storage&& storage,
      DispatchKeySet ks,
      const caffe2::TypeMeta data_type);

  // To create a TensorImpl on PrivateUse1 backend, pass in the following ks to TensorImpl creation.
  DispatchKeySet ks = c10::DispatchKeySet{c10::DispatchKey::PrivateUse1, c10::DispatchKey::AutogradPrivateUse1};


Note that ``TensorImpl`` class above assumes your Tensor is backed by a storage like CPU/CUDA. We also
provide ``OpaqueTensorImpl`` for backends without a storage. And you might need to tweak/override certain
methods to fit your customized hardware.
One example in pytorch repo is `Vulkan TensorImpl <https://github.com/pytorch/pytorch/blob/1.7/aten/src/ATen/native/vulkan/VulkanOpaqueTensorImpl.h>`_.


.. note::
   Once the prototype is done and you plan to do regular releases for your backend extension,  please feel free to
   submit a PR to ``pytorch/pytorch`` to reserve a dedicated dispath key for your backend.


Get the full list of PyTorch operators
--------------------------------------

PyTorch provides a full list of extensible C++ operators in generated file
``build/aten/src/ATen/RegistrationDeclarations.h``.
This file is only available after building PyTorch from source.
Here's a snippet of the file:

.. code-block:: cpp

  Tensor abs(const Tensor & self); // {"schema": "aten::abs(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}
  Tensor & abs_(Tensor & self); // {"schema": "aten::abs_(Tensor(a!) self) -> Tensor(a!)", "dispatch": "True", "default": "True"}
  Tensor & abs_out(Tensor & out, const Tensor & self); // {"schema": "aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
  Tensor absolute(const Tensor & self); // {"schema": "aten::absolute(Tensor self) -> Tensor", "dispatch": "False", "default": "False"}
  Tensor & absolute_(Tensor & self); // {"schema": "aten::absolute_(Tensor(a!) self) -> Tensor(a!)", "dispatch": "False", "default": "False"}
  Tensor & absolute_out(Tensor & out, const Tensor & self); // {"schema": "aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "False", "default": "False"}
  Tensor angle(const Tensor & self); // {"schema": "aten::angle(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}
  Tensor & angle_out(Tensor & out, const Tensor & self); // {"schema": "aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
  Tensor sgn(const Tensor & self); // {"schema": "aten::sgn(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}

There're multiple fields associated with a single operator. Let's break it down using ``abs_out`` as an example:

* ``Tensor & abs_out(Tensor & out, const Tensor & self);`` is the C++ signature of the operator, your C++
  kernel should match this signature exactly.
* ``aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)`` is the unique schema representing the operator,
  which also contains aliasing and mutation annotations compared to the C++ signature.  This is the unique identifier
  the dispatcher uses to find an operator.
* ``dispatch`` and ``default`` are boolean fields that provide information about what native PyTorch kernels
  can do, thus implies whether it's required for backend extenders to implement the kernel.
  More details can be found in :ref:`register kernels for the new backend<register-kernel>`.


.. _register-kernel:

Register kernels for the new backend
------------------------------------

To register your kernels to PyTorch dispatcher, you can use the
``TORCH_LIBRARY_IMPL`` API described in
`Registering a Dispatched Operator in C++ <dispatcher>`_:

.. code-block:: cpp

  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl(<schema_my_op1>, &my_op1);
    m.impl(<schema_my_op2>, &my_op2);
    m.impl(<schema_my_op2_backward>, &my_op2_backward);
  }

Now let's zoom in and what operator requires a kernel from a customized backend and what's
inside the kernels exactly.

PyTorch currently has more than 1600 operators and it’s still growing.  It’s unrealistic
for backend extensions to keep up with this speed.  Even for native backends like CPU
or CUDA, it often requires a lot of work to write dedicated kernels for every new op.

Fortunately, some native PyTorch kernels are written in a way that they decompose to
combination of several known operators. In other words, you only need to implement
a set of known operators (ops that require registration below) instead of all PyTorch operators.

PyTorch operators can be classified into two categories:

* Ops that require registration: PyTorch native implementation for these ops is backend specific
  and thus it’s required to provide a kernel for customized backend.  Otherwise calling such op
  on the customized backend will error out.
    * In ``RegistrationDeclarations.h`` these operators have ``dispatch`` set to True *and* ``default`` set to False
      in the metadata found in their accompanying comments.


* Registration is optional: backend extenders can skip registering to these ops without sacrificing any support.
  However, if a backend extender wants to override the default kernel provided by PyTorch, they can still
  register their customized kernel to their backend and the dispatcher will use it for your backend only.
  For example, current implementation of PyTorch's ``max_pool2d`` returns ``indices`` as part of forward outputs which
  creates overhead in torch_xla, so torch_xla registers its own kernel for ``max_pool2d`` instead.
    * In ``RegistrationDeclarations.h`` these operators have ``dispatch`` set to False *or* ``default`` set to True
      in the metadata found in their accompanying comments.



Autograd support for the new backend
------------------------------------

Gradient formulas are mostly purely mathematical and thus are general for all backends.
PyTorch often registers a kernel to alias dispatch key Autograd, which means it can be used by all backends.

For these operators you don't have to worry about their derivative formulas,
you can just write forward definitions for operators in ``RegistrationDeclarations.h`` and PyTorch handles
backward for you automatically.

.. code-block:: cpp


  Tensor my_op1(const Tensor& self, const Tensor& other) {
    // call your backend-specific APIs to implement my_op so that
    // it matches PyTorch's native behavior
  }
  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl(<schema_my_op1>, &my_op);
  }


In some cases, PyTorch backward kernel implementations are also device specific so that they can squeeze out
max performance out of each backend. For those operators you’ll see op_backward showing up in
``RegistrationDeclarations.h`` as *required registration* as well.

.. code-block:: cpp


  Tensor my_op2_backward(const Tensor& self, const Tensor& other) {
    // call your backend-specific APIs to implement my_op2_backward so that
    // it matches PyTorch's native behavior
  }

  // Note backward kernel is still registered to PrivateUse1 instead of AutogradPrivateUse1.
  // PyTorch will wrap your backward kernel with proper autograd setup and then link to it in
  // my_op2's AutogradPrivateUse1 kernel.
  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl(<schema_my_op2>, &my_op2);
    m.impl(<schema_my_op2_backward>, &my_op2_backward);
  }


In a few *rare* cases, PyTorch’s gradient formula for certain operators may have assumptions that don’t generalize
for all backends. In those cases backend extenders can optionally override PyTorch Autograd layer by registering
a kernel from torch::autograd::Function to the corresponding dispatch key (for example, AutogradPrivateUse1 if
you're using PrivateUse1 for your backend):


.. code-block:: cpp


  class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
    public:
    static Tensor forward(AutogradContext *ctx, torch::Tensor self, torch::Tensor other) {
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

  // Register the autograd kernel to AutogradPrivateUse1
  TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl(<myadd_schema>, &myadd_autograd);
  }

  // Register the inference kernel to PrivateUse1
  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl(<myadd_schema>, &myadd);
  }



With this trick you have full control over both training and inference behavior for ``my_add`` operator in your backend.
Here's `an example <https://github.com/pytorch/xla/blob/r1.7/torch_xla/csrc/aten_autograd_ops.h>`_ in the ``pytorch/xla`` repository.


Build an extension
------------------

Out-of-tree backend is supported by adding a C++ extension to PyTorch.
Once you have kernels and registrations ready, you can build a C++ extension by
writing a ``setup.py`` script that uses ``setuptools`` to compile C++ code.  Here's a simplified example from
`pytorch/xla repo <https://github.com/pytorch/xla/blob/master/setup.py>`_::

  from setuptools import setup
  from torch.utils.cpp_extension import BuildExtension, CppExtension

  setup(
      name='torch_xla',
      ext_modules=[
          CppExtension(
              '_XLAC',
              torch_xla_sources,
              include_dirs=include_dirs,
              extra_compile_args=extra_compile_args,
              library_dirs=library_dirs,
              extra_link_args=extra_link_args + \
                  [make_relative_rpath('torch_xla/lib')],
          ),
      ],
      cmdclass={
          'build_ext': Build,  # Build is a derived class of BuildExtension
      }
      # more configs...
  )


See `our C++ extension tutorial <https://pytorch.org/tutorials/advanced/cpp_extension.html#building-with-setuptools>`_
for more details.


Custom operator support
-----------------------

Your new backend should work seamlessly with
`customized operators extended in python <https://pytorch.org/docs/stable/notes/extending.html>`_
without writing any new kernels as long as the customized operator is composed of existing
PyTorch operators (which are already supported by your backend).

For `custom operators extended in C++ <cpp_autograd>`_ they often come with a
`backend specific C++ kernel implementation e.g. nms kernel in torchvsion <https://github.com/pytorch/vision/blob/master/torchvision/csrc/ops/cuda/nms_kernel.cu>`_
as well as `a customized Python API e.g. torch.ops.torchvision.nms <https://github.com/pytorch/vision/blob/master/torchvision/csrc/ops/nms.cpp#L18>`_.
To support these operators, backend extenders will need to write a C++ kernel for your backend and properly
register it to the corresponding namespace in the dispatcher similar to supporting PyTorch native operators.
Alternatively you could also add a customized API in your extension e.g ``torch_xla.core.functions.nms`` for
these adhoc requests.

JIT support
-----------

As we mentioned in `Registering a Dispatched Operator in C++ <dispatcher>`_, kernels registered through `m.impl()` API
support being called in both unboxed and boxed ways. In other words your customized backend can also work with our
JIT tracing/scripting frontend just like the in-tree backends like CPU or CUDA do.  You could potentially also write specialized optimization
passes for your backend on a JIT graph.  But we will not discuss it here since we haven't finalized the integration point
in JIT, so the current backend support will focus on the eager frontend for now.


Testing your backend against native PyTorch backends
----------------------------------------------------

PyTorch lets tests run on multiple device types using its `generic device type testing framework <https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_device_type.py>`_.
You can find details about `how tests use it <https://github.com/pytorch/pytorch/blob/5a8198eb3c594aa18352930fd21f3c25bd7b7100/torch/testing/_internal/common_device_type.py#L23>`_
and information about `how to add a new device type <https://github.com/pytorch/pytorch/blob/5a8198eb3c594aa18352930fd21f3c25bd7b7100/torch/testing/_internal/common_device_type.py#L369>`_.
Once added, PyTorch tests using the generic device type testing framework will be run using your device type, too.
See `this Wiki page <https://github.com/pytorch/pytorch/wiki/Writing-tests-that-run-on-all-available-device-types>`_ for an example of how tests are instantiated.

Running PyTorch’s existing test suites with your device type is important to ensure correctness,
but not all PyTorch features are supported by every device type.  The generic device type testing
framework allows for considerable customization so that device types can select which tests to run,
which dtypes they support, and even which precisions to use when comparing tensors for equality.

An example device type that uses the generic device type testing framework and doesn’t ship with
PyTorch is XLA.  See `its extension of the generic device type testing framework <https://github.com/pytorch/xla/blob/master/test/pytorch_test_base.py>`_,
which contains examples of block listing tests, block listing dtypes, and overriding test precision.

The generic device type testing framework is actively developed. To request a feature please file an
issue on PyTorch’s Github.


Backward Compatibility
----------------------

Currently PyTorch can’t guarantee backward compatibility for registered operators.
Operators, as well as their schemas, might be added/modified/deleted as needed.  Registered
kernels must be *exactly* the same as PyTorch version.  If PyTorch adds more parameters (
even with defaults) for an operator, your old registration won't work until it's updated
to match PyTorch's new signature.

As a result, we *highly recommend* out-of-tree backend extenders only sync with major PyTorch
releases to minimize interruptions in development.  PyTorch is on a quarterly release cadence.
Backend extenders should join the *#announcement* channel at `pytorch.slack.com <http://pytorch.slack.com/>`_
to get latest updates on releases.

Known issues & additional notes
-------------------------------

*  Not all test suites are device generic yet. Extensible test classes can be found by searching
   ``instantiate_device_type_tests`` in PyTorch codebase, e.g
   ``TestTorchDeviceType, TestViewOps, TestTensorDeviceOps, TestTypePromotion`` etc.
* There's no extension point in C++ for serializing a python Tensor object on customized backend. Currently
  you can only extend it by modifying `PyTorch Tensor __reduce_ex__ method <https://github.com/pytorch/pytorch/blob/5640b79bf8a5412a0209a919c05c811d5427cc12/torch/tensor.py#L83-L150>`_
  or monkey patching in out-of-tree repository.
* If your backend doesn't allow direct memory access, you should pay additional attention to supporting
  view ops since they're supposed to share storage. Changes to view tensor need to propagated to its
  base tensor and vice versa.
* There's no extension point in C++ for Optimizer if your backend doesn't work with the native PyTorch
  Optimizers, e.g. need to carry the states to be updated in backward like torch-xla. Such use cases
  currently can only be done through adding customized API or monkey patching in out-of-tree repository.

Future Work
-----------

Making every component in PyTorch extensible for an out-of-tree backend seamless
requires a lot of changes to PyTorch internals.  Here are a few items that we're
actively working on might improve the experience in the future:

* Improve test coverage of generic testing framework.
* Improve ``Math`` kernel coverage and more comprehensive tests to make sure ``Math``
  kernel bahavior matches other backends like ``CPU/CUDA``.
* Refactor ``RegistrationDeclarations.h`` to carry the minimal information and reuse
  PyTorch's codegen as much as possible.
* Support a backend fallback kernel to automatic convert inputs to CPU and convert the
  result back to the customized backend. This will allow "full" operator coverage even
  though you don't have kernels written for every operator.


Stay in touch
-------------

Please use `PyTorch dev discussions <https://dev-discuss.pytorch.org/>`_ for questions and discussions. If you have
any feature requests or bug reports, please `file an issue on github <https://github.com/pytorch/pytorch/issues>`_.

If you're interested in helping in any of the future work items above (e.g adding more ``Math``
kernels for PyTorch operators in C++), please reach out to us through Github or Slack!

