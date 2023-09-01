Facilitating New Backend Integration by PrivateUse1
===================================================

In this tutorial we will walk through some necessary steps to integrate a new backend
living outside ``pytorch/pytorch`` repo by ``PrivateUse1``. Note that this tutorial assumes that
you already have a basic understanding of PyTorch.
you are an advanced user of PyTorch.

.. note::

   This tutorial only involves the parts related to the PrivateUse1 mechanism that facilitates the integration of new devices,
   and other parts will not be covered. At the same time, not all the modules involved in this tutorial are required,
   and you can choose the modules that are helpful to you according to your actual needs.


What is PrivateUse1?
--------------------

Prior to Pytorch 2.0, PyTorch provided three reserved dispatch keys (and their corresponding Autograd keys)
for prototyping out-of-tree backend extensions, the three dispatch keys are as follows:

* ``PrivateUse1/AutogradPrivateUse1``
* ``PrivateUse2/AutogradPrivateUse2``
* ``PrivateUse3/AutogradPrivateUse3``

After the prototype verification is passed, you can apply for a private key for the new backend, such as CUDA, XLA, MPS, and so on.

However, with the rapid development of PyTorch, more and more hardware manufacturers are trying to
integrate their backends into PyTorch, which might cause the following problems:

* Every new backend integration involves a lot of file modification
* There is currently a hard limit on the number of Dispatch Keys (``DispatchKeySet`` 64-bit limit)

.. note::

   There is also a problem with integrating the new backend into PyTorch through the PrivateUse1 Key, as it is impossible
   to integrate many backends at the same time. Fortunately, these out-of-tree backends are rarely used simultaneously.


In view of the above reasons, the community began to recommend new backend to be integrated
into the PyTorch via ``PrivateUse1``.

However, the previous ``PrivateUse1`` mechanism is not fully capable of integrating with the new backend, because it
lacks some related support in certain modules, such as Storage, AMP, Distributed, and so on.

With the arrival of Pytorch 2.1.0, a series of optimizations and enhancements have been made
for ``PrivateUse1`` in terms of new backend integration, and it is now possible to support the integration
of new devices rapidly and efficiently.

How to integrate new backend via PrivateUse1
--------------------------------------------

In this section, we will discuss the details of integrating the new backend into Pytorch via ``PrivateUse1``,
which mainly consists of the following parts:

1. Register kernels for the new backend.
2. Register generator for the new backend.
3. Register device guard for the new backend.
4. Register serialization and deserialization functions for new backend metadata.
5. Other Modules.

Register kernels for the new backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The new backend may have some high-performance implementations of operator, which can be registered to the dispatcher
by ``TORCH_LIBRARY_IMPL`` API described in `Registering a Dispatched Operator in C++ <dispatcher>`_. This involves
several situations:

1. Register all the forward operators supported by the new backend to the dispatcher, and register the fallback
   at the same time, so that when the new backend does not support some operators, these operators can fall back
   to the CPU for execution to ensure the availability of functions.

.. code-block:: cpp

  at::Tensor wrapper_Custom_Tensor_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    // Implementation of add kernel in new backend
    ...
  }

  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    ...
    m.impl("add.Tensor", TORCH_FN(wrapper_Custom_Tensor_add));
    ...
  }

  void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // Add some hints about new devices that do not support and need to fall back to cpu
    at::native::cpu_fallback(op, stack);
  }

  TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  }

2. Register kernels from ``torch::autograd::Function`` to the dispatcher by ``AutogradPrivateUse1``, if it is necessary for
   new backend to override ``PyTorch Autograd layer``, the dispatcher and autograd system will automatically call the forward and
   backward implementations of these operators.

.. code-block:: cpp

  class CumtomSeluFunction : public torch::autograd::Function<CumtomSeluFunction> {
    // Implementation of selu kernel in new backend
  }

  at::Tensor wrapper_AutogradCumstom__selu(const at::Tensor & self) {
    return CumtomSeluFunction::apply(self);
  }

  TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    ...
    m.impl("selu", TORCH_FN(wrapper_AutogradCustom__selu));
    ...
  }

3. Register kernels which want to support `automatic mixed precision (AMP) <https://pytorch.org/docs/stable/amp.html>`_ and
   fallback mechanism to the dispatcher by ``AutocastPrivateUse1``, the autocast system will automatically call these kernels when needed.

.. code-block:: cpp

  TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
    ...
    KERNEL_PRIVATEUSEONE(<operator>, <policy>)
    ...
  }

  TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
  }

What needs to be added is that if you want to support AMP in a new backend, you need to register a new ``BackendModule`` by
``torch._register_device_module("backend_name", BackendModule)``, and the ``BackendModule`` needs to have the following APIs:

* ``get_amp_supported_dtype() -> List[torch.dtype]``
    get the supported dtypes on the new backend in AMP, which might support one more ``dtype``.
* ``is_autocast_enabled() -> bool``
    check the AMP is enabled or not on the new backend.
* ``get_autocast_dtype() -> torch.dtype``
    get the supported ``dtype`` on the new backend in AMP, which is set by ``set_autocast_dtype`` or the
    default ``dtype``, and the default ``dtype`` is ``torch.float16``.
* ``set_autocast_enabled(bool) -> None``
    enable or disable AMP on the new backend.
* ``set_autocast_dtype(dtype) -> None``
    set the supported ``dtype`` on the new backend in AMP, and the ``dtype`` be contained in the ``dtypes`` got
    from ``get_amp_supported_dtype``.

Register generator for the new backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is necessary to support generators corresponding to new devices. Currently, ``PrivateUse1`` can dynamically
register custom generators, which are mainly divided into the following steps.

1. Inherit the ``GeneratorImpl`` class to implement the generator class corresponding to the new backend,
   and implement various general methods.
2. Define a new backend ``builder`` with a single parameter: ``device index``.
3. Call ``REGISTER_GENERATOR_PRIVATEUSE1`` macro to complete dynamic registration.

.. code-block:: cpp

  struct CustomGeneratorImpl : public c10::GeneratorImpl {
    // Implementation of generator in new backend
  }

  at::Generator make_custom_generator(c10::DeviceIndex device_index) {
    return at::make_generator<CustomGeneratorImpl>(device_index);
  }

  REGISTER_GENERATOR_PRIVATEUSE1(make_cumstom_generator)

Register device guard for the new backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch provides functionalities related to device, stream, and event switching via ``DeviceGuard``.
This function is also applicable to ``PrivateUse1`` Key.

1. Inherit the ``DeviceGuardImplInterface`` class to implement the various general methods corresponding to the new backend.
2. Call ``C10_REGISTER_GUARD_IMPL`` macro to complete dynamic registration.

.. code-block:: cpp

  struct CustomGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    // Implementation of guard in new backend
  }

  C10_REGISTER_GUARD_IMPL(PrivateUse1, CustomGuardImpl);

Register serialization and deserialization functions for new backend metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch is currently able to dynamically register serialization/deserialization functions to support the serialization and deserialization
of new backend additional metadata named ``backend_meta_`` in class ``TensorImpl.ExtraMeta``. You can refer to the following steps:

1. Inherit the ``BackendMeta`` class to implement ``CustomBackendMetadata`` corresponding to the new backend and
   various fields of the new backend can be customized in the class.
2. Implement the serialization and deserialization functions of the new backend, the function signatures are 
   ``void(const at::Tensor&, std::unordered_map<std::string, bool>&)``.
3. Call the ``TensorBackendMetaRegistry`` macro to complete dynamic registration.

.. code-block:: cpp

  struct CustomBackendMetadata : public c10::BackendMeta {
    // Implementation of backend metadata in new backend
  }

  void for_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
    // Implementation of serialization
  }

  void for_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
    // Implementation of deserialization
  }

  TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1, &for_serialization, &for_deserialization);

Other Modules
^^^^^^^^^^^^^

In addition to the above-mentioned parts, there are some other modules that can be expanded through ``PrivateUse1``,
such as ``distributed collective communication``, ``benchmark timer``, and others, which will be added in the future.
One example about ``PrivateUse1`` integration is `Ascend NPU <https://github.com/ascend/pytorch>`_.


How to Improve User Experience with Privateuse1
-----------------------------------------------

The primary goal of integrating new devices through ``PrivateUse1`` is to meet the basic functional requirements,
and the next thing to do is to improve usability, which mainly involves the following aspects.

1. Register new backend module to Pytorch.
2. Generate methods and properties related to the new backend.
3. Generate methods and properties related to the new backend.

Register new backend module to Pytorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some CUDA-related interfaces in PyTorch can be called through the following form: ``torch.cuda.xxx``. Therefore, in order to
comply with user habits, the new backend implemented through the ``PrivateUse1`` mechanism should also provide similar interfaces.

For example, using ``Ascend NPU``:

.. code-block:: python

  torch._register_device_module('npu', torch_npu.npu)

After doing the above operations, users can call some exclusive APIs of ``Ascend NPU`` through ``torch.npu.xxx``

Rename PrivateUse1 to a custom name for the new backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PrivateUse1`` Key is the internal mechanism of the new backend integrated into PyTorch. For users, compared with ``PrivateUse1``,
the custom name strongly related to the new backend should be more friendly.

Taking the ``Ascend NPU`` as an example, the first usage will be more user-friendly.

.. code-block:: python

  torch.rand((2,2),device='npu:0')
  torch.rand((2,2),device='privateuse1:0')

Now, PyTorch provides a new C++/Python API for the self-named ``PrivateUse1`` backend, which is very simple to use.

.. tab-set-code::

  .. code-block:: python

      torch.rename_privateuse1_backend("npu")

  .. code-block:: C++

      c10::register_privateuse1_backend("npu")

Generate methods and properties related to the new backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After renaming ``PrivateUse1`` to a custome name, automatically generate properties and methods related to the new backend name
in the ``Tensor, nn, Storage`` modules for the new backend.

Here is an example for ``Ascend NPU``:

.. code-block:: python

  torch.rename_privateuse1_backend("npu")
  unsupported_dtype = [torch.quint8]
  torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True, unsupported_dtype=unsupported_dtype)

Then, you can use the following methods and properties:

.. code-block:: python

  torch.Tensor.npu()
  torch.Tensor.is_npu
  torch.Storage.npu()
  torch.Storage.is_npu
  ...

Future Work
-----------

The improvement of the ``PrivateUse1`` mechanism is still in progress, so the integration method of ``PrivateUse1``
of the new module will be added in turn. Here are a few items that we are actively working on:

* Add the integration method of ``distributed collective communication``.
* Add the integration method of ``benchmark timer``.

Conclusion
----------

This tutorial walked you through the process of integrating new backends into PyTorch via ``PrivateUse1``, including but not limited to
operator registration, generator registration, device guard registration, and so on. At the same time, some methods are introduced
to improve the user experience.
