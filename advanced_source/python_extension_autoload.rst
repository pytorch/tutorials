Out-of-tree extension autoloading in Python
===========================================

What is it?
-----------

The extension autoloading mechanism enables PyTorch to automatically
load out-of-tree backend extensions without explicit import statements. This
mechanism is very useful for users. On the one hand, it improves the user
experience and enables users to adhere to the familiar PyTorch device
programming model without needing to explicitly load or import device-specific
extensions. On the other hand, it facilitates effortless
adoption of existing PyTorch applications with zero-code changes on
out-of-tree devices. For more information,
see `[RFC] Autoload Device Extension <https://github.com/pytorch/pytorch/issues/122468>`_.

.. note::

    This feature is enabled by default and can be disabled using
    ``export TORCH_DEVICE_BACKEND_AUTOLOAD=0``.
    If you get an error like this: "Failed to load the backend extension",
    this error has nothing to do with PyTorch, you should disable this feature
    and ask the out-of-tree extension maintainer for help.

How to apply this mechanism to out-of-tree extensions?
------------------------------------------------------

For example, if you have a backend named ``foo`` and a package named
``torch_foo``. Make sure your package is based on PyTorch 2.5+ and includes
the following in its ``__init__.py``:

.. code-block:: python

    def _autoload():
        print("No need to import torch_foo anymore! Check things are working with `torch.foo.is_available()`.")

Then the only thing you need to do is add an entry point to your Python
package:

.. code-block:: python

    setup(
        name="torch_foo",
        version="1.0",
        entry_points={
            "torch.backends": [
                "torch_foo = torch_foo:_autoload",
            ],
        }
    )

Now the ``torch_foo`` module can be imported when running import torch:

.. code-block:: python

    >>> import torch
    No need to import torch_foo anymore! Check things are working with `torch.foo.is_available()`.
    >>> torch.foo.is_available()
    True

Examples
^^^^^^^^

Here we take Intel Gaudi HPU and Huawei Ascend NPU as examples to determine how to
integrate your out-of-tree extension with PyTorch based on the autoloading
mechanism.

`habana_frameworks.torch`_ is a Python package that enables users to run
PyTorch programs on Intel Gaudi via the PyTorch ``HPU`` device key.
``import habana_frameworks.torch`` is no longer necessary after this mechanism
is applied.

.. _habana_frameworks.torch: https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html

.. code-block:: diff

    import torch
    import torchvision.models as models
    - import habana_frameworks.torch # <-- extra import
    model = models.resnet50().eval().to("hpu")
    input = torch.rand(128, 3, 224, 224).to("hpu")
    output = model(input)

`torch_npu`_ enables users to run PyTorch programs on Huawei Ascend NPU, it
leverages the ``PrivateUse1`` device key and exposes the device name
as ``npu`` to the end users.

.. _torch_npu: https://github.com/Ascend/pytorch

Define an entry point in `torch_npu/setup.py`_:

.. _torch_npu/setup.py: https://github.com/Ascend/pytorch/blob/c164fbd5bb74790191ff8496b77d620fddf806d8/setup.py#L618

.. code-block:: diff

    setup(
        name="torch_npu",
        version="2.5",
    +   entry_points={
    +       'torch.backends': [
    +           'torch_npu = torch_npu:_autoload',
    +       ],
    +   }
    )

``import torch_npu`` is also no longer needed after applying this mechanism:

.. code-block:: diff

    import torch
    import torchvision.models as models
    - import torch_npu # <-- extra import
    model = models.resnet50().eval().to("npu")
    input = torch.rand(128, 3, 224, 224).to("npu")
    output = model(input)

How it works
------------

.. image:: ../_static/img/python_extension_autoload_impl.png
   :alt: Autoloading implementation
   :align: center

This mechanism is implemented based on Python's `Entry points
<https://packaging.python.org/en/latest/specifications/entry-points/>`_
mechanism. We discover and load all of the specific entry points
in ``torch/__init__.py`` that are defined by out-of-tree extensions.
Its implementation is in `[RFC] Add support for device extension autoloading
<https://github.com/pytorch/pytorch/pull/127074>`_.

Conclusion
----------

This tutorial has guided you through the out-of-tree extension autoloading
mechanism, including its usage and implementation.
