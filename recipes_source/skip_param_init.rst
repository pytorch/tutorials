Skipping Module Parameter Initialization
========================================

Prerequisites
-------------

PyTorch 1.9.0

Introduction
------------

When a module is created, its learnable parameters are initialized according
to a default initialization scheme associated with the module type. For example, the `weight`
parameter for a :class:`torch.nn.Linear` module is initialized from a
`uniform(-1/sqrt(in_features), 1/sqrt(in_features))` distribution. If some other initialization
scheme is desired, this has traditionally required re-initializing the parameters
after module instantiation:

::

    import torch

    # Initializes weight from the default distribution: uniform(-1/sqrt(10), 1/sqrt(10)).
    m = torch.nn.Linear(10, 5)

    # Re-initialize weight from a different distribution.
    torch.nn.init.orthogonal_(m.weight)

In this case, the initialization done during construction is wasted computation, and it may be non-trivial if
the `weight` parameter is large.

As of PyTorch 1.9.0, it is possible to skip initialization during construction, avoiding this wasted computation.

Skipping Initialization
-----------------------

The following pattern skips parameter initialization when creating a module instance:

::

    import torch

    # Initialize module on the meta device; all torch.nn.init ops have
    # no-op behavior on the meta device.
    m = torch.nn.Linear(10, 5, device='meta')

    # Materialize an uninitialized (empty) form of the module on the CPU device.
    # The result of this is a module instance with uninitialized parameters.
    m.to_empty(device='cpu')

It works by instantiating the module onto a "meta" device, which has tensor shape information
but does not allocate any storage. The `torch.nn.init` ops are specially implemented for this meta device
so that they have no-op behavior. This results in the parameter intialization logic being essentially skipped.

Note that this pattern only works for modules that support a `device` kwarg during construction.
While this is the case for all modules provided in `torch.nn`, it's likely that user-defined modules
have not been written in this way. The next section demonstrates how to update your module
to support skipping initialization through the addition of a `device` constructor kwarg.

Updating Modules to Support Skipping Initialization
---------------------------------------------------

You can opt in to the parameter initialization skipping functionality for your custom module
simply by adhering to the following two requirements:

  1. The module must accept a `device` kwarg in its constructor that is passed to any parameters
  or buffers created during construction.

  2. The module must not perform any computation on parameters or buffers in its constructor except
  initialization (i.e. functions from `torch.nn.init`).

The following example demonstrates a module updated to support the `device`
kwarg by passing it along to any created parameters, buffers, or submodules:

::

    import torch

    class MyModule(torch.nn.Module):
      def __init__(self, foo, bar, device=None):
        super().__init__()

        # ==== Case 1: Module creates parameters directly. ====
        # Pass device along to any created parameters.
        self.param1 = torch.nn.Parameter(torch.empty((foo, bar), device=device))
        self.register_parameter('param2', torch.nn.Parameter(torch.empty(bar, device=device)))

        # To ensure support for the meta device, avoid using ops except those in
        # torch.nn.init on parameters in your module's constructor.
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.param1)
            torch.nn.init.uniform_(self.param2)


        # ==== Case 2: Module creates submodules. ====
        # Pass device along recursively. All submodules will need to support
        # them as well; this is the case for all torch.nn provided modules.
        self.fc = torch.nn.Linear(bar, 5, device=device)

        # This also works with containers.
        self.linears = torch.nn.Sequential(
            torch.nn.Linear(5, 5, device=device),
            torch.nn.Linear(5, 1, device=device)
        )


        # ==== Case 3: Module creates buffers. ====
        # Pass device along during buffer tensor creation.
        self.register_buffer('some_buffer', torch.ones(7, device=device))

    ...

    m = MyModule(10, 5, device='meta')
