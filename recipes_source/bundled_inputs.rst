(beta) Bundling inputs to PyTorch Models
==================================================================

**Author**: `Jacob Szwejbka <https://github.com/JacobSzwejbka>`_

Introduction
------------

This tutorial introduces the steps to use PyTorch's utility to bundle example or trivial inputs directly into your TorchScript Module.

The interface of the model remains unchanged (other than adding a few methods), so it can still be safely deployed to production. The advantage of this standardized interface is that tools that run models can use it instead of having some sort of external file (or worse, document) that tells you how to run the model properly.

Common case
-------------------

One of the common cases—bundling an input to a model that only uses 'forward' for inference.

1. **Prepare model**: Convert your model to TorchScript through either tracing or scripting

.. code:: python

    import torch
    import torch.jit
    import torch.utils
    import torch.utils.bundled_inputs

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.lin = nn.Linear(10, 1)

        def forward(self, x):
            return self.lin(x)

    model = Net()
    scripted_module = torch.jit.script(model)

2. **Create example input and attach to model**

.. code:: python

    # For each method create a list of inputs and each input is a tuple of arguments
    sample_input = [(torch.zeros(1,10),)]

    # Create model with bundled inputs, if type(input) is list then the input is bundled to 'forward'
    bundled_model = bundle_inputs(scripted_module, sample_input)


3. **Run model with input as arguments**

.. code:: python

    sample_inputs = bundled_model.get_all_bundled_inputs()

    print(bundled_model(*sample_inputs[0]))


Uncommon case
--------------

An uncommon case would be bundling and retrieving inputs for functions beyond 'forward'.

1. **Prepare model**: Convert your model to TorchScript through either tracing or scripting

.. code:: python

    import torch
    import torch.jit
    import torch.utils
    import torch.utils.bundled_inputs
    from typing import Dict

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.lin = nn.Linear(10, 1)

        def forward(self, x):
            return self.lin(x)

        @torch.jit.export
        def foo(self, x: Dict[String, int]):
            return x['a'] + x['b']


    model = Net()
    scripted_module = torch.jit.script(model)

2. **Create example input and attach to model**

.. code:: python

    # For each method create a list of inputs and each input is a tuple of arguments
    example_dict = {'a' : 1, 'b' : 2}
    sample_input = {
        scripted_module.forward : [(torch.zeros(1,10),)],
        scripted_module.foo : [(example_dict,)]
    }

    # Create model with bundled inputs, if type(sample_input) is Dict then each callable key is mapped to its corresponding bundled input
    bundled_model = bundle_inputs(scripted_module, sample_input)


3. **Retrieve inputs and run model on them**

.. code:: python

    all_info = bundled_model.get_bundled_inputs_functions_and_info()

    # The return type for get_bundled_inputs_functions_and_info is complex, but essentially we are retrieving the name
    # of a function we can use to get the bundled input for our models method
    for func_name in all_info.keys():
        input_func_name = all_info[func_name]['get_inputs_function_name'][0]
        func_to_run = getattr(bundled_model, input_func_name)
        # retrieve input
        sample_input = func_to_run()
        model_function = getattr(bundled_model, func_name)
        for i in range(len(sample_input)):
            print(model_function(*sample_input[i]))

Inflatable args
-------------------
Attaching inputs to models can result in nontrivial size increases. Inflatable args are a way to compress and decompress inputs to minimize this impact.

.. note:: Any automatic compression, or parsing of inflatable args only happens to top level arguments in the input tuple.

   - ie if your model takes in a List type of inputs you would need to create an inflatable arg that returned a list not create a list of inflatable args.

1. **Existing Inflatable args**

The following input types are compressed automatically without requiring an explicit inflatable arg:
    - Small contiguous tensors are cloned to have small storage.
    - Inputs from torch.zeros, torch.ones, or torch.full are moved to their compact representations.

.. code:: python

    # bundle_randn will generate a random tensor when the model is asked for bundled inputs
    sample_inputs = [(torch.utils.bundled_inputs.bundle_randn((1,10)),)]
    bundled_model = bundle_inputs(scripted_module, sample_inputs)
    print(bundled_model.get_all_bundled_inputs())

2. **Creating your own**

Inflatable args are composed of 2 parts, the deflated (compressed) argument, and an expression or function definition to inflate them.

.. code:: python

    def create_example(*size, dtype=None):
        """Generate a tuple of 2 random tensors both of the specified size"""

        deflated_input = (torch.zeros(1, dtype=dtype).expand(*size), torch.zeros(1, dtype=dtype).expand(*size))

        # {0} is how you access your deflated value in the inflation expression
        return torch.utils.bundled_inputs.InflatableArg(
            value=stub,
            fmt="(torch.randn_like({0}[0]), torch.randn_like({0}[1]))",
        )

3. **Using a function instead**
    If you need to create a more complicated input providing a function is an easy alternative

.. code:: python

        sample = dict(
            a=torch.zeros([10, 20]),
            b=torch.zeros([1, 1]),
            c=torch.zeros([10, 20]),
        )

        def condensed(t):
            ret = torch.empty_like(t).flatten()[0].clone().expand(t.shape)
            assert ret.storage().size() == 1
            return ret

        # An example of how to create an inflatable arg for a complex model input like Optional[Dict[str, Tensor]]
        # here we take in a normal input, deflate it, and define an inflater function that converts the mapped tensors to random values
        def bundle_optional_dict_of_randn(template: Optional[Dict[str, Tensor]]):
            return torch.utils.bundled_inputs.InflatableArg(
                value=(
                    None
                    if template is None
                    else {k: condensed(v) for (k, v) in template.items()}
                ),
                fmt="{}",
                fmt_fn="""
                def {}(self, value: Optional[Dict[str, Tensor]]):
                    if value is not None:
                        output = {{}}
                        for k, v in value.items():
                            output[k] = torch.randn_like(v)
                        return output
                    else:
                        return None
                """,
            )

        sample_inputs = (
            bundle_optional_dict_of_randn(sample),
        )


Learn More
----------
- To learn more about PyTorch Mobile, please refer to `PyTorch Mobile Home Page <https://pytorch.org/mobile/home/>`_
