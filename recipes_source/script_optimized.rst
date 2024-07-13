Script and Optimize for Mobile Recipe
=====================================

This recipe demonstrates how to convert a PyTorch model to TorchScript which can run in a high-performance C++ environment such as iOS and Android, and how to optimize the converted TorchScript model for mobile deployment.

Introduction
------------

After a PyTorch model is trained and optionally but preferably quantized (see `Quantization Recipe <quantization.html>`_ for more details), one essential step before the model can be used in iOS and Android apps is to convert the Python-dependent model to TorchScript, which can then further be optimized for mobile apps. Conversion to TorchScript can be as simple as a single call, or as complicated as changing the original model in many different places.

Pre-requisites
--------------

PyTorch 1.6.0 or 1.7.0

Conversion to TorchScript
-------------------------

There are two basic ways to convert a PyTorch model to TorchScript, using `trace` and `script`. Mixing `trace` and `script` may also be needed in some cases - see `here <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#mixing-scripting-and-tracing>`_ for more information.

Use the `trace` Method
^^^^^^^^^^^^^^^^^^^^^^

To use the `trace` method on a model, an example or dummy input for the model needs to be specified, the actual input size needs to be the same as the example input size, and the model definition cannot have control flow such as `if` or `for`. The reason for these constraints is that running `trace` on a model with an example input simply calls the model's `forward` method with the input and all operations executed in the model layers are recorded, creating the trace of the model.

::

    import torch

    dummy_input = torch.rand(1, 3, 224, 224)
    torchscript_model = torch.jit.trace(model_quantized, dummy_input)


Use the `script` Method
^^^^^^^^^^^^^^^^^^^^^^^

For the example above, calling `script` below makes no difference:

::

    torchscript_model = torch.jit.script(model_quantized)

But if a model has some flow control, then `trace` won't correctly record all the possible traces. Take some code snippet of an example model definition from `here <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_ for example:

::

    import torch

    class MyDecisionGate(torch.nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x
            else:
                return -x

    x = torch.rand(3, 4)
    traced_cell = torch.jit.trace(MyDecisionGate(), x)
    print(traced_cell.code)

The code above will output:

::

    TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can''t record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!

    if x.sum() > 0:
  def forward(self,
      x: Tensor) -> Tensor:
    return x


Note that "the trace might not generalize to other inputs" warning above means that if the model has any kind of data-dependent control flow, `trace` is not the right answer. But if we replace the last two lines of the Python code snippet above (before the code output) with:

::

    scripted_cell = torch.jit.script(MyDecisionGate())
    print(scripted_cell.code)

The scripted model as shown by the `print` result below will be covering all possible inputs, thus generalizing to other inputs:

::

    def forward(self,
        x: Tensor) -> Tensor:
      _0 = bool(torch.gt(torch.sum(x, dtype=None), 0))
      if _0:
        _1 = x
      else:
        _1 = torch.neg(x)
      return _1


This is another example of using `trace` and `script` - it converts the model trained in the PyTorch tutorial `NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION <https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`_:

::

    encoder = EncoderRNN(input_lang.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)

    # method 1: using trace with example inputs

    encoder_input=torch.tensor([1])
    encoder_hidden=torch.zeros(1, 1, hidden_size)

    decoder_input1=torch.tensor([[0]])
    decoder_input2=torch.zeros(1, 1, hidden_size)
    decoder_input3=torch.zeros(MAX_LENGTH, hidden_size)

    traced_encoder = torch.jit.trace(encoder, (encoder_input, encoder_hidden))
    traced_decoder = torch.jit.trace(decoder, (decoder_input1, decoder_input2, decoder_input3))

    # method 2: using script

    scripted_encoder = torch.jit.script(encoder)
    scripted_decoder = torch.jit.script(decoder)

So is it true that one can simply always use the `script` call and the model is converted to TorchScript? The answer is no, because TorchScript is actually a subset of Python and to make `script` work, the PyTorch model definition must only use the language features of that TorchScript subset of Python. `TorchScript Language Reference <https://pytorch.org/docs/master/jit_language_reference.html#language-reference>`_ covers all the details of what is supported in TorchScript. Below we will describe some of the common errors when using the `script` method.


Fix Common Errors When Using the `script` Method
----------------------------------------------------

If you apply the `script` method to a non-trivial model, chances are you may encounter several types of errors. Check out `this tutorial <https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html>`_ for a complete example of converting a chatbot model to TorchScript. But follow the steps below to fix common errors when you run the `script` method:

1. RuntimeError `attribute lookup is not defined on python value of type`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this error, pass the value of the model as a parameter in the constructor. This is because when calling `script` on a model that accepts another model as a parameter, the model passed is actually of type `TracedModule` or `ScriptModule`, not of type `Module`, making the the model attribute not defined when scripting.

For example, the `LuongAttnDecoderRNN` module in the tutorial above has an attribute `n_layers`, and the `GreedySearchDecoder` module refers to the `n_layers` attribute of a `decoder` instance of the `LuongAttnDecoderRNN` module, so in order to make `script` work, the `GreedySearchDecoder` module's constructor needs to be changed from:

::

    def __init__(self, encoder, decoder):

to:

::

    def __init__(self, encoder, decoder, decoder_n_layers):
      ...
      self._decoder_n_layers = decoder_n_layers


and the `GreedySearchDecoder`'s `forward` method needs to refer `self._decoder_n_layers` instead of `decoder.n_layers`.

2. RuntimeError `python value of type '...' cannot be used as a value.`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The complete error message for this one continues with `Perhaps it is a closed over global variable? If so, please consider passing it in as an argument or use a local variable instead.`, store global variables' values as attributes in the model constructor (there's no need to add them to a special list called `__constants__`). The reason is that global values can be used conveniently in normal model training and inference, but the global values are not accessible during the scripting.

For example, `device` and `SOS_token` are global variables, and to make `script` work, they need to be added to the `GreedySearchDecoder`'s constructor:

::

    self._device = device
    self._SOS_token = SOS_token

and referred to as `self._device` and `self._SOS_token` instead of `device` and `SOS_token` in the `GreedySearchDecoder`'s `forward` method.

3. RuntimeError `all inputs of range must be '...', found Tensor (inferred) in argument`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The error message continues with: `add type definitions for each of the module's forward method arguments. Because all parameters to a TorchScript function are of the `torch.Tensor` type by default, you need to specifically declare the type for each parameter that is not of type 'Tensor'. For a complete list of TorchScript-supported types, see `here <https://pytorch.org/docs/master/jit_language_reference.html#supported-type>`_.

For example, the `GreedySearchDecoder`'s `forward` method signature needs to be changed from:

::

    def forward(self, input_seq, input_length, max_length):

to:

::

    def forward(self, input_seq, input_length, max_length : int):

After using the `trace` or `script` method above, and fixing possible errors, you should have a TorchScript model ready to be optimized for mobile.


Optimize a TorchScript Model
--------------------------------------

Simply run the following code snippet to optimize a TorchScript model generated with the `trace` and/or `script` method:

::

    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_torchscript_model = optimize_for_mobile(torchscript_model)

The optimized model can then be saved and deployed in mobile apps:

::

    optimized_torchscript_model.save("optimized_torchscript_model.pth")

By default, for the CPU backend, `optimize_for_mobile` performs the following types of optimizations:

* `Conv2D and BatchNorm fusion` which folds Conv2d-BatchNorm2d into Conv2d;

* `Insert and fold prepacked ops` which rewrites the model graph to replace 2D convolutions and linear ops with their prepacked counterparts.

* `ReLU and hardtanh fusion` which rewrites graph by finding ReLU/hardtanh ops and fuses them together.

* `Dropout removal` which removes dropout nodes from this module when training is false.

* `Conv packed params hoisting` which moves convolution packed params to the root module, so that the convolution structs can be deleted. This decreases model size without impacting numerics.

For the Vulkan backend,`optimize_for_mobile` performs the following type of optimization:

* `Automatic GPU transfer` which rewrites the graph so that moving input and output data to and from the GPU becomes part of the model.

Optimization types can be disabled by passing an optimization blocklist as an argument to `optimize_for_mobile`.

Learn More
-----------------
1. The official `TorchScript Language Reference <https://pytorch.org/docs/stable/jit_language_reference.html>`_.
2. The `torch.utils.mobile_optimizer` `API documentation <https://pytorch.org/docs/stable/mobile_optimizer.html>`_.
