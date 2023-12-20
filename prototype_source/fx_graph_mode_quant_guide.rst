(prototype) FX Graph Mode Quantization User Guide
===========================================================

**Author**: `Jerry Zhang <https://github.com/jerryzh168>`_

FX Graph Mode Quantization requires a symbolically traceable model.
We use the FX framework to convert a symbolically traceable nn.Module instance to IR,
and we operate on the IR to execute the quantization passes.
Please post your question about symbolically tracing your model in `PyTorch Discussion Forum <https://discuss.pytorch.org/c/quantization/17>`_

Quantization will only work on the symbolically traceable parts of your model.
The data dependent control flow-if statements / for loops, and so on using symbolically traced values-are one common pattern which is not supported.
If your model is not symbolically traceable end to end, you have a couple of options to enable FX Graph Mode Quantization only on a part of the model.
You can use any combination of these options:

1. Non traceable code doesn’t need to be quantized
    a. Symbolically trace only the code that needs to be quantized
    b. Skip symbolic tracing the non-traceable code

2. Non traceable code needs to be quantized
    a. Refactor your code to make it symbolically traceable
    b. Write your own observed and quantized submodule


If the code that is not symbolically traceable does not need to be quantized, we have the following two options
to run FX Graph Mode Quantization:


Symbolically trace only the code that needs to be quantized
-----------------------------------------------------------------
When the whole model is not symbolically traceable but the submodule we want to quantize is
symbolically traceable, we can run quantization only on that submodule.

before:

.. code:: python

  class M(nn.Module):
      def forward(self, x):
          x = non_traceable_code_1(x)
          x = traceable_code(x)
          x = non_traceable_code_2(x)
          return x

after:

.. code:: python
    
  class FP32Traceable(nn.Module):
      def forward(self, x):
          x = traceable_code(x)
          return x

  class M(nn.Module):
      def __init__(self):
          self.traceable_submodule = FP32Traceable(...)
      def forward(self, x):
          x = self.traceable_code_1(x)
          # We'll only symbolic trace/quantize this submodule
          x = self.traceable_submodule(x)
          x = self.traceable_code_2(x)
          return x

quantization code:

.. code:: python

  qconfig_mapping = QConfigMapping().set_global(qconfig)
  model_fp32.traceable_submodule = \
    prepare_fx(model_fp32.traceable_submodule, qconfig_mapping, example_inputs)

Note if original model needs to be preserved, you will have to
copy it yourself before calling the quantization APIs.


Skip symbolically trace the non-traceable code
---------------------------------------------------
When we have some non-traceable code in the module, and this part of code doesn’t need to be quantized,
we can factor out this part of the code into a submodule and skip symbolically trace that submodule.


before

.. code:: python

  class M(nn.Module):

      def forward(self, x):
          x = self.traceable_code_1(x)
          x = non_traceable_code(x)
          x = self.traceable_code_2(x)
          return x


after, non-traceable parts moved to a module and marked as a leaf

.. code:: python

  class FP32NonTraceable(nn.Module):

      def forward(self, x):
          x = non_traceable_code(x)
          return x

  class M(nn.Module):

      def __init__(self):
          ...
          self.non_traceable_submodule = FP32NonTraceable(...)

      def forward(self, x):
          x = self.traceable_code_1(x)
          # we will configure the quantization call to not trace through
          # this submodule
          x = self.non_traceable_submodule(x)
          x = self.traceable_code_2(x)
          return x

quantization code:

.. code:: python

  qconfig_mapping = QConfigMapping.set_global(qconfig)

  prepare_custom_config_dict = {
      # option 1
      "non_traceable_module_name": "non_traceable_submodule",
      # option 2
      "non_traceable_module_class": [MNonTraceable],
  }
  model_prepared = prepare_fx(
      model_fp32,
      qconfig_mapping,
      example_inputs,
      prepare_custom_config_dict=prepare_custom_config_dict,
  )

If the code that is not symbolically traceable needs to be quantized, we have the following two options:

Refactor your code to make it symbolically traceable
--------------------------------------------------------
If it is easy to refactor the code and make the code symbolically traceable,
we can refactor the code and remove the use of non-traceable constructs in python.

More information about symbolic tracing support can be found `here <https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing>`_.

before:

.. code:: python

  def transpose_for_scores(self, x):
      new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
      x = x.view(*new_x_shape)
      return x.permute(0, 2, 1, 3)


This is not symbolically traceable because in x.view(*new_x_shape)
unpacking is not supported, however, it is easy to remove the unpacking
since x.view also supports list input.


after:

.. code:: python

  def transpose_for_scores(self, x):
      new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
      x = x.view(new_x_shape)
      return x.permute(0, 2, 1, 3)


This can be combined with other approaches and the quantization code
depends on the model.

Write your own observed and quantized submodule
-----------------------------------------------------

If the non-traceable code can’t be refactored to be symbolically traceable,
for example it has some loops that can’t be eliminated, like nn.LSTM,
we’ll need to factor out the non-traceable code to a submodule (we call it CustomModule in fx graph mode quantization) and
define the observed and quantized version of the submodule (in post training static quantization or quantization aware training for static quantization)
or define the quantized version (in post training dynamic and weight only quantization)


before:

.. code:: python

  class M(nn.Module):

      def forward(self, x):
          x = traceable_code_1(x)
          x = non_traceable_code(x)
          x = traceable_code_1(x)
          return x

after:

1. Factor out non_traceable_code to FP32NonTraceable
non-traceable logic, wrapped in a module

.. code:: python

  class FP32NonTraceable:
      ...

2. Define observed version of
FP32NonTraceable

.. code:: python

  class ObservedNonTraceable:

      @classmethod
      def from_float(cls, ...):
          ...

3. Define statically quantized version of FP32NonTraceable
and a class method "from_observed" to convert from ObservedNonTraceable
to StaticQuantNonTraceable

.. code:: python

  class StaticQuantNonTraceable:

      @classmethod
      def from_observed(cls, ...):
          ...


.. code:: python

  # refactor parent class to call FP32NonTraceable
  class M(nn.Module):

     def __init__(self):
          ...
          self.non_traceable_submodule = FP32NonTraceable(...)

      def forward(self, x):
          x = self.traceable_code_1(x)
          # this part will be quantized manually
          x = self.non_traceable_submodule(x)
          x = self.traceable_code_1(x)
          return x


quantization code:


.. code:: python

  # post training static quantization or
  # quantization aware training (that produces a statically quantized module)v
  prepare_custom_config_dict = {
      "float_to_observed_custom_module_class": {
          "static": {
              FP32NonTraceable: ObservedNonTraceable,
          }
      },
  }

  model_prepared = prepare_fx(
      model_fp32,
      qconfig_mapping,
      example_inputs,
      prepare_custom_config_dict=prepare_custom_config_dict)

calibrate / train (not shown)

.. code:: python

  convert_custom_config_dict = {
      "observed_to_quantized_custom_module_class": {
          "static": {
              ObservedNonTraceable: StaticQuantNonTraceable,
          }
      },
  }
  model_quantized = convert_fx(
      model_prepared,
      convert_custom_config_dict)

post training dynamic/weight only quantization
in these two modes we don't need to observe the original model, so we
only need to define thee quantized model

.. code:: python

   class DynamicQuantNonTraceable: # or WeightOnlyQuantMNonTraceable
      ...
      @classmethod
      def from_observed(cls, ...):
          ...

      prepare_custom_config_dict = {
          "non_traceable_module_class": [
              FP32NonTraceable
          ]
      }


.. code:: python

  # The example is for post training quantization
  model_fp32.eval()
  model_prepared = prepare_fx(
      model_fp32,
      qconfig_mapping,
      example_inputs,
      prepare_custom_config_dict=prepare_custom_config_dict)

  convert_custom_config_dict = {
      "observed_to_quantized_custom_module_class": {
          "dynamic": {
              FP32NonTraceable: DynamicQuantNonTraceable,
          }
      },
  }
  model_quantized = convert_fx(
      model_prepared,
      convert_custom_config_dict)

You can also find examples for custom modules in test ``test_custom_module_class`` in ``torch/test/quantization/test_quantize_fx.py``.
