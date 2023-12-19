PyTorch 2 Export Quantization-Aware Training (QAT) with X86 Backend through Inductor
========================================================================================

**Author**: `Leslie Fang <https://github.com/leslie-fang-intel>`_, `Jiong Gong <https://github.com/jgong5>`_

Prerequisites
^^^^^^^^^^^^^^^

-  `PyTorch 2 Export Quantization-Aware Training tutorial <https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html>`_
-  `PyTorch 2 Export Post Training Quantization with X86 Backend through Inductor tutorial <https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_x86_inductor.html>`_
-  `TorchInductor and torch.compile concepts in PyTorch <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_


This tutorial demonstrates the process of performing PT2 export quantization-aware training (QAT) on X86 CPU
with X86InductorQuantizer, and subsequently lowering the quantized model into Inductor.
For more comprehensive details about PyTorch 2 Export Quantization-Aware Training in general, please refer to the
dedicated tutorial on `PyTorch 2 Export Quantization-Aware Training <https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html>`_.
For a deeper understanding of X86InductorQuantizer, please consult the tutorial of
`PyTorch 2 Export Post Training Quantization with X86 Backend through Inductor <https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_x86_inductor.html>`_.

The PyTorch 2 Export QAT flow looks like the following—it is similar
to the post training quantization (PTQ) flow for the most part:

.. code:: python

  import torch
  from torch._export import capture_pre_autograd_graph
  from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
  )
  import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
  from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

  class M(torch.nn.Module):
     def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1000)

     def forward(self, x):
        return self.linear(x)


  example_inputs = (torch.randn(1, 1024),)
  m = M()

  # Step 1. program capture
  # NOTE: this API will be updated to torch.export API in the future, but the captured
  # result shoud mostly stay the same
  exported_model = capture_pre_autograd_graph(m, example_inputs)
  # we get a model with aten ops

  # Step 2. quantization-aware training
  # Use Backend Quantizer for X86 CPU
  quantizer = X86InductorQuantizer()
  quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=True))
  prepared_model = prepare_qat_pt2e(exported_model, quantizer)

  # train omitted

  converted_model = convert_pt2e(prepared_model)
  # we have a model with aten ops doing integer computations when possible

  # move the quantized model to eval mode, equivalent to `m.eval()`
  torch.ao.quantization.move_exported_model_to_eval(converted_model)

  # Lower the model into Inductor
  with torch.no_grad():
    optimized_model = torch.compile(converted_model)
    _ = optimized_model(*example_inputs)

Please note that since the Inductor ``freeze`` feature does not turn on by default yet, need to run example code with ``TORCHINDUCTOR_FREEZING=1``.

For example:

::

    TORCHINDUCTOR_FREEZING=1 python example_x86inductorquantizer_qat.py
