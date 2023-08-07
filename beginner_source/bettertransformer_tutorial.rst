Fast Transformer Inference with Better Transformer
===============================================================

**Author**: `Michael Gschwind <https://github.com/mikekgfb>`__

This tutorial introduces Better Transformer (BT) as part of the PyTorch 1.12 release. 
In this tutorial, we show how to use Better Transformer for production 
inference with torchtext.  Better Transformer is a production ready fastpath to
accelerate deployment of Transformer models with high performance on CPU and GPU.
The fastpath feature works transparently for models based either directly on 
PyTorch core ``nn.module`` or with torchtext.  

Models which can be accelerated by Better Transformer fastpath execution are those
using the following PyTorch core ``torch.nn.module`` classes ``TransformerEncoder``, 
``TransformerEncoderLayer``, and ``MultiHeadAttention``.  In addition, torchtext has 
been updated to use the core library modules to benefit from fastpath acceleration.
(Additional modules may be enabled with fastpath execution in the future.)

Better Transformer offers two types of acceleration:

* Native multihead attention (MHA) implementation for CPU and GPU to improve overall execution efficiency.  
* Exploiting sparsity in NLP inference.  Because of variable input lengths, input
  tokens may contain a large number of padding tokens for which processing may be
  skipped, delivering significant speedups.

Fastpath execution is subject to some criteria. Most importantly, the model 
must be executed in inference mode and operate on input tensors that do not collect 
gradient tape information (e.g., running with torch.no_grad). 

To follow this example in Google Colab, `click here 
<https://colab.research.google.com/drive/1KZnMJYhYkOMYtNIX5S3AGIYnjyG0AojN?usp=sharing>`__.

Better Transformer Features in This Tutorial
--------------------------------------------

* Load pretrained models (created before PyTorch version 1.12 without Better Transformer)
* Run and benchmark inference on CPU with and without BT fastpath (native MHA only)
* Run and benchmark inference on (configurable) DEVICE with and without BT fastpath (native MHA only)
* Enable sparsity support
* Run and benchmark inference on (configurable) DEVICE with and without BT fastpath (native MHA + sparsity)

Additional Information
-----------------------
Additional information about Better Transformer may be found in the PyTorch.Org blog  
`A Better Transformer for Fast Transformer Inference
<https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference//>`__.



1. Setup

1.1 Load pretrained models

We download the XLM-R model from the predefined torchtext models by following the instructions in
`torchtext.models <https://pytorch.org/text/main/models.html>`__.  We also set the DEVICE to execute 
on-accelerator tests.  (Enable GPU execution for your environment as appropriate.)

.. code-block:: python 

    import torch
    import torch.nn as nn

    print(f"torch version: {torch.__version__}")

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"torch cuda available: {torch.cuda.is_available()}")

    import torch, torchtext
    from torchtext.models import RobertaClassificationHead
    from torchtext.functional import to_tensor
    xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
    classifier_head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim = 1024)
    model = xlmr_large.get_model(head=classifier_head)
    transform = xlmr_large.transform()

1.2 Dataset Setup

We set up two types of inputs: a small input batch and a big input batch with sparsity.

.. code-block:: python

    small_input_batch = [
                   "Hello world", 
                   "How are you!"
    ]
    big_input_batch = [
                   "Hello world", 
                   "How are you!", 
                   """`Well, Prince, so Genoa and Lucca are now just family estates of the
    Buonapartes. But I warn you, if you don't tell me that this means war,
    if you still try to defend the infamies and horrors perpetrated by
    that Antichrist- I really believe he is Antichrist- I will have
    nothing more to do with you and you are no longer my friend, no longer
    my 'faithful slave,' as you call yourself! But how do you do? I see
    I have frightened you- sit down and tell me all the news.`

    It was in July, 1805, and the speaker was the well-known Anna
    Pavlovna Scherer, maid of honor and favorite of the Empress Marya
    Fedorovna. With these words she greeted Prince Vasili Kuragin, a man
    of high rank and importance, who was the first to arrive at her
    reception. Anna Pavlovna had had a cough for some days. She was, as
    she said, suffering from la grippe; grippe being then a new word in
    St. Petersburg, used only by the elite."""
    ]

Next, we select either the small or large input batch, preprocess the inputs and test the model. 

.. code-block:: python

    input_batch=big_input_batch

    model_input = to_tensor(transform(input_batch), padding_value=1)
    output = model(model_input)
    output.shape

Finally, we set the benchmark iteration count:

.. code-block:: python

    ITERATIONS=10

2. Execution

2.1  Run and benchmark inference on CPU with and without BT fastpath (native MHA only)

We run the model on CPU, and collect profile information:  

* The first run uses traditional ("slow path") execution.
* The second run enables BT fastpath execution by putting the model in inference mode using `model.eval()` and disables gradient collection with `torch.no_grad()`.

You can see an improvement (whose magnitude will depend on the CPU model) when the model is executing on CPU.  Notice that the fastpath profile shows most of the execution time
in the native `TransformerEncoderLayer` implementation `aten::_transformer_encoder_layer_fwd`.

.. code-block:: python

    print("slow path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
      for i in range(ITERATIONS):  
        output = model(model_input)
    print(prof)

    model.eval()

    print("fast path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
      with torch.no_grad():
        for i in range(ITERATIONS):
          output = model(model_input)
    print(prof)


2.2  Run and benchmark inference on (configurable) DEVICE with and without BT fastpath (native MHA only)

We check the BT sparsity setting:

.. code-block:: python

    model.encoder.transformer.layers.enable_nested_tensor
    

We disable the BT sparsity:

.. code-block:: python

    model.encoder.transformer.layers.enable_nested_tensor=False    
    
 
We run the model on DEVICE, and collect profile information for native MHA execution on DEVICE:  

* The first run uses traditional ("slow path") execution.
* The second run enables BT fastpath execution by putting the model in inference mode using `model.eval()`
  and disables gradient collection with `torch.no_grad()`.

When executing on a GPU, you should see a significant speedup, in particular for the small input batch setting:

.. code-block:: python

    model.to(DEVICE)
    model_input = model_input.to(DEVICE)

    print("slow path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
      for i in range(ITERATIONS):  
        output = model(model_input)
    print(prof)

    model.eval()

    print("fast path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
      with torch.no_grad():
        for i in range(ITERATIONS):
          output = model(model_input)
    print(prof)
    

2.3 Run and benchmark inference on (configurable) DEVICE with and without BT fastpath (native MHA + sparsity)

We enable sparsity support:

.. code-block:: python

    model.encoder.transformer.layers.enable_nested_tensor = True

We run the model on DEVICE, and collect profile information for native MHA and sparsity support execution on DEVICE:  

* The first run uses traditional ("slow path") execution.
* The second run enables BT fastpath execution by putting the model in inference mode using `model.eval()` and disables gradient collection with `torch.no_grad()`.

When executing on a GPU, you should see a significant speedup, in particular for the large input batch setting which includes sparsity:

.. code-block:: python

    model.to(DEVICE)
    model_input = model_input.to(DEVICE)

    print("slow path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
      for i in range(ITERATIONS):  
        output = model(model_input)
    print(prof)

    model.eval()

    print("fast path:")
    print("==========")
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
      with torch.no_grad():
        for i in range(ITERATIONS):
          output = model(model_input)
    print(prof)


Summary
-------

In this tutorial, we have introduced fast transformer inference with 
Better Transformer fastpath execution in torchtext using PyTorch core 
Better Transformer support for Transformer Encoder models.  We have 
demonstrated the use of Better Transformer with models trained prior to 
the availability of BT fastpath execution.  We have demonstrated and 
benchmarked the use of both BT fastpath execution modes, native MHA execution
and BT sparsity acceleration. 


