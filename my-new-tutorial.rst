Fast Transformer Inference with Better Transformer
===============================================================

**Author**: `Michael Gschwind <https://github.com/mikekgfb>`__

This tutorial introduces Better Transformer (BT) as part of the PyTorch 1.12 release. 
In this tutorial, we show how to use Better Transformer for production 
inference with torchtext.  Better Transformer is a production ready fastpath to
accelerate deployment of Transformer models with high performance on CPU and GPU.
The fastpath feature works transparently for models based either directly on 
PyTorch core nn.module or with torchtext.  

Models which can be accelerated by Better Transformer ``fastpath`` execution are those
using the following PyTorch core `torch.nn.module` classes `TransformerEncoder`, 
`TransformerEncoderLayer`, and `MultiHeadAttention`.  In addition, torchtext has 
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
