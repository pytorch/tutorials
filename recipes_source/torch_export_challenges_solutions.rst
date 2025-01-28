Understanding the ``torch.export`` Flow and Solutions to Common Challenges
==========================================================================
**Authors:** `Ankith Gunapal <https://github.com/agunapal>`__, `Jordi Ramon <https://github.com/JordiFB>`__, `Marcos Carranza <https://github.com/macarran>`__

In the `Introduction to torch.export Tutorial <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`__ , we learned how to use `torch.export <https://pytorch.org/docs/stable/export.html>`__.
This tutorial expands on the previous one and explores the process of exporting popular models with code, as well as addresses common challenges that may arise with ``torch.export``.

In this tutorial, you will learn how to export models for these use cases:

* Video classifier (`MViT <https://pytorch.org/vision/main/models/video_mvit.html>`__)
* Automatic Speech Recognition (`OpenAI Whisper-Tiny <https://huggingface.co/openai/whisper-tiny>`__)
* Image Captioning (`BLIP <https://github.com/salesforce/BLIP>`__)
* Promptable Image Segmentation (`SAM2 <https://ai.meta.com/sam2/>`__)

Each of the four models were chosen to demonstrate unique features of `torch.export`, as well as some practical considerations
and issues faced in the implementation.

Prerequisites
-------------

* PyTorch 2.4 or later
* Basic understanding of ``torch.export`` and PyTorch Eager inference.


Key requirement for ``torch.export``: No graph break
----------------------------------------------------

`torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__ speeds up PyTorch code by using JIT to compile PyTorch code into optimized kernels. It optimizes the given model
using ``TorchDynamo`` and creates an optimized graph , which is then lowered into the hardware using the backend specified in the API.
When TorchDynamo encounters unsupported Python features, it breaks the computation graph, lets the default Python interpreter
handle the unsupported code, and then resumes capturing the graph. This break in the computation graph is called a `graph break <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#torchdynamo-and-fx-graphs>`__.

One of the key differences between ``torch.export`` and ``torch.compile`` is that ``torch.export`` doesn’t support graph breaks
which means that the entire model or part of the model that you are exporting needs to be a single graph. This is because handling graph breaks
involves interpreting the unsupported operation with default Python evaluation, which is incompatible with what ``torch.export`` is
designed for. You can read details about the differences between the various PyTorch frameworks in this `link <https://pytorch.org/docs/main/export.html#existing-frameworks>`__

You can identify graph breaks in your program by using the following command:

.. code:: sh

   TORCH_LOGS="graph_breaks" python <file_name>.py

You will need to modify your program to get rid of graph breaks. Once resolved, you are ready to export the model.
PyTorch runs `nightly benchmarks <https://hud.pytorch.org/benchmark/compilers>`__ for `torch.compile` on popular HuggingFace and TIMM models.
Most of these models have no graph breaks.

The models in this recipe have no graph breaks, but fail with `torch.export`.

Video Classification
--------------------

MViT is a class of models based on `MultiScale Vision Transformers <https://arxiv.org/abs/2104.11227>`__. This model has been trained for video classification using the `Kinetics-400 Dataset <https://arxiv.org/abs/1705.06950>`__.
This model with a relevant dataset can be used for action recognition in the context of gaming.


The code below exports MViT by tracing with ``batch_size=2`` and then checks if the ExportedProgram can run with ``batch_size=4``.

.. code:: python

   import numpy as np
   import torch
   from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b
   import traceback as tb

   model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

   # Create a batch of 2 videos, each with 16 frames of shape 224x224x3.
   input_frames = torch.randn(2,16, 224, 224, 3)
   # Transpose to get [1, 3, num_clips, height, width].
   input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

   # Export the model.
   exported_program = torch.export.export(
       model,
       (input_frames,),
   )

   # Create a batch of 4 videos, each with 16 frames of shape 224x224x3.
   input_frames = torch.randn(4,16, 224, 224, 3)
   input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
   try:
       exported_program.module()(input_frames)
   except Exception:
       tb.print_exc()


Error: Static batch size
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

       raise RuntimeError(
   RuntimeError: Expected input at *args[0].shape[0] to be equal to 2, but got 4


By default, the exporting flow will trace the program assuming that all input shapes are static, so if you run the program with
input shapes that are different than the ones you used while tracing, you will run into an error.

Solution
~~~~~~~~

To address the error, we specify the first dimension of the input (``batch_size``) to be dynamic , specifying the expected range of ``batch_size``.
In the corrected example shown below, we specify that the expected ``batch_size`` can range from 1 to 16.
One detail to notice that ``min=2``  is not a bug and is explained in `The 0/1 Specialization Problem <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk>`__. A detailed description of dynamic shapes
for ``torch.export`` can be found in the export tutorial. The code shown below demonstrates how to export mViT with dynamic batch sizes:

.. code:: python

   import numpy as np
   import torch
   from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b
   import traceback as tb


   model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

   # Create a batch of 2 videos, each with 16 frames of shape 224x224x3.
   input_frames = torch.randn(2,16, 224, 224, 3)

   # Transpose to get [1, 3, num_clips, height, width].
   input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

   # Export the model.
   batch_dim = torch.export.Dim("batch", min=2, max=16)
   exported_program = torch.export.export(
       model,
       (input_frames,),
       # Specify the first dimension of the input x as dynamic
       dynamic_shapes={"x": {0: batch_dim}},
   )

   # Create a batch of 4 videos, each with 16 frames of shape 224x224x3.
   input_frames = torch.randn(4,16, 224, 224, 3)
   input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
   try:
       exported_program.module()(input_frames)
   except Exception:
       tb.print_exc()


Automatic Speech Recognition
---------------

**Automatic Speech Recognition** (ASR) is the use of machine learning to transcribe spoken language into text.
`Whisper <https://arxiv.org/abs/2212.04356>`__ is a Transformer based encoder-decoder model from OpenAI, which was trained on 680k hours of labelled data for ASR and speech translation.
The code below tries to export ``whisper-tiny`` model for ASR.


.. code:: python

   import torch
   from transformers import WhisperProcessor, WhisperForConditionalGeneration
   from datasets import load_dataset

   # load model
   model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

   # dummy inputs for exporting the model
   input_features = torch.randn(1,80, 3000)
   attention_mask = torch.ones(1, 3000)
   decoder_input_ids = torch.tensor([[1, 1, 1 , 1]]) * model.config.decoder_start_token_id

   model.eval()

   exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(input_features, attention_mask, decoder_input_ids,))



Error: strict tracing with TorchDynamo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'DynamicCache' object has no attribute 'key_cache'


By default ``torch.export`` traces your code using `TorchDynamo <https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html>`__, a byte-code analysis engine,  which symbolically analyzes your code and builds a graph.
This analysis provides a stronger guarantee about safety but not all Python code is supported. When we export the ``whisper-tiny`` model  using the
default strict mode, it typically returns an error in Dynamo due to an unsupported feature. To understand why this errors in Dynamo, you can refer to this `GitHub issue <https://github.com/pytorch/pytorch/issues/144906>`__.

Solution
~~~~~~~~

To address the above error , ``torch.export`` supports  the ``non_strict`` mode where the program is traced using the Python interpreter, which works similar to
PyTorch eager execution. The only difference is that all ``Tensor`` objects will be replaced by ``ProxyTensors``, which will record all their operations into
a graph. By using ``strict=False``, we are able to export the program.

.. code:: python

   import torch
   from transformers import WhisperProcessor, WhisperForConditionalGeneration
   from datasets import load_dataset

   # load model
   model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

   # dummy inputs for exporting the model
   input_features = torch.randn(1,80, 3000)
   attention_mask = torch.ones(1, 3000)
   decoder_input_ids = torch.tensor([[1, 1, 1 , 1]]) * model.config.decoder_start_token_id

   model.eval()

   exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(input_features, attention_mask, decoder_input_ids,), strict=False)

Image Captioning
----------------

**Image Captioning** is the task of defining the contents of an image in words. In the context of gaming, Image Captioning can be used to enhance the
gameplay experience by dynamically generating text description of the various game objects in the scene, thereby providing the gamer with additional
details. `BLIP <https://arxiv.org/pdf/2201.12086>`__ is a popular model for Image Captioning `released by SalesForce Research <https://github.com/salesforce/BLIP>`__. The code below tries to export BLIP with ``batch_size=1``.


.. code:: python

   import torch
   from models.blip import blip_decoder

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   image_size = 384
   image = torch.randn(1, 3,384,384).to(device)
   caption_input = ""

   model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
   model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
   model.eval()
   model = model.to(device)

   exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(image,caption_input,), strict=False)



Error: Cannot mutate tensors with frozen storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While exporting a model, it might fail because the model implementation might contain certain Python operations which are not yet supported by ``torch.export``.
Some of these failures may have a workaround. BLIP is an example where the original model errors, which can be resolved by making a small change in the code.
``torch.export`` lists the common cases of supported and unsupported operations in `ExportDB <https://pytorch.org/docs/main/generated/exportdb/index.html>`__ and shows how you can modify your code to make it export compatible.

.. code:: console

   File "/BLIP/models/blip.py", line 112, in forward
       text.input_ids[:,0] = self.tokenizer.bos_token_id
     File "/anaconda3/envs/export/lib/python3.10/site-packages/torch/_subclasses/functional_tensor.py", line 545, in __torch_dispatch__
       outs_unwrapped = func._op_dk(
   RuntimeError: cannot mutate tensors with frozen storage



Solution
~~~~~~~~

Clone the `tensor <https://github.com/salesforce/BLIP/blob/main/models/blip.py#L112>`__ where export fails.

.. code:: python

   text.input_ids = text.input_ids.clone() # clone the tensor
   text.input_ids[:,0] = self.tokenizer.bos_token_id

.. note::
   This constraint has been relaxed in PyTorch 2.7 nightlies. This should work out-of-the-box in PyTorch 2.7

Promptable Image Segmentation
-----------------------------

**Image segmentation** is a computer vision technique that divides a digital image into distinct groups of pixels, or segments, based on their characteristics.
`Segment Anything Model (SAM) <https://ai.meta.com/blog/segment-anything-foundation-model-image-segmentation/>`__) introduced promptable image segmentation, which predicts object masks given prompts that indicate the desired object. `SAM 2 <https://ai.meta.com/sam2/>`__ is
the first unified model for segmenting objects across images and videos. The `SAM2ImagePredictor <https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py#L20>`__ class provides an easy interface to the model for prompting
the model. The model can take as input both point and box prompts, as well as masks from the previous iteration of prediction. Since SAM2 provides strong
zero-shot performance for object tracking, it can be used for tracking game objects in a scene.


The tensor operations in the predict method of `SAM2ImagePredictor <https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py#L20>`__  are happening in the `_predict <https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py#L291>`__ method. So, we try to export like this.

.. code:: python

   ep = torch.export.export(
       self._predict,
       args=(unnorm_coords, labels, unnorm_box, mask_input, multimask_output),
       kwargs={"return_logits": return_logits},
       strict=False,
   )


Error: Model is not of type ``torch.nn.Module``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch.export`` expects the module to be of type ``torch.nn.Module``. However, the module we are trying to export is a class method. Hence it errors.

.. code:: console

   Traceback (most recent call last):
     File "/sam2/image_predict.py", line 20, in <module>
       masks, scores, _ = predictor.predict(
     File "/sam2/sam2/sam2_image_predictor.py", line 312, in predict
       ep = torch.export.export(
     File "python3.10/site-packages/torch/export/__init__.py", line 359, in export
       raise ValueError(
   ValueError: Expected `mod` to be an instance of `torch.nn.Module`, got <class 'method'>.


Solution
~~~~~~~~

We write a helper class, which inherits from ``torch.nn.Module`` and call the ``_predict method`` in the ``forward`` method of the class. The complete code can be found `here <https://github.com/anijain2305/sam2/blob/ued/sam2/sam2_image_predictor.py#L293-L311>`__.

.. code:: python

   class ExportHelper(torch.nn.Module):
       def __init__(self):
           super().__init__()

       def forward(_, *args, **kwargs):
           return self._predict(*args, **kwargs)

    model_to_export = ExportHelper()
    ep = torch.export.export(
         model_to_export,
         args=(unnorm_coords, labels, unnorm_box, mask_input,  multimask_output),
         kwargs={"return_logits": return_logits},
         strict=False,
         )

Conclusion
----------

In this tutorial, we have learned how to use ``torch.export`` to export models for popular use cases by addressing challenges through correct configuration and simple code modifications.
Once you are able to export a model, you can lower the ``ExportedProgram`` into your hardware using `AOTInductor <https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html>`__ in case of servers and `ExecuTorch <https://pytorch.org/executorch/stable/index.html>`__ in case of edge device.
To learn more about ``AOTInductor`` (AOTI), please refer to the `AOTI tutorial <https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html>`__.
To learn more about ``ExecuTorch`` , please refer to the `ExecuTorch tutorial <https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html>`__.
