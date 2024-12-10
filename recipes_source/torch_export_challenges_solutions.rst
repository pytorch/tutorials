Demonstration of torch.export flow, common challenges and the solutions to address them
=======================================================================================
**Authors:** `Ankith Gunapal`, `Jordi Ramon`, `Marcos Carranza`

In a previous `tutorial <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`__ , we learnt how to use `torch.export <https://pytorch.org/docs/stable/export.html>`__.
This tutorial builds on the previous tutorial and explores the process of exporting popular models with code & addresses common challenges one might face with `torch.export`.

You will learn how to export models for these usecases

* Video classifier (MViT)
* Pose Estimation (Yolov11 Pose)
* Image Captioning (BLIP)
* Promptable Image Segmentation (SAM2)

Each of the four models were chosen to demonstrate unique features of `torch.export`, some practical considerations
& issues faced in the implementation.

Prerequisites
-------------

* PyTorch 2.4 or later
* Basic understanding of ``torch.export`` and PyTorch Eager inference.


Key requirement for `torch.export`: No graph break
------------------------------------------------

`torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__ speeds up PyTorch code by JIT compiling PyTorch code into optimized kernels. It optimizes the given model
using TorchDynamo and creates an optimized graph , which is then lowered into the hardware using the backend specified in the API.
When TorchDynamo encounters unsupported Python features, it breaks the computation graph, lets the default Python interpreter
handle the unsupported code, then resumes capturing the graph. This break in the computation graph is called a `graph break <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#torchdynamo-and-fx-graphs>`__.

One of the key differences between `torch.export` and `torch.compile` is that `torch.export` doesn’t support graph breaks
i.e the entire model or part of the model that you are exporting needs to be a single graph. This is because handling graph breaks
involves interpreting the unsupported operation with default Python evaluation, which is incompatible with what torch.export is
designed for.

You can identify graph breaks in your program by using the following

.. code:: console

   TORCH_LOGS="graph_breaks" python <file_name>.py

You will need to modify your program to get rid of graph breaks. Once resolved, you are ready to export the model.
PyTorch runs `nightly benchmarks <https://hud.pytorch.org/benchmark/compilers>`__ for `torch.compile` on popular HuggingFace and TIMM models.
Most of these models have no graph break.

The models in this recipe have no graph break, but fail with `torch.export`

Video Classification
--------------------

MViT is a class of models based on `MultiScale Vision Transformers <https://arxiv.org/abs/2104.11227>`__. This has been trained for video classification using the `Kinetics-400 Dataset <https://arxiv.org/abs/1705.06950>`__.
This model with a relevant dataset can be used for action recognition in the context of gaming.


The code below exports MViT by tracing with `batch_size=2` and then checks if the ExportedProgram can run with `batch_size=4`

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

.. code:: console

       raise RuntimeError(
   RuntimeError: Expected input at *args[0].shape[0] to be equal to 2, but got 4


By default, the exporting flow will trace the program assuming that all input shapes are static, so if you run the program with
inputs shapes that are different than the ones you used while tracing, you will run into an error.

Solution
~~~~~~~~

To address the error, we specify the first dimension of the input (`batch_size`) to be dynamic , specifying the expected range of `batch_size`.
In the corrected example shown below, we specify that the expected `batch_size` can range from 1 to 16.
One detail to notice that `min=2`  is not a bug and is explained in `The 0/1 Specialization Problem <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk>`__. A detailed description of dynamic shapes
for torch.export can be found in the export tutorial. The code shown below demonstrates how to export mViT with dynamic batch sizes.

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





Pose Estimation
---------------

Pose Estimation is a popular Computer Vision concept that can be used to identify the location of joints of a human in a 2D image.
`Ultralytics <https://docs.ultralytics.com/tasks/pose/>`__ has published a Pose Estimation model based on `YOLO11 <https://docs.ultralytics.com/models/yolo11/>`__. This has been trained on the `COCO Dataset <https://cocodataset.org/#keypoints-2017>`__. This model can be used
for analyzing human pose for determining action or intent. The code below tries to export the YOLO11 Pose model with `batch_size=1`


.. code:: python

   from ultralytics import YOLO
   import torch
   from torch.export import export

   pose_model = YOLO("yolo11n-pose.pt")  # Load model
   pose_model.model.eval()

   inputs = torch.rand((1,3,640,640))
   exported_program: torch.export.ExportedProgram= export(pose_model.model, args=(inputs,))


Error: strict tracing with TorchDynamo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   torch._dynamo.exc.InternalTorchDynamoError: PendingUnbackedSymbolNotFound: Pending unbacked symbols {zuf0} not in returned outputs FakeTensor(..., size=(6400, 1)) ((1, 1), 0).


By default `torch.export` traces your code using `TorchDynamo <https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html>`__, a byte-code analysis engine,  which symbolically analyzes your code and builds a graph.
This analysis provides a stronger guarantee about safety but not all python code is supported. When we export the `yolo11n-pose` model  using the
default strict mode, it errors.

Solution
~~~~~~~~

To address the above error `torch.export` supports non_strict mode where the program is traced using the python interpreter, which works similar to
PyTorch eager execution, the only difference is that all Tensor objects will be replaced by ProxyTensors, which will record all their operations into
a graph. By using `strict=False`, we are able to export the program.

.. code:: python

   from ultralytics import YOLO
   import torch
   from torch.export import export

   pose_model = YOLO("yolo11n-pose.pt")  # Load model
   pose_model.model.eval()

   inputs = torch.rand((1,3,640,640))
   exported_program: torch.export.ExportedProgram= export(pose_model.model, args=(inputs,), strict=False)



Image Captioning
----------------

Image Captioning is the task of defining the contents of an image in words. In the context of gaming, Image Captioning can be used to enhance the
gameplay experience by dynamically generating text description of the various game objects in the scene, thereby providing the gamer with additional
details. `BLIP <https://arxiv.org/pdf/2201.12086>`__ is a popular model for Image Captioning `released by SalesForce Research <https://github.com/salesforce/BLIP>`__. The code below tries to export BLIP with `batch_size=1`


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



Error: Unsupported python operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While exporting a model, it might fail because the model implementation might contain certain python operations which are not yet supported by `torch.export`.
Some of these failures may have a workaround. BLIP is an example where the original model errors and making a small change in the code resolves the issue.
`torch.export` lists the common cases of supported and unsupported operations in `ExportDB <https://pytorch.org/docs/main/generated/exportdb/index.html>`__ and shows how you can modify your code to make it export compatible.

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



Promptable Image Segmentation
-----------------------------

Image segmentation is a computer vision technique that divides a digital image into distinct groups of pixels, or segments, based on their characteristics.
Segment Anything Model(`SAM <https://ai.meta.com/blog/segment-anything-foundation-model-image-segmentation/>`__) introduced promptable image segmentation, which predicts object masks given prompts that indicate the desired object. `SAM 2 <https://ai.meta.com/sam2/>`__ is
the first unified model for segmenting objects across images and videos. The `SAM2ImagePredictor <https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py#L20>`__ class provides an easy interface to the model for prompting
the model. The model can take as input both point and box prompts, as well as masks from the previous iteration of prediction. Since SAM2 provides strong
zero-shot performance for object tracking, it can be used for tracking game objects in a scene. The code below tries to export SAM2ImagePredictor with batch_size=1


The tensor operations in the predict method of `SAM2ImagePredictor <https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py#L20>`__  are happening in the `_predict <https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py#L291>`__ method. So, we try to export this.

.. code:: python

   ep = torch.export.export(
       self._predict,
       args=(unnorm_coords, labels, unnorm_box, mask_input, multimask_output),
       kwargs={"return_logits": return_logits},
       strict=False,
   )


Error: Model is not of type `torch.nn.Module`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`torch.export` expects the module to be of type `torch.nn.Module`. However, the module we are trying to export is a class method. Hence it errors.

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

We write a helper class, which inherits from `torch.nn.Module` and call the `_predict method` in the `forward` method of the class. The complete code can be found `here <https://github.com/anijain2305/sam2/blob/ued/sam2/sam2_image_predictor.py#L293-L311>`__.

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

In this tutorial, we have learned how to use `torch.export` to export models for popular use cases by addressing challenges through correct configuration & simple code modifications.
