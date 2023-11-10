# -*- coding: utf-8 -*-
"""
`Introduction to ONNX <intro_onnx.html>`_ ||
**Exporting a PyTorch model to ONNX** ||
`Extending the ONNX Registry <onnx_registry_tutorial.html>`_

Export a PyTorch model to ONNX
==============================

**Author**: `Thiago Crepaldi <https://github.com/thiagocrepaldi>`_

.. note::
    As of PyTorch 2.1, there are two versions of ONNX Exporter.

    * ``torch.onnx.dynamo_export`` is the newest (still in beta) exporter based on the TorchDynamo technology released with PyTorch 2.0
    * ``torch.onnx.export`` is based on TorchScript backend and has been available since PyTorch 1.2.0

"""

###############################################################################
# In the `60 Minute Blitz <https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>`_,
# we had the opportunity to learn about PyTorch at a high level and train a small neural network to classify images.
# In this tutorial, we are going to expand this to describe how to convert a model defined in PyTorch into the
# ONNX format using TorchDynamo and the ``torch.onnx.dynamo_export`` ONNX exporter.
#
# While PyTorch is great for iterating on the development of models, the model can be deployed to production
# using different formats, including `ONNX <https://onnx.ai/>`_ (Open Neural Network Exchange)!
#
# ONNX is a flexible open standard format for representing machine learning models which standardized representations
# of machine learning allow them to be executed across a gamut of hardware platforms and runtime environments
# from large-scale cloud-based supercomputers to resource-constrained edge devices, such as your web browser and phone.
#
# In this tutorial, we’ll learn how to:
#
# 1. Install the required dependencies.
# 2. Author a simple image classifier model.
# 3. Export the model to ONNX format.
# 4. Save the ONNX model in a file.
# 5. Visualize the ONNX model graph using `Netron <https://github.com/lutzroeder/netron>`_.
# 6. Execute the ONNX model with `ONNX Runtime`
# 7. Compare the PyTorch results with the ones from the ONNX Runtime.
#
# 1. Install the required dependencies
# ------------------------------------
# Because the ONNX exporter uses ``onnx`` and ``onnxscript`` to translate PyTorch operators into ONNX operators,
# we will need to install them.
#
#  .. code-block:: bash
#
#   pip install onnx
#   pip install onnxscript
#
# 2. Author a simple image classifier model
# -----------------------------------------
#
# Once your environment is set up, let’s start modeling our image classifier with PyTorch,
# exactly like we did in the `60 Minute Blitz <https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>`_.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

######################################################################
# 3. Export the model to ONNX format
# ----------------------------------
#
# Now that we have our model defined, we need to instantiate it and create a random 32x32 input.
# Next, we can export the model to ONNX format.

torch_model = MyModel()
torch_input = torch.randn(1, 1, 32, 32)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)

######################################################################
# As we can see, we didn't need any code change to the model.
# The resulting ONNX model is stored within ``torch.onnx.ONNXProgram`` as a binary protobuf file.
#
# 4. Save the ONNX model in a file
# --------------------------------
#
# Although having the exported model loaded in memory is useful in many applications,
# we can save it to disk with the following code:

onnx_program.save("my_image_classifier.onnx")

######################################################################
# You can load the ONNX file back into memory and check if it is well formed with the following code:

import onnx
onnx_model = onnx.load("my_image_classifier.onnx")
onnx.checker.check_model(onnx_model)

######################################################################
# 5. Visualize the ONNX model graph using Netron
# ----------------------------------------------
#
# Now that we have our model saved in a file, we can visualize it with `Netron <https://github.com/lutzroeder/netron>`_.
# Netron can either be installed on macos, Linux or Windows computers, or run directly from the browser.
# Let's try the web version by opening the following link: https://netron.app/.
#
# .. image:: ../../_static/img/onnx/netron_web_ui.png
#   :width: 70%
#   :align: center
#
#
# Once Netron is open, we can drag and drop our ``my_image_classifier.onnx`` file into the browser or select it after
# clicking the **Open model** button.
#
# .. image:: ../../_static/img/onnx/image_clossifier_onnx_modelon_netron_web_ui.png
#   :width: 50%
#
#
# And that is it! We have successfully exported our PyTorch model to ONNX format and visualized it with Netron.
#
# 6. Execute the ONNX model with ONNX Runtime
# -------------------------------------------
#
# The last step is executing the ONNX model with `ONNX Runtime`, but before we do that, let's install ONNX Runtime.
#
#  .. code-block:: bash
#
#   pip install onnxruntime
#
# The ONNX standard does not support all the data structure and types that PyTorch does,
# so we need to adapt PyTorch input's to ONNX format before feeding it to ONNX Runtime.
# In our example, the input happens to be the same, but it might have more inputs
# than the original PyTorch model in more complex models.
#
# ONNX Runtime requires an additional step that involves converting all PyTorch tensors to Numpy (in CPU)
# and wrap them on a dictionary with keys being a string with the input name as key and the numpy tensor as the value.
#
# Now we can create an *ONNX Runtime Inference Session*, execute the ONNX model with the processed input
# and get the output. In this tutorial, ONNX Runtime is executed on CPU, but it could be executed on GPU as well.

import onnxruntime

onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
print(f"Input length: {len(onnx_input)}")
print(f"Sample input: {onnx_input}")

ort_session = onnxruntime.InferenceSession("./my_image_classifier.onnx", providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}

onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

####################################################################
# 7. Compare the PyTorch results with the ones from the ONNX Runtime
# ------------------------------------------------------------------
#
# The best way to determine whether the exported model is looking good is through numerical evaluation
# against PyTorch, which is our source of truth.
#
# For that, we need to execute the PyTorch model with the same input and compare the results with ONNX Runtime's.
# Before comparing the results, we need to convert the PyTorch's output to match ONNX's format.

torch_outputs = torch_model(torch_input)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")

######################################################################
# Conclusion
# ----------
#
# That is about it! We have successfully exported our PyTorch model to ONNX format,
# saved the model to disk, viewed it using Netron, executed it with ONNX Runtime
# and finally compared its numerical results with PyTorch's.
#
# Further reading
# ---------------
#
# The list below refers to tutorials that ranges from basic examples to advanced scenarios,
# not necessarily in the order they are listed.
# Feel free to jump directly to specific topics of your interest or
# sit tight and have fun going through all of them to learn all there is about the ONNX exporter.
#
# .. include:: /beginner_source/onnx/onnx_toc.txt
#
# .. toctree::
#    :hidden:
#