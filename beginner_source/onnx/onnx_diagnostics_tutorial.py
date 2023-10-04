# -*- coding: utf-8 -*-

"""
`Introduction to ONNX <intro_onnx.html>`_ ||
`Export a PyTorch model to ONNX <export_simple_model_to_onnx_tutorial.html>`_ ||
`Introduction to ONNX Registry <onnx_registry_tutorial.html>`_ ||
**Introduction to ONNX Diagnostics**

Introduction to ONNX Diagnostics
================================

**Author:** Bowen Bao (bowbao@microsoft.com)
"""

###############################################################################
# Overview
# --------
#
# Welcome to this tutorial on ONNX Diagnostics. ONNX diagnostics goes beyond regular logs through the adoption of
# `Static Analysis Results Interchange Format (aka SARIF) <https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html>`__
# to help users debug and improve their model using a GUI, such as
# Visual Studio Code's `SARIF Viewer <https://marketplace.visualstudio.com/items?itemName=MS-SarifVSCode.sarif-viewer>`_,
# or `online SARIF Viewer <https://microsoft.github.io/sarif-web-component/>`_, etc.
# Each diagnostic rule is documented at `ONNX Diagnostic Rules <https://pytorch.org/docs/main/onnx_dynamo.html#diagnosing-issues-with-sarif>`_.
#
# Benefits of ONNX Diagnostics include:
#
# - Machine-parsable `Static Analysis Results Interchange Format (SARIF) <https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html>`__ output.
# - A clearer and structured approach for adding and tracking diagnostic rules.
# - Serving as a foundation for future enhancements leveraging the diagnostics.
#
# This tutorial will guide you on:
#
# 1. Setting up the SARIF Viewer.
# 2. Generating a SARIF diagnostic log from ONNX export.
# 3. Loading and exploring SARIF diagnostics in SARIF Viewer.
# 4. Increasing diagnostic verbosity for in-depth debugging.
#

######################################################################
# Setting up the SARIF Viewer
# ---------------------------
#
# Various tools can visualize SARIF files, including the`SARIF Viewer <https://marketplace.visualstudio.com/items?itemName=MS-SarifVSCode.sarif-viewer>`_
# extension for VSCode and the `online SARIF Viewer <https://microsoft.github.io/sarif-web-component/>`_.
# This tutorial will focus on the VSCode SARIF Viewer extension.
#
# To install:
#
# - Open VSCode.
# - Search for "SARIF Viewer" in the extension marketplace.
# - Install the extension.
#
# .. image:: ../_static/img/onnx/diagnostics_install_sarif_viewer.png
#

######################################################################
# Generating a SARIF diagnostic log from ONNX export
# --------------------------------------------------
#
# Let's begin by exporting a simple model:
#

import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x, y):
        return self.linear(x) + y


input_x = torch.randn(3, 3)
input_y = torch.randint(0, 10, (3, 3))
model = Model().eval()

export_output = torch.onnx.dynamo_export(model, input_x, input_y)
export_output.save("model.onnx")
export_output.save_diagnostics("log.sarif")

######################################################################
# The SARIF file is saved as "log.sarif" in the current directory.
#

######################################################################
# Loading and exploring SARIF diagnostics in SARIF Viewer
# -------------------------------------------------------
#
# SARIF logs (\*.sarif) can be loaded in various ways:
#
# - Simply open the file, and the SARIF Viewer extension will automatically process it.
# - Use the `sarif.showPanel` command and then select `Open SARIF log`.
# Let's open the SARIF file (log.sarif) we just generated.
#
# The SARIF Viewer provides comprehensive tools to navigate and understand the diagnostics.
# The top panel groups diagnostics by rules, offering a snapshot of the diagnostic rule name,
# severity, message, and its source code location.
#
# .. image:: ../_static/img/onnx/diagnostics_sarif_viewer_top_panel.png
#
# Filters can be adjusted from the top panel. For example, we can filter the diagnostics by severity level.
# .. image:: ../_static/img/onnx/diagnostics_sarif_viewer_filter.png
#
# By selecting a diagnostic, you'll access detailed information in the bottom panel.
# Here, you'll find the full diagnostic message and further insights into the diagnostic rule.
# .. image:: ../_static/img/onnx/diagnostics_sarif_viewer_bottom_panel.png
#
# The detailed information includes:
#
# - **Complete Diagnostic Message:** This includes comprehensive details specific to the code section
#   from which this diagnostic originated. Often, it will also show the function signature related to
#   the diagnostic occurrence. The next section will discuss how to capture even more information by
#   increasing the diagnostic verbosity.
# - **Rule Description:** This elaborates on the diagnostic rule, providing context about the identified
#   issue, insights into its origin, and potential solutions. For a deeper dive into each rule, refer to
#   `ONNX Diagnostic Rules <https://pytorch.org/docs/main/onnx_dynamo.html#diagnosing-issues-with-sarif>`_.
#

######################################################################
# Increasing diagnostic verbosity for in-depth debugging
# ------------------------------------------------------
#
# For a more detailed diagnostic output, adjust the `verbosity_level` in the `ExportOptions`.
# The default is set to `logging.INFO`. A higher verbosity level captures more diagnostic details.
#

import logging

export_output = torch.onnx.dynamo_export(
    model,
    input_x,
    input_y,
    export_options=torch.onnx.ExportOptions(
        diagnostic_options=torch.onnx.DiagnosticOptions(verbosity_level=logging.DEBUG)
    ),
)
export_output.save_diagnostics("debug.sarif")

######################################################################
# You can also adjust the verbosity level to `DEBUG` without modifying your code.
# Simply set the environment variable `TORCH_LOGS="onnx_diagnostics"`. For instance:
#
# .. code-block:: bash
#
#   TORCH_LOGS="onnx_diagnostics" python onnx_diagnostics_tutorial.py
#
# When you load the updated SARIF file into the viewer, you'll notice richer details.
# As an illustration, the `fx-pass` diagnostic for the `InsertTypePromotion` pass now
# shows a graph comparison between the original FX graph with its transformed version.
#
# .. image:: ../_static/img/onnx/diagnostics_sarif_viewer_fx_pass.png
#
# However, bear in mind that enhanced diagnostic details could have a performance impact
# on the ONNX export. Under `DEBUG` level, the SARIF log files will be bulkier, leading
# to a lengthier export time.
#

######################################################################
# Summary
# -------
#
# In this tutorial, we learned how to generate ONNX diagnostics and explore it to debug
# ONNX export issues. We also learned how to increase the verbosity of the diagnostics to get more details.
# For further in-depth reading, feel free to checkout the diagnostic docs which are located at `ONNX Diagnostic Rules <https://pytorch.org/docs/main/onnx_dynamo.html#diagnosing-issues-with-sarif>`_.
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
