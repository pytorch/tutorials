# -*- coding: utf-8 -*-

"""
Template Tutorial
=================

**Author:** `FirstName LastName <https://github.com/username>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * Item 1
      * Item 2
      * Item 3

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * PyTorch v2.0.0
      * GPU ???
      * Other items 3

If you have a video, add it here like this:

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/IC0_FRiX-sw" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

To test your tutorial locally, you can do one of the following:

*  You can control specific files that generate the results by using
   ``GALLERY_PATTERN`` environment variable. The GALLERY_PATTERN variable
   respects regular expressions.
   For example to run only ``neural_style_transfer_tutorial.py``,
   use the following command:

   .. code-block:: sh

      GALLERY_PATTERN="neural_style_transfer_tutorial.py" make html

   or

   .. code-block:: sh

      GALLERY_PATTERN="neural_style_transfer_tutorial.py" sphinx-build . _build

* Make a copy of this repository and add only your
  tutorial to the `beginner_source` directory removing all other tutorials.
  Then run ``make html``.
  
Verify that all outputs were generated correctly in the created HTML.
"""

#########################################################################
# Overview
# --------
#
# Describe Why is this topic important? Add Links to relevant research papers.
#
# This tutorial walks you through the process of....
#
# Steps
# -----
#
# Example code (the output below is generated automatically):
# 
import torch
x = torch.rand(5, 3)
print(x)

######################################################################
# (Optional) Additional Exercises
# -------------------------------
#
# Add additional practice exercises for users to test their knowledge.
# Example: `NLP from Scratch <https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#exercises>`__.
#

######################################################################
# Conclusion
# ----------
# 
# Summarize the steps and concepts covered. Highlight key takeaways.
#
# Further Reading
# ---------------
#
# * Link1
# * Link2

