"""
TODO: Add Recipe Title
=======================

TODO: 
      * Include 1-2 sentences summing up what the user can expect from the recipe.
      * For example - “This samples demonstrates how to...”

Introduction
--------------
TODO: 
      * Add why is this topic important?
      * Ex: Provide a summary of how Integrated Gradients works and how you will teach users to implement it using Captum in this tutorial

Setup
----------------------
TODO: 
      * Call out any required setup or data downloads


TODO: List Steps
-----------------
TODO: 
      * Use the steps you introduced in the Learning Objectives
      * Break down the steps as well as add prose for context
      * Add comments in the code to help clarify for readers what each section is doing
      * Link back to relevant pytorch documentation
      * Think of it akin to creating a really practical Medium post

TIPS: 
      * To denote a word or phrase as code, enclose it in double backticks (``). ``torch.Tensor``
      * You can **bold** or *italicize* text for emphasis. 
      * Add python code directly in the file. The output will render and build on the site in a separate code block. 
        Below is an example of python code with comments.  
        You can build this python file to see the resulting html by following the README.md at github.com/pytorch/tutorials
"""

import torch

###############################################################
# Because of the line of pound sign delimiters above,  this comment will show up as plain text between the code.
x = torch.ones(2, 2, requires_grad=True)
# Since this is a single line comment, it will show up as a comment in the code block
print(x)



###############################################################
# .. Note::
#
#       You can add Notes using this syntax




########################################################################
# Learn More
# ----------------------------
# TODO:
#      * Link to any additional resources (e.g. Docs, other Tutorials, external resources) if readers want to learn more
#      * There are different ways add hyperlinks - 
#      * For example, pasting the url works:  Read more about the ``autograd.Function`` at https://pytorch.org/docs/stable/autograd.html#function. 
#      * or link to other files in this repository by their titles such as :doc:`data_parallel_tutorial`.
#      * There are also ways to add internal and external links. Check out this resource for more tips: https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#id4
#
