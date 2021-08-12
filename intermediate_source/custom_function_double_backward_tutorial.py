# -*- coding: utf-8 -*-
"""
Double Backward with Custom Functions
=====================================

It is sometimes useful to run backwards twice through backward graph, for
example to compute higher-order gradients. It takes an understanding of
autograd and some care to support double backwards, however. Functions
that support performing backward a single time are not necessarily
equipped to support double backward. In this tutorial we show how to
write a custom autograd function that supports double backward, and
point out some things to look out for.
"""
######################################################################
# When writing a custom autograd function to backward through twice,
# it is important to know when operations performed in a custom function
# are recorded by autograd, when they aren't, and most importantly, how
# `save_for_backward` works with all of this.

