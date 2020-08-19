# -*- coding: utf-8 -*-
"""
Automatic Mixed Precision in PyTorch
*******************************************************
**Author**: `Michael Carilli <https://github.com/mcarilli>`_

``torch.cuda.amp`` provides convenience methods for mixed precision,
where some operations use the ``torch.float32`` (``float``) datatype and other operations
use ``torch.float16`` (``half``). Some ops, like linear layers and convolutions,
are much faster in ``float16``. Other ops, like reductions, often require the dynamic
range of ``float32``.  Mixed precision tries to match each op to its appropriate datatype.
which can reduce your network's runtime and memory footprint.

Ordinarily, "automatic mixed precision training" uses :class:`torch.cuda.amp.autocast` and
:class:`torch.cuda.amp.GradScaler` together.
Here we'll walk through adding ``autocast`` and ``GradScaler`` to a toy network.
First we'll cover typical use, then describe more advanced cases.

.. contents:: :local:
"""

######################################################################
# Without torch.cuda.amp, the following simple network executes all
# ops in default precision (torch.float32):

import torch

######################################################################
# Adding autocast
# ---------------
#


######################################################################
# Adding GradScaler
# -----------------
# 







######################################################################
# Advanced topics
# ---------------
#



#
# know by creating `an issue <https://github.com/pytorch/pytorch/issues>`_.
