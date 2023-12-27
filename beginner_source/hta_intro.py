# -*- coding: utf-8 -*-
"""
Introduction to Holistic Trace Analysis
------------
**Author:** `Anupam Bhatnagar <https://github.com/anupambhatnagar>`_

.. note::
    Visualizations have been set to False to keep the notebook size small. When
    running the notebook locally set the visualize variable to True to display
    the plots.

"""

##############################################################
# Setup and loading traces
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this demo we analyze the traces from a distributed training job which used 8 GPUs. To run the code on your laptop:
# 
# 1) Install Holistic Trace Analysis via pip. `pip install HolisticTraceAnalysis`
# 2) [Optional and recommended] Setup a conda environment. See here for details.
# 3) Edit the `hta_install_dir` vairable below to the folder in your local `HolisticTraceAnalysis` installation.

from hta.trace_analysis import TraceAnalysis
hta_install_dir = "/path/to/HolisticTraceAnalysis"
trace_dir = hta_install_dir + "/tests/data/vision_transformer/"
analyzer = TraceAnalysis(trace_dir=trace_dir)


##############################################################
# Temporal Breakdown
# ~~~~~~~~~~~~~~~~~~
# 
# The temporal breakdown feature gives a breakdown of time spent by the GPU as follows:
# 
# 1) Idle time - GPU idle
# 2) Compute time - GPU busy with computation events
# 3) Non compute time - GPU busy with communication or memory events

time_spent_df = analyzer.get_temporal_breakdown(visualize=False)
print(time_spent_df)


##############################################################
# Kernel Breakdown
# ~~~~~~~~~~~~~~~~
#
# This feature computes the following:
#
# 1) Breakdown of time spent among kernel types (Computation, Communication, Memory) across all ranks.
# 2) Kernels taking the most time on each rank by kernel type.
# 3) Distribution of average time across ranks for the kernels taking the most time.
