# -*- coding: utf-8 -*-
"""
.. _cuda-graph-annotations-tutorial:

CUDA Graph Kernel Annotations and Profiling
============================================

**Author**: `PyTorch Team <https://github.com/pytorch>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to capture CUDA graphs with kernel annotations
       * How to profile annotated graphs
       * How to post-process traces with semantic kernel lanes
       * How to visualize graph execution with custom stream assignments

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch 2.0 or later
       * CUDA-capable GPU
       * Driver/CUDA-compat >= 13.1 for full annotation support
       * cuda-python package

CUDA graphs are a powerful optimization technique that can significantly reduce
kernel launch overhead by capturing and replaying sequences of CUDA operations.
However, when profiling CUDA graphs, all kernels appear on the same stream,
making it difficult to understand the logical structure of your computation.

This tutorial demonstrates how to use **kernel annotations** to add semantic
labels to kernels within CUDA graphs. These annotations can be merged back into
profiler traces to create custom visualization lanes, making it easier to
understand and debug complex graph executions.
"""

###############################################################################
# Overview
# --------
#
# When you capture operations into a CUDA graph, the profiler shows all
# kernels executing on a single stream. This makes it hard to distinguish
# between different logical components of your model (e.g., attention vs MLP).
#
# Kernel annotations solve this by:
#
# 1. **Marking** kernels during graph capture with semantic labels
# 2. **Profiling** the graph replay to collect execution traces
# 3. **Post-processing** traces to merge annotations and create custom lanes
#
# The result is a trace where kernels are organized into meaningful groups,
# making it much easier to identify performance bottlenecks.
#
# .. image:: /_static/img/cuda_graph_annotations_before_after.png
#
# Requirements
# ------------
#
# For this tutorial, you'll need:
#
# - PyTorch 2.13+
# - A CUDA GPU
# - Driver/CUDA-compat >= 13.1 for annotation support
# - The ``cuda-bindings`` package >= 13.3.0 (``pip install cuda-python``)
#
# The cuda-bindings package provides the Python bindings for CUDA runtime APIs.
# Version 13.3.0+ is required for the ``cudaGraphNodeGetToolsId`` API that
# enables kernel annotations. If you have an older version, the tutorial will
# run but annotations will be disabled with a warning message explaining how
# to upgrade.
#
# On older drivers or cuda-bindings versions, the capture and profiling will
# still work, but ``mark_kernels`` will be a no-op and no semantic lanes will
# appear in the final trace.

import copy
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity
from torch.cuda._graph_annotations import (
    get_kernel_annotations,
    mark_kernels,
    _is_tools_id_unavailable,
)
from torch.cuda._annotate_cuda_graph_trace import (
    annotate_trace,
    load_trace,
    save_trace,
    _fix_overlapping_timestamps,
    _move_overlapping_to_stream,
)

###############################################################################
# Building a Model
# ----------------
#
# Let's create a simple transformer block as our example model. We'll annotate
# different parts of the computation (QKV projection, attention, output
# projection, MLP) to see them as separate lanes in the profiler.

def build_transformer_block():
    """Create a simple transformer block with parameters."""
    device = "cuda"
    torch.manual_seed(0)

    # Model dimensions
    batch_size, seq_len, dim, num_heads = 4, 256, 1024, 8
    head_dim = dim // num_heads

    # Initialize parameters
    params = {
        "x": torch.randn(batch_size, seq_len, dim, device=device),
        "Wqkv": torch.randn(dim, 3 * dim, device=device) / math.sqrt(dim),
        "Wo": torch.randn(dim, dim, device=device) / math.sqrt(dim),
        "W1": torch.randn(dim, 4 * dim, device=device) / math.sqrt(dim),
        "W2": torch.randn(4 * dim, dim, device=device) / math.sqrt(4 * dim),
    }

    def forward():
        """Forward pass with annotated regions."""
        B, T, D, H = batch_size, seq_len, dim, num_heads
        hd = head_dim

        # Annotate QKV projection
        with mark_kernels({"name": "qkv_proj"}):
            qkv = params["x"] @ params["Wqkv"]

        # Reshape for multi-head attention
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, H, hd).transpose(1, 2)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)

        # Annotate attention computation (optionally on a custom stream)
        with mark_kernels({"name": "attention", "stream": 62}):
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
            attn = torch.softmax(scores, dim=-1)
            ctx = (attn @ v).transpose(1, 2).reshape(B, T, D)

        # Annotate output projection
        with mark_kernels({"name": "out_proj"}):
            o = ctx @ params["Wo"]

        # Annotate MLP (on another custom stream)
        with mark_kernels({"name": "mlp", "stream": 61}):
            return torch.nn.functional.gelu(o @ params["W1"]) @ params["W2"]

    return forward

###############################################################################
# The ``mark_kernels`` Context Manager
# -------------------------------------
#
# The key API is ``mark_kernels()``, which takes a dictionary with:
#
# - ``name``: A string label for this kernel group (becomes the lane name)
# - ``stream`` (optional): A virtual stream ID for visualization
#
# Any CUDA kernels launched within the context will be tagged with these
# annotations. Later, when we post-process the profiler trace, these tags
# will be used to organize kernels into custom lanes.

###############################################################################
# Capturing a CUDA Graph with Annotations
# ----------------------------------------
#
# To capture a graph with annotations enabled, we pass
# ``enable_annotations=True`` to ``torch.cuda.graph()``. This automatically
# handles the annotation lifecycle: enabling, resolving, and remapping.

def capture_graph_with_annotations(model_fn):
    """Capture the model into a CUDA graph with annotations enabled."""
    # Warm up on a side stream before capture
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            model_fn()

    torch.cuda.current_stream().wait_stream(warmup_stream)

    # Capture with annotations enabled
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, enable_annotations=True):
        output = model_fn()

    num_annotations = len(get_kernel_annotations())
    print(f"Captured graph with {num_annotations} annotated nodes")

    return graph, output

###############################################################################
# Profiling the Graph
# -------------------
#
# After capturing the graph, we replay it a few times to warm up, then profile
# subsequent replays. The profiler will record kernel execution times, which
# we'll later merge with our annotations.

def profile_graph(graph, output_dir):
    """Profile graph replays and save the trace."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Warm up replays
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    # Profile several replays
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()

    # Export the raw trace
    trace_path = output_dir / "trace_raw.json.gz"
    prof.export_chrome_trace(str(trace_path))
    print(f"Saved raw trace to {trace_path}")

    return trace_path

###############################################################################
# Saving Annotation Metadata
# ---------------------------
#
# We need to save the annotation metadata in a pickle file that the
# post-processing tool can discover. The file should be named
# ``kernel_annotations_rank0_fwd_bwd.pkl`` and placed where the trace tool
# can find it.

def save_annotations(output_dir):
    """Save kernel annotations to a pickle file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    annotations_path = output_dir / "kernel_annotations_rank0_fwd_bwd.pkl"

    annotations = dict(get_kernel_annotations())
    with open(annotations_path, "wb") as f:
        pickle.dump(annotations, f)

    print(f"Saved {len(annotations)} annotations to {annotations_path}")
    return annotations_path

###############################################################################
# Post-Processing: Merging Annotations into Traces
# -------------------------------------------------
#
# The final step is to merge the annotations back into the trace. This involves:
#
# 1. Loading the raw trace and annotations
# 2. Calling ``annotate_trace()`` to apply the annotations
# 3. Running cleanup passes to handle overlapping kernels
# 4. Saving the annotated trace
#
# The result is a trace where kernels are organized by your semantic labels.

def post_process_trace(raw_trace_path, annotations_path, output_dir):
    """Merge annotations into the trace and apply cleanup."""
    output_dir = Path(output_dir)

    # Load raw trace and annotations
    raw_trace = load_trace(raw_trace_path)
    with open(annotations_path, "rb") as f:
        annotations = pickle.load(f)

    # Make a copy for post-processing
    annotated_trace = copy.deepcopy(raw_trace)

    # Apply annotations
    num_annotated = annotate_trace(annotated_trace, annotations)
    print(f"Annotated {num_annotated} kernels in the trace")

    # Cleanup passes: move overlapping kernels and fix timestamps
    _move_overlapping_to_stream(annotated_trace)
    _fix_overlapping_timestamps(annotated_trace)

    # Save the annotated trace
    annotated_path = output_dir / "trace_annotated.json.gz"
    save_trace(annotated_trace, annotated_path)
    print(f"Saved annotated trace to {annotated_path}")

    return annotated_path, raw_trace, annotated_trace

###############################################################################
# Comparing Before and After
# ---------------------------
#
# To see the impact of annotations, let's count how kernels are distributed
# across thread IDs (which represent visualization lanes in the trace).

def compare_traces(raw_trace, annotated_trace):
    """Compare kernel distribution before and after annotation."""
    def count_lanes(trace):
        """Count kernels per lane (tid)."""
        counter = Counter(
            event["tid"]
            for event in trace["traceEvents"]
            if event.get("cat") == "kernel"
        )
        return dict(sorted(counter.items()))

    raw_lanes = count_lanes(raw_trace)
    annotated_lanes = count_lanes(annotated_trace)

    print("\n" + "="*60)
    print("BEFORE annotation - kernels per lane (tid -> count):")
    for tid, count in raw_lanes.items():
        print(f"  Stream {tid}: {count} kernels")

    print("\nAFTER annotation - kernels per lane (tid -> count):")
    for tid, count in annotated_lanes.items():
        print(f"  Stream {tid}: {count} kernels")
    print("="*60)

###############################################################################
# Putting It All Together
# ------------------------
#
# Now let's run the complete workflow: build a model, capture it with
# annotations, profile it, and post-process the trace.

def main():
    """End-to-end CUDA graph annotation and profiling demo."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this tutorial")

    # Check if annotation support is available
    # PyTorch will log a warning if cuda-bindings version is too old
    supported = not _is_tools_id_unavailable()
    print(f"Annotation support available: {supported}")
    if not supported:
        print("NOTE: Annotation API not available.")
        print("This could be due to:")
        print("  - Driver/CUDA-compat < 13.1")
        print("  - Outdated cuda-bindings (check PyTorch warnings above)")
        print("Annotations will not be recorded, but the demo will still run.")
        print("Any lane changes you see are from cleanup passes, not annotations.\n")

    output_dir = Path("traces")

    # Build the model
    print("\n1. Building transformer block model...")
    model_fn = build_transformer_block()

    # Capture graph with annotations
    print("\n2. Capturing CUDA graph with annotations...")
    graph, output = capture_graph_with_annotations(model_fn)

    # Save annotations
    print("\n3. Saving annotation metadata...")
    annotations_path = save_annotations(output_dir)

    # Profile the graph
    print("\n4. Profiling graph replays...")
    raw_trace_path = profile_graph(graph, output_dir)

    # Post-process the trace
    print("\n5. Post-processing: merging annotations into trace...")
    annotated_path, raw_trace, annotated_trace = post_process_trace(
        raw_trace_path, annotations_path, output_dir
    )

    # Compare before and after
    print("\n6. Comparing traces...")
    compare_traces(raw_trace, annotated_trace)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Raw trace:       {raw_trace_path}")
    print(f"Annotated trace: {annotated_path}")
    print(f"Annotations:     {annotations_path}")
    print("\nOpen the annotated trace in chrome://tracing to visualize")
    print("the semantic kernel lanes.")
    print("="*60)

###############################################################################
# Visualizing Results
# -------------------
#
# To view the annotated trace:
#
# 1. Open Chrome/Chromium browser
# 2. Navigate to ``chrome://tracing``
# 3. Click "Load" and select the ``trace_annotated.json.gz`` file
# 4. You should see kernels organized into custom lanes like "qkv_proj",
#    "attention", "out_proj", and "mlp"
#
# The custom stream IDs (61, 62) specified in ``mark_kernels`` appear as
# separate lanes, making it easy to see which operations run concurrently
# or sequentially.

###############################################################################
# Advanced Usage: Multiple Graphs
# --------------------------------
#
# For more complex applications with multiple graphs or distributed training:
#
# - Use different annotation names for different components
# - The pickle filename can include rank information for multi-GPU setups
# - Apply the same post-processing workflow to each trace
# - Annotations persist across graph replays, so you capture once and can
#   profile many times

###############################################################################
# Understanding the Cleanup Passes
# ---------------------------------
#
# The post-processing applies two cleanup functions:
#
# 1. ``_move_overlapping_to_stream()``: If kernels on the same lane overlap
#    in time, move one to a different lane. This prevents visual overlap in
#    the trace viewer.
#
# 2. ``_fix_overlapping_timestamps()``: Adjust timestamps slightly if
#    overlapping kernels would cause confusion. This is a last resort to
#    ensure the trace renders correctly.
#
# These passes ensure that the trace is both accurate and readable, even
# when the original execution has complex concurrency patterns.

###############################################################################
# Performance Considerations
# ---------------------------
#
# Kernel annotations add minimal overhead:
#
# - Annotation marking happens during graph capture (one-time cost)
# - Graph replay performance is identical to unannotated graphs
# - Post-processing is offline and doesn't affect runtime
#
# The main cost is the profiling itself, which you would do anyway when
# optimizing performance. Annotations simply make the profiler output more
# useful by adding semantic structure.

###############################################################################
# Troubleshooting
# ---------------
#
# **No annotations in the trace?**
#
# - Check that your driver/CUDA-compat >= 13.1
# - Verify that ``enable_annotations=True`` was passed to ``torch.cuda.graph()``
# - Ensure ``cuda-python`` is installed
#
# **Kernels still overlapping in the trace?**
#
# - The cleanup passes should handle this automatically
# - If issues persist, try assigning explicit stream IDs in ``mark_kernels``
#
# **Annotations not showing up in specific kernels?**
#
# - Some operations may not launch kernels (e.g., tensor views)
# - Only kernels launched within the ``mark_kernels`` context are annotated
# - Verify the operation actually produces CUDA kernels using ``torch.profiler``

###############################################################################
# Conclusion
# ----------
#
# CUDA graph kernel annotations provide a powerful way to add semantic
# structure to your profiling traces. By marking logical components of your
# model during graph capture and merging these annotations in post-processing,
# you can create visualizations that make it much easier to understand and
# optimize complex CUDA graph executions.
#
# Key takeaways:
#
# - Use ``mark_kernels()`` to label regions during graph capture
# - Enable annotations with ``enable_annotations=True``
# - Post-process traces with ``annotate_trace()`` and cleanup passes
# - View results in chrome://tracing for intuitive visualization
#
# This technique is especially valuable for large models with many components,
# distributed training setups, or any scenario where understanding the
# execution structure is critical for performance optimization.

if __name__ == "__main__":
    main()
