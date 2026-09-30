Note

Go to the end
to download the full example code.

# CUDA Graph Kernel Annotations and Profiling

**Author**: [Shangdi Yu](https://github.com/yushangdi)

 What you will learn

- How to capture CUDA graphs with kernel annotations
- How to profile annotated graphs
- How to export traces with inline annotations and semantic kernel lanes
- How to visualize graph execution with custom stream assignments
- How to use `mark_stream` to recover logical stream lanes
- How to annotate communication collectives with the metadata
(collective type, message size, group, rank) that eager NCCL
traces expose but CUDA graphs drop

 Prerequisites

- PyTorch 2.15+ (use a nightly build until 2.15 is released)
- CUDA-capable GPU
- Driver/CUDA-compat >= 13.1 for annotation support
- cuda-bindings >= 13.1.0

CUDA graphs are a powerful optimization technique that can significantly reduce
kernel launch overhead by capturing and replaying sequences of CUDA operations.
However, when profiling CUDA graphs, all kernels appear on the same stream,
making it difficult to understand the logical structure of your computation.

This tutorial demonstrates how to use **kernel annotations** to add semantic
labels to kernels within CUDA graphs. The profiler can include these annotations
directly when exporting traces and create custom visualization lanes, making it easier to
understand and debug complex graph executions.

Annotations are not limited to compute kernels. One of the most valuable uses
is annotating **communication collectives**. In eager mode, the profiler
attaches rich metadata to every NCCL kernel - the collective type, message
size, process group, and ranks - so you can see exactly what each comm is
doing. Under CUDA graphs that metadata is lost: the collective replays as an
opaque kernel. This tutorial shows how to re-attach that metadata with
annotations so graphed comms read just like eager ones.

## Overview

CUDA graph kernel annotations allow you to add semantic labels to kernels
during graph capture. These labels help you understand what each kernel does
when profiling, making it easy to identify which parts of your model (e.g.,
attention, MLP, normalization) are executing at any given time.

Without annotations, profiler traces show all kernels on a single stream with
auto-generated names, making it difficult to understand the logical structure
of your computation. With annotations, you can:

1. **Label kernel groups** with meaningful names during capture
2. **Assign custom stream IDs** for visual organization
3. **Export labels directly in profiler traces** for semantic visualization

The result is a profiler trace where kernels are labeled and organized by
their function, making it much easier to identify performance bottlenecks
and understand execution flow.

**Before annotations:** All kernels appear on a single stream with
auto-generated names, making it difficult to understand which operations
belong to which logical component of your model.

[![CUDA graph trace before annotations showing all kernels on one stream](../_images/cuda_graph_trace_before.png)](../_images/cuda_graph_trace_before.png)

**After annotations:** Kernels are organized into semantic lanes (streams 61
and 62) with meaningful labels like "attention" and "mlp", making it easy to
identify different components and understand the execution structure.

[![CUDA graph trace after annotations showing kernels organized by function](../_images/cuda_graph_trace_after.png)](../_images/cuda_graph_trace_after.png)

As another example, here is an AllReduce kernel with annotated metadata:

[![AllReduce kernel with annotated metadata](../_images/annotated_cudagraph.png)](../_images/annotated_cudagraph.png)

## Requirements

For this tutorial, you'll need:

- PyTorch 2.15+ (use a nightly build until 2.15 is released)
- A CUDA GPU
- Driver/CUDA-compat >= 13.1 for annotation support
- The `cuda-bindings` package >= 13.1.0 (`pip install "cuda-bindings>=13.1.0"`)

The cuda-bindings package provides the Python bindings for CUDA runtime APIs.
Version 13.1.0+ is required for the `cudaGraphNodeGetToolsId` API that
enables kernel annotations. If you have an older version, the tutorial will
run but annotations will be disabled with a warning message explaining how
to upgrade.

On older drivers or cuda-bindings versions, the capture and profiling will
still work, but `mark_kernels` will be a no-op and no semantic lanes will
appear in the final trace.

## Building a Model

Let's create a simple transformer block as our example model. We'll annotate
different parts of the computation (QKV projection, attention, output
projection, MLP) to see them as separate lanes in the profiler.

## The `mark_kernels` Context Manager

The key API is `mark_kernels()`, which takes a dictionary with:

- `name`: A string label stored in the kernel event's `args` (also used
as the lane name when a custom stream is assigned)
- `stream` (optional): A virtual stream ID for visualization

Any CUDA kernels launched within the context will be tagged with these
annotations. When we export the profiler trace with `graph_lanes="all"`,
these tags organize kernels into custom lanes. The stream IDs control only
visualization; they do not change where the kernels execute.

## Capturing a CUDA Graph with Annotations

To capture a graph with annotations enabled, we pass
`enable_annotations=True` to `torch.cuda.graph()`. This automatically
handles the annotation lifecycle: enabling, resolving, and remapping.

## Profiling the Graph

After capturing the graph, we replay it a few times to warm up, then profile
subsequent replays. Pass the recorded annotations directly to
`export_chrome_trace()` to include them in the exported kernel events.
We also export a raw trace from the same profile for comparison.

`cuda_graph_annotations` selects the Python exporter automatically when the
mapping is nonempty. We explicitly select it for both exports so they also
work with an empty mapping. Both files can be opened in [https://ui.perfetto.dev/](https://ui.perfetto.dev/).

## Choosing the Trace Layout

`graph_lanes="all"` places graphed events with a `stream` annotation on
that display lane. Other graphed events go to `default_stream` (7 by
default). Moved events retain their actual execution stream in
`args["original_stream"]`. This groups attention on lane 62 and MLP on
lane 61 in our example.

To keep the recorded stream layout, omit `graph_lanes` or set it to
`"none"`. Annotation metadata is still included in each matching event's
`args`, and any annotated stream is stored as `args["annotated_stream"]`.
This is useful when inspecting concurrency on the original streams.

An empty annotation mapping is treated as no annotations. Since
`graph_lanes="all"` requires a nonempty mapping, the example uses
`"none"` when annotation support is unavailable.

Keep the captured graph alive until export finishes: destroying or resetting
it removes its entries from the live annotation registry. There is no need
to save annotations to a separate file or post-process the exported trace.

## Comparing Before and After

To see the impact of annotations, let's count how kernels are distributed
across thread IDs (which represent visualization lanes in the trace).

## Putting It All Together

Now let's run the complete workflow: build a model, capture it with
annotations, profile it, and export the annotated trace.

```
# Example output:
# if __name__ == "__main__":
# main()
#
# Annotation support available: True
#
# 1. Building transformer block model...
#
# 2. Capturing CUDA graph with annotations...
# Captured graph with 13 annotated nodes
#
# 3. Profiling graph replays and exporting traces...
# Saved raw trace to traces/trace_raw.json.gz
# Saved annotated trace to traces/trace_annotated.json.gz
#
# 4. Comparing traces...
#
# ============================================================
# BEFORE annotation - kernels per lane (tid -> count):
# Stream 7: 65 kernels
#
# AFTER annotation - kernels per lane (tid -> count):
# Stream 7: 10 kernels
# Stream 61: 15 kernels
# Stream 62: 40 kernels
# ============================================================
#
# ============================================================
# SUMMARY
# ============================================================
# Raw trace: traces/trace_raw.json.gz
# Annotated trace: traces/trace_annotated.json.gz
#
# Open the annotated trace in https://ui.perfetto.dev/ to visualize
# the semantic kernel lanes.
# ============================================================
```

## Recording Logical Streams with `mark_stream`

When your computation already uses multiple CUDA streams, `mark_stream`
switches to a stream and records a logical lane ID for its kernels. This
helper currently lives in the private `torch.cuda._graph_annotations`
module. Use `mark_kernels` with an explicit `stream` field when you only
want to change the display layout, as in the transformer example above.

The block below runs two independent projections on separate CUDA streams,
then combines their results. Each side stream waits for the current stream
inside its `mark_stream` scope. These waits make the side streams join the
capture before launching work. The current stream waits for both branches
before reading their outputs and ending capture. `mark_stream` does not
insert these dependencies for you.

Reuse the capture and profiling helpers to export this graph. With
`graph_lanes="all"`, the projections appear on distinct logical lanes
named `left_projection` and `right_projection`, even if graph replay
schedules them on different hardware streams. The `combine` kernel goes
to the default lane, 7. Moved events retain `args["original_stream"]`.

Lane IDs are assigned automatically and may vary depending on which streams
have already been marked. Reusing the same stream reuses its lane ID. Passing
the current stream to `mark_stream` adds the label without assigning a new
lane. With `graph_lanes="none"`, the original layout is preserved and the
recorded lane IDs appear in `args["annotated_stream"]` instead.

Run `stream_annotation_demo()` and open
`traces_streams/trace_annotated.json.gz` in [https://ui.perfetto.dev/](https://ui.perfetto.dev/).

## Annotating Communication Collectives

In eager mode the profiler **automatically intercepts** NCCL collectives and
records rich metadata: collective type, input/output message sizes, the process
group, its size, and the participating ranks.

Under CUDA graphs that automatic interception stops working. The collective is
captured once and then replayed as an opaque kernel node. The profiler cannot
intercept graph replay, so it has nothing to attach the NCCL metadata to. The
kernels still show up in the trace (e.g., `ncclDevKernel_AllReduce_Sum_f32_RING_LL`),
but they are opaque: you cannot tell what collective type it is, how many bytes
moved, or which process group it belongs to.

Annotations close this gap. By wrapping the collective in `mark_kernels`
with the same fields the profiler auto-attaches in eager mode, we manually
re-attach that metadata to the graphed kernel. After export, a
graphed collective reads just like an eager one. The helper below builds the
metadata dict; using the field names the profiler uses in eager
(`In msg nelems`, `Group size`, `Process Group Name`, ...) keeps the
annotated trace consistent with non-graphed traces.

## A Block That Mixes Compute and Communication

A tensor- or data-parallel layer interleaves matmuls with collectives. Here
the projection output is all-reduced across the group, mirroring the comm in
a tensor-parallel linear. The collective is annotated with
`annotate_collective` and lands on its own lane.

## Running the Communication Demo

```
# Example output (2 GPUs):
# if __name__ == "__main__":
# comm_annotation_demo()
#
# Building compute + collective block...
# Capturing CUDA graph with annotations...
# Captured graph with 3 annotated nodes
# Saved raw trace to traces_comm/trace_raw.json.gz
# Saved annotated trace to traces_comm/trace_annotated.json.gz
#
# The all_reduce runs a real NCCL kernel
# (``ncclDevKernel_AllReduce_Sum_f32_RING_LL``) across the two ranks:
#
# Annotated collective kernels (metadata restored):
# ncclDevKernel_AllReduce_Sum_f32_RING_LL
# In msg nelems: 1048576
# Out msg nelems: 1048576
# Group size: 2
# dtype: float32
# Process Group Name: 0
# Process Group Description: default_pg
# Process Group Ranks: [0, 1]
# stream: 60
#
# In the trace viewer, the all-reduce sits on its own dedicated comm lane
# (stream 60), and selecting it shows the collective type, message sizes, group,
# and ranks -- the same fields you would see in an eager trace, now recovered
# for a CUDA-graphed collective. This metadata is LOST without annotations.
```

## Performance Considerations

Kernel annotations add minimal overhead:

- Annotation marking happens during graph capture (one-time cost)
- Graph replay performance is identical to unannotated graphs
- Annotations are added during trace export, after profiling finishes

The main cost is the profiling itself, which you would do anyway when
optimizing performance. Annotations simply make the profiler output more
useful by adding semantic structure.

## Troubleshooting

**No annotations in the trace?**

- Check that your driver/CUDA-compat >= 13.1
- Verify that `enable_annotations=True` was passed to `torch.cuda.graph()`
- Ensure `cuda-bindings>=13.1.0` is installed
- Pass `cuda_graph_annotations=get_kernel_annotations()` to
`export_chrome_trace()` while the graph is still alive

**Annotations not showing up in specific kernels?**

- Some operations may not launch kernels (e.g., tensor views)
- Only kernels launched within the `mark_kernels` context are annotated
- Verify the operation actually produces CUDA kernels using `torch.profiler`

## Conclusion

CUDA graph kernel annotations provide a powerful way to add semantic
structure to your profiling traces. By marking logical components of your
model during graph capture and including these annotations during export,
you can create visualizations that make it much easier to understand and
optimize complex CUDA graph executions.

Key takeaways:

- Use `mark_kernels()` to label regions during graph capture
- Use `mark_stream()` to record logical lanes when switching CUDA streams
- Enable annotations with `enable_annotations=True`
- Annotate communication collectives to recover the NCCL metadata
(collective type, message size, group, rank) that CUDA graphs drop but
eager traces expose
- Pass annotations directly to `export_chrome_trace()`
- Use `graph_lanes="all"` to organize graphed kernels into semantic lanes
- View results in [https://ui.perfetto.dev/](https://ui.perfetto.dev/) for intuitive visualization

This technique is especially valuable for large models with many components,
distributed training setups, or any scenario where understanding the
execution structure is critical for performance optimization.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: cuda_graph_annotations_tutorial.ipynb`](../_downloads/93c170f8ef9d2c0e3ebe2db9ba616e9f/cuda_graph_annotations_tutorial.ipynb)

[`Download Python source code: cuda_graph_annotations_tutorial.py`](../_downloads/8891ea63335e99147b5909553baa119b/cuda_graph_annotations_tutorial.py)

[`Download zipped: cuda_graph_annotations_tutorial.zip`](../_downloads/f5e06826050964a8e3c15c270666f021/cuda_graph_annotations_tutorial.zip)