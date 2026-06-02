# CUDA Graph Annotations Tutorial

## Overview

A new tutorial has been added to the PyTorch tutorials repository that demonstrates how to use CUDA graph kernel annotations for enhanced profiling and visualization.

## File Location

- **Tutorial file**: `advanced_source/cuda_graph_annotations_tutorial.py`
- **Added to index**: `index.rst` (line ~518)
- **Added to deep-dive**: `deep-dive.rst` (profiling section)

## Tutorial Content

The tutorial covers:

1. **Introduction to CUDA Graph Annotations**: Why they're useful for profiling complex graph executions
2. **Building an Example Model**: A simple transformer block with multiple annotated regions
3. **Using `mark_kernels()`**: The key API for annotating kernel regions with semantic labels
4. **Graph Capture with Annotations**: How to enable annotations during CUDA graph capture
5. **Profiling**: Recording execution traces of graph replays
6. **Post-Processing**: Merging annotations back into traces for custom visualization lanes
7. **Visualization**: How to view annotated traces in chrome://tracing
8. **Troubleshooting**: Common issues and solutions

## Key Features

- **Complete end-to-end workflow**: From model definition to visualization
- **Practical example**: Uses a realistic transformer block
- **Custom stream assignments**: Shows how to organize kernels into semantic lanes
- **Before/after comparison**: Demonstrates the value of annotations
- **Comprehensive documentation**: Includes requirements, troubleshooting, and advanced usage

## Tutorial Style

The tutorial follows PyTorch tutorials conventions:
- Uses Sphinx Gallery format with docstring sections
- Includes grid cards for learning objectives and prerequisites
- Code is well-commented and organized into logical sections
- Provides practical, runnable examples
- Explains both the "how" and "why"

## Integration

The tutorial has been integrated into:
1. Main index (`index.rst`) under Model Optimization/Profiling
2. Deep Dive section (`deep-dive.rst`) alongside other profiling tutorials
3. Tagged appropriately: `Model-Optimization`, `Best-Practice`, `Profiling`, `CUDA`

## Building

To build just this tutorial:
```bash
cd tutorials
GALLERY_PATTERN="cuda_graph_annotations_tutorial.py" make html
```

To build all tutorials:
```bash
cd tutorials
make docs
```

## Requirements

- PyTorch 2.0+
- CUDA-capable GPU
- Driver/CUDA-compat >= 13.1 for full annotation support
- cuda-python package (`pip install cuda-python`)

Note: The tutorial gracefully handles older drivers by detecting annotation support and providing appropriate messages.

## Next Steps

1. Test the full build in CI
2. Add a thumbnail image to `_static/img/thumbnails/cropped/`
3. Verify the tutorial renders correctly in the documentation site
4. Consider adding more advanced examples (multi-GPU, distributed training)
