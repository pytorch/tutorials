# -*- coding: utf-8 -*-

"""
Mosaic: Memory Profiling for PyTorch
====================================

**Author:** `Basil Wong <https://github.com/basilwong>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to capture and analyze PyTorch memory snapshots
       * Identify memory savings from activation checkpointing
       * Debug unexpected memory usage from abandoned code
       * Integrate memory analysis into training pipelines

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.0.0 or later
       * CUDA-capable GPU
       * Basic understanding of PyTorch training loops

This tutorial demonstrates how to use `Mosaic <https://github.com/facebookresearch/mosaic>`_, a post-processing memory
snapshot analysis tool for PyTorch. Mosaic helps analyze GPU memory usage in
distributed deep learning, providing detailed insights into memory allocations,
peak usage, and memory imbalances across parallel workers.

Mosaic was instrumental in debugging OOM issues during the
`405B LLaMA training <https://ai.meta.com/blog/meta-llama-3-1/>`_
and is now open source.

"""

######################################################################
# Introduction to Mosaic
# ======================
#
# Overview
# --------
#
# In distributed deep learning, understanding GPU memory usage is critical
# for optimizing training efficiency and debugging Out-of-Memory (OOM) errors.
# Mosaic is a post-analysis tool for memory usage designed to work with
# large-scale jobs. It helps analyze PyTorch memory snapshots captured during
# the execution of PyTorch training jobs, providing detailed insights into
# memory allocations, peak usage, and memory imbalances across parallel workers.
#
# Getting Started
# ---------------
#
# Clone the mosaic repository and install from the mosaic directory:
#
# .. code-block:: bash
#
#    git clone https://github.com/facebookresearch/mosaic
#    cd mosaic
#    python3 -m venv venv
#    source venv/bin/activate
#    pip3 install -r requirements.txt
#    pip3 install -e .
#
# Alternatively, install directly via pip:
#
# .. code-block:: bash
#
#    pip install git+https://github.com/facebookresearch/mosaic.git
#
# Simple Usage Examples
# ---------------------
#
# **1. Peak Memory Usage Analysis**
#
# When addressing memory problems like OOM errors, focusing on peak memory
# usage is crucial. The ``mosaic_get_memory_usage_peak`` command presents a
# stack trace of the memory allocations that contributed to the peak memory
# usage:
#
# .. code-block:: bash
#
#    mosaic_get_memory_usage_peak --snapshot <path to snapshot>
#
# **2. Categorical Memory Profiling**
#
# Mosaic classifies allocations into categories (activation, backward,
# optimizer, etc.):
#
# - **Activation Memory:** Tensors saved for backward pass
# - **Gradient Memory:** Gradients computed during backpropagation
# - **Optimizer State:** Adam/SGD momentum and variance buffers
# - **Parameter Memory:** Model weights
#
# .. code-block:: bash
#
#    mosaic_get_memory_profile --snapshot <path> --out-path <html> \
#        --profile categories
#
# An example HTML output looks like:
#
# .. figure:: /_static/img/mosaic/mosaic-categorical-memory-profiling-no-allocation-ordering.png
#    :alt: Mosaic categorical memory profiling without allocation ordering
#    :align: center
#    :width: 600px
#
#    Categorical memory profiling showing memory breakdown by type
#    (activation, gradient, optimizer, etc.)
#
# To maintain allocation order for the categories, add ``--preserve-allocation-order``:
#
# .. code-block:: bash
#
#    mosaic_get_memory_profile --snapshot <path> --out-path <html> \
#        --profile categories --preserve-allocation-order
#
# .. figure:: /_static/img/mosaic/mosaic-categorical-memory-profiling-allocation-ordering.png
#    :alt: Mosaic categorical memory profiling with allocation ordering preserved
#    :align: center
#    :width: 600px
#
#    Categorical profiling with ``--preserve-allocation-order`` shows memory
#    allocations in chronological order
#
# **3. Custom Dictionary Profiling**
#
# For targeted analysis via regex pattern matching:
#
# .. code-block:: bash
#
#    mosaic_get_memory_profile --snapshot <path> --profile custom \
#        --custom-profile '{"ncclx": "ncclx"}'
#
# This is invaluable for tracking specific kernels, optimizers, or custom code patterns:
#
# .. figure:: /_static/img/mosaic/mosaic-categorical-memory-profiling-ncclx.png
#    :alt: Mosaic custom dictionary profiling with ncclx pattern
#    :align: center
#    :width: 600px
#
#    Custom profiling with regex patterns to track specific operations like
#    NCCL communications
#

######################################################################
# Dependencies and Imports
# ========================
#
# Let's set up the required dependencies and imports for this tutorial.

import subprocess
import sys
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader, Dataset

# Install dependencies if needed
try:
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "transformers"]
    )
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

try:
    from mosaic.libmosaic.analyzer.memory_abstract import MemoryAbstract
except ImportError:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/facebookresearch/mosaic.git",
        ]
    )
    from mosaic.libmosaic.analyzer.memory_abstract import MemoryAbstract

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

######################################################################
# Shared Utilities
# ================
#
# These helper classes and functions are used throughout the tutorial.


class RandomTokenDataset(Dataset):
    """Generates random token sequences for training.

    This dataset creates random input sequences suitable for language model
    training, simulating real training data without requiring actual text.
    """

    def __init__(self, vocab_size, seq_length=512, num_samples=100, seed=None):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):  # noqa: ARG002
        if self.generator is not None:
            input_ids = torch.randint(
                0, self.vocab_size, (self.seq_length,), generator=self.generator
            )
        else:
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


@contextmanager
def capture_memory_snapshot(output_path):
    """Context manager to capture and save PyTorch CUDA memory snapshots.

    This captures all GPU memory allocations during the context and saves
    them to a pickle file for later analysis with Mosaic.

    Args:
        output_path: Path to save the memory snapshot pickle file.
    """
    torch.cuda.memory._record_memory_history(max_entries=100000)
    try:
        yield
    finally:
        snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        with open(output_path, "wb") as f:
            pickle.dump(snapshot, f)
        print(f"✓ Memory snapshot saved to {output_path}")


######################################################################
# Case 1: Understanding Memory Differences with Activation Checkpointing
# =======================================================================
#
# This section demonstrates how to use Mosaic to analyze and compare GPU
# memory usage between different model configurations.
#
# **What we'll do:**
#
# 1. Train GPT-2 and capture a memory snapshot (baseline)
# 2. Enable activation checkpointing and train again (modified)
# 3. Use Mosaic to identify exactly where memory savings occur
#

######################################################################
# Training Function for Activation Checkpointing Comparison
# ----------------------------------------------------------


def run_training_ac(
    activation_checkpointing: bool,
    snapshot_path: str,
    batch_size: int = 4,
    seq_length: int = 512,
    num_steps: int = 5,
):
    """Run training loop and capture memory snapshot.

    Args:
        activation_checkpointing: Whether to enable gradient checkpointing.
        snapshot_path: Path to save the memory snapshot.
        batch_size: Training batch size.
        seq_length: Sequence length for input tokens.
        num_steps: Number of training steps to run.

    Returns:
        Peak GPU memory usage in GB.
    """
    # Clear any previous memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")

    # Load model
    print(f"Loading GPT-2 (activation_checkpointing={activation_checkpointing})...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if activation_checkpointing:
        model.gradient_checkpointing_enable()
        print("Activation checkpointing is ENABLED")
    else:
        print("Activation checkpointing is DISABLED")

    model = model.to(device)
    model.train()

    # Create dataset and dataloader
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = RandomTokenDataset(
        vocab_size=tokenizer.vocab_size,
        seq_length=seq_length,
        num_samples=100,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Training loop with memory capture
    print(f"Running {num_steps} training steps...")

    with capture_memory_snapshot(snapshot_path):
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"  Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"✓ Peak GPU memory: {peak_memory_gb:.2f} GB")

    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()

    return peak_memory_gb


######################################################################
# Run Baseline Training (Without Activation Checkpointing)
# ---------------------------------------------------------
#
# .. note::
#
#    This tutorial requires a CUDA-capable GPU. If you're running in
#    Google Colab, make sure to select a GPU runtime:
#    Runtime → Change runtime type → Hardware accelerator → GPU

if not torch.cuda.is_available():
    print("=" * 60)
    print("WARNING: No CUDA GPU detected!")
    print("=" * 60)
    print("\nThis tutorial requires a CUDA-capable GPU for memory profiling.")
    print("\nIf you're running in Google Colab:")
    print("  1. Go to Runtime → Change runtime type")
    print("  2. Set Hardware accelerator to 'GPU'")
    print("  3. Click 'Save' and re-run the notebook")
    print("\nSkipping GPU memory profiling examples...")
    HAS_CUDA = False
else:
    HAS_CUDA = True
    print("=" * 60)
    print("BASELINE: Training WITHOUT Activation Checkpointing")
    print("=" * 60)

    baseline_memory = run_training_ac(
        activation_checkpointing=False,
        snapshot_path="snapshot_baseline.pickle",
        batch_size=4,
        seq_length=512,
        num_steps=5,
    )

######################################################################
# Run Modified Training (With Activation Checkpointing)
# ------------------------------------------------------

if HAS_CUDA:
    print("\n" + "=" * 60)
    print("MODIFIED: Training WITH Activation Checkpointing")
    print("=" * 60)

    ac_memory = run_training_ac(
        activation_checkpointing=True,
        snapshot_path="snapshot_with_ac.pickle",
        batch_size=4,
        seq_length=512,
        num_steps=5,
    )

    # Summary
    print("\n" + "=" * 60)
    print("MEMORY COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Baseline (no AC):     {baseline_memory:.2f} GB")
    print(f"With AC:              {ac_memory:.2f} GB")
    if baseline_memory > 0:
        saved_pct = 100 * (baseline_memory - ac_memory) / baseline_memory
        print(
            f"Memory Saved:         {baseline_memory - ac_memory:.2f} GB ({saved_pct:.1f}%)"
        )

######################################################################
# Generate Categorical Memory Profiles with Mosaic
# -------------------------------------------------
#
# Use Mosaic to generate HTML profiles for both snapshots.

if HAS_CUDA:
    print("\n" + "=" * 60)
    print("MOSAIC: Categorical Memory Profiling")
    print("=" * 60)

    # Generate HTML profiles using subprocess
    subprocess.run(
        [
            "mosaic_get_memory_profile",
            "--snapshot",
            "snapshot_baseline.pickle",
            "--out-path",
            "profile_baseline.html",
            "--profile",
            "categories",
            "--preserve-allocation-order",
            "--plotter_sampling_rate",
            "20",
        ],
        check=True,
    )

    print()

    subprocess.run(
        [
            "mosaic_get_memory_profile",
            "--snapshot",
            "snapshot_with_ac.pickle",
            "--out-path",
            "profile_with_ac.html",
            "--profile",
            "categories",
            "--preserve-allocation-order",
            "--plotter_sampling_rate",
            "20",
        ],
        check=True,
    )

    print("\n✓ Generated profile_baseline.html")
    print("✓ Generated profile_with_ac.html")
    print("\nDownload these files to view the interactive memory profiles.")

######################################################################
# Results Interpretation: Activation Checkpointing
# -------------------------------------------------
#
# What We Observed
# ~~~~~~~~~~~~~~~~
#
# Based on the Mosaic categorical profiling results:
#
# .. list-table:: Memory Comparison Results
#    :header-rows: 1
#
#    * - Metric
#      - Baseline
#      - With Activation Checkpointing
#      - Difference
#    * - **Total Peak Memory**
#      - **4.62 GB**
#      - **2.55 GB**
#      - **2.07 GB (45% reduction)**
#    * - Activation Memory
#      - 2.93 GB
#      - 872.79 MB
#      - **2.08 GB saved (71% reduction)**
#    * - Backward/Gradient Memory
#      - 793.39 MB
#      - 785.27 MB
#      - 8 MB (minimal change)
#    * - Optimizer State
#      - 949.4 MB
#      - 949.4 MB
#      - No change
#    * - Unknown
#      - 32 KB
#      - 32 KB
#      - No change
#
# Key Insights
# ~~~~~~~~~~~~
#
# **Primary Finding:** Activation memory dropped from **2.93 GB → 872 MB**
# (71% reduction), which accounts for nearly all the total memory savings.
#
# Why Does This Happen?
# ~~~~~~~~~~~~~~~~~~~~~
#
# **Activation checkpointing** is a memory optimization technique that:
#
# 1. **Without AC (Baseline):** All intermediate activations from the forward
#    pass are stored in memory for use during backpropagation. GPT-2 has 12
#    transformer layers, each storing multiple activations (attention outputs,
#    MLP outputs, etc.). For batch_size=4, seq_length=512, this adds up quickly.
#
# 2. **With AC (Optimized):** Only activations at checkpoint boundaries are
#    stored; intermediate activations are recomputed during the backward pass.
#    This dramatically reduces activation memory (71% in our case) while other
#    memory categories remain unchanged.
#
# How Mosaic Helped
# ~~~~~~~~~~~~~~~~~
#
# Mosaic's categorical profiling immediately identified:
#
# - Activation memory is the category with the largest difference (2.08 GB saved)
# - Backward/Gradient memory stayed nearly constant (793 MB → 785 MB)
# - Optimizer state remained unchanged (949 MB) - expected since model
#   parameters don't change
#
# **Without Mosaic:** You would need to manually instrument your code, track
# allocations, and categorize them yourself.
#
# **With Mosaic:** You get instant categorical breakdowns with exact numbers,
# making it trivial to identify/quantify memory optimizations.
#

######################################################################
# Case 2: Debugging Unexpected Memory Usage
# ==========================================
#
# This section demonstrates how to use Mosaic to debug when your model is
# using more memory than expected and you're not sure why.
#
# **What we'll do:**
#
# 1. Train GPT-2 and capture a memory snapshot.
# 2. Train GPT-2 with a bug that introduces additional memory and capture
#    a memory snapshot.
# 3. Use Mosaic to identify potential culprits introducing additional memory.
#

######################################################################
# The Buggy Model
# ---------------
#
# This model has **abandoned debug code** that creates unnecessary GPU memory
# overhead. Someone added projection layers to "analyze hidden states" during
# debugging, but forgot to remove them before training.


class GPT2WithDebugOverhead(GPT2LMHeadModel):
    """GPT2 with abandoned 'feature analysis' code that bloats peak memory."""

    def __init__(self, config):
        super().__init__(config)

        # BUG: Large projection layers from an abandoned experiment
        self.debug_projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(config.n_embd, config.n_embd * 4)
                for _ in range(config.n_layer)
            ]
        )

        debug_params = sum(p.numel() for p in self.debug_projections.parameters())
        print(f"  [DEBUG] Added {config.n_layer} debug projection layers")
        print(f"  [DEBUG] Extra parameters: {debug_params:,}")

    def forward(self, input_ids=None, labels=None, **kwargs):
        # Run normal GPT-2 forward with hidden states
        outputs = super().forward(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        # BUG: Project all hidden states through debug layers
        projected = []
        for _layer_idx, (hidden, proj) in enumerate(
            zip(outputs.hidden_states[1:], self.debug_projections)
        ):
            proj_hidden = proj(hidden)
            projected.append(proj_hidden)

        # Tie to loss so gradients flow through
        debug_regularization = sum(p.mean() for p in projected) * 1e-10

        return CausalLMOutputWithCrossAttentions(
            loss=outputs.loss + debug_regularization,
            logits=outputs.logits,
        )


######################################################################
# Training Functions for Debug Comparison
# ----------------------------------------


def run_training_clean(snapshot_path, num_steps=3):
    """Training with the normal model."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")

    print("Loading clean model (no debug overhead)...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.train()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = RandomTokenDataset(
        vocab_size=tokenizer.vocab_size, seq_length=512, seed=42
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print("Running training (should contain no debug overhead)...")

    with capture_memory_snapshot(snapshot_path):
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"  Step {step + 1}, Loss: {loss.item():.4f}")

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {peak_memory:.2f} GB")

    del model, optimizer
    torch.cuda.empty_cache()

    return peak_memory


def run_training_with_bug(snapshot_path, num_steps=3):
    """Training with the buggy model."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")

    print("Loading buggy model with debug overhead...")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2WithDebugOverhead(config).to(device)

    # Load pretrained weights
    pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(pretrained.state_dict(), strict=False)
    del pretrained
    torch.cuda.empty_cache()

    model.train()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = RandomTokenDataset(
        vocab_size=tokenizer.vocab_size, seq_length=512, seed=42
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print("Running training (WITH debug overhead bug)...")

    with capture_memory_snapshot(snapshot_path):
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"  Step {step + 1}, Loss: {loss.item():.4f}")

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {peak_memory:.2f} GB")

    del model, optimizer
    torch.cuda.empty_cache()

    return peak_memory


######################################################################
# Run Training for Baseline (Clean Model)
# ----------------------------------------

if HAS_CUDA:
    print("\n" + "=" * 60)
    print("Training with baseline model")
    print("=" * 60)

    baseline_memory_debug = run_training_clean(
        "snapshot_debug_baseline.pickle", num_steps=3
    )

######################################################################
# Run Training WITH the Bug
# --------------------------

if HAS_CUDA:
    print("\n" + "=" * 60)
    print("Training with debug projection overhead (BUG)")
    print("=" * 60)

    buggy_memory = run_training_with_bug("snapshot_with_bug.pickle", num_steps=3)

######################################################################
# Use Mosaic to Find the Problem
# -------------------------------
#
# Analyze both snapshots to identify the source of extra memory usage.

if HAS_CUDA:
    print("\n" + "=" * 60)
    print("MOSAIC: Analyzing the Baseline Snapshot")
    print("=" * 60)

    subprocess.run(
        ["mosaic_get_memory_usage_peak", "--snapshot", "snapshot_debug_baseline.pickle"],
        check=True,
    )

    print("\n" + "=" * 60)
    print("MOSAIC: Analyzing the Buggy Snapshot")
    print("=" * 60)

    subprocess.run(
        ["mosaic_get_memory_usage_peak", "--snapshot", "snapshot_with_bug.pickle"],
        check=True,
    )

######################################################################
# Analyzing The Mosaic Output
# ----------------------------
#
# When you run Mosaic's peak memory analysis, it shows stack traces for each
# memory allocation. Let's look at how to find abandoned or unnecessary code
# that's bloating the memory.
#
# 1. Optimizer State Allocations Delta
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the buggy snapshot output, we can see that the first two stack traces
# represent the **optimizer state allocations** (like ``zeros_like`` for Adam
# optimizer state). See ``torch/optim/adam.py`` in the stack trace.
#
# In the snapshot of the buggy model we can see around a total of 0.21 GB
# more memory:
#
# .. list-table:: Optimizer State Comparison
#    :header-rows: 1
#
#    * - Version
#      - Stack Trace Position
#      - Calls
#      - Memory (per trace)
#    * - Buggy model
#      - 1st and 2nd
#      - 172 calls
#      - 0.569 GB + 0.569 GB
#    * - Baseline
#      - 2nd and 3rd
#      - 148 calls
#      - 0.464 GB + 0.464 GB
#
# **What this tells us:** The optimizer is tracking more tensors! This is your
# first clue that there are extra parameters or tensors in the computation graph.
#
# 2. Additional Activation Allocations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The buggy version shows **extra allocations** that don't appear in the
# baseline model. Scrolling down the Mosaic output of the buggy model we can
# see additional stack traces which contain:
#
# 1. ``torch::autograd::Engine::evaluate_function``: We're in the backward pass
# 2. ``AddmmBackward0::apply``: Computing gradients for an addmm operation
# 3. ``empty_cuda`` at the bottom: Allocating a new CUDA tensor to store
#    the gradient
#
# - 0.176 GB from matrix multiply gradients (``AddmmBackward0``, ``mm_mat1_backward``)
#
# Memory Total Explanation
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# **Total Peak Dynamic Memory Usage:** This is the peak memory that changes
# during execution, measured relative to the starting point of the snapshot.
# It tracks memory allocations that occur during the traced execution timeline.
#
# **Total Static Memory Usage:** This is the "starting memory" or baseline
# memory that exists before tracing begins. It's estimated by the PyTorch
# visualizer and remains constant throughout the snapshot (doesn't come with
# stack traces).
#
# .. note::
#
#    In the snapshots you may observe differences in total *static* memory
#    usage, which accounts for the remaining difference.
#
# **Total Overall Peak Memory Usage:** Dynamic + Static
#

if HAS_CUDA:
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Baseline (clean model):           {baseline_memory_debug:.2f} GB")
    print(f"With bug (debug projections):     {buggy_memory:.2f} GB")
    print(
        f"Extra memory from bug:            {buggy_memory - baseline_memory_debug:.2f} GB"
    )

######################################################################
# Case 3: Integrating Memory Analysis into Your Training Pipeline
# ================================================================
#
# This section demonstrates how to use Mosaic to automatically capture memory
# snapshots during training, get structured memory breakdown data for
# monitoring/dashboards, and build automated memory monitoring for large-scale
# training using Mosaic **programmatically** (as a Python dependency).
#
# Mosaic integrates memory analysis directly into your training pipeline.
#

######################################################################
# Training with Automatic Memory Capture
# ---------------------------------------


def run_training_with_memory_capture(
    batch_size=4,
    seq_length=512,
    num_steps=5,
    snapshot_path="training_snapshot.pickle",
):
    """Run training and automatically capture memory snapshot."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.train()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = RandomTokenDataset(tokenizer.vocab_size, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print(f"Running {num_steps} training steps with memory capture...")

    with capture_memory_snapshot(snapshot_path):
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            outputs.loss.backward()
            optimizer.step()
            print(f"  Step {step + 1}/{num_steps}, Loss: {outputs.loss.item():.4f}")

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"✓ PyTorch reported peak memory: {peak_memory_gb:.3f} GB")

    del model, optimizer
    torch.cuda.empty_cache()

    return snapshot_path


if HAS_CUDA:
    print("\n" + "=" * 60)
    print("CASE 3: Pipeline Integration")
    print("=" * 60)

    pipeline_snapshot_path = run_training_with_memory_capture(batch_size=4, seq_length=512)

######################################################################
# Mosaic Memory Analysis via Python API
# --------------------------------------
#
# Instead of using CLI commands, we can use Mosaic's Python API directly
# for programmatic integration.

if HAS_CUDA:
    print("\n" + "=" * 60)
    print("MOSAIC MEMORY ANALYSIS (via Python API)")
    print("=" * 60)

    # Load and analyze the memory snapshot
    memory_abstract = MemoryAbstract(memory_snapshot_file=pipeline_snapshot_path)
    memory_abstract.load_memory_snapshot()

    # Analyze peak memory usage
    memory_abstract.memory_snapshot.analyze_memory_snapshot(opt="memory_peak")

    # Get results
    dynamic_peak = memory_abstract.memory_snapshot.dynamic_memory_peak
    static_memory = memory_abstract.memory_snapshot.static_memory
    overall_peak = dynamic_peak + static_memory

    print(f"Peak dynamic memory: {dynamic_peak / 1024**3:.3f} GiB")
    print(f"Static memory: {static_memory / 1024**3:.3f} GiB")
    print(f"Overall peak memory: {overall_peak / 1024**3:.3f} GiB")

    print("✓ Analysis complete using Mosaic Python API")

######################################################################
# Reusable Memory Analysis Function
# ----------------------------------
#
# Create a reusable function for analyzing training memory snapshots.


def analyze_training_memory(snapshot_path):
    """Analyze a memory snapshot using Mosaic's Python API.

    Returns a structured dictionary with memory breakdown.

    Args:
        snapshot_path: Path to the memory snapshot pickle file.

    Returns:
        Dictionary containing memory analysis results.
    """
    # Load snapshot
    memory_abstract = MemoryAbstract(memory_snapshot_file=snapshot_path)
    memory_abstract.load_memory_snapshot()

    # Analyze peak memory
    memory_abstract.memory_snapshot.analyze_memory_snapshot(opt="memory_peak")

    # Extract results
    dynamic_peak = memory_abstract.memory_snapshot.dynamic_memory_peak
    static_memory = memory_abstract.memory_snapshot.static_memory
    overall_peak = dynamic_peak + static_memory

    return {
        "snapshot_path": snapshot_path,
        "dynamic_peak_memory_bytes": dynamic_peak,
        "static_memory_bytes": static_memory,
        "overall_peak_memory_bytes": overall_peak,
        "dynamic_peak_memory_gib": dynamic_peak / 1024**3,
        "static_memory_gib": static_memory / 1024**3,
        "overall_peak_memory_gib": overall_peak / 1024**3,
    }


if HAS_CUDA:
    analysis = analyze_training_memory(pipeline_snapshot_path)
    print("\nMemory Analysis Result:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

######################################################################
# Complete Training Pipeline with Memory Monitoring
# --------------------------------------------------
#
# This demonstrates a production-ready training pipeline with integrated
# Mosaic memory monitoring that can be used in CI/CD, monitoring dashboards,
# or capacity planning.


def training_pipeline_with_memory_monitoring(
    model_name: str,
    batch_size: int,
    seq_length: int,
    num_steps: int = 5,
    snapshot_path: str = "pipeline_snapshot.pickle",
) -> dict:
    """Complete training pipeline with integrated Mosaic memory monitoring.

    Can be integrated into CI/CD, monitoring dashboards, or capacity planning.

    Args:
        model_name: HuggingFace model name to use.
        batch_size: Training batch size.
        seq_length: Sequence length for input tokens.
        num_steps: Number of training steps.
        snapshot_path: Path to save the memory snapshot.

    Returns:
        Dictionary containing training and memory analysis report.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup
    print(f"Loading model: {model_name}")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Training with memory capture
    print(f"Running {num_steps} training steps...")
    with capture_memory_snapshot(snapshot_path):
        for step in range(num_steps):
            input_ids = torch.randint(
                0, tokenizer.vocab_size, (batch_size, seq_length)
            ).to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"  Step {step + 1}/{num_steps}, Loss: {outputs.loss.item():.4f}")

    pytorch_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    # Mosaic analysis using Python API
    print("Analyzing memory with Mosaic...")
    memory_abstract = MemoryAbstract(memory_snapshot_file=snapshot_path)
    memory_abstract.load_memory_snapshot()
    memory_abstract.memory_snapshot.analyze_memory_snapshot(opt="memory_peak")

    dynamic_peak = memory_abstract.memory_snapshot.dynamic_memory_peak
    static_memory = memory_abstract.memory_snapshot.static_memory
    overall_peak = dynamic_peak + static_memory

    report = {
        "model": model_name,
        "config": {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "num_steps": num_steps,
        },
        "pytorch_peak_memory_gb": pytorch_peak_gb,
        "mosaic_analysis": {
            "dynamic_peak_gib": dynamic_peak / 1024**3,
            "static_memory_gib": static_memory / 1024**3,
            "overall_peak_gib": overall_peak / 1024**3,
        },
        "snapshot_path": snapshot_path,
    }

    del model, optimizer
    torch.cuda.empty_cache()

    return report


# Run the pipeline
if HAS_CUDA:
    report = training_pipeline_with_memory_monitoring(
        "gpt2", batch_size=4, seq_length=512, num_steps=5
    )

    print("\n" + "=" * 60)
    print("PIPELINE REPORT")
    print("=" * 60)
    print(f"Model: {report['model']}")
    print(f"Config: {report['config']}")
    print(f"PyTorch Peak Memory: {report['pytorch_peak_memory_gb']:.3f} GB")
    print(f"Mosaic Dynamic Peak: {report['mosaic_analysis']['dynamic_peak_gib']:.3f} GiB")
    print(f"Mosaic Overall Peak: {report['mosaic_analysis']['overall_peak_gib']:.3f} GiB")

######################################################################
# CI/CD and Dashboard Integration Patterns
# -----------------------------------------
#
# These patterns show how to integrate Mosaic analysis into automated
# workflows.

import json

######################################################################
# Pattern 1: CI/CD Memory Regression Testing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def check_memory_regression(report, threshold_gib=5.0):
    """Check if memory usage exceeds threshold for CI/CD pipelines.

    Args:
        report: Memory analysis report from training_pipeline_with_memory_monitoring.
        threshold_gib: Maximum allowed memory in GiB.

    Raises:
        AssertionError: If memory exceeds threshold.
    """
    peak = report["mosaic_analysis"]["overall_peak_gib"]
    assert peak < threshold_gib, (
        f"Memory regression! {peak:.2f} GiB > {threshold_gib} GiB"
    )
    print(f"Memory check passed: {peak:.2f} GiB < {threshold_gib} GiB threshold")


######################################################################
# Pattern 2: Export to JSON for Dashboards
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if HAS_CUDA:
    check_memory_regression(report, threshold_gib=8.0)

    with open("memory_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("Memory report exported to memory_report.json")

######################################################################
# Conclusion
# ==========
#
# This tutorial demonstrated three key use cases for Mosaic memory profiling:
#
# **Case 1: Activation Checkpointing Analysis**
#
# - Used Mosaic to compare memory usage between baseline and optimized models
# - Identified that activation checkpointing reduced activation memory by 71%
# - Mosaic's categorical profiling made it trivial to pinpoint memory savings
#
# **Case 2: Debugging Unexpected Memory Usage**
#
# - Created a "buggy" model with abandoned debug code
# - Used ``mosaic_get_memory_usage_peak`` to identify extra allocations
# - Stack traces revealed optimizer state tracking extra parameters
#
# **Case 3: Pipeline Integration**
#
# - Demonstrated programmatic usage via Mosaic's Python API
# - Showed integration patterns for CI/CD and dashboards with structured reports
#
# Further Reading
# ---------------
#
# * `Mosaic GitHub Repository <https://github.com/facebookresearch/mosaic>`_
# * `PyTorch Memory Management Documentation <https://pytorch.org/docs/stable/notes/cuda.html#memory-management>`_
# * `Understanding CUDA Memory Usage <https://pytorch.org/docs/stable/torch_cuda_memory.html>`_
# * `Activation Checkpointing in PyTorch <https://pytorch.org/docs/stable/checkpoint.html>`_
# * `PyTorch Memory Snapshot Visualizer <https://pytorch.org/memory_viz>`_
#
