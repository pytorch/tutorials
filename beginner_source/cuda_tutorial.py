# -*- coding: utf-8 -*-

"""
Accelerating PyTorch with CUDA
==============================

**Author:** `Your Name <https://github.com/codomposer>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to check for CUDA availability in PyTorch
       * Moving tensors and models to GPU
       * Best practices for GPU training
       * Troubleshooting common CUDA issues
       * Performance comparisons between CPU and GPU

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.0.0
       * CUDA-compatible GPU (optional for understanding)
       * Basic PyTorch knowledge
       * Understanding of neural networks

This tutorial provides a comprehensive guide to CUDA acceleration in PyTorch.
You'll learn how to leverage GPUs for faster training and inference, with practical
examples and best practices for efficient GPU usage.

"""

#########################################################################
# Overview
# --------
#
# CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform
# that allows us to use GPUs for general purpose computing. PyTorch provides seamless
# integration with CUDA, making it easy to accelerate your deep learning workloads.
#
# This tutorial will walk you through:
# - Checking CUDA availability and system setup
# - Moving tensors and models between CPU and GPU
# - Training neural networks on GPU with performance comparisons
# - Best practices for GPU memory management and optimization
# - Troubleshooting common CUDA-related issues
# - Advanced topics like multi-GPU training

######################################################################
# Checking CUDA Availability
# --------------------------
#
# Before using CUDA, it's essential to verify that your system supports it.
# PyTorch provides several functions to check CUDA availability and get GPU information.

import torch
import time
import torch.nn as nn

# Check PyTorch and CUDA versions
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    # Get information about each GPU
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

    print(f"Current GPU device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. This tutorial will demonstrate concepts, but GPU acceleration won't be available.")

######################################################################
# Device Management
# -----------------
#
# PyTorch uses a device abstraction to handle CPU and GPU operations seamlessly.
# The torch.device object represents the device on which a tensor or model should be allocated.

# Create device object - this is the recommended way
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# You can also specify a particular GPU if you have multiple
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    device = torch.device('cuda:0')  # Use first GPU
    print(f"Using specific GPU: {device}")

######################################################################
# Moving Tensors to GPU
# ---------------------
#
# Tensors can be moved between CPU and GPU using several methods.
# The most common approaches are using .to() method or .cuda()/.cpu() methods.

# Create a tensor on CPU by default
x_cpu = torch.randn(1000, 1000)
print(f"Tensor device: {x_cpu.device}")
print(f"Tensor dtype: {x_cpu.dtype}")
print(f"Tensor shape: {x_cpu.shape}")

# Method 1: Using .cuda() - moves to current GPU
if torch.cuda.is_available():
    x_gpu = x_cpu.cuda()
    print(f"After .cuda(): {x_gpu.device}")

# Method 2: Using .to() - more flexible, recommended approach
x_gpu2 = x_cpu.to(device)  # Moves to the device object we created
print(f"After .to(device): {x_gpu2.device}")

# Method 3: Direct creation on GPU
if torch.cuda.is_available():
    x_gpu_direct = torch.randn(1000, 1000, device='cuda')
    print(f"Direct GPU creation: {x_gpu_direct.device}")

# Method 4: Using device argument in tensor creation
x_gpu_device = torch.randn(1000, 1000, device=device)
print(f"Created on device: {x_gpu_device.device}")

######################################################################
# Moving Models to GPU
# -------------------
#
# Neural networks and their parameters can also be moved to GPU.
# All parameters and buffers of the model need to be on the same device.

class SimpleModel(nn.Module):
    """
    A simple linear regression model for demonstration.

    This model consists of a single linear layer that maps input features
    to a single output value.
    """
    def __init__(self, input_size=10, output_size=1):
        """
        Initialize the model.

        Args:
            input_size (int): Number of input features
            output_size (int): Number of output features
        """
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        return self.linear(x)

# Create model on CPU
model_cpu = SimpleModel()
print(f"Model parameters on CPU: {next(model_cpu.parameters()).device}")

# Move entire model to GPU
model_gpu = model_cpu.to(device)
print(f"Model parameters on GPU: {next(model_gpu.parameters()).device}")

# Verify all parameters are on GPU
print("All parameters on same device:",
      all(param.device == device for param in model_gpu.parameters()))

######################################################################
# Performance Comparison: CPU vs GPU
# ----------------------------------
#
# Let's compare the performance of matrix operations on CPU vs GPU to demonstrate
# the speedup potential of CUDA.

def benchmark_operation(operation_name, operation_func, iterations=10):
    """
    Benchmark a PyTorch operation by running it multiple times and averaging.

    Args:
        operation_name (str): Name of the operation for display
        operation_func (callable): Function that performs the operation
        iterations (int): Number of times to run the operation

    Returns:
        float: Average time in seconds
    """
    times = []
    for _ in range(iterations):
        start_time = time.time()
        result = operation_func()
        # Synchronize GPU operations to get accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"{operation_name}: {avg_time:.4f} seconds (avg of {iterations} runs)")
    return avg_time

# Create large matrices for computation
size = 5000
A = torch.randn(size, size)
B = torch.randn(size, size)

# CPU operations
def cpu_matrix_mult():
    return torch.mm(A, B)

cpu_time = benchmark_operation("CPU Matrix Multiplication", cpu_matrix_mult)

# GPU operations
if torch.cuda.is_available():
    A_gpu = A.to(device)
    B_gpu = B.to(device)

    def gpu_matrix_mult():
        return torch.mm(A_gpu, B_gpu)

    gpu_time = benchmark_operation("GPU Matrix Multiplication", gpu_matrix_mult)
    speedup = cpu_time / gpu_time
    print(".2f")
else:
    print("GPU not available for comparison")

######################################################################
# GPU Training Example
# -------------------
#
# Here's a complete example of training a neural network on GPU with proper
# device management and performance monitoring.

def create_dataset(num_samples=1000, input_size=10):
    """
    Create a synthetic regression dataset.

    Args:
        num_samples (int): Number of data samples
        input_size (int): Number of input features

    Returns:
        tuple: (X, y) tensors
    """
    # Create random input features
    X = torch.randn(num_samples, input_size)

    # Create target with some linear relationship plus noise
    true_weights = torch.randn(input_size, 1)
    y = X @ true_weights + 0.1 * torch.randn(num_samples, 1)

    return X, y

# Create training data
X_train, y_train = create_dataset(1000, 10)

# Create validation data
X_val, y_val = create_dataset(200, 10)

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

# Create model and move to device
model = SimpleModel(input_size=10, output_size=1).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training parameters
num_epochs = 100
batch_size = 32
print_every = 20

print(f"Training on device: {device}")
print(f"Training data shape: {X_train.shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters

    # Validation every few epochs
    if (epoch + 1) % print_every == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        print(f'Epoch [{epoch+1:3d}/{num_epochs}], '
              f'Train Loss: {loss.item():.4f}, '
              f'Val Loss: {val_loss.item():.4f}')

# Final evaluation
model.eval()
with torch.no_grad():
    final_outputs = model(X_val)
    final_loss = criterion(final_outputs, y_val)
    print(f'\nFinal validation loss: {final_loss.item():.4f}')

######################################################################
# GPU Memory Management
# --------------------
#
# GPU memory is limited and must be managed carefully. PyTorch provides
# tools to monitor and manage GPU memory usage.

if torch.cuda.is_available():
    print("\nGPU Memory Information:")
    print(torch.cuda.memory_summary(device=device, abbreviated=True))

    # Check memory usage before and after creating large tensor
    print(f"Memory allocated before: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")

    large_tensor = torch.randn(10000, 10000, device=device)
    print(f"Memory allocated after: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")

    # Delete tensor and clear cache
    del large_tensor
    torch.cuda.empty_cache()
    print(f"Memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")

######################################################################
# Best Practices for GPU Training
# -------------------------------
#
# Here are some essential best practices for efficient GPU usage:

# 1. Use device-agnostic code
def create_model_and_data(device):
    """Create model and data in a device-agnostic way."""
    model = SimpleModel().to(device)
    X, y = create_dataset(100, 10)
    X, y = X.to(device), y.to(device)
    return model, X, y

# 2. Use DataLoader with pin_memory for faster GPU transfers
from torch.utils.data import TensorDataset, DataLoader

def create_data_loader(X, y, batch_size=32, pin_memory=False):
    """
    Create a DataLoader with optional pinned memory for faster GPU transfers.

    Args:
        X, y: Input and target tensors
        batch_size: Batch size for training
        pin_memory: Whether to pin memory for faster GPU transfers

    Returns:
        DataLoader: Configured data loader
    """
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size,
                       shuffle=True, pin_memory=pin_memory)
    return loader

# 3. Use gradient clipping to prevent exploding gradients
def train_with_gradient_clipping(model, dataloader, num_epochs=10, max_grad_norm=1.0):
    """
    Training loop with gradient clipping.

    Args:
        model: Neural network model
        dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        max_grad_norm: Maximum gradient norm for clipping
    """
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

# 4. Use mixed precision training for faster computation (if available)
try:
    from torch.cuda.amp import autocast, GradScaler

    def train_mixed_precision(model, dataloader):
        """Training with automatic mixed precision."""
        scaler = GradScaler()

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            # Use autocast for automatic mixed precision
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

            # Scale gradients for numerical stability
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    print("Mixed precision training is available")
except ImportError:
    print("Mixed precision training requires PyTorch >= 1.6")

######################################################################
# Common Issues and Solutions
# --------------------------
#
# **RuntimeError: CUDA out of memory**
# Solution: Reduce batch size, use gradient accumulation, or clear unnecessary tensors
def gradient_accumulation_training(model, dataloader, accumulation_steps=4):
    """
    Training with gradient accumulation to reduce memory usage.

    Args:
        model: Neural network model
        dataloader: DataLoader for training data
        accumulation_steps: Number of steps to accumulate gradients
    """
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    optimizer.zero_grad()
    for i, (batch_X, batch_y) in enumerate(dataloader):
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss = loss / accumulation_steps  # Normalize loss
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# **RuntimeError: Input and target tensors should be on the same device**
# Solution: Always ensure all tensors are on the same device
def safe_model_forward(model, x):
    """
    Safe forward pass that checks device compatibility.

    Args:
        model: Neural network model
        x: Input tensor

    Returns:
        Output tensor
    """
    # Ensure input is on same device as model
    if x.device != next(model.parameters()).device:
        x = x.to(next(model.parameters()).device)

    return model(x)

# **Slow training on GPU**
# Solutions:
# 1. Check if data loading is the bottleneck - use pin_memory=True
# 2. Use larger batch sizes when possible
# 3. Use mixed precision training
# 4. Profile your code with torch.profiler

######################################################################
# Multi-GPU Training (Advanced)
# ----------------------------
#
# For systems with multiple GPUs, PyTorch supports data parallelism.

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"\nMulti-GPU setup detected: {torch.cuda.device_count()} GPUs available")

    # DataParallel automatically splits data across GPUs
    model_multi = nn.DataParallel(SimpleModel())

    # Move to GPUs (DataParallel handles this)
    model_multi = model_multi.cuda()

    print("Model wrapped with DataParallel")
    print(f"Model device: {next(model_multi.parameters()).device}")
else:
    print("\nSingle GPU or CPU setup - DataParallel not demonstrated")

######################################################################
# Conclusion
# ----------
#
# In this comprehensive tutorial, we covered:
#
# - Checking CUDA availability and system configuration
# - Device management and tensor/model movement between CPU and GPU
# - Performance comparisons demonstrating GPU speedup
# - Complete GPU training example with validation
# - GPU memory management and monitoring
# - Best practices including device-agnostic code, data loading, and gradient clipping
# - Troubleshooting common CUDA issues with practical solutions
# - Advanced topics like mixed precision and multi-GPU training
#
# CUDA acceleration can dramatically speed up your PyTorch workflows, especially
# for large models and datasets. Always write device-agnostic code and follow
# memory management best practices for optimal performance and compatibility.

######################################################################
# Further Reading
# ---------------
#
# * `PyTorch CUDA Documentation <https://pytorch.org/docs/stable/cuda.html>`__
# * `CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`__
# * `GPU Memory Management <https://pytorch.org/docs/stable/cuda.html#memory-management>`__
# * `Mixed Precision Training <https://pytorch.org/docs/stable/amp.html>`__
# * `Multi-GPU Training <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`__
# * `PyTorch Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`__
