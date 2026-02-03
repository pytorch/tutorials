"""
Offline batch inference at scale with PyTorch and Ray Data
==========================================================

**Author:** `Ricardo Decal <https://github.com/crypdick>`__

This tutorial shows how to run batch inference using a pretrained PyTorch model
with Ray Data for scalable, production-ready data processing.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to create a production-ready PyTorch offline batch inference pipeline. We will cover two use cases: batch predictions and batch embeddings.
       * How to scale the pipeline from your laptop to a cluster with thousands of nodes and GPUs with no code changes.
       * Configure resource allocation (CPU/GPU) per worker, including fractional GPU usage
       * Measure and benchmark throughput for batch inference pipelines
       * How Ray Data can self-heal from failures with built-in fault tolerance
       * Monitor batch jobs with the Ray dashboard for real-time insights

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.9+ and ``torchvision``
       * Ray Data (``ray[data]``) v2.52.1+
       * A GPU is recommended for higher throughput but is not required

`Ray Data <https://docs.ray.io/en/latest/data/data.html>`__ is a
scalable framework for data processing in production.
It's built on top of `Ray <https://docs.ray.io/en/latest/index.html>`__,
which is a unified framework for scaling AI and Python applications that
simplifies the complexities of distributed computing. Ray is also open
source and part of the PyTorch Foundation.


Setup
-----

To install the dependencies:

"""

# %%bash
# pip install "ray[data]" torch torchvision

######################################################################
# Start by importing the required libraries:

import os
import time

import numpy as np
from PIL import Image
import ray
import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

######################################################################
# Load the dataset with Ray Data
# ------------------------------
#
# Ray Data can read images directly from cloud storage (S3, GCS) or local paths.
# Here we use a subset of the ImageNette dataset hosted on S3:

s3_uri = "s3://anonymous@air-example-data-2/imagenette2/train/n01440764/"

ds = ray.data.read_images(s3_uri, mode="RGB")
print(ds)

######################################################################
# Under the hood, ``read_images()`` reads the data **lazily** and distributes
# the work across all available nodes. This approach leverages every node's
# network bandwidth and starts processing immediately without waiting for
# the entire dataset to download.
#
# After loading, Ray divides the data into **blocks** and dispatches them to
# workers. This block-based architecture enables streaming execution: as soon
# as a block finishes one stage, it can move to the next without waiting for
# the entire dataset.

######################################################################
# Ray Data provides useful methods to explore your data without loading it all into memory.
# The ``schema()`` method shows the column names and data types:

print(ds.schema())

######################################################################
# The ``take_batch()`` method lets you grab a small sample to inspect:

sample_batch = ds.take_batch(5)
print(f"Batch keys: {sample_batch.keys()}")
print(f"Image shape: {sample_batch['image'][0].shape}")

######################################################################
# Let's visualize one of the images:

img = Image.fromarray(sample_batch["image"][0])
img.show()

######################################################################
# Part 1: Batch Predictions
# =========================
#
# Define the preprocessing function
# ---------------------------------
#
# First, we define a preprocessing function that transforms raw images into tensors.
# This function operates on individual rows and will be applied lazily via ``ds.map()``.

weights = EfficientNet_V2_S_Weights.DEFAULT
preprocess = weights.transforms()


def preprocess_image(row: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Transform a raw image into a tensor suitable for the model."""
    # Convert numpy array to PIL Image for torchvision transforms
    pil_image = Image.fromarray(row["image"])
    # Apply the model's preprocessing transforms and convert back to numpy
    tensor = preprocess(pil_image)
    return {
        "original_image": row["image"],
        "transformed_image": tensor.numpy(),
    }


######################################################################
# Apply the preprocessing with ``ds.map()``. This operation is **lazy**—Ray
# Data only executes the transformation when downstream operations demand
# the results. Lazy execution allows Ray to optimize the entire pipeline
# before any work begins.
#
# ``ds.map()`` applies the transformation to each record in parallel across
# the cluster. Whenever possible, Ray avoids transferring objects across
# network connections to take advantage of **zero-copy reads**, avoiding
# serialization and deserialization overhead.

ds = ds.map(preprocess_image)
print(ds.schema())

######################################################################
# Define the model class for batch inference
# ------------------------------------------
#
# For batch inference, we wrap our model in a class. By passing a class to
# ``map_batches()``, Ray creates **Actor** processes that recycle state between
# batches. The model loads once when the Actor starts and remains warm for all
# subsequent batches—avoiding repeated model initialization overhead.
#
# Separating preprocessing (CPU) from model inference (GPU) is a key pattern
# for high-throughput pipelines. This **decoupling** prevents GPUs from
# blocking on CPU work and allows you to scale each stage independently
# based on where your bottlenecks are.

class EfficientNetClassifier:
    """A callable class for batch image classification with EfficientNet."""

    def __init__(self):
        self.weights = EfficientNet_V2_S_Weights.DEFAULT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = efficientnet_v2_s(weights=self.weights).to(self.device)
        self.model.eval()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on a batch of preprocessed images."""
        # Stack the preprocessed images into a batch tensor
        images = torch.tensor(batch["transformed_image"], device=self.device)

        with torch.inference_mode():
            logits = self.model(images)
            predictions = logits.argmax(dim=1).cpu().numpy()

        # Map class indices to human-readable labels
        categories = self.weights.meta["categories"]
        predicted_labels = np.array([categories[idx] for idx in predictions])

        return {
            "predicted_label": predicted_labels,
            "original_image": batch["original_image"],
        }


######################################################################
# Configure resource allocation and scaling
# -----------------------------------------
#
# Ray Data allows you to specify **resource allocation** per worker, such as the
# number of CPUs or GPUs. Ray handles the orchestration of these resources across
# your cluster, automatically placing workers on nodes with available capacity.
# This **heterogeneous compute** support lets you mix different node types
# (CPU-only machines, different GPU models) in the same cluster, and Ray
# schedules work appropriately.
#
# Ray also supports `fractional
# GPUs <https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#fractional-accelerators>`__,
# allowing multiple workers to share a single GPU when models are small
# enough to fit in memory together.
#
# For example, on a cluster of 10 machines with 4 GPUs each, setting
# ``num_gpus=0.5`` would schedule 2 workers per GPU, giving you 80 workers
# across the cluster. The same code that runs on your laptop with a single GPU
# scales to this multi-node setup with only a configuration change—no code
# modifications required.
#
# Run batch inference with map_batches
# ------------------------------------
#
# The ``map_batches()`` method applies our model to batches of data in parallel.
# Key parameters:
#
# - ``compute``: Use ``ActorPoolStrategy`` for GPU inference to maintain persistent workers
# - ``num_gpus``: GPUs per model replica (set to 0 for CPU-only)
# - ``num_cpus``: CPUs per worker (useful for CPU-intensive preprocessing)
# - ``batch_size``: Number of images per batch (tune based on GPU memory)
#
# The ``num_gpus`` parameter tells Ray to place each replica on a node with an
# available GPU. If a worker fails, Ray automatically restarts the task on
# another node with the required resources.

num_gpus_per_worker = 1  # Set to 0 for CPU-only
num_cpus_per_worker = 1
num_workers = 2  # Number of parallel workers

ds = ds.map_batches(
    EfficientNetClassifier,
    num_gpus=num_gpus_per_worker,
    num_cpus=num_cpus_per_worker,
    batch_size=16,  # Adjust based on available GPU memory
)

######################################################################
# Inspect the predictions:

prediction_batch = ds.take_batch(5)

for image, label in zip(prediction_batch["original_image"], prediction_batch["predicted_label"]):
    img = Image.fromarray(image)
    img.show()
    print(f"Prediction: {label}")



# Get the total number of images in the dataset
num_images = ds.count()
print(f"Total images in dataset: {num_images}")


######################################################################
# Save predictions to disk
# ------------------------
#
# Write results to Parquet format for downstream processing. The
# ``write_parquet()`` call triggers execution of the pipeline and streams
# results to disk as they become available.
#
# Ray Data automatically shards the output into multiple files for efficient
# parallel reads in downstream steps. For distributed workloads writing to
# shared storage (S3, GCS, NFS), all workers write in parallel.


# Write predictions to parquet to trigger execution
output_dir = os.path.join(os.getcwd(), "predictions")
os.makedirs(output_dir, exist_ok=True)

# Drop original images now that we've inspected them
ds = ds.drop_columns(["original_image"])
# Write predictions to parquet. This is a blocking operation that triggers the execution of the pipeline.
start_time = time.time()
# ds.write_parquet(f"local://{output_dir}")
ds.materialize()  # FIXME

######################################################################
# Performance benchmarking
# ------------------------
#
# Measuring throughput is important for understanding how your batch inference
# performs at scale. Ray Data provides built-in execution stats that show
# processing rates, resource utilization, and bottlenecks.
#
# Note that Ray Data uses **streaming execution**: blocks flow through the
# pipeline as soon as they're ready, rather than waiting for entire stages
# to complete. This means the first results appear quickly even on large
# datasets, and memory usage stays bounded since intermediate data doesn't
# accumulate.

elapsed = time.time() - start_time

print(f"Processed {num_images} images in {elapsed:.2f} seconds")
print(f"Throughput: {num_images/elapsed:.2f} images/second")

# Display execution stats after write completes
print("\nExecution statistics:")
print(ds.stats())

# Clear ds for the next example
del ds

######################################################################
# The stats show important metrics like:
#
# - Wall time and CPU time per operation
# - Peak memory usage
# - Data throughput (MB/s)
# - Number of tasks and blocks processed
#
# This information helps identify bottlenecks and optimize your pipeline.
# For example, if preprocessing is slow, you might increase ``num_cpus`` or
# optimize your preprocessing function.



######################################################################
# Part 2: Batch Embeddings
# ========================
#
# Embeddings are dense vector representations useful for similarity search,
# clustering, and downstream ML tasks. To extract embeddings, we modify the
# model to return the features before the final classification layer.
#
# Define the embedding model class
# --------------------------------
#
# The key modification is replacing the classifier head with an Identity layer,
# so the model outputs the penultimate layer's features instead of class logits.

class EfficientNetEmbedder:
    """A callable class for extracting embeddings from EfficientNet."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT).to(self.device)

        # Replace the classifier head with Identity to get embeddings
        # EfficientNet_v2_s has a classifier attribute: Sequential(Dropout, Linear)
        # The Linear layer outputs 1000 classes from 1280-dim features
        self.model.classifier = torch.nn.Identity()

        self.model.eval()
        self.embedding_dim = 1280  # EfficientNet_v2_s feature dimension

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Extract embeddings from a batch of preprocessed images."""
        images = torch.tensor(batch["transformed_image"], device=self.device)

        with torch.inference_mode():
            embeddings = self.model(images).cpu().numpy()

        return {
            "embedding": embeddings,
        }


######################################################################
# Run batch embedding extraction:

ds = ray.data.read_images(s3_uri, mode="RGB")
ds = ds.map(preprocess_image)
ds = ds.drop_columns(["original_image"])
ds = ds.map_batches(
    EfficientNetEmbedder,
    num_gpus=1,
    batch_size=16,
)

######################################################################
# Inspect the embeddings:

embedding_batch = ds.take_batch(3)
print(f"Embedding shape: {embedding_batch['embedding'].shape}")
print(f"First embedding (truncated): {embedding_batch['embedding'][0][:10]}...")

######################################################################
# Save embeddings to disk:

embeddings_output_dir = os.path.join(os.getcwd(), "embeddings")
os.makedirs(embeddings_output_dir, exist_ok=True)
ds.materialize()  # FIXME
# ds.write_parquet(f"local://{embeddings_output_dir}")
print(f"Embeddings saved to: {embeddings_output_dir}")

# Collect execution stats after write
print("\nExecution statistics for embeddings:")
print(ds.stats())


######################################################################
# Fault Tolerance
# ---------------
#
# In production, process and machine failures are inevitable during long-running
# batch jobs. Ray Data is designed to handle failures gracefully and continue
# processing without losing progress.
#
# Ray Data provides several fault tolerance mechanisms:
#
# - **Task retry**: If a task fails (e.g., due to an out-of-memory error or
#   network issue), Ray automatically retries it on another worker.
# - **Actor reconstruction**: If a worker actor crashes, Ray creates a new
#   actor and reassigns pending tasks to it.
# - **Lineage-based recovery**: Ray tracks the lineage of data transformations,
#   so if a node fails, only the lost partitions need to be recomputed rather
#   than restarting the entire job.
#
# Ray Data can recover from larger infrastructure failures, such as entire nodes
# failing. For very large batch jobs, you can enable checkpointing to save
# intermediate results and resume from the last checkpoint if the job fails.
#
# For more information about Ray Data's fault tolerance, see the
# `Ray Data fault tolerance guide <https://docs.ray.io/en/latest/data/fault-tolerance.html>`__.

######################################################################
# Monitor your batch jobs
# -----------------------
#
# Monitoring is critical when running large-scale batch inference. The `Ray
# dashboard <https://docs.ray.io/en/latest/ray-observability/getting-started.html>`__
# displays Ray Data metrics like processing throughput, task status, and error
# rates. It also shows cluster resource usage (CPU, GPU, memory) and overall
# job health in real time.
#
# To access the dashboard:
#
# 1. Start Ray with ``ray start --head`` (if running on a cluster)
# 2. Open your browser to ``http://localhost:8265`` (default port)
# 3. Navigate to the "Jobs" tab to see your Ray Data job
# 4. Click on the job to see detailed metrics and task execution timeline
#
# The dashboard lets you:
#
# - Monitor progress of your batch job in real time
# - Inspect logs from individual workers across the cluster
# - Identify bottlenecks in your data pipeline
# - View resource utilization (CPU, GPU, memory) per worker
# - Debug failures with detailed error messages and stack traces
#
# For debugging, Ray offers `distributed debugging
# tools <https://docs.ray.io/en/latest/ray-observability/index.html>`__
# that let you attach a debugger to running workers across the cluster.
# For more information, see the `Ray Data monitoring
# documentation <https://docs.ray.io/en/latest/data/monitoring-your-workload.html>`__.

######################################################################
# Conclusion
# ----------
#
# In this tutorial, you learned how to:
#
# - Load image data with Ray Data from cloud storage using **distributed
#   ingestion** that leverages all nodes' network bandwidth
# - Explore datasets using ``schema()`` and ``take_batch()``
# - Separate CPU preprocessing from GPU inference to **maximize hardware
#   utilization** and enable independent scaling of each stage
# - Configure **resource allocation** and **fractional GPU usage** to
#   efficiently scale across heterogeneous clusters
# - Run scalable batch predictions with a pretrained EfficientNet model
# - Extract embeddings by modifying the model's classification head
# - Measure and benchmark throughput for batch inference pipelines
# - Understand Ray Data's **fault tolerance** mechanisms
# - Monitor batch jobs using the Ray dashboard
#
# The key advantage of Ray Data is that **the same code runs everywhere**:
# from a laptop to a multi-node cluster with heterogeneous GPU types. Ray
# handles parallelization, batching, resource management, and failure recovery
# automatically—you focus on your model and transformations while Ray handles
# the distributed systems complexity.
#
# Further Reading
# ---------------
#
# Ray Data has more production features that are out of scope for this
# tutorial but are worth checking out:
#
# - `Streaming batch inference <https://docs.ray.io/en/latest/data/batch_inference.html#streaming-batch-inference>`__
#   for processing datasets larger than cluster memory by streaming blocks through
#   the pipeline with bounded memory usage.
# - `Advanced preprocessing <https://docs.ray.io/en/latest/data/transforming-data.html>`__
#   with vectorized operations and custom partitioning strategies for optimal performance.
# - `Integration with Ray Train <https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html>`__
#   to build end-to-end training and inference pipelines.
# - `Checkpointing and resuming <https://docs.ray.io/en/latest/data/saving-data.html#resuming-from-failures>`__
#   for very large batch jobs that may span multiple hours or days.
# - `Custom data sources <https://docs.ray.io/en/latest/data/creating-datasets.html>`__
#   to read from databases, APIs, or custom file formats.
#
# For more information, see the `Ray Data
# documentation <https://docs.ray.io/en/latest/data/data.html>`__ and
# `Ray Data examples <https://docs.ray.io/en/latest/data/examples/index.html>`__.
