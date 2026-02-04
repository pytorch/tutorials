"""
Offline batch inference at scale with PyTorch and Ray Data
==========================================================

**Author:** `Ricardo Decal <https://github.com/crypdick>`__

This tutorial shows how to run batch inference using a pretrained PyTorch model
with Ray Data for scalable, production-ready data processing.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to create a production-ready PyTorch offline batch inference pipeline. 
         We will cover two use cases: batch predictions and batch embeddings.
       * How to scale the pipeline from your laptop to a cluster with thousands of nodes 
         and GPUs with no code changes.
       * How Ray Data can process data that is much larger than the cluster's shared memory.
       * How to configure resource allocation (CPU/GPU) and fractional resources.
       * How to measure and benchmark throughput for batch inference pipelines
       * How Ray Data can self-heal from failures with built-in fault tolerance.
       * How to monitor batch jobs with the Ray dashboard for real-time insights.

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.9+ and ``torchvision``
       * Ray Data (``ray[data]``) v2.52.1+
       * A GPU is recommended for higher throughput but is not required

`Ray Data <https://docs.ray.io/en/latest/data/data.html>`__ is a
scalable framework for data processing in production.
It's built on top of `Ray <https://docs.ray.io/en/latest/index.html>`__, a
unified framework for scaling AI and Python applications that
simplifies the complexities of distributed computing. Ray is also open-source
and part of the PyTorch Foundation.

Setup
-----

To install the dependencies:

"""

# pip install "ray[data]" torch torchvision

###############################################################################
# Start by importing the required libraries:

import os

import numpy as np
from PIL import Image
import ray
import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

###############################################################################
# Load the dataset with Ray Data
# ------------------------------
#
# Ray Data can read images directly from cloud storage (S3, GCS) or local paths.
# Here we use a subset of the ImageNette dataset hosted on S3:

s3_uri = "s3://anonymous@air-example-data-2/imagenette2/train/"

ds = ray.data.read_images(s3_uri, mode="RGB")
print(ds)

###############################################################################
# Under the hood, ``read_images()`` spreads the downloads across all available
# nodes, using all the network bandwidth available to the cluster.
#
# Ray divides the data into **blocks** and dispatches them to
# workers. This block-based architecture enables **streaming execution**: as soon
# as a stage outputs a block, the next stage can begin processing immediately it without
# waiting for previous stages to process the entire dataset. This allows you to utilize
# all your cluster's resources and evict intermediate data from the cluster's shared memory
# as soon as it's no longer needed, making room for more data to be processed.
#
# Ray Data provides useful methods to explore your data without loading it all into memory.
# The ``schema()`` method shows the column names and data types:

print(ds.schema())

###############################################################################
# The ``take_batch()`` method lets you copy a small sample for inspection:

sample_batch = ds.take_batch(5)
first_img = sample_batch["image"][0]
print(f"Image shape: {first_img.shape}")
img = Image.fromarray(first_img)
img.show()

###############################################################################
# Part 1: Batch Predictions
# =========================
#
# Define the preprocessing function
# ---------------------------------
#
# First, we define a preprocessing function that transforms raw images into preprocessed tensors.
# We will use the same preprocessing function that the model used during training. In this case,
# the EfficientNet preprocessing function includes resizing, normalization, and conversion to tensor.

weights = EfficientNet_V2_S_Weights.DEFAULT
preprocess = weights.transforms()


def preprocess_image(row: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Transform a raw image into a tensor suitable for the model."""
    # Convert numpy array to PIL Image for torchvision transforms
    pil_image = Image.fromarray(row["image"])
    # Apply the model's preprocessing transforms
    tensor = preprocess(pil_image)
    # Convert the tensor back to ndarray (a zero-copy operation since the tensor is on CPU).
    return {
        "original_image": row["image"],
        "transformed_image": tensor.numpy(),
    }



###############################################################################
# Apply the preprocessing function with ``ds.map()``. This operation is **lazy**,
# meaning that Ray Data will not begin this stage until a non-lazy operation
# demands the results (in this case, when ``ds.write_parquet()`` is called).
# Lazy execution allows Ray to intelligently optimize the entire pipeline
# before any work begins.

ds = ds.map(preprocess_image)
print(ds.schema())


###############################################################################
# Define the model class for batch inference
# ------------------------------------------
#
# For batch inference, we wrap our model in a class. By passing a class to
# ``map_batches()``, Ray creates **Actor** processes that recycle state between
# batches. The model loads once when the Actor starts and remains warm for all
# subsequent batches, avoiding repeated model initialization overhead.
#
# Separating preprocessing (CPU) from model inference (GPU) is a key pattern
# for high-throughput pipelines. This decoupling prevents GPUs from
# blocking on CPU work and allows you to scale stages independently
# or eliminate bottlenecks. Ray takes care of moving the data to a node
# with the appropriate resources if the current node doesn't have the
# required resources.

class Classifier:
    def __init__(self):
        self.weights = EfficientNet_V2_S_Weights.DEFAULT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = efficientnet_v2_s(weights=self.weights).to(self.device)
        self.model.eval()
        self.categories = np.array(self.weights.meta["categories"])

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on a batch of preprocessed images."""
        # Stack the preprocessed images into a batch tensor
        images = torch.tensor(batch["transformed_image"], device=self.device)

        with torch.inference_mode():
            # Process the whole batch at once
            logits = self.model(images)
            predictions = logits.argmax(dim=1).cpu().numpy()

        # Map class indices to human-readable labels
        predicted_labels = self.categories[predictions]

        return {
            "predicted_label": predicted_labels,
            "original_image": batch["original_image"],
        }



###############################################################################
# Configure resource allocation and scaling
# -----------------------------------------
#
# Ray Data allows you to specify **resource allocation** per worker, such as the
# number of CPUs or GPUs. Ray handles the orchestration of these resources across
# your cluster, automatically placing workers on nodes with available capacity. This
# means that when you move your workload from your laptop to a large cluster, you
# don't need to change your code since Ray will automatically detect the resources
# available in the cluster and scale the workload accordingly.
#
# This flexibility enables you to mix different node types into your cluster, such as
# different accelerators or CPU-only machines. This is useful for multi-modal workloads or
# when you want to optimize the hardware utilization of different stages of your pipeline.
#
# Ray also supports `fractional
# resource allocation <https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#fractional-accelerators>`__,
# allowing multiple workers to share a single GPU when models are small
# enough to fit in memory together.
#
# For example, on a cluster of 10 machines with 4 GPUs each, setting
# ``num_gpus=0.5`` would schedule 2 workers per GPU, giving you 80 workers
# across the cluster.

###############################################################################
# Run batch inference with map_batches
# ------------------------------------
#
# The ``map_batches()`` method applies our model to batches of data in parallel.
# This enables you to speed up stages of your pipeline that can benefit from vectorized operations,
# which GPUs are particularly good at parallelizing.
#
# The ``num_gpus`` parameter tells Ray to place each replica on a node with an
# available GPU. If a worker fails, Ray automatically restarts the task on
# another node with the required resources. The ``batch_size`` parameter tells Ray how many
# images to process at each invocation of the actor. If you run into CUDA out of memory errors,
# you can try reducing the ``batch_size``, increasing the ``num_gpus`` per worker, or
# using a GPU with more memory.

num_gpus_per_worker = 1  # Set to 0 for CPU-only
num_cpus_per_worker = 1

ds = ds.map_batches(
    Classifier,
    num_gpus=num_gpus_per_worker,
    num_cpus=num_cpus_per_worker,
    batch_size=128,  # Adjust based on available GPU memory
)

###############################################################################
# Inspect a few predictions:

prediction_batch = ds.take_batch(5)

for image, label in zip(prediction_batch["original_image"], prediction_batch["predicted_label"]):
    img = Image.fromarray(image)
    img.show()
    print(f"Prediction: {label}")



# Get the total number of images in the dataset
num_images = ds.count()
print(f"Total images in dataset: {num_images}")


###############################################################################
# Run the pipeline and save the predictions to disk
# -------------------------------------------------
#
# The ``write_parquet()`` method is a blocking operation that triggers the execution of the
# pipeline we defined above. As the pipeline streams results, the ``write_parquet()`` method
# writes them to shards. Sharding the results is desirable because afterwards you can read
# the shards in parallel. Writing to shared storage such as S3, GCS, or NFS is efficient because
# different workers can upload shards in parallel and utilizes your cluster's upload bandwidth.

# Write predictions to parquet to trigger execution
output_dir = os.path.join(os.getcwd(), "predictions")
os.makedirs(output_dir, exist_ok=True)

# Drop original images now that we've inspected them
ds = ds.drop_columns(["original_image"])
# Write predictions to parquet. This is a blocking operation that triggers the execution of the pipeline.
# ds.write_parquet(f"local://{output_dir}")
# print(f"Wrote {len(os.listdir(output_dir))} shards to {output_dir}")
ds.materialize()  # FIXME

###############################################################################
# Performance benchmarking
# ------------------------
#
# Measuring throughput is important for understanding how your batch inference
# performs at scale. Ray Data provides fine-grained execution statistics for both
# the overall pipeline as well as invidivual operations with the ``stats()`` method.

print("\nExecution statistics:")
print(ds.stats())

# Clear ds for the next example
del ds


###############################################################################
# For a single stage, the report looks like this:
#
# ```text
# Operator 3 Map(preprocess_image)->MapBatches(drop_columns): 58 tasks executed, 58 blocks produced in 9.65s
#
# * Remote wall time: 369.14ms min, 1.85s max, 634.59ms mean, 36.81s total
# * Remote cpu time: 369.57ms min, 696.42ms max, 551.0ms mean, 31.96s total
# * UDF time: 733.07ms min, 3.69s max, 1.26s mean, 73.33s total
# * Peak heap memory usage (MiB): 720.84 min, 1478.72 max, 1129 mean
# * Output num rows per block: 44 min, 54 max, 48 mean, 2794 total
# * Output size bytes per block: 77857120 min, 95551920 max, 85240122 mean, 4943927120 total
# * Output rows per task: 44 min, 54 max, 48 mean, 58 tasks used
# * Tasks per node: 8 min, 40 max, 19 mean; 3 nodes used
# * Operator throughput:
#         * Total input num rows: 3358 rows
#         * Total output num rows: 2794 rows
#         * Ray Data throughput: 289.43 rows/s
#         * Estimated single task throughput: 75.91 rows/s
# ```
#
# This information helps identify bottlenecks and optimize your pipeline.

###############################################################################
# Part 2: Batch Embeddings
# ========================
#
# Embeddings are dense vector representations useful for similarity search,
# clustering, and downstream ML tasks. To extract embeddings, we modify the
# model to return the features before the final classification layer.

###############################################################################
# Define the embedding model class
# --------------------------------
#
# The key modification is replacing the classifier head with an Identity layer,
# so the model outputs the penultimate layer's features instead of class logits.

class Embedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT).to(self.device)

        # Replace the classifier head with Identity to get embeddings
        self.model.classifier = torch.nn.Identity()
        self.model.eval()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Extract embeddings from a batch of preprocessed images."""
        # Stack the preprocessed images into a batch tensor
        images = torch.tensor(batch["transformed_image"], device=self.device)

        with torch.inference_mode():
            # Process the whole batch at once
            embeddings = self.model(images)
        # Return the embeddings as a numpy array
        return {"embedding": embeddings.cpu().numpy()}



###############################################################################
# Run batch embedding extraction:

ds = ray.data.read_images(s3_uri, mode="RGB")
# TODO: map batches version
ds = ds.map(preprocess_image)
ds = ds.drop_columns(["original_image"])
ds = ds.map_batches(
    Embedder,
    num_gpus=1,
    batch_size=16,
)

###############################################################################
# Inspect the embeddings:

embedding_batch = ds.take_batch(3)
print(f"Embedding shape: {embedding_batch['embedding'].shape}")
print(f"First embedding (truncated): {embedding_batch['embedding'][0][:10]}...")

###############################################################################
# Save embeddings to disk:

embeddings_output_dir = os.path.join(os.getcwd(), "embeddings")
os.makedirs(embeddings_output_dir, exist_ok=True)
ds.materialize()  # FIXME
# ds.write_parquet(f"local://{embeddings_output_dir}")
print(f"Embeddings saved to: {embeddings_output_dir}")

# Collect execution stats after write
print("\nExecution statistics for embeddings:")
print(ds.stats())


###############################################################################
# Fault Tolerance
# ---------------
#
# In production, process and machine failures are inevitable during long-running
# batch jobs. Ray Data is designed to handle failures gracefully and continue
# processing without losing progress.
#
# Ray Data provides several fault tolerance mechanisms:
#
# * **Backpressure**: Ray Data has multiple backpressure mechanisms to prevent a job from
#   exhausting the cluster's shared memory. For instance, Ray Data can detect if a stage
#   becomes a bottleneck, and throttle upstream stages to downstream to prevent queue buildup
#   and exhausting memory.
# * **Disk spilling**: If the cluster's shared memory is exhaused, Ray Data will spill data
#   from RAM to disk to prevent the job from failing due to out-of-memory errors.
# * **Task retry**: If a task fails (e.g., due to a network issue), Ray automatically
#   retries.
# * **Actor reconstruction**: If an actor crashes, Ray creates a new
#   actor and reassigns pending tasks to it.
# * **Lineage-based recovery**: Ray Data tracks the lineage of data transformations,
#   so if a node fails, will recompute the lost data rather than
#   than restarting the entire job.
#
# Ray Data can recover from larger infrastructure failures, such as entire nodes
# failing.

###############################################################################
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
# TODO: screenshots of the dashboard
#
# The dashboard lets you:
#
# * Monitor progress of your batch job in real time
# * Inspect logs from individual workers across the cluster
# * Identify bottlenecks in your data pipeline
# * View resource utilization (CPU, GPU, memory) per worker
# * Debug failures with detailed error messages and stack traces
#
# For debugging, Ray offers `distributed debugging
# tools <https://docs.ray.io/en/latest/ray-observability/index.html>`__
# that let you attach a debugger to running workers across the cluster.
# For more information, see the `Ray Data monitoring
# documentation <https://docs.ray.io/en/latest/data/monitoring-your-workload.html>`__.

###############################################################################
# Conclusion
# ----------
#
# In this tutorial, you learned how to:
#
# * Load image data with Ray Data from cloud storage using **distributed
#   ingestion** that leverages all nodes' network bandwidth
# * Explore datasets using ``schema()`` and ``take_batch()``
# * Separate CPU preprocessing from GPU inference to **maximize hardware
#   utilization** and enable independent scaling of each stage
# * Configure **resource allocation** and **fractional GPU usage** to
#   efficiently scale across heterogeneous clusters
# * Run scalable batch predictions with a pretrained EfficientNet model
# * Extract embeddings by modifying the model's classification head
# * Measure and benchmark throughput for batch inference pipelines
# * Understand Ray Data's **fault tolerance** mechanisms
# * Monitor batch jobs using the Ray dashboard
#
# The key advantage of Ray Data is that **the same code runs everywhere**:
# from a laptop to a multi-node cluster with heterogeneous GPU types. Ray
# handles parallelization, batching, resource management, and failure recovery
# automatically—you focus on your model and transformations while Ray handles
# the distributed systems complexity.

###############################################################################
# Further Reading
# ---------------
#
# Ray Data has more production features that are out of scope for this
# tutorial but are worth checking out:
#
# * `Custom aggregations <https://docs.ray.io/en/latest/data/aggregating-data.html#custom-aggregations>`__
# * `Integration with Ray Train <https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html>`__
#   to build end-to-end training and inference pipelines.
# * `Reading and writing custom file types <https://docs.ray.io/en/latest/data/custom-datasource-example.html>`__
