"""
Offline batch inference at scale with PyTorch and Ray Data
==========================================================

**Author:** `Ricardo Decal <https://github.com/crypdick>`__

This tutorial shows how to run batch inference using a pretrained PyTorch model
with Ray Data for scalable, production-ready data processing.


.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` You will learn how to:
       :class-card: card-prerequisites

       * Create a production-ready PyTorch offline batch inference pipeline.
       * Scale the pipeline from your laptop to a cluster with thousands of nodes
         and GPUs with no code changes.
       * Use Ray Data to process data that is much larger than the cluster's shared memory.
       * Configure resource allocation (CPU/GPU) and fractional resources.
       * Measure and benchmark throughput for batch inference pipelines.
       * Use Ray Data fault tolerance to self-heal from failures.
       * Monitor batch jobs with the Ray dashboard.

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.9+ and ``torchvision``.
       * Ray Data (``ray[data]``) v2.52.1+.
       * A GPU is recommended for higher throughput but is not required.

`Ray Data <https://docs.ray.io/en/latest/data/data.html>`__ is a
scalable framework for data processing in production.
Ray Data builds on top of `Ray <https://docs.ray.io/en/latest/index.html>`__, a
unified framework for scaling AI and Python applications that
simplifies the complexities of distributed computing. Ray is also open source
and part of the PyTorch Foundation.

Setup
-----

To install the dependencies, run ``pip install "ray[data]" torch torchvision``.

"""

###############################################################################
# Start by importing the required libraries:

import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ray
import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


# Reduce Ray Data verbosity
ray.data.DataContext.get_current().enable_progress_bars = False
ray.data.DataContext.get_current().print_on_execution_start = False

###############################################################################
# Load the dataset with Ray Data
# ------------------------------
#
# Ray Data can read image files directly from cloud storage such as Amazon S3 and Google Cloud Platform (GCP) Storage, or from local paths. This tutorial uses a subset of the ImageNette dataset stored on S3:

s3_uri = "s3://anonymous@air-example-data-2/imagenette2/train/"

ds = ray.data.read_images(s3_uri, mode="RGB")
print(ds)

###############################################################################
# Behind the scenes, ``read_images()`` spreads the downloads across all available
# nodes, using all the network bandwidth available to the cluster.
#
# Ray divides the data into **blocks** and dispatches them to
# workers. This block-based architecture enables **streaming execution**: as soon
# as a stage outputs a block, the next stage can begin processing it immediately without
# waiting for previous stages to process the entire dataset. This is key to Ray Data's efficiency,
# because it prevents hardware from sitting idle
# or parking intermediate data in memory waiting for processing.
#
# Ray Data provides useful methods to explore your data without loading it all into memory. For example, the ``schema()`` method shows the column names and data types:

print(ds.schema())

###############################################################################
# The ``take_batch()`` method lets you copy a small sample for inspection:

sample_batch = ds.take_batch(5)
first_img_array = sample_batch["image"][0]
print(f"Image shape: {first_img_array.shape}")
plt.imshow(first_img_array)

###############################################################################
# Part 1: Batch predictions
# =========================
#
# Define the preprocessing function
# ---------------------------------
#
# First, define a preprocessing function that transforms raw input image files into preprocessed tensors.
# Use the same preprocessing function that the model used during training. In this case,
# the EfficientNet preprocessing function includes resizing, normalization, and conversion to tensor.

weights = EfficientNet_V2_S_Weights.DEFAULT
preprocess = weights.transforms()


def preprocess_image(row: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Transform a raw image into a tensor suitable for the model."""
    # Convert numpy array to a PIL image for torchvision transforms
    pil_image = Image.fromarray(row["image"])
    # Apply the model's preprocessing transforms
    tensor = preprocess(pil_image)
    # Convert the tensor back to ndarray, a zero-copy operation since the tensor is on CPU.
    return {
        "original_image": row["image"],
        "transformed_image": tensor.numpy(),
    }



###############################################################################
# Apply the preprocessing function with ``ds.map()``. This operation is **lazy**,
# meaning that Ray Data doesn't begin this stage until a non-lazy operation
# demands the results, such as when ``ds.write_parquet()`` runs.
# Lazy execution lets Ray intelligently optimize the entire pipeline
# before any work begins.

ds = ds.map(preprocess_image)
print(ds.schema())


###############################################################################
# The schema of the dataset shows that there are two columns: "original_image" and "transformed_image",
# both of which are tensor arrays. The "transformed_image" should be cropped into a square.
#
# Define the model class for batch inference
# ------------------------------------------
#
# For batch inference, wrap the model in a class. By passing a class to
# ``map_batches()``, Ray creates **Actor** processes that recycle state between
# batches. The model loads once when the Actor starts and remains warm for all
# subsequent batches, avoiding repeated model initialization overhead.
#
# Separating preprocessing (CPU) from model inference (GPU) is a key pattern
# for high-throughput pipelines. This decoupling prevents GPUs from
# blocking on CPU work and lets you scale stages independently
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
# Ray Data lets you specify **resource allocation** per worker, such as the
# number of CPUs or GPUs. Ray handles the orchestration of these resources across
# your cluster, automatically placing workers on nodes with available capacity. This
# means that scaling a batch inference job from a laptop to a large cluster doesn't require code changes, since Ray automatically detects the resources available in the cluster and scales the job accordingly.
#
# This flexibility enables you to mix different node types into your cluster, such as
# different accelerators or CPU-only machines. This is useful for multi-modal data pipelines or
# when you want to optimize the hardware use of different stages of your pipeline.
#
# Ray also supports `fractional
# resource allocation <https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#fractional-accelerators>`__,
# letting multiple workers share a single GPU when models are small
# enough to fit in memory together.
#
# For example, on a cluster of 10 machines with 4 GPUs each, setting
# ``num_gpus=0.5`` schedules 2 workers per GPU, resulting in 80 workers
# across the cluster.

###############################################################################
# Run batch inference with map_batches
# ------------------------------------
#
# The ``map_batches()`` method applies the model to batches of data in parallel.
# This speeds up stages of your pipeline that can benefit from vectorized operations,
# which GPUs are particularly good at parallelizing.
#
# The ``num_gpus`` parameter tells Ray to place each replica on a node with an
# available GPU. If a worker fails, Ray automatically restarts the task on
# another node with the required resources. The ``batch_size`` parameter tells Ray how many
# images to process at each invocation of the actor. If you run into CUDA out-of-memory errors,
# try reducing the ``batch_size``, increasing the ``num_gpus`` per worker, or
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

for img_array, label in zip(prediction_batch["original_image"], prediction_batch["predicted_label"]):
    img = Image.fromarray(img_array)
    img.show()
    print(f"Prediction: {label}")



# Get the total number of input images in the dataset
num_images = ds.count()
print(f"Total images in dataset: {num_images}")


###############################################################################
# Run the pipeline and save the predictions to disk
# -------------------------------------------------
#
# The ``write_parquet()`` method is a blocking operation that triggers the execution of the
# pipeline defined earlier. As the pipeline streams results, the ``write_parquet()`` method
# writes them to shards. Sharding the results is desirable because afterwards you can read
# the shards in parallel. Writing to shared storage such as Amazon S3, Google Cloud Platform (GCP) Storage, or network file systems such as Network File System (NFS) is efficient because
# different workers can upload shards in parallel and use your cluster's upload bandwidth.

# Write predictions to Parquet to trigger execution
output_dir = os.path.join(os.getcwd(), "predictions")
os.makedirs(output_dir, exist_ok=True)

# Drop original image data now that we've inspected it
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
# the overall pipeline and individual operations with the ``stats()`` method.

print("\nExecution statistics:")
print(ds.stats())

# Clear ds for the next example
del ds


###############################################################################
# For a single stage, the report looks like this:
#
# ```markdown
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
# Part 2: Batch embeddings
# ========================
#
# Embeddings are dense vector representations useful for similarity search,
# clustering, and downstream ML tasks. To extract embeddings, modify the
# model to return the features before the final classification layer.

###############################################################################
# Define the embedding model class
# --------------------------------
#
# ML models can also extract internal representations of the data. These representations, sometimes called embeddings, latent representations,
# or features, are a compressed representation of the data that distills the semantic meaning of the data into a lower-dimensional space. These
# representations are useful for similarity search, clustering, and other ML tasks.
#
# To extract the penultimate layer's features, replace the model's final classification head with an identity layer. This layer is
# essentially a no-op that passes the data through unchanged.

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
print(f"Embedding batch shape: {embedding_batch['embedding'].shape}")
print(f"First embedding vector: {embedding_batch['embedding'][0][:10]}...")

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
# Fault tolerance
# ---------------
#
# In production, machine failures are inevitable during long-running
# batch jobs. Ray Data handles failures gracefully and continues
# processing without losing progress.
#
# Ray Data provides several fault tolerance mechanisms:
#
# * **Backpressure**: Ray Data has multiple backpressure mechanisms to prevent a job from
#   exhausting the cluster's shared memory. For example, Ray Data can detect if a stage
#   becomes a bottleneck and throttle upstream stages to prevent queue buildup
#   and memory exhaustion.
# * **Disk spilling**: If the cluster's shared memory runs out, Ray Data spills data
#   from RAM to disk to prevent the job from failing due to out-of-memory errors.
# * **Task retry**: If a task fails (for example, due to a network issue), Ray automatically
#   retries.
# * **Actor reconstruction**: If an actor crashes, Ray creates a new
#   actor and reassigns pending tasks to it.
# * **Lineage-based recovery**: Ray Data tracks the lineage of data transformations,
#   so if a node fails, Ray recomputes the lost data rather than
#   restarting the entire job.
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
# rates. It also shows cluster resource usage for CPU, GPU, and memory and overall
# job health.
#
# To view the dashboard, open the link printed in the logs after Ray initializes.
# Typically, this link is
# ``http://localhost:8265``.
#
# TODO: Add screenshots of the Ray dashboard.
#
# The dashboard lets you:
#
# * Monitor the progress of your batch job
# * Inspect logs from individual workers across the cluster
# * Identify bottlenecks in your data pipeline
# * View resource usage for CPU, GPU, and memory per worker
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
# In this tutorial, you:
#
# * Loaded image data with Ray Data from cloud storage using **distributed
#   ingestion** that leverages all nodes' network bandwidth.
# * Explored datasets using ``schema()`` and ``take_batch()``.
# * Separated CPU preprocessing from GPU inference to independently scale
#   each stage, eliminating bottlenecks and maximizing hardware use.
# * Configured **resource allocation** to
#   efficiently scale across heterogeneous clusters.
# * Ran scalable batch predictions with a pretrained EfficientNet model.
# * Extracted embeddings by modifying the model's classification head.
# * Measured and benchmarked throughput for batch inference pipelines.
# * Learned about Ray Data's **fault tolerance** mechanisms.
# * Monitored batch jobs using the Ray dashboard.
#
# Ray Data handles the complexity of distributed systems and resource allocation
# so that you can focus on defining your data pipeline.

###############################################################################
# Further reading
# ---------------
#
# Ray Data has more production features that are out of scope for this
# tutorial but are worth checking out:
#
# * `Custom aggregations <https://docs.ray.io/en/latest/data/aggregating-data.html#custom-aggregations>`__
# * `Integration with Ray Train <https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html>`__
#   to build end-to-end training and inference pipelines.
# * `Reading and writing custom file types <https://docs.ray.io/en/latest/data/custom-datasource-example.html>`__
