Note

Go to the end
to download the full example code.

# Distributed training at scale with PyTorch and Ray Train

**Author:** [Ricardo Decal](https://github.com/crypdick)

This tutorial shows how to distribute PyTorch training across multiple GPUs
using Ray Train and Ray Data for scalable, production-ready model training.

 You will learn how to:

- Pre-train a GPT-2 (~124M-parameter) language model using PyTorch
and Hugging Face Transformers.
- Distribute training across multiple GPUs with Ray Train with minimal code changes.
- Stream training data from Hugging Face datasets with Ray Data's distributed workers.
- Save and load distributed checkpoints.
- Scale from a single node to a multinode cluster with minimal code changes.
- Optimize cost and performance with heterogeneous clusters.
- Monitor training with the Ray dashboard.

 Prerequisites

- PyTorch v2.9+.
- Ray Train (`ray[train]`) v2.52.1+.
- `tiktoken`, `datasets`, and `transformers` (Hugging Face).
- One or more GPUs are recommended but not required. This tutorial is tested on a `g4dn.12xlarge` instance, which has 4 NVIDIA T4 GPUs (16GB of memory per GPU).

[Ray Train](https://docs.ray.io/en/latest/train/train.html) is a
scalable framework for distributed deep learning.
Ray Train builds on top of [Ray](https://docs.ray.io/en/latest/index.html), a
unified framework for scaling AI and Python applications that
simplifies the complexities of distributed computing. Ray is also open source
and part of the PyTorch Foundation.

Ray Train enables you to
scale from a single GPU to hundreds of GPUs without rewriting your training
loop. Combined with [Ray Data](https://docs.ray.io/en/latest/data/data.html)
for streaming data ingestion, you get an end-to-end distributed training
pipeline that handles data loading, sharding, gradient synchronization,
checkpointing, and fault tolerance.

## Setup

To install the dependencies, run `pip install "ray[train]" torch tiktoken datasets transformers`.

Then, import the required libraries:

```
# Enable smoke test to run this tutorial quickly.

# Reduce Ray Data verbosity
```

## Load the dataset with Ray Data

This tutorial uses the [Wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext)
dataset, a collection of over 100 million tokens from verified Good and
Featured articles on Wikipedia.

The `ray.data.from_huggingface()` function converts a Hugging Face
dataset into a Ray Dataset, enabling distributed streaming and
preprocessing across all available nodes.

```
# Limit dataset size for fast iteration during smoke tests.
```

The schema can look like this:

```
Column Type
------ ----
text string
```

This means that the dataset has one column called `text` and it is a string.

### Inspect raw data

Use `take(n)` to fetch a small number of rows for inspection.
Each row is a dictionary with the column names as keys.

You'll see output like this:

```
Row 0: ''
Row 1: ' = Valkyria Chronicles III = '
```

Each row in Wikitext-103 is a single line from a Wikipedia article.
Consecutive rows belong to the same article, with empty rows separating
paragraphs. New articles begin with a title line like
`= Article Title =`. The tokenization step below inserts an
`<|endoftext|>` separator token before each title line so the model
learns to reset context at article boundaries.

## Tokenize and chunk the data

Language models consume fixed-length sequences of token IDs. The
preprocessing step converts raw text into token ID sequences for
next-token prediction.

This tutorial uses `tiktoken` with the GPT-2 encoding (vocabulary size
50,257). `tiktoken` is a fast, standalone tokenizer that has no
dependency on the Hugging Face `transformers` library.

The `tokenize_and_chunk` function does the following:

- Tokenizes each batch of text, concatenating into a single stream.
Article title lines (for example, `= Article Title =`) trigger an
`<|endoftext|>` separator so the model resets context at article
boundaries.
- Splits the stream into fixed-length blocks of `block_size` tokens.
- Returns `input_ids` for each block. During training, the same
tensor serves as both input and label because `GPT2LMHeadModel`
shifts the labels internally when computing the cross-entropy loss.

Apply the tokenization with `map_batches()`. This operation is **lazy**,
meaning that Ray Data defers execution until a downstream consumer requests the
results. Lazy execution lets Ray optimize the entire pipeline before any
work begins.

```
# These do not trigger execution.
```

Inspect the tokenized output with `take(2)`:

Each row now contains a fixed-length `input_ids` array of 256 tokens.

### Streaming execution

Internally, Ray divides the data into **blocks** and dispatches them to
workers. This block-based architecture enables **streaming execution**: as
soon as a stage outputs a block, the next stage can begin processing it
immediately without waiting for previous stages to finish the entire
dataset. This means the `map_batches` tokenization above runs in a
streaming pipeline with the training loop, so the full dataset never needs
to fit in memory at once.

When training starts, Ray Data logs the execution plan. For this tutorial
one possible plan is:

```
Execution plan: InputDataBuffer[Input]
 -> TaskPoolMapOperator[MapBatches(tokenize_and_chunk)]
 -> OutputSplitter[split(4, equal=True)]
```

This tells you exactly how Ray Data will stream through tokenization
and split the data across 4 trainer workers.

## Define the transformer model

The model is a decoder-only transformer language model using Hugging Face's
`GPT2LMHeadModel`. The hyperparameters below are for the standard GPT-2 "small" architecture.

Verify the model size:

You can see approximately 123.8M parameters.

## Define the distributed training function

The training function runs on each worker process. Ray Train
manages the distributed setup: it wraps the model in
`DistributedDataParallel`, shards the data across workers, and
synchronizes gradients automatically.

The key Ray Train integration points are:

- `ray.train.get_dataset_shard("train")` retrieves the worker's portion of the
dataset, and Ray Data automatically splits the dataset across all workers.
- `ray.train.torch.prepare_model(model)` wraps the model in
`DistributedDataParallel` and moves it to the correct GPU.
- `shard.iter_torch_batches(batch_size=...)` returns an iterator
of `dict[str, torch.Tensor]` batches, with tensors automatically placed on the worker's GPU. Setting `prefetch_batches=2` opportunistically fetches 2 batches ahead of the current batch.
- `ray.train.report(metrics, checkpoint=...)` reports metrics to the driver and saves a checkpoint.

## Configure and launch distributed training

The `TorchTrainer` brings everything together. Running `trainer.fit()` finally
triggers the execution of the full data pipeline and training loop. The Trainer accepts:

- `train_func_per_worker`: the function each worker executes.
- `train_loop_config`: a dictionary of hyperparameters forwarded
to the training function.
- `datasets`: a dictionary of Ray Datasets. Ray Train automatically
splits each dataset across workers.
- `scaling_config`: specifies the number of workers and whether to
use GPUs.

Setting `num_workers=4` launches 4 parallel workers, one per GPU. Ray
Train handles `torch.distributed` initialization, NCCL backend setup,
and `DistributedDataParallel` wrapping behind the scenes. In the logs,
you see each worker assigned a rank and device:

```
Started training worker group of size 4:

* (ip=10.0.176.183, pid=25636) world_rank=0, local_rank=0, node_rank=0
* (ip=10.0.176.183, pid=25637) world_rank=1, local_rank=1, node_rank=0
...
Moving model to device: cuda:0
Wrapping provided model in DistributedDataParallel.
```

`batch_size_per_worker` is the number of sequences each worker
processes per gradient step. With 4 workers and a per-worker batch size
of 16, the **effective global batch size** is 4 × 16 = 64 sequences,
or 64 × 256 = 4,096 tokens per optimizer step.

## Inspect results

After training, the `Result` object contains the final metrics and
checkpoint. `result.metrics` comes from the last
`ray.train.report()` call. `result.checkpoint` contains the
checkpoint from the last `ray.train.report()` call.

`result.metrics` contains the metrics dict from the last
`ray.train.report()` call:

```
{'train_loss': 7.0646, 'val_loss': 7.6051, 'epoch': 4,
 'epoch_time_sec': 12.34, 'epoch_tokens': 20480, 'tokens_per_sec': 1759.8}
```

The per-worker logs show training loss, validation loss, and throughput
metrics for each epoch. With random weights and only a few steps, expect
a high loss (~10-11).

### Checkpointing

In production training, you can enable checkpointing to make
your training jobs robust to unexpected failures. Checkpointing
permits you to take advantage of Ray Train's fault tolerance mechanisms described in the
Fault tolerance section.

Ray Train offers several checkpointing optimizations. Asynchronous
uploading enables you to continue training while checkpoints stream to remote storage
in the background.
Distributed checkpointing uploads shards from each worker in parallel, avoiding
a gather step into a single worker's memory that risks OOM errors for large models.

For a full guide on checkpointing with Ray Train, see the
[Ray Train checkpointing guide](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html).

## Scaling to a multi-node cluster

The code above runs on a single 4-GPU machine. Scaling to a multi-node
cluster requires only two changes:

1. **Increase ``num_workers``** to match the total number of GPUs in the cluster.
2. **Set a shared storage path** so that all nodes can access checkpoints.

For example, to train on a cluster of 4 nodes with 4 GPUs each
(16 GPUs total):

```
trainer = TorchTrainer(
 train_loop_per_worker=train_func_per_worker,
 train_loop_config={...},
 datasets={"train": train_ds, "validation": val_ds},
 scaling_config=ScalingConfig(
 num_workers=16, # 4 nodes x 4 GPUs
 use_gpu=True,
 ),
 run_config=RunConfig(
 # Shared storage accessible from all nodes
 storage_path="s3://my-bucket/ray-checkpoints",
 checkpoint_config=CheckpointConfig(num_to_keep=2),
 ),
)
```

Ray Train automatically:

- Launches workers across all available nodes, bringing up new nodes if needed in an autoscaling Ray cluster.
- Shards data across all workers.

No changes to the training function are needed. The same
`train_func_per_worker` runs identically whether on 1 GPU or 256 GPUs.

This tutorial uses `DistributedDataParallel` (DDP), which replicates
the full model on every GPU. For larger models that don't fit on a
single GPU, you can switch to
[FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html)
(FSDP) to shard parameters, gradients, and optimizer states across
workers by setting `prepare_model(parallel_strategy="fsdp")`.

### Heterogeneous clusters: separate data and training resources

Because Ray Data and Ray Train are separate systems, they don't have to
share the same machines. By default, Ray Data preprocessing and training
workers all run on the same nodes. However, you can optionally add
**CPU-only nodes** to your cluster and Ray Data automatically
schedules preprocessing tasks on them, keeping your expensive GPU nodes
free for training.

This is useful when data preprocessing is a bottleneck. If you notice
low GPU use because workers are waiting on data, you can add
cheaper CPU-only nodes to the cluster and Ray Data scales out
preprocessing to them.

For more information, see [Configuring data ingest](https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html).

## Fault tolerance

Long-running distributed training jobs are vulnerable to hardware
failures. These include hardware failures, network failures, or preemption.
Without fault tolerance, any of these events can force you to restart
training from scratch, wasting time and compute.

Ray Train has features that handle these failures automatically. When a worker process
crashes, Ray Train restarts it in place and resumes training. If an
entire node goes down, Ray Train provisions a replacement and
recovers from the most recent checkpoint so that only a small amount
of work is lost. This makes it practical to interrupt training jobs and resume
them later.

To enable automatic failure recovery, configure `FailureConfig` in
your `RunConfig`. The `max_failures` parameter controls how many
consecutive failures Ray Train tolerates before giving up:

```
from ray.train import FailureConfig

run_config = RunConfig(
 storage_path="s3://my-bucket/ray-checkpoints",
 failure_config=FailureConfig(max_failures=3),
 checkpoint_config=CheckpointConfig(num_to_keep=2),
)
```

For more details, see the [Ray Train fault tolerance guide](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html).

## Monitor your training jobs

Monitoring is critical when running distributed training.
The [Ray dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
displays real-time metrics including:

- Training loss and validation metrics per epoch
- GPU utilization and memory usage per worker
- Data loading throughput
- Worker status and error logs

To view the dashboard, open the link printed in the logs after Ray
initializes. Typically, this link is `http://localhost:8265`.

The dashboard lets you:

- Monitor training progress across all workers
- Inspect logs from individual workers
- Identify data loading or communication bottlenecks
- View resource use for CPU, GPU, and memory per worker
- Debug failures with detailed error messages and stack traces

For more information, see the [Ray Train monitoring
documentation](https://docs.ray.io/en/latest/train/user-guides/monitoring-logging.html).

## Conclusion

In this tutorial, you:

- Pre-trained a GPT-2 (~124M-parameter) language model using
Hugging Face Transformers and PyTorch.
- Loaded and preprocessed the Wikitext-103 dataset using Ray Data
with distributed streaming.
- Ran distributed training across 4 GPUs using Ray Train's
`TorchTrainer` with only minimal changes to a standard PyTorch
training loop.
- Learned how to save and load distributed checkpoints for model
recovery.
- Learned how to scale to multi-node clusters by changing
`ScalingConfig` and `RunConfig`.
- Learned how heterogeneous clusters let you run data preprocessing
on CPU nodes and training on GPU nodes for cost and performance
optimization.
- Learned about Ray Train's **fault tolerance** mechanisms for
production training jobs.
- Monitored training with the Ray dashboard.

## Further reading

- [Ray Train documentation](https://docs.ray.io/en/latest/train/train.html)
- [Ray Data for training](https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html)
- [Saving and loading checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [Ray Train fault tolerance](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html)
- [Ray cluster setup](https://docs.ray.io/en/latest/cluster/getting-started.html)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: distributed_training_with_ray_tutorial.ipynb`](../_downloads/98ac2c5de546105f5c566c256db2aaec/distributed_training_with_ray_tutorial.ipynb)

[`Download Python source code: distributed_training_with_ray_tutorial.py`](../_downloads/21b5b21c91510182086d4452006d94f6/distributed_training_with_ray_tutorial.py)

[`Download zipped: distributed_training_with_ray_tutorial.zip`](../_downloads/f1139ec4839789c284a04b5b8b0c1782/distributed_training_with_ray_tutorial.zip)