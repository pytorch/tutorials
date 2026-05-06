Note

Go to the end
to download the full example code.

# Introduction to TorchRec

**TorchRec** is a PyTorch library tailored for building scalable and efficient recommendation systems using embeddings.
This tutorial guides you through the installation process, introduces the concept of embeddings, and highlights their importance in
recommendation systems. It offers practical demonstrations on implementing embeddings with PyTorch
and TorchRec, focusing on handling large embedding tables through distributed training and advanced optimizations.

 What you will learn

- Fundamentals of embeddings and their role in recommendation systems
- How to set up TorchRec to manage and implement embeddings in PyTorch environments
- Explore advanced techniques for distributing large embedding tables across multiple GPUs

 Prerequisites

- PyTorch v2.5 or later with CUDA 11.8 or later
- Python 3.9 or later
- [FBGEMM](https://github.com/pytorch/fbgemm)

## Install Dependencies

Before running this tutorial in Google Colab, make sure to install the
following dependencies:

```
!pip3 install --pre torch --index-url https://download.pytorch.org/whl/cu121 -U
!pip3 install fbgemm_gpu --index-url https://download.pytorch.org/whl/cu121
!pip3 install torchmetrics==1.0.3
!pip3 install torchrec --index-url https://download.pytorch.org/whl/cu121
```

Note

If you are running this in Google Colab, make sure to switch to a GPU runtime type.
For more information,
see [Enabling CUDA](https://pytorch.org/tutorials/beginner/colab#enabling-cuda)

### Embeddings

When building recommendation systems, categorical features typically
have massive cardinality, posts, users, ads, and so on.

In order to represent these entities and model these relationships,
**embeddings** are used. In machine learning, **embeddings are a vectors
of real numbers in a high-dimensional space used to represent meaning in
complex data like words, images, or users**.

### Embeddings in RecSys

Now you might wonder, how are these embeddings generated in the first
place? Well, embeddings are represented as individual rows in an
**Embedding Table**, also referred to as embedding weights. The reason
for this is that embeddings or embedding table weights are trained just
like all of the other weights of the model via gradient descent!

Embedding tables are simply a large matrix for storing embeddings, with
two dimensions (B, N), where:

- B is the number of embeddings stored by the table
- N is the number of dimensions per embedding (N-dimensional embedding).

The inputs to embedding tables represent embedding lookups to retrieve
the embedding for a specific index or row. In recommendation systems, such
as those used in many large systems, unique IDs are not only used for
specific users, but also across entities like posts and ads to serve as
lookup indices to respective embedding tables!

Embeddings are trained in RecSys through the following process:

- **Input/lookup indices are fed into the model, as unique IDs**. IDs are
hashed to the total size of the embedding table to prevent issues when
the ID > number of rows
- Embeddings are then retrieved and **pooled, such as taking the sum or
mean of the embeddings**. This is required as there can be a variable number of
embeddings per example while the model expects consistent shapes.
- The **embeddings are used in conjunction with the rest of the model to
produce a prediction**, such as [Click-Through Rate
(CTR)](https://support.google.com/google-ads/answer/2615875?hl=en)
for an ad.
- The loss is calculated with the prediction and the label
for an example, and **all weights of the model are updated through
gradient descent and backpropagation, including the embedding weights**
that were associated with the example.

These embeddings are crucial for representing categorical features, such
as users, posts, and ads, in order to capture relationships and make
good recommendations. The [Deep learning recommendation
model](https://arxiv.org/abs/1906.00091) (DLRM) paper talks more
about the technical details of using embedding tables in RecSys.

This tutorial introduces the concept of embeddings, showcase
TorchRec specific modules and data types, and depict how distributed training
works with TorchRec.

#### Embeddings in PyTorch

In PyTorch, we have the following types of embeddings:

- [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding): An embedding table where forward pass returns the
embeddings themselves as is.
- [`torch.nn.EmbeddingBag`](https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag): Embedding table where forward pass returns
embeddings that are then pooled, for example, sum or mean, otherwise known
as **Pooled Embeddings**.

In this section, we will go over a very brief introduction to performing
embedding lookups by passing in indices into the table.

```
# Initialize our embedding table

# Pass in pre-generated weights just for example, typically weights are randomly initialized

# Print out the tables, we should see the same weights as above

# Lookup rows (ids for embedding ids) from the embedding tables
# 2D tensor with shape (batch_size, ids for each batch)

# Print out the embedding lookups
# You should see the specific embeddings be the same as the rows (ids) of the embedding tables above

# ``nn.EmbeddingBag`` default pooling is mean, so should be mean of batch dimension of values above

# ``nn.EmbeddingBag`` is the same as ``nn.Embedding`` but just with pooling (mean, sum, and so on)
# We can see that the mean of the embeddings of embedding_collection is the same as the output of the embedding_bag_collection
```

Congratulations! Now you have a basic understanding of how to use
embedding tables -- one of the foundations of modern recommendation
systems! These tables represent entities and their relationships. For
example, the relationship between a given user and the pages and posts
they have liked.

## TorchRec Features Overview

In the section above we've learned how to use embedding tables, one of the foundations of
modern recommendation systems! These tables represent entities and
relationships, such as users, pages, posts, etc. Given that these
entities are always increasing, a **hash** function is typically applied
to make sure the IDs are within the bounds of a certain embedding table.
However, in order to represent a vast amount of entities and reduce hash
collisions, these tables can become quite massive (think about the number of ads
for example). In fact, these tables can become so massive that they
won't be able to fit on 1 GPU, even with 80G of memory.

In order to train models with massive embedding tables, sharding these
tables across GPUs is required, which then introduces a whole new set of
problems and opportunities in parallelism and optimization. Luckily, we have
the TorchRec library <[https://docs.pytorch.org/torchrec/overview.html](https://docs.pytorch.org/torchrec/overview.html)>`__ that has encountered, consolidated, and addressed
many of these concerns. TorchRec serves as a **library that provides
primitives for large scale distributed embeddings**.

Next, we will explore the major features of the TorchRec
library. We will start with `torch.nn.Embedding` and will extend that to
custom TorchRec modules, explore distributed training environment with
generating a sharding plan for embeddings, look at inherent TorchRec
optimizations, and extend the model to be ready for inference in C++.
Below is a quick outline of what this section consists of:

- TorchRec Modules and Data Types
- Distributed Training, Sharding, and Optimizations

Let's begin with importing TorchRec:

This section goes over TorchRec Modules and data types including such
entities as `EmbeddingCollection` and `EmbeddingBagCollection`,
`JaggedTensor`, `KeyedJaggedTensor`, `KeyedTensor` and more.

### From `EmbeddingBag` to `EmbeddingBagCollection`

We have already explored [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding) and [`torch.nn.EmbeddingBag`](https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag).
TorchRec extends these modules by creating collections of embeddings, in
other words modules that can have multiple embedding tables, with
`EmbeddingCollection` and `EmbeddingBagCollection`
We will use `EmbeddingBagCollection` to represent a group of
embedding bags.

In the example code below, we create an `EmbeddingBagCollection` (EBC)
with two embedding bags, 1 representing **products** and 1 representing **users**.
Each table, `product_table` and `user_table`, is represented by a 64 dimension
embedding of size 4096.

Let's inspect the forward method for `EmbeddingBagCollection` and the
module's inputs and outputs:

```
# Let's look at the ``EmbeddingBagCollection`` forward method
# What is a ``KeyedJaggedTensor`` and ``KeyedTensor``?
```

### TorchRec Input/Output Data Types

TorchRec has distinct data types for input and output of its modules:
`JaggedTensor`, `KeyedJaggedTensor`, and `KeyedTensor`. Now you
might ask, why create new data types to represent sparse features? To
answer that question, we must understand how sparse features are
represented in code.

Sparse features are otherwise known as `id_list_feature` and
`id_score_list_feature`, and are the **IDs** that will be used as
indices to an embedding table to retrieve the embedding for that ID. To
give a very simple example, imagine a single sparse feature being Ads
that a user interacted with. The input itself would be a set of Ad IDs
that a user interacted with, and the embeddings retrieved would be a
semantic representation of those Ads. The tricky part of representing
these features in code is that in each input example, **the number of
IDs is variable**. One day a user might have interacted with only one ad
while the next day they interact with three.

A simple representation is shown below, where we have a `lengths`
tensor denoting how many indices are in an example for a batch and a
`values` tensor containing the indices themselves.

```
# Batch Size 2
# 1 ID in example 1, 2 IDs in example 2

# Values (IDs) tensor: ID 5 is in example 1, ID 7, 1 is in example 2
```

Next, let's look at the offsets as well as what is contained in each batch

```
# Lengths can be converted to offsets for easy indexing of values

# ``JaggedTensor`` is just a wrapper around lengths/offsets and values tensors!

# Automatically compute offsets from lengths

# Convert to list of values

# ``__str__`` representation

# ``JaggedTensor`` represents IDs for 1 feature, but we have multiple features in an ``EmbeddingBagCollection``
# That's where ``KeyedJaggedTensor`` comes in! ``KeyedJaggedTensor`` is just multiple ``JaggedTensors`` for multiple id_list_feature_offsets
# From before, we have our two features "product" and "user". Let's create ``JaggedTensors`` for both!

# Q1: How many batches are there, and which values are in the first batch for ``product_jt`` and ``user_jt``?

# Look at our feature keys for the ``KeyedJaggedTensor``

# Look at the overall lengths for the ``KeyedJaggedTensor``

# Look at all values for ``KeyedJaggedTensor``

# Can convert ``KeyedJaggedTensor`` to dictionary representation

# ``KeyedJaggedTensor`` string representation

# Q2: What are the offsets for the ``KeyedJaggedTensor``?

# Now we can run a forward pass on our ``EmbeddingBagCollection`` from before

# Result is a ``KeyedTensor``, which contains a list of the feature names and the embedding results

# The results shape is [2, 128], as batch size of 2. Reread previous section if you need a refresher on how the batch size is determined
# 128 for dimension of embedding. If you look at where we initialized the ``EmbeddingBagCollection``, we have two tables "product" and "user" of dimension 64 each
# meaning embeddings for both features are of size 64. 64 + 64 = 128

# Nice to_dict method to determine the embeddings that belong to each feature
```

Congrats! You now understand TorchRec modules and data types.
Give yourself a pat on the back for making it this far. Next, we will
learn about distributed training and sharding.

### Distributed Training and Sharding

Now that we have a grasp on TorchRec modules and data types, it's time
to take it to the next level.

Remember, the main purpose of TorchRec is to provide primitives for
distributed embeddings. So far, we've only worked with embedding tables
on a single device. This has been possible given how small the embedding tables
have been, but in a production setting this isn't generally the case.
Embedding tables often get massive, where one table can't fit on a single
GPU, creating the requirement for multiple devices and a distributed
environment.

In this section, we will explore setting up a distributed environment,
exactly how actual production training is done, and explore sharding
embedding tables, all with TorchRec.

**This section will also only use 1 GPU, though it will be treated in a
distributed fashion. This is only a limitation for training, as training
has a process per GPU. Inference does not run into this requirement**

In the example code below, we set up our PyTorch distributed environment.

Warning

If you are running this in Google Colab, you can only call this cell once,
calling it again will cause an error as you can only initialize the process
group once.

```
# Set up environment variables for distributed training
# RANK is which GPU we are on, default 0

# How many devices in our "world", colab notebook can only handle 1 process

# Localhost as we are training locally

# Port for distributed training

# nccl backend is for GPUs, gloo is for CPUs
```

### Distributed Embeddings

We have already worked with the main TorchRec module:
`EmbeddingBagCollection`. We have examined how it works along with how
data is represented in TorchRec. However, we have not yet explored one
of the main parts of TorchRec, which is **distributed embeddings**.

GPUs are the most popular choice for ML workloads by far today, as they
are able to do magnitudes more floating point operations/s
([FLOPs](https://en.wikipedia.org/wiki/FLOPS)) than CPU. However,
GPUs come with the limitation of scarce fast memory (HBM which is
analogous to RAM for CPU), typically, ~10s of GBs.

A RecSys model can contain embedding tables that far exceed the memory
limit for 1 GPU, hence the need for distribution of the embedding tables
across multiple GPUs, otherwise known as **model parallel**. On the
other hand, **data parallel** is where the entire model is replicated on
each GPU, which each GPU taking in a distinct batch of data for
training, syncing gradients on the backwards pass.

Parts of the model that **require less compute but more memory
(embeddings) are distributed with model parallel** while parts that
**require more compute and less memory (dense layers, MLP, etc.) are
distributed with data parallel**.

### Sharding

In order to distribute an embedding table, we split up the embedding
table into parts and place those parts onto different devices, also
known as "sharding".

There are many ways to shard embedding tables. The most common ways are:

- Table-Wise: the table is placed entirely onto one device
- Column-Wise: columns of embedding tables are sharded
- Row-Wise: rows of embedding tables are sharded

### Sharded Modules

While all of this seems like a lot to deal with and implement, you're in
luck. **TorchRec provides all the primitives for easy distributed
training and inference**! In fact, TorchRec modules have two corresponding
classes for working with any TorchRec module in a distributed
environment:

- **The module sharder**: This class exposes a `shard` API
that handles sharding a TorchRec Module, producing a sharded module.
* For `EmbeddingBagCollection`, the sharder is `EmbeddingBagCollectionSharder `
- **Sharded module**: This class is a sharded variant of a TorchRec module.
It has the same input/output as a the regular TorchRec module, but much
more optimized and works in a distributed environment.
* For `EmbeddingBagCollection`, the sharded variant is ShardedEmbeddingBagCollection

Every TorchRec module has an unsharded and sharded variant.

- The unsharded version is meant to be prototyped and experimented with.
- The sharded version is meant to be used in a distributed environment for
distributed training and inference.

The sharded versions of TorchRec modules, for example
`EmbeddingBagCollection`, will handle everything that is needed for Model
Parallelism, such as communication between GPUs for distributing
embeddings to the correct GPUs.

Refresher of our `EmbeddingBagCollection` module

```
# Corresponding sharder for ``EmbeddingBagCollection`` module

# ``ProcessGroup`` from torch.distributed initialized 2 cells above
```

### Planner

Before we can show how sharding works, we must know about the
**planner**, which helps us determine the best sharding configuration.

Given a number of embedding tables and a number of ranks, there are many
different sharding configurations that are possible. For example, given
2 embedding tables and 2 GPUs, you can:

- Place 1 table on each GPU
- Place both tables on a single GPU and no tables on the other
- Place certain rows and columns on each GPU

Given all of these possibilities, we typically want a sharding
configuration that is optimal for performance.

That is where the planner comes in. The planner is able to determine
given the number of embedding tables and the number of GPUs, what is the optimal
configuration. Turns out, this is incredibly difficult to do manually,
with tons of factors that engineers have to consider to ensure an
optimal sharding plan. Luckily, TorchRec provides an auto planner when
the planner is used.

The TorchRec planner:

- Assesses memory constraints of hardware
- Estimates compute based on memory fetches as embedding lookups
- Addresses data specific factors
- Considers other hardware specifics like bandwidth to generate an optimal sharding plan

In order to take into consideration all these variables, The TorchRec
planner can take in [various amounts of data for embedding tables,
constraints, hardware information, and
topology](https://github.com/pytorch/torchrec/blob/main/torchrec/distributed/planner/planners.py#L147-L155)
to aid in generating the optimal sharding plan for a model, which is
routinely provided across stacks.

To learn more about sharding, see our [sharding
tutorial](https://pytorch.org/tutorials/advanced/sharding.html).

```
# In our case, 1 GPU and compute on CUDA device

# Run planner to get plan for sharding
```

### Planner Result

As you can see above, when running the planner there is quite a bit of output.
We can see a lot of stats being calculated along with where our
tables end up being placed.

The result of running the planner is a static plan, which can be reused
for sharding! This allows sharding to be static for production models
instead of determining a new sharding plan everytime. Below, we use the
sharding plan to finally generate our `ShardedEmbeddingBagCollection`.

```
# The static plan that was generated

# Shard the ``EmbeddingBagCollection`` module using the ``EmbeddingBagCollectionSharder``
```

## GPU Training with `LazyAwaitable`

Remember that TorchRec is a highly optimized library for distributed
embeddings. A concept that TorchRec introduces to enable higher
performance for training on GPU is a
LazyAwaitable `.
You will see ``LazyAwaitable` types as outputs of various sharded
TorchRec modules. All a `LazyAwaitable` type does is delay calculating some
result as long as possible, and it does it by acting like an async type.

```
# Demonstrate a ``LazyAwaitable`` type:

# The output of our sharded ``EmbeddingBagCollection`` module is an `Awaitable`?

# Now we have our ``KeyedTensor`` after calling ``.wait()``
# If you are confused as to why we have a ``KeyedTensor ``output,
# give yourself a refresher on the unsharded ``EmbeddingBagCollection`` module

# Same output format as unsharded ``EmbeddingBagCollection``
```

### Anatomy of Sharded TorchRec modules

We have now successfully sharded an `EmbeddingBagCollection` given a
sharding plan that we generated! The sharded module has common APIs from
TorchRec which abstract away distributed communication/compute amongst
multiple GPUs. In fact, these APIs are highly optimized for performance
in training and inference. **Below are the three common APIs for
distributed training/inference** that are provided by TorchRec:

- `input_dist`: Handles distributing inputs from GPU to GPU.
- `lookups`: Does the actual embedding lookup in an optimized,
batched manner using FBGEMM TBE (more on this later).
- `output_dist`: Handles distributing outputs from GPU to GPU.

The distribution of inputs and outputs is done through [NCCL
Collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html),
namely
[All-to-Alls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#all-to-all),
which is where all GPUs send and receive data to and from one another.
TorchRec interfaces with PyTorch distributed for collectives and
provides clean abstractions to the end users, removing the concern for
the lower level details.

The backwards pass does all of these collectives but in the reverse
order for distribution of gradients. `input_dist`, `lookup`, and
`output_dist` all depend on the sharding scheme. Since we sharded in a
table-wise fashion, these APIs are modules that are constructed by
TwPooledEmbeddingSharding.

```
# Distribute input KJTs to all other GPUs and receive KJTs

# Distribute output embeddings to all other GPUs and receive embeddings
```

### Optimizing Embedding Lookups

In performing lookups for a collection of embedding tables, a trivial
solution would be to iterate through all the `nn.EmbeddingBags` and do
a lookup per table. This is exactly what the standard, unsharded
`EmbeddingBagCollection` does. However, while this solution
is simple, it is extremely slow.

[FBGEMM](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu) is a
library that provides GPU operators (otherwise known as kernels) that
are very optimized. One of these operators is known as **Table Batched
Embedding** (TBE), provides two major optimizations:

- Table batching, which allows you to look up multiple embeddings with
one kernel call.
- Optimizer Fusion, which allows the module to update itself given the
canonical pytorch optimizers and arguments.

The `ShardedEmbeddingBagCollection` uses the FBGEMM TBE as the lookup
instead of traditional `nn.EmbeddingBags` for optimized embedding
lookups.

### `DistributedModelParallel`

We have now explored sharding a single `EmbeddingBagCollection`! We were
able to take the `EmbeddingBagCollectionSharder` and use the unsharded
`EmbeddingBagCollection` to generate a
`ShardedEmbeddingBagCollection` module. This workflow is fine, but
typically when implementing model parallel,
DistributedModelParallel
(DMP) is used as the standard interface. When wrapping your model (in
our case `ebc`), with DMP, the following will occur:

1. Decide how to shard the model. DMP will collect the available
sharders and come up with a plan of the optimal way to shard the
embedding table(s) (for example, `EmbeddingBagCollection`)
2. Actually shard the model. This includes allocating memory for each
embedding table on the appropriate device(s).

DMP takes in everything that we've just experimented with, like a static
sharding plan, a list of sharders, etc. However, it also has some nice
defaults to seamlessly shard a TorchRec model. In this toy example,
since we have two embedding tables and one GPU, TorchRec will place both
on the single GPU.

### Sharding Best Practices

Currently, our configuration is only sharding on 1 GPU (or rank), which
is trivial: just place all the tables on 1 GPUs memory. However, in real
production use cases, embedding tables are **typically sharded on
hundreds of GPUs**, with different sharding methods such as table-wise,
row-wise, and column-wise. It is incredibly important to determine a
proper sharding configuration (to prevent out of memory issues) while
keeping it balanced not only in terms of memory but also compute for
optimal performance.

### Adding in the Optimizer

Remember that TorchRec modules are hyperoptimized for large scale
distributed training. An important optimization is in regards to the
optimizer.

TorchRec modules provide a seamless API to fuse the
backwards pass and optimize step in training, providing a significant
optimization in performance and decreasing the memory used, alongside
granularity in assigning distinct optimizers to distinct model
parameters.

## Optimizer Classes

TorchRec uses `CombinedOptimizer`, which contains a collection of
`KeyedOptimizers`. A `CombinedOptimizer` effectively makes it easy
to handle multiple optimizers for various sub groups in the model. A
`KeyedOptimizer` extends the `torch.optim.Optimizer` and is
initialized through a dictionary of parameters exposes the parameters.
Each `TBE` module in a `EmbeddingBagCollection` will have it's own
`KeyedOptimizer` which combines into one `CombinedOptimizer`.

## Fused optimizer in TorchRec

Using `DistributedModelParallel`, the **optimizer is fused, which
means that the optimizer update is done in the backward**. This is an
optimization in TorchRec and FBGEMM, where the optimizer embedding
gradients are not materialized and applied directly to the parameters.
This brings significant memory savings as embedding gradients are
typically size of the parameters themselves.

You can, however, choose to make the optimizer `dense` which does not
apply this optimization and let's you inspect the embedding gradients or
apply computations to it as you wish. A dense optimizer in this case
would be your [canonical PyTorch model training loop with
optimizer.](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

Once the optimizer is created through `DistributedModelParallel`, you
still need to manage an optimizer for the other parameters not
associated with TorchRec embedding modules. To find the other
parameters,
use `in_backward_optimizer_filter(model.named_parameters())`.
Apply an optimizer to those parameters as you would a normal Torch
optimizer and combine this and the `model.fused_optimizer` into one
`CombinedOptimizer` that you can use in your training loop to
`zero_grad` and `step` through.

## Adding an Optimizer to `EmbeddingBagCollection`

We will do this in two ways, which are equivalent, but give you options
depending on your preferences:

1. Passing optimizer kwargs through `fused_params` in sharder.
2. Through `apply_optimizer_in_backward`, which converts the optimizer
parameters to `fused_params` to pass to the `TBE` in the `EmbeddingBagCollection` or `EmbeddingCollection`.

```
# Option 1: Passing optimizer kwargs through fused parameters

# We initialize the sharder with

# Initialize sharder with ``fused_params``

# We'll use same plan and unsharded EBC as before but this time with our new sharder

# Looking at the optimizer of each, we can see that the learning rate changed, which indicates our optimizer has been applied correctly.
# If seen, we can also look at the TBE logs of the cell to see that our new optimizer is indeed being applied

# Option 2: Applying optimizer through apply_optimizer_in_backward
# Note: we need to call apply_optimizer_in_backward on unsharded model first and then shard it

# We can achieve the same result as we did in the previous

# Now when we print the optimizer, we will see our new learning rate, you can verify momentum through the TBE logs as well if outputted

# We can also check through the filter other parameters that aren't associated with the "fused" optimizer(s)
# Practically, just non TorchRec module parameters. Since our module is just a TorchRec EBC
# there are no other parameters that aren't associated with TorchRec

# Here we do a dummy backwards call and see that parameter updates for fused
# optimizers happen as a result of the backward pass

# We don't call an optimizer.step(), so for the loss to have changed here,
# that means that the gradients were somehow updated, which is what the
# fused optimizer automatically handles for us
```

## Conclusion

In this tutorial, you have done training a distributed RecSys model
If you are interested in the inference the [TorchRec repo](https://github.com/pytorch/torchrec/tree/main/torchrec/inference) has a
full example of how to run the TorchRec in Inference mode.

For more information, please see our
[dlrm](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/)
example, which includes multinode training on the Criteo 1TB
dataset using the methods described in [Deep Learning Recommendation Model
for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091).

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: torchrec_intro_tutorial.ipynb`](../_downloads/8d83c5b10438a4bcc94963daaddeaeec/torchrec_intro_tutorial.ipynb)

[`Download Python source code: torchrec_intro_tutorial.py`](../_downloads/b07b6a647a3bf9e6882df8ca2cc20e8b/torchrec_intro_tutorial.py)

[`Download zipped: torchrec_intro_tutorial.zip`](../_downloads/d02270f901ca302f496cc62a6fbdb225/torchrec_intro_tutorial.zip)