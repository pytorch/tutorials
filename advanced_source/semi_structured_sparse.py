# -*- coding: utf-8 -*-

"""
(prototype) Accelerating BERT with semi-structured (2:4) sparsity
=================================================================

**Author**: `Jesse Cai <https://github.com/jcaip>`_

Like other forms of sparsity, **semi-structured sparsity** is a model
optimization technique that seeks to reduce the memory overhead and
latency of a neural network at the expense of some model accuracy. It is
also known as **fine-grained structured sparsity** or **2:4 structured
sparsity**.

Semi-structured sparsity derives its name from its unique sparsity
pattern, where n out of every 2n elements are pruned. We most often see
n=2, hence 2:4 sparsity Semi-structured sparsity is particularly
interesting because it can be efficiently accelerated on GPUs and
doesn’t degrade model accuracy as much as other sparsity patterns.

With the introduction of
`semi-structured sparsity support <https://pytorch.org/docs/2.1/sparse.html#sparse-semi-structured-tensors>`_,
it is possible to prune and accelerate a semi-structured sparse model
without leaving PyTorch. We will explain this process in this tutorial.

.. image:: ../../_static/img/pruning_flow.jpg

By the end of this tutorial, we will have sparsified a BERT
question-answering model to be 2:4 sparse, fine-tuning it to recover
nearly all F1 loss (86.92 dense vs 86.48 sparse). Finally, we will
accelerate this 2:4 sparse model for inference, yielding a 1.3x speedup.

Requirements
------------

-  PyTorch >= 2.1.
-  A NVIDIA GPU with semi-structured sparsity support (Compute
   Capability 8.0+).

This tutorial is designed for beginners to semi-structured sparsity and
sparsity in general. For users with existing 2:4 sparse models,
accelerating ``nn.Linear`` layers for inference with
``to_sparse_semi_structured`` is quite straightforward. Here is an example: 

"""

import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
SparseSemiStructuredTensor._FORCE_CUTLASS = True

# mask Linear weight to be 2:4 sparse
mask = torch.Tensor([0, 0, 1, 1]).tile((3072, 2560)).cuda().bool()
linear = torch.nn.Linear(10240, 3072).half().cuda().eval()
linear.weight = torch.nn.Parameter(mask * linear.weight)

x = torch.rand(3072, 10240).half().cuda()

with torch.inference_mode():
    dense_output = linear(x)
    dense_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # accelerate via SparseSemiStructuredTensor
    linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))

    sparse_output = linear(x)
    sparse_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # sparse and dense matmul are numerically equivalent
    # On an A100 80GB, we see: `Dense: 0.870ms Sparse: 0.630ms | Speedup: 1.382x`
    assert torch.allclose(sparse_output, dense_output, atol=1e-3)
    print(f"Dense: {dense_t:.3f}ms Sparse: {sparse_t:.3f}ms | Speedup: {(dense_t / sparse_t):.3f}x")


######################################################################
# What problem does semi-structured sparsity solve?
# -------------------------------------------------
# 
# The general motivation behind sparsity is simple: if there are zeros in
# your network, you can optimize efficiency by not storing or computing those
# parameters. However, the specifics of sparsity are tricky. Zeroing out
# parameters doesn’t affect the latency / memory overhead of our model out
# of the box.
# 
# This is because the dense tensor still contains the pruned (zero)
# elements, which the dense matrix multiplication kernel will still
# operate on this elements. In order to realize performance gains, we need
# to swap out dense kernels for sparse kernels, which skip calculation
# involving pruned elements.
# 
# To do this, these kernels work on sparse matrices, which do not store
# the pruned elements and store the specified elements in a compressed
# format.
# 
# For semi-structured sparsity, we store exactly half of the original
# parameters along with some compressed metadata about how the elements
# were arranged.
# 
# .. image:: https://developer-blogs.nvidia.com/wp-content/uploads/2023/06/2-4-structured-sparsity-pattern.png
#    :align: center :width: 80%
# 
#    Image sourced from `NVIDIA blog post <https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/>`_ on semi-structured sparsity.
# 
# There are many different sparse layouts, each with their own benefits
# and drawbacks. The 2:4 semi-structured sparse layout is particularly
# interesting for two reasons:
# 
# * Unlike previous sparse formats,
# semi-structured sparsity was designed to be efficiently accelerated on
# GPUs. In 2020, NVIDIA introduced hardware support for semi-structured
# sparsity with their Ampere architecture, and have also released fast
# sparse kernels via
# CUTLASS `cuSPARSELt <https://docs.nvidia.com/cuda/cusparselt/index.html>`__.
# * At the same time, semi-structured sparsity tends to have a milder
# impact on model accuracy compared to other sparse formats, especially
# when accounting for more advanced pruning / fine-tuning methods. NVIDIA
# has shown in their `white paper <https://arxiv.org/abs/2104.08378>`_
# that a simple paradigm of magnitude pruning once to be 2:4 sparse and
# then retraining the model yields nearly identical model accuracies.
# 
# Semi-structured exists in a sweet spot, providing a 2x (theoretical)
# speedup at a much lower sparsity level (50%), while still being granular
# enough to preserve model accuracy.
# 
# +---------------------+-------------+--------+------------+-------------+
# | Network             | Data Set    | Metric | Dense FP16 | Sparse FP16 |
# +=====================+=============+========+============+=============+
# | ResNet-50           | ImageNet    | Top-1  | 76.1       | 76.2        |
# +---------------------+-------------+--------+------------+-------------+
# | ResNeXt-101_32x8d   | ImageNet    | Top-1  | 79.3       | 79.3        |
# +---------------------+-------------+--------+------------+-------------+
# | Xception            | ImageNet    | Top-1  | 79.2       | 79.2        |
# +---------------------+-------------+--------+------------+-------------+
# | SSD-RN50            | COCO2017    | bbAP   | 24.8       | 24.8        |
# +---------------------+-------------+--------+------------+-------------+
# | MaskRCNN-RN50       | COCO2017    | bbAP   | 37.9       | 37.9        |
# +---------------------+-------------+--------+------------+-------------+
# | FairSeq Transformer | EN-DE WMT14 | BLEU   | 28.2       | 28.5        |
# +---------------------+-------------+--------+------------+-------------+
# | BERT-Large          | SQuAD v1.1  | F1     | 91.9       | 91.9        |
# +---------------------+-------------+--------+------------+-------------+
# 
# Semi-structured sparsity has an additional advantage from a workflow
# perspective. Because the sparsity level is fixed at 50%, it is easier to
# decompose the problem of sparsifying a model into two distinct
# subproblems:
# 
# -  Accuracy - How can we find a set of 2:4 sparse weights that minimize
#    the accuracy degradation of our model?
# -  Performance - How can we accelerate our 2:4 sparse weights for
#    inference and reduced memory overhead?
# 
# .. math::
# 
#    \begin{bmatrix}
#       1 & 1 & 0 & 0 \\
#       0 & 0 & 1 & 1 \\
#       1 & 0 & 0 & 0 \\
#       0 & 0 & 1 & 1 \\
#       \end{bmatrix}
# 
# The natural handoff point between these two problems are zeroed-out
# dense tensors. Our inference solution is designed to compress and
# accelerate tensors in this format. We anticipate many users coming up
# with custom masking solution, as this is an active area of research.
# 
# Now that we’ve learned a little more about semi-structured sparsity,
# let’s apply it to a BERT model trained on a question answering task,
# SQuAD.
# 
# Intro & Setup
# -------------
# 
# Let’s start by importing all the packages we need.
# 

!pip install datasets transformers evaluate accelerate pandas

import collections
import datasets
import evaluate
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.ao.pruning import WeightNormSparsifier
import transformers

# force CUTLASS use if ``cuSPARSELt`` is not available
SparseSemiStructuredTensor._FORCE_CUTLASS = True
torch.manual_seed(100)


######################################################################
# We’ll also need to define some helper functions that are specific to the
# dataset / task at hand. These were adapted from
# ``this <https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt>``\ \_
# Hugging Face course as a reference.
# 

def preprocess_validation_function(examples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def preprocess_train_function(examples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, (offset, answer) in enumerate(zip(offset_mapping, answers)):
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30
    metric = evaluate.load("squad")

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    # for example in ``tqdm`` (examples):
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0
                    # or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


######################################################################
# Now that those are defined, we just need one additional helper function,
# which will help us benchmark our model.
# 

def measure_execution_time(model, batch_sizes, dataset):
    dataset_for_model = dataset.remove_columns(["example_id", "offset_mapping"])
    dataset_for_model.set_format("torch")
    batch_size_to_time_sec = {}
    for batch_size in batch_sizes:
        batch = {
            k: dataset_for_model[k][:batch_size].cuda()
            for k in dataset_for_model.column_names
        }

        with torch.no_grad():
            baseline_predictions = model(**batch)
            timer = benchmark.Timer(
                stmt="model(**batch)", globals={"model": model, "batch": batch}
            )
            p50 = timer.blocked_autorange().median * 1000
            batch_size_to_time_sec[batch_size] = p50

            model_c = torch.compile(model, fullgraph=True)
            timer = benchmark.Timer(
                stmt="model(**batch)", globals={"model": model_c, "batch": batch}
            )
            p50 = timer.blocked_autorange().median * 1000
            batch_size_to_time_sec[f"{batch_size}_compile"] = p50
            new_predictions = model_c(**batch)
            
    return batch_size_to_time_sec



######################################################################
# We will get started by loading our model and tokenizer, and then setting
# up our dataset.
# 

# load model
model_name = "bert-base-cased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
print(f"Loading tokenizer: {model_name}")
print(f"Loading model: {model_name}")

# set up train and val dataset
squad_dataset = datasets.load_dataset("squad")
tokenized_squad_dataset = {}
tokenized_squad_dataset["train"] = squad_dataset["train"].map(
    lambda x: preprocess_train_function(x, tokenizer), batched=True
)
tokenized_squad_dataset["validation"] = squad_dataset["validation"].map(
    lambda x: preprocess_validation_function(x, tokenizer),
    batched=True,
    remove_columns=squad_dataset["train"].column_names,
)
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)


######################################################################
# Establishing a baseline
# =======================
# 
# Next, we’ll train a quick baseline of our model on SQuAD. This task asks
# our model to identify spans, or segments of text, in a given context
# (Wikipedia articles) that answer a given question. Running the following
# code gives me an F1 score of 86.9. This is quite close to the reported
# NVIDIA score and the difference is likely due to BERT-base
# vs. BERT-large or fine-tuning hyperparameters.
# 

training_args = transformers.TrainingArguments(
    "trainer",
    num_train_epochs=1,
    lr_scheduler_type="constant",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=512,
    logging_steps=50, 
)

trainer = transformers.Trainer(
    model,
    training_args,
    train_dataset=tokenized_squad_dataset["train"],
    eval_dataset=tokenized_squad_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# batch sizes to compare for eval
batch_sizes = [4, 16, 64, 256]
# 2:4 sparsity require fp16, so we cast here for a fair comparison
with torch.autocast("cuda"):
    with torch.no_grad():
        predictions = trainer.predict(tokenized_squad_dataset["validation"])
        start_logits, end_logits = predictions.predictions
        fp16_baseline = compute_metrics(
            start_logits,
            end_logits,
            tokenized_squad_dataset["validation"],
            squad_dataset["validation"],
        )
        fp16_time = measure_execution_time(
            model,
            batch_sizes,
            tokenized_squad_dataset["validation"],
        )

print("fp16", fp16_baseline)
print("cuda_fp16 time", fp16_time)

import pandas as pd
df = pd.DataFrame(trainer.state.log_history)
df.plot.line(x='step', y='loss', title="Loss vs. # steps", ylabel="loss")


######################################################################
# Pruning BERT to be 2:4 sparse
# -----------------------------
# 
# Now that we have our baseline, it’s time we prune BERT. There are many
# different pruning strategies, but one of the most common is **magnitude
# pruning**, which seeks to remove the weights with the lowest L1 norm.
# Magnitude pruning was used by NVIDIA in all their results and is a
# common baseline.
# 
# To do this, we will use the ``torch.ao.pruning`` package, which contains
# a weight-norm (magnitude) sparsifier. These sparsifiers work by applying
# mask parametrizations to the weight tensors in a model. This lets them
# simulate sparsity by masking out the pruned weights.
# 
# We’ll also have to decide what layers of the model to apply sparsity to,
# which in this case is all of the ``nn.Linear`` layers, except for the
# task-specific head outputs. That’s because semi-structured sparsity has
# ``shape constraints <https://pytorch.org/docs/2.1/sparse.html#constructing-sparse-semi-structured-tensors>``\ \_,
# and the task-specific ``nn.Linear`` layers do not satisfy them.
# 

sparsifier = WeightNormSparsifier(
    # apply sparsity to all blocks
    sparsity_level=1.0,
    # shape of 4 elements is a block
    sparse_block_shape=(1, 4),
    # two zeros for every block of 4
    zeros_per_block=2
)

# add to config if ``nn.Linear`` and in the BERT model.
sparse_config = [
    {"tensor_fqn": f"{fqn}.weight"}
    for fqn, module in model.named_modules()
    if isinstance(module, nn.Linear) and "layer" in fqn
]


######################################################################
# The first step for pruning the model is to insert parametrizations for
# masking the weights of the model. This is done by the prepare step.
# Anytime we try to access the ``.weight`` we will get ``mask * weight``
# instead.
# 

# Prepare the model, insert fake-sparsity parametrizations for training
sparsifier.prepare(model, sparse_config)
print(model.bert.encoder.layer[0].output)


######################################################################
# Then, we’ll take a single pruning step. All pruners implement a
# ``update_mask()`` method that updates the mask with the logic being
# determined by the pruner implementation. The step method calls this
# ``update_mask`` functions for the weights specified in the sparse
# config.
# 
# We will also evaluate the model to show the accuracy degradation of
# zero-shot pruning, or pruning without fine-tuning / retraining.
# 

sparsifier.step()
with torch.autocast("cuda"):
    with torch.no_grad():
        predictions = trainer.predict(tokenized_squad_dataset["validation"])
    pruned = compute_metrics(
        *predictions.predictions,
        tokenized_squad_dataset["validation"],
        squad_dataset["validation"],
    )
print("pruned eval metrics:", pruned)


######################################################################
# In this state, we can start fine-tuning the model, updating the elements
# that wouldn’t be pruned to better account for the accuracy loss. Once
# we’ve reached a satisfied state, we can call ``squash_mask`` to fuse the
# mask and the weight together. This will remove the parametrizations and
# we are left with a zeroed-out 2:4 dense model.
# 

trainer.train()
sparsifier.squash_mask()
torch.set_printoptions(edgeitems=4)
print(model.bert.encoder.layer[0].intermediate.dense.weight[:8, :8])

df["sparse_loss"] = pd.DataFrame(trainer.state.log_history)["loss"]
df.plot.line(x='step', y=["loss", "sparse_loss"], title="Loss vs. # steps", ylabel="loss")


######################################################################
# Accelerating 2:4 sparse models for inference
# --------------------------------------------
# 
# Now that we have a model in this format, we can accelerate it for
# inference just like in the QuickStart Guide.
# 

model = model.cuda().half()
# accelerate for sparsity
for fqn, module in model.named_modules():
    if isinstance(module, nn.Linear) and "layer" in fqn:
        module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))

with torch.no_grad():
    predictions = trainer.predict(tokenized_squad_dataset["validation"])
start_logits, end_logits = predictions.predictions
metrics_sparse = compute_metrics(
    start_logits,
    end_logits,
    tokenized_squad_dataset["validation"],
    squad_dataset["validation"],
)
print("sparse eval metrics: ", metrics_sparse)
sparse_perf = measure_execution_time(
    model,
    batch_sizes,
    tokenized_squad_dataset["validation"],
)
print("sparse perf metrics: ", sparse_perf)


######################################################################
# Retraining our model after magnitude pruning has recovered nearly all of
# the F1 that has been lost when the model was pruned. At the same time we
# have achieved a 1.28x speedup for ``bs=16``. Note that not all shapes are
# amenable to performance improvements. When batch sizes are small and
# limited time is spent in compute sparse kernels may be slower than their
# dense counterparts.
# 
# Because semi-structured sparsity is implemented as a tensor subclass, it
# is compatible with ``torch.compile``. When composed with
# ``to_sparse_semi_structured``, we are able to achieve a total 2x speedup
# on BERT.
# 
# | =============== ====== ========== =============== ======== Metrics
#   fp16 2:4 sparse delta / speedup compiled =============== ======
#   ========== =============== ======== Exact Match (%) 78.53 78.44 -0.09
# | F1 (%) 86.93 86.49 -0.44
# | Time (bs=4) 11.10 15.54 0.71x no Time (bs=16) 19.35 15.74 1.23x no
#   Time (bs=64) 72.71 59.41 1.22x no Time (bs=256) 286.65 247.63 1.14x no
#   Time (bs=4) 7.59 7.46 1.02x yes Time (bs=16) 11.47 9.68 1.18x yes Time
#   (bs=64) 41.57 36.92 1.13x yes Time (bs=256) 159.22 142.23 1.12x yes
#   =============== ====== ========== =============== ========
# 
# Conclusion
# ==========
# 
# In this tutorial, we have shown how to prune BERT to be 2:4 sparse and
# how to accelerate a 2:4 sparse model for inference. By taking advantage
# of our ``SparseSemiStructuredTensor`` subclass, we were able to achieve a
# 1.3x speedup over the fp16 baseline, and up to 2x with
# ``torch.compile``. We also demonstrated the benefits of 2:4 sparsity by
# fine-tuning BERT to recover any lost F1 (86.92 dense vs 86.48 sparse).
# 
