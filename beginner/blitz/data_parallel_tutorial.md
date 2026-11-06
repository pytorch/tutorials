Note

Go to the end
to download the full example code.

# Optional: Data Parallelism

**Authors**: [Sung Kim](https://github.com/hunkim) and [Jenny Kang](https://github.com/jennykang)

In this tutorial, we will learn how to use multiple GPUs using `DataParallel`.

It's very easy to use GPUs with PyTorch. You can put the model on a GPU:

```
device = torch.device("cuda:0")
model.to(device)
```

Then, you can copy all your tensors to the GPU:

```
mytensor = my_tensor.to(device)
```

Please note that just calling `my_tensor.to(device)` returns a new copy of
`my_tensor` on GPU instead of rewriting `my_tensor`. You need to assign it to
a new tensor and use that tensor on the GPU.

It's natural to execute your forward, backward propagations on multiple GPUs.
However, Pytorch will only use one GPU by default. You can easily run your
operations on multiple GPUs by making your model run parallelly using
`DataParallel`:

```
model = nn.DataParallel(model)
```

That's the core behind this tutorial. We will explore it in more detail below.

## Imports and parameters

Import PyTorch modules and define parameters.

```
# Parameters and DataLoaders
```

Device

## Dummy DataSet

Make a dummy (random) dataset. You just need to implement the
getitem

## Simple Model

For the demo, our model just gets an input, performs a linear operation, and
gives an output. However, you can use `DataParallel` on any model (CNN, RNN,
Capsule Net etc.)

We've placed a print statement inside the model to monitor the size of input
and output tensors.
Please pay attention to what is printed at batch rank 0.

## Create Model and DataParallel

This is the core part of the tutorial. First, we need to make a model instance
and check if we have multiple GPUs. If we have multiple GPUs, we can wrap
our model using `nn.DataParallel`. Then we can put our model on GPUs by
`model.to(device)`

## Run the Model

Now we can see the sizes of input and output tensors.

## Results

If you have no GPU or one GPU, when we batch 30 inputs and 30 outputs, the model gets 30 and outputs 30 as
expected. But if you have multiple GPUs, then you can get results like this.

### 2 GPUs

If you have 2, you will see:

```
# on 2 GPUs
Let's use 2 GPUs!
 In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
 In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
 In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
 In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
 In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
 In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
 In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
 In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

### 3 GPUs

If you have 3 GPUs, you will see:

```
Let's use 3 GPUs!
 In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
 In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
 In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
 In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
 In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
 In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
 In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
 In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
 In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

### 8 GPUs

If you have 8, you will see:

```
Let's use 8 GPUs!
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
 In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
 In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
 In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
 In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
 In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
 In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

## Summary

DataParallel splits your data automatically and sends job orders to multiple
models on several GPUs. After each model finishes their job, DataParallel
collects and merges the results before returning it to you.

For more information, please check out
[https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html).

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: data_parallel_tutorial.ipynb`](../../_downloads/7f37028fb3517ca10f3388f4bb4889b8/data_parallel_tutorial.ipynb)

[`Download Python source code: data_parallel_tutorial.py`](../../_downloads/e85264945029fe236addcb864bf5f13f/data_parallel_tutorial.py)

[`Download zipped: data_parallel_tutorial.zip`](../../_downloads/0bf0ebd2bc2f524b57b0d2c313c08e38/data_parallel_tutorial.zip)