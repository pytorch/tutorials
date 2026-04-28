Note

Go to the end
to download the full example code.

# How to use TensorBoard with PyTorch

TensorBoard is a visualization toolkit for machine learning experimentation.
TensorBoard allows tracking and visualizing metrics such as loss and accuracy,
visualizing the model graph, viewing histograms, displaying images and much more.
In this tutorial we are going to cover TensorBoard installation,
basic usage with PyTorch, and how to visualize data you logged in TensorBoard UI.

## Installation

PyTorch should be installed to log models and metrics into TensorBoard log
directory. The following command will install PyTorch 1.4+ via
Anaconda (recommended):

```
$ conda install pytorch torchvision -c pytorch
```

or pip

```
$ pip install torch torchvision
```

## Using TensorBoard in PyTorch

Let's now try using TensorBoard with PyTorch! Before logging anything,
we need to create a `SummaryWriter` instance.

Writer will output to `./runs/` directory by default.

## Log scalars

In machine learning, it's important to understand key metrics such as
loss and how they change during training. Scalar helps to save
the loss value of each training step, or the accuracy after each epoch.

To log a scalar value, use
`add_scalar(tag, scalar_value, global_step=None, walltime=None)`.
For example, lets create a simple linear regression training, and
log loss value using `add_scalar`

Call `flush()` method to make sure that all pending events
have been written to disk.

See [torch.utils.tensorboard tutorials](https://pytorch.org/docs/stable/tensorboard.html)
to find more TensorBoard visualization types you can log.

If you do not need the summary writer anymore, call `close()` method.

## Run TensorBoard

Install TensorBoard through the command line to visualize data you logged

```
pip install tensorboard
```

Now, start TensorBoard, specifying the root log directory you used above.
Argument `logdir` points to directory where TensorBoard will look to find
event files that it can display. TensorBoard will recursively walk
the directory structure rooted at `logdir`, looking for `.*tfevents.*` files.

```
tensorboard --logdir=runs
```

Go to the URL it provides OR to [http://localhost:6006/](http://localhost:6006/)

[![../../_images/tensorboard_scalars.png](../../_images/tensorboard_scalars.png)](../../_images/tensorboard_scalars.png)

This dashboard shows how the loss and accuracy change with every epoch.
You can use it to also track training speed, learning rate, and other
scalar values. It's helpful to compare these metrics across different
training runs to improve your model.

## Learn More

- [torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html) docs
- [Visualizing models, data, and training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) tutorial

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: tensorboard_with_pytorch.ipynb`](../../_downloads/d493dae89f8804b07cdf678f7d0c2dc6/tensorboard_with_pytorch.ipynb)

[`Download Python source code: tensorboard_with_pytorch.py`](../../_downloads/9d3fdce6265a4c437c6242553a2aa24d/tensorboard_with_pytorch.py)

[`Download zipped: tensorboard_with_pytorch.zip`](../../_downloads/82f146a9a7043d0916fafe05af930ac8/tensorboard_with_pytorch.zip)