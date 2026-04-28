Note

Go to the end
to download the full example code.

# PyTorch: Control Flow + Weight Sharing

To showcase the power of PyTorch dynamic graphs, we will implement a very strange
model: a third-fifth order polynomial that on each forward pass
chooses a random number between 4 and 5 and uses that many orders, reusing
the same weights multiple times to compute the fourth and fifth order.

```
# Create Tensors to hold input and outputs.

# Construct our model by instantiating the class defined above

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: dynamic_net.ipynb`](../../_downloads/6b6889455ef3d6c74e64c3fc1c12815b/dynamic_net.ipynb)

[`Download Python source code: dynamic_net.py`](../../_downloads/788cabaf416dc69c7b2faffc0f744ae1/dynamic_net.py)

[`Download zipped: dynamic_net.zip`](../../_downloads/1c6f6c6ee3f3ad95b0ff8553232076fb/dynamic_net.zip)