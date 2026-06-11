Note

Go to the end
to download the full example code.

# PyTorch: Custom nn Modules

A third order polynomial, trained to predict \(y=\sin(x)\) from \(-\pi\)
to \(\pi\) by minimizing squared Euclidean distance.

This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.

```
# Create Tensors to hold input and outputs.

# Construct our model by instantiating the class defined above

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined
# with torch.nn.Parameter) which are members of the model.

# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: polynomial_module.ipynb`](../../_downloads/fdb76f84e688e2ecc24fa38edfa41aea/polynomial_module.ipynb)

[`Download Python source code: polynomial_module.py`](../../_downloads/4dbaf9210d9de48b066fe57085912ccf/polynomial_module.py)

[`Download zipped: polynomial_module.zip`](../../_downloads/50b31e19a63de93f4e85a4f9cc45b844/polynomial_module.zip)