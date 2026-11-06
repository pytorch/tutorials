Note

Go to the end
to download the full example code.

# Changing default device

It is common practice to write PyTorch code in a device-agnostic way,
and then switch between CPU and CUDA depending on what hardware is available.
Typically, to do this you might have used if-statements and `cuda()` calls
to do this:

Note

This recipe requires PyTorch 2.0.0 or later.

PyTorch now also has a context manager which can take care of the
device transfer automatically. Here is an example:

You can also set it globally like this:

This function imposes a slight performance cost on every Python
call to the torch API (not just factory functions). If this
is causing problems for you, please comment on
[this issue](https://github.com/pytorch/pytorch/issues/92701)

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: changing_default_device.ipynb`](../../_downloads/f942955edbd28653c694b70c32b24ff2/changing_default_device.ipynb)

[`Download Python source code: changing_default_device.py`](../../_downloads/716d809000230e12afb363157e62e635/changing_default_device.py)

[`Download zipped: changing_default_device.zip`](../../_downloads/e717733f10a91502e39a911d23df58aa/changing_default_device.zip)