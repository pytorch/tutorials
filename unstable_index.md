# Unstable

API unstable features are not available as part of binary distributions
like PyPI or Conda (except maybe behind run-time flags). To test these
features we would, depending on the feature, recommend building PyTorch
from source (main) or using the nightly wheels that are made
available on [pytorch.org](https://pytorch.org).

*Level of commitment*: We are committing to gathering high bandwidth
feedback only on these features. Based on this feedback and potential
further engagement between community members, we as a community will
decide if we want to upgrade the level of commitment or to fail fast.

---

[#### Using torch.vmap

Learn about torch.vmap, an autovectorizer for PyTorch operations.

vmap

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/vmap_recipe.html)

[#### Nested Tensor

Learn about nested tensors, the new way to batch heterogeneous-length data

NestedTensor

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/nestedtensor.html)

[#### MaskedTensor Overview

Learn about masked tensors, the source of truth for specified and unspecified values

MaskedTensor

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/maskedtensor_overview.html)

[#### Masked Tensor Sparsity

Learn about how to leverage sparse layouts (e.g. COO and CSR) in MaskedTensor

MaskedTensor

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/maskedtensor_sparsity.html)

[#### Masked Tensor Advanced Semantics

Learn more about Masked Tensor's advanced semantics (reductions and comparing vs. NumPy's MaskedArray)

MaskedTensor

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/maskedtensor_advanced_semantics.html)

[#### MaskedTensor: Simplifying Adagrad Sparse Semantics

See a showcase on how masked tensors can enable sparse semantics and provide for a cleaner dev experience

MaskedTensor

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/maskedtensor_adagrad.html)

[#### Inductor Cpp Wrapper Tutorial

Speed up your models with Inductor Cpp Wrapper

Model-Optimization

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/inductor_cpp_wrapper_tutorial.html)

[#### Inductor Windows CPU Tutorial

Speed up your models with Inductor On Windows CPU

Model-Optimization

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/inductor_windows.html)

[#### Use max-autotune compilation on CPU to gain additional performance boost

Tutorial for max-autotune mode on CPU to gain additional performance boost

Model-Optimization

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/max_autotune_on_CPU_tutorial.html)

[#### Flight Recorder Tutorial

Debug stuck jobs easily with Flight Recorder

Distributed, Debugging, FlightRecorder

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/flight_recorder_tutorial.html)

[#### Context Parallel Tutorial

Parallelize the attention computation along sequence dimension

Distributed, Context Parallel

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/context_parallel.html)

[#### Out-of-tree extension autoloading in Python

Learn how to improve the seamless integration of out-of-tree extension with PyTorch based on the autoloading mechanism.

Extending-PyTorch, Frontend-APIs

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/python_extension_autoload.html)

[#### (prototype) Using GPUDirect Storage

Learn how to use GPUDirect Storage in PyTorch.

GPUDirect-Storage

![](_static/img/thumbnails/cropped/generic-pytorch-logo.png)](unstable/gpu_direct_storage.html)