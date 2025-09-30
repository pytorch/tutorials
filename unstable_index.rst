Unstable
========

API unstable features are not available as part of binary distributions
like PyPI or Conda (except maybe behind run-time flags). To test these
features we would, depending on the feature, recommend building PyTorch
from source (main) or using the nightly wheels that are made
available on `pytorch.org <https://pytorch.org>`_.

*Level of commitment*: We are committing to gathering high bandwidth
feedback only on these features. Based on this feedback and potential
further engagement between community members, we as a community will
decide if we want to upgrade the level of commitment or to fail fast.


.. raw:: html

   <div id="tutorial-cards-container">

    <nav class="navbar navbar-expand-lg navbar-light tutorials-nav col-12">
        <div class="tutorial-tags-container">
            <div id="dropdown-filter-tags">
                <div class="tutorial-filter-menu">
                    <div class="tutorial-filter filter-btn all-tag-selected" data-tag="all">All</div>
                </div>
            </div>
        </div>
    </nav>

    <hr class="tutorials-hr">

    <div class="row">

    <div id="tutorial-cards">
    <div class="list">

.. Add prototype tutorial cards below this line

.. vmap

.. customcarditem::
   :header: Using torch.vmap
   :card_description: Learn about torch.vmap, an autovectorizer for PyTorch operations.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/vmap_recipe.html
   :tags: vmap

.. NestedTensor

.. customcarditem::
   :header: Nested Tensor
   :card_description: Learn about nested tensors, the new way to batch heterogeneous-length data
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/nestedtensor.html
   :tags: NestedTensor

.. MaskedTensor

.. customcarditem::
   :header: MaskedTensor Overview
   :card_description: Learn about masked tensors, the source of truth for specified and unspecified values
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/maskedtensor_overview.html
   :tags: MaskedTensor

.. customcarditem::
   :header: Masked Tensor Sparsity
   :card_description: Learn about how to leverage sparse layouts (e.g. COO and CSR) in MaskedTensor
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/maskedtensor_sparsity.html
   :tags: MaskedTensor

.. customcarditem::
   :header: Masked Tensor Advanced Semantics
   :card_description: Learn more about Masked Tensor's advanced semantics (reductions and comparing vs. NumPy's MaskedArray)
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/maskedtensor_advanced_semantics.html
   :tags: MaskedTensor

.. customcarditem::
   :header: MaskedTensor: Simplifying Adagrad Sparse Semantics
   :card_description: See a showcase on how masked tensors can enable sparse semantics and provide for a cleaner dev experience
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/maskedtensor_adagrad.html
   :tags: MaskedTensor

.. Model-Optimization

.. customcarditem::
   :header: Inductor Cpp Wrapper Tutorial
   :card_description: Speed up your models with Inductor Cpp Wrapper
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/inductor_cpp_wrapper_tutorial.html
   :tags: Model-Optimization

.. customcarditem::
   :header: Inductor Windows CPU Tutorial
   :card_description: Speed up your models with Inductor On Windows CPU
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/inductor_windows.html
   :tags: Model-Optimization

.. customcarditem::
   :header: Use max-autotune compilation on CPU to gain additional performance boost
   :card_description: Tutorial for max-autotune mode on CPU to gain additional performance boost
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/max_autotune_on_CPU_tutorial.html
   :tags: Model-Optimization

.. Distributed
.. customcarditem::
   :header: Flight Recorder Tutorial
   :card_description: Debug stuck jobs easily with Flight Recorder
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/flight_recorder_tutorial.html
   :tags: Distributed, Debugging, FlightRecorder

.. customcarditem::
   :header: Context Parallel Tutorial
   :card_description: Parallelize the attention computation along sequence dimension
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/context_parallel.html
   :tags: Distributed, Context Parallel

.. Integration
.. customcarditem::
   :header: Out-of-tree extension autoloading in Python
   :card_description: Learn how to improve the seamless integration of out-of-tree extension with PyTorch based on the autoloading mechanism.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/python_extension_autoload.html
   :tags: Extending-PyTorch, Frontend-APIs

.. GPUDirect Storage
.. customcarditem::
   :header: (prototype) Using GPUDirect Storage
   :card_description: Learn how to use GPUDirect Storage in PyTorch.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: unstable/gpu_direct_storage.html
   :tags: GPUDirect-Storage

.. End of tutorial card section

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:

   unstable/context_parallel
   unstable/flight_recorder_tutorial
   unstable/inductor_cpp_wrapper_tutorial
   unstable/inductor_windows
   unstable/vmap_recipe
   unstable/nestedtensor
   unstable/maskedtensor_overview
   unstable/maskedtensor_sparsity
   unstable/maskedtensor_advanced_semantics
   unstable/maskedtensor_adagrad
   unstable/python_extension_autoload
   unstable/gpu_direct_storage.html
   unstable/max_autotune_on_CPU_tutorial
   unstable/skip_param_init.html
