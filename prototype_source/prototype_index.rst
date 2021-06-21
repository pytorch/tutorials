PyTorch Prototype Recipes
---------------------------------------------
Prototype features are not available as part of binary distributions like PyPI or Conda (except maybe behind run-time flags). To test these features we would, depending on the feature, recommend building from master or using the nightly wheels that are made available on `pytorch.org <https://pytorch.org>`_.

*Level of commitment*: We are committing to gathering high bandwidth feedback only on these features. Based on this feedback and potential further engagement between community members, we as a community will decide if we want to upgrade the level of commitment or to fail fast.


.. raw:: html

        </div>
    </div>

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

.. Quantization

.. customcarditem::
   :header: FX Graph Mode Quantization User Guide
   :card_description: Learn about FX Graph Mode Quantization.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../prototype/fx_graph_mode_quant_guide.html
   :tags: FX,Quantization

.. customcarditem::
   :header: FX Graph Mode Post Training Dynamic Quantization
   :card_description: Learn how to do post training dynamic quantization in graph mode based on torch.fx. 
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../prototype/fx_graph_mode_ptq_dynamic.html
   :tags: FX,Quantization

.. customcarditem::
   :header: FX Graph Mode Post Training Static Quantization
   :card_description: Learn how to do post training static quantization in graph mode based on torch.fx. 
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../prototype/fx_graph_mode_ptq_static.html
   :tags: FX,Quantization

.. customcarditem::
   :header: Graph Mode Dynamic Quantization on BERT
   :card_description: Learn how to do post training dynamic quantization with graph mode quantization on BERT models.
   :image: ../_static/img/thumbnails/cropped/graph-mode-dynamic-bert.png
   :link: ../prototype/graph_mode_dynamic_bert_tutorial.html
   :tags: Text,Quantization

.. customcarditem::
   :header: PyTorch Numeric Suite Tutorial
   :card_description: Learn how to use the PyTorch Numeric Suite to support quantization debugging efforts.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../prototype/numeric_suite_tutorial.html
   :tags: Debugging,Quantization

.. Mobile

.. customcarditem::
   :header: Use iOS GPU in PyTorch
   :card_description: Learn how to run your models on iOS GPU.
   :image: ../_static/img/thumbnails/cropped/ios.png
   :link: ../prototype/ios_gpu_workflow.html
   :tags: Mobile

.. customcarditem::
   :header: Convert MobileNetV2 to NNAPI
   :card_description: Learn how to prepare a computer vision model to use Android’s Neural Networks API (NNAPI).
   :image: ../_static/img/thumbnails/cropped/android.png
   :link: ../prototype/nnapi_mobilenetv2.html
   :tags: Mobile

.. customcarditem::
   :header: PyTorch Vulkan Backend User Workflow
   :card_description: Learn how to use the Vulkan backend on mobile GPUs.
   :image: ../_static/img/thumbnails/cropped/android.png
   :link: ../prototype/vulkan_workflow.html
   :tags: Mobile
   
.. Modules

.. customcarditem::
   :header: Skipping Module Parameter Initialization in PyTorch 1.10
   :card_description: Describes skipping parameter initialization during module construction in PyTorch 1.10, avoiding wasted computation.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../prototype/skip_param_init.html
   :tags: Modules

.. TorchScript

.. customcarditem::
   :header: Model Freezing in TorchScript
   :card_description: Freezing is the process of inlining Pytorch module parameters and attributes values into the TorchScript internal representation.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../prototype/torchscript_freezing.html
   :tags: TorchScript

.. vmap

.. customcarditem::
   :header: Using torch.vmap
   :card_description: Learn about torch.vmap, an autovectorizer for PyTorch operations.
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../prototype/vmap_recipe.html
   :tags: vmap

.. End of tutorial card section

.. raw:: html

    </div>

    <div class="pagination d-flex justify-content-center"></div>

    </div>

    </div>

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :hidden:

   prototype/fx_graph_mode_quant_guide.html
   prototype/fx_graph_mode_ptq_dynamic.html
   prototype/fx_graph_mode_ptq_static.html
   prototype/graph_mode_dynamic_bert_tutorial.html
   prototype/ios_gpu_workflow.html
   prototype/nnapi_mobilenetv2.html
   prototype/numeric_suite_tutorial.html
   prototype/torchscript_freezing.html
   prototype/vmap_recipe.html
   prototype/vulkan_workflow.html
