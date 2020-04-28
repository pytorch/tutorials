PyTorch Recipes
---------------------------------------------
Recipes are bite-sized bite-sized, actionable examples of how to use specific PyTorch features, different from our full-length tutorials.

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

.. Add recipe cards below this line

.. Basics

.. customcarditem::
   :header: Loading data in PyTorch
   :card_description: Learn how to use PyTorch packages to prepare and load common datasets for your model.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/loading_data_recipe.html
   :tags: Basics


.. customcarditem::
   :header: Defining a Neural Network
   :card_description: Learn how to use PyTorch's torch.nn package to create and define a neural network the MNIST dataset.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/defining_a_neural_network.html
   :tags: Basics

.. customcarditem::
   :header: What is a state_dict in PyTorch
   :card_description: Learn how state_dict objects, Python dictionaries, are used in saving or loading models from PyTorch.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/what_is_state_dict.html
   :tags: Basics

.. customcarditem::
   :header: Saving and loading models for inference in PyTorch
   :card_description: Learn about the two approaches for saving and loading models for inference in PyTorch - via the state_dict and via the entire model. 
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/saving_and_loading_models_for_inference.html
   :tags: Basics


.. customcarditem::
   :header: Saving and loading a general checkpoint in PyTorch
   :card_description: Saving and loading a general checkpoint model for inference or resuming training can be helpful for picking up where you last left off. In this recipe, explore how to save and load multiple checkpoints.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/saving_and_loading_a_general_checkpoint.html
   :tags: Basics

.. customcarditem::
   :header: Saving and loading multiple models in one file using PyTorch
   :card_description: In this recipe, learn how saving and loading multiple models can be helpful for reusing models that you have previously trained. 
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/saving_multiple_models_in_one_file.html
   :tags: Basics

.. customcarditem::
   :header: Warmstarting model using parameters from a different model in PyTorch
   :card_description: Learn how warmstarting the training process by partially loading a model or loading a partial model can help your model converge much faster than training from scratch.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html
   :tags: Basics

.. customcarditem::
   :header: Saving and loading models across devices in PyTorch
   :card_description: Learn how saving and loading models across devices (CPUs and GPUs) is relatively straightforward using PyTorch. 
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/save_load_across_devices.html
   :tags: Basics

.. customcarditem::
   :header: Zeroing out gradients in PyTorch
   :card_description: Learn when you should zero out graidents and how doing so can help increase the accuracy of your model. 
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/zeroing_out_gradients.html
   :tags: Basics


.. Interpretability

.. customcarditem::
   :header: Model Interpretability using Captum
   :card_description: Learn how to use Captum attribute the predictions of an image classifier to their corresponding image features and visualize the attribution results.
   :image: _static/img/thumbnails/captum_teaser.png
   :link: ../recipes/recipes/Captum_Recipe.html
   :tags: Interpretability, Captum

.. Quantization

.. customcarditem::
   :header: Dynamic Quantization
   :card_description:  Apply dynamic quantization to a simple LSTM model.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/dynamic_quantization.html
   :tags: Quantization, Text, Model-Optimization

.. Production Development

.. customcarditem::
   :header: TorchScript for Deployment
   :card_description: Learn how to export your trained model in TorchScript format and how to load your TorchScript model in C++ and do inference.
   :image: _static/img/thumbnails/pytorch-logo-flat.png
   :link: ../recipes/recipes/torchscript_inference.html
   :tags: TorchScript
   


.. End of recipe card section

.. raw:: html

    </div>

    </div>

    </div>

    </div>

.. .. galleryitem:: beginner/saving_loading_models.py
