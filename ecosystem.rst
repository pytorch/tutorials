Ecosystem
=========

Explore tutorials that cover tools and frameworks in
the PyTorch ecosystem. These practical guides will help you leverage
PyTorch's extensive ecosystem for everything from experimentation
to production deployment.

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

.. Add tutorial cards below this line
.. customcarditem::
   :header: Hyperparameter Tuning Tutorial
   :card_description: Learn how to use Ray Tune to find the best performing set of hyperparameters for your model.
   :image: _static/img/ray-tune.png
   :link: beginner/hyperparameter_tuning_tutorial.html
   :tags: Model-Optimization,Best-Practice,Ecosystem,Ray-Distributed,Parallel-and-Distributed-Training

.. customcarditem::
   :header: Serving PyTorch Tutorial
   :card_description: Deploy and scale a PyTorch model with Ray Serve.
   :image: _static/img/ray-serve.png
   :link: beginner/ray_serve_tutorial.html
   :tags: Production,Best-Practice,Ray-Distributed,Ecosystem

.. customcarditem::
   :header: Multi-Objective Neural Architecture Search with Ax
   :card_description: Learn how to use Ax to search over architectures find optimal tradeoffs between accuracy and latency.
   :image: _static/img/ax_logo.png
   :link: intermediate/ax_multiobjective_nas_tutorial.html
   :tags: Model-Optimization,Best-Practice,Ax,TorchX,Ecosystem

.. customcarditem::
   :header: Performance Profiling in TensorBoard
   :card_description: Learn how to use the TensorBoard plugin to profile and analyze your model's performance.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: intermediate/tensorboard_profiler_tutorial.html
   :tags: Model-Optimization,Best-Practice,Profiling,TensorBoard,Ecosystem

.. customcarditem::
   :header: Real Time Inference on Raspberry Pi 4
   :card_description: This tutorial covers how to run quantized and fused models on a Raspberry Pi 4 at 30 fps.
   :image: _static/img/thumbnails/cropped/realtime_rpi.png
   :link: intermediate/realtime_rpi.html
   :tags: Model-Optimization,Image/Video,Quantization,Ecosystem

.. End of tutorial card section
.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :maxdepth: 2
   :hidden:

   beginner/hyperparameter_tuning_tutorial
   beginner/ray_serve_tutorial
   intermediate/ax_multiobjective_nas_tutorial
   intermediate/tensorboard_profiler_tutorial
   intermediate/realtime_rpi
