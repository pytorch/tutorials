:orphan:

Domains
=======

This section contains specialized tutorials focused on applying
PyTorch to specific application areas. These guides demonstrate
how to use domain-specific libraries like torchvision, torchaudio, and
others. This section is for developers looking to implement PyTorch
in particular fields of deep learning.

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

.. Add cards below this line

.. customcarditem::
   :header: TorchVision Object Detection Finetuning Tutorial
   :card_description: Finetune a pre-trained Mask R-CNN model.
   :image: _static/img/thumbnails/cropped/TorchVision-Object-Detection-Finetuning-Tutorial.png
   :link: intermediate/torchvision_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Transfer Learning for Computer Vision Tutorial
   :card_description: Train a convolutional neural network for image classification using transfer learning.
   :image: _static/img/thumbnails/cropped/Transfer-Learning-for-Computer-Vision-Tutorial.png
   :link: beginner/transfer_learning_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Optimizing Vision Transformer Model
   :card_description: Apply cutting-edge, attention-based transformer models to computer vision tasks.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: beginner/vt_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Adversarial Example Generation
   :card_description: Train a convolutional neural network for image classification using transfer learning.
   :image: _static/img/thumbnails/cropped/Adversarial-Example-Generation.png
   :link: beginner/fgsm_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: DCGAN Tutorial
   :card_description: Train a generative adversarial network (GAN) to generate new celebrities.
   :image: _static/img/thumbnails/cropped/DCGAN-Tutorial.png
   :link: beginner/dcgan_faces_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Spatial Transformer Networks Tutorial
   :card_description: Learn how to augment your network using a visual attention mechanism.
   :image: _static/img/stn/Five.gif
   :link: intermediate/spatial_transformer_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Inference on Whole Slide Images with TIAToolbox
   :card_description: Learn how to use the TIAToolbox to perform inference on whole slide images.
   :image: _static/img/thumbnails/cropped/TIAToolbox-Tutorial.png
   :link: intermediate/tiatoolbox_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Semi-Supervised Learning Tutorial Based on USB
   :card_description: Learn how to train semi-supervised learning algorithms (on custom data) using USB and PyTorch.
   :image: _static/img/usb_semisup_learn/code.png
   :link: advanced/usb_semisup_learn.html
   :tags: Image/Video

.. Reinforcement Learning

.. customcarditem::
   :header: Reinforcement Learning (DQN)
   :card_description: Learn how to use PyTorch to train a Deep Q Learning (DQN) agent on the CartPole-v0 task from the OpenAI Gym.
   :image: _static/img/cartpole.gif
   :link: intermediate/reinforcement_q_learning.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Reinforcement Learning (PPO) with TorchRL
   :card_description: Learn how to use PyTorch and TorchRL to train a Proximal Policy Optimization agent on the Inverted Pendulum task from Gym.
   :image: _static/img/invpendulum.gif
   :link: intermediate/reinforcement_ppo.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Train a Mario-playing RL Agent
   :card_description: Use PyTorch to train a Double Q-learning agent to play Mario.
   :image: _static/img/mario.gif
   :link: intermediate/mario_rl_tutorial.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Recurrent DQN
   :card_description: Use TorchRL to train recurrent policies
   :image: _static/img/rollout_recurrent.png
   :link: intermediate/dqn_with_rnn_tutorial.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Code a DDPG Loss
   :card_description: Use TorchRL to code a DDPG Loss
   :image: _static/img/half_cheetah.gif
   :link: advanced/coding_ddpg.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Writing your environment and transforms
   :card_description: Use TorchRL to code a Pendulum
   :image: _static/img/pendulum.gif
   :link: advanced/pendulum.html
   :tags: Reinforcement-Learning

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------   


.. toctree::
   :maxdepth: 1
   :includehidden:
   :hidden:
   :caption: Image and Video

   intermediate/torchvision_tutorial
   beginner/transfer_learning_tutorial
   beginner/fgsm_tutorial
   beginner/dcgan_faces_tutorial
   intermediate/spatial_transformer_tutorial
   beginner/vt_tutorial
   intermediate/tiatoolbox_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Reinforcement Learning

   intermediate/reinforcement_q_learning
   intermediate/reinforcement_ppo
   intermediate/dqn_with_rnn_tutorial.html
   intermediate/mario_rl_tutorial
   advanced/pendulum
   advanced/coding_ddpg.html

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Recommendation Systems

   intermediate/torchrec_intro_tutorial
   advanced/sharding

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Audio

   See Audio tutorials on the audio website <https://pytorch.org/audio/stable/index.html>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: ExecuTorch

   See ExecuTorch tutorials on the audio website <https://pytorch.org/executorch/stable/index.html>
