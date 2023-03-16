.. _new-release_colab::

Notes for Running in Colab
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   This tutorial requires PyTorch 2.0.0 or later. If you are running this
   in Google Colab, verify that you have the required ``torch`` and
   compatible domain libraties installed by running ``!pip list``.
   If the installed version of PyTorch is lower than required,
   unistall it and reinstall again by running the following commands:

   .. code-block:: python

      !pip3 uninstall --yes torch torchaudio torchvision torchtext torchdata
      !pip3 install torch torchaudio torchvision torchtext torchdata
