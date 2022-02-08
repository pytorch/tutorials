Real Time Inference on Raspberry Pi 4 (30 fps!)
=================================================
**Author**: `Tristan Rice <https://github.com/d4l3k>`_

PyTorch has out of the box support for Raspberry Pi 4. This tutorial will guide
you on how to setup a Raspberry Pi 4 for running PyTorch and run a MobileNet v2
classification model in real time (30 fps+) on the CPU.

This was all tested with Raspberry Pi 4 Model B 4GB but should work with the 2GB
variant as well as on the 3B with reduced performance.

.. image:: https://user-images.githubusercontent.com/909104/152895495-7e9910c1-2b9f-4299-a788-d7ec43a93424.jpg

Prerequisites
~~~~~~~~~~~~~~~~

To follow this tutorial you'll need a Raspberry Pi 4, a camera for it and all
the other standard accessories.

* `Raspberry Pi 4 Model B 2GB+ <https://www.raspberrypi.com/products/raspberry-pi-4-model-b/>`_
* `Raspberry Pi Camera Module <https://www.raspberrypi.com/products/camera-module-v2/>`_
* Heat sinks and Fan (optional but recommended)
* 5V 3A USB-C Power Supply
* SD card (at least 8gb)
* SD card read/writer


Raspberry Pi 4 Setup
~~~~~~~~~~~~~~~~~~~~~~~

PyTorch only provides pip packages for Arm 64bit (aarch64) so you'll need to install a 64 bit version of the OS on your Raspberry Pi

You can download the latest arm64 Raspberry Pi OS from https://downloads.raspberrypi.org/raspios_arm64/images/ and install it via rpi-imager.

**32-bit Raspberry Pi OS will not work.**

.. image:: https://user-images.githubusercontent.com/909104/152866212-36ce29b1-aba6-4924-8ae6-0a283f1fca14.gif

Installation will take at least a few minutes depending on your internet speed and sdcard speed. Once it's done it should look like:

.. image:: https://user-images.githubusercontent.com/909104/152867425-c005cff0-5f3f-47f1-922d-e0bbb541cd25.png

Time to put your sdcard in your Raspberry Pi, connect the camera and boot it up.

.. image:: https://user-images.githubusercontent.com/909104/152869862-c239c980-b089-4bd5-84eb-0a1e5cf22df2.png


Once that boots and you complete the initial setup you'll need to edit the ``/boot/config.txt`` file to enable the camera.

.. code:: toml

    # This enables the extended features such as the camera.
    start_x=1

    # This needs to be at least 128M for the camera processing, if it's bigger you can just leave it as is.
    gpu_mem=128

    # You need to commment/remove the existing camera_auto_detect line since this causes issues with OpenCV/V4L2 capture.
    #camera_auto_detect=1

And then reboot. After you reboot the video4linux2 device ``/dev/video0`` should exist.

Installing PyTorch and OpenCV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch and all the other libraries we need have ARM 64-bit/aarch64 variants so you can just install them via pip and have it work like any other Linux system.

.. code:: shell

    $ pip install torch torchvision torchaudio
    $ pip install opencv-contrib-python
    $ pip install numpy --upgrade

.. image:: https://user-images.githubusercontent.com/909104/152874260-95a7a8bd-0f9b-438a-9c0b-5b67729e233f.png


We can now check that everything installed correctly:

.. code:: shell

  $ python3 -c "import torch; print(torch.__version__)"
  1.10.0+cpu

.. image:: https://user-images.githubusercontent.com/909104/152874271-d7057c2d-80fd-4761-aed4-df6c8b7aa99f.png


Video Capture
~~~~~~~~~~~~~~

For video capture we're going to be using OpenCV to stream the video frames
instead of the more common ``picamera``. `picamera` isn't available on 64-bit
Raspberry Pi OS and it's much slower than OpenCV. OpenCV directly accesses the
``/dev/video0`` device to grab frames.

The model we're using (MobileNetV2) takes in image sizes of ``224x224`` so we
can request that directly from OpenCV at 36fps. We're targeting 30fps for the
model but we request a slightly higher framerate than that so there's always
enough frames.

.. code:: python

  import cv2
  from PIL import Image

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
  cap.set(cv2.CAP_PROP_FPS, 36)

OpenCV returns a ``numpy`` array in BGR so we need to read and do a bit of
shuffling to get it into the expected RGB format.

.. code:: python

    ret, image = cap.read()
    # convert opencv output from BGR to RGB
    image = image[:, :, [2, 1, 0]]

NOTE: You can get even more performance by training the model directly with OpenCV's BGR data format to remove the conversion step.

Image Preprocessing
~~~~~~~~~~~~~~~~~~~~

We need to take the frames and transform them into the format the model expects. This is the same processing as you would do on any machine with the standard torchvision transforms.

.. code:: python

    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

MobileNetV2: Quantization and JIT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance we want a model that's quantized and fused. Quantized
means that it does the computation using int8 which is much more performant than
the standard float32 math. Fused means that consecutive operations have been
fused together into a more performant version where possible. Commonly things
like activations (``ReLU``) can be merged into the layer before (``Conv2d``)
during inference.

The aarch64 version of pytorch requires using the ``qnnpack`` engine.

.. code:: python

    import torch
    torch.backends.quantized.engine = 'qnnpack'

For this example we'll use a prequantized and fused version of MobileNetV2 that's provided out of the box by torchvision.

.. code:: python

    from torchvision import models
    net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)

We then want to jit the model to reduce Python overhead and fuse any ops. Jit gives us ~30fps instead of ~20fps without it.

.. code:: python

    net = torch.jit.script(net)
    net.eval()

Putting It Together
~~~~~~~~~~~~~~~~~~~~~~~~~

We can now put all the pieces together and run it:

.. code:: python

    import time

    import torch
    import numpy as np
    from torchvision import models, transforms

    import cv2
    from PIL import Image

    torch.backends.quantized.engine = 'qnnpack'

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv2.CAP_PROP_FPS, 36)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
    # jit model to take it from ~20fps to ~30fps
    net = torch.jit.script(net)
    net.eval()

    started = time.time()
    last_logged = time.time()
    frame_count = 0

    with torch.no_grad():
        while True:
            # read frame
            ret, image = cap.read()
            if not ret:
                raise RuntimeError("failed to read frame")

            # convert opencv output from BGR to RGB
            image = image[:, :, [2, 1, 0]]
            permuted = image

            # preprocess
            input_tensor = preprocess(image)

            # create a mini-batch as expected by the model
            input_batch = input_tensor.unsqueeze(0)

            # run model
            output = net(input_batch)
            # do something with output ...

            # log model performance
            frame_count += 1
            now = time.time()
            if now - last_logged > 1:
                print(f"{frame_count / (now-last_logged)} fps")
                last_logged = now
                frame_count = 0

Running it shows that we're hovering at ~30 fps.

.. image:: https://user-images.githubusercontent.com/909104/152892609-7d115705-3ec9-4f8d-beed-a51711503a32.png

This is with all the default settings in Raspberry Pi OS. If you disabled the UI
and all the other background services that are enabled by default it's more
performant and stable.

If we check ``htop`` we see that we have almost 100% utilization.

.. image:: https://user-images.githubusercontent.com/909104/152892630-f094b84b-19ba-48f6-8632-1b954abc59c7.png

Next Steps
~~~~~~~~~~~~~

You can create your own model or fine tune an existing one. If you fine tune on
one of the models from
`torchvision.models.quantized
<https://pytorch.org/vision/stable/models.html#quantized-models>`_
most of the work to fuse and quantize has already been done for you so you can
directly deploy with good performance on a Raspberry Pi.

See more:

* `Quantization <https://pytorch.org/docs/stable/quantization.html>`_ for more information on how to quantize and fuse your model.
* :ref:`beginner/transfer_learning_tutorial` for how to use transfer learning to fine tune a pre-existing model to your dataset.
