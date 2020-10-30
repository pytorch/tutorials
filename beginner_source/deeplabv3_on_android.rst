Image Segmentation DeepLabV3 on Android
=================================================

**Author**: `Jeff Tang <https://github.com/jeffxtang>`_

Introduction
------------

Semantic image segmentation is a computer vision task that uses semantic labels to mark specific regions of an input image. The PyTorch semantic image segmentation `DeepLabV3 model <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_ can be used to label image regions with `20 semantic classes <http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html>`_ including, for example, bicycle, bus, car, dog, and person. It is evident that image segmentation models can be very useful in applications such as autonomous driving and scene understanding.

In this tutorial, we will provide a step-by-step guide on how to prepare and run the PyTorch DeepLabV3 model on Android, taking you from the beginning of having a model you may want to use on Android to the end of having a complete Android app using the model. We will also cover practical and general tips on how to check if your next favorable pre-trained PyTorch models can run on Android, and what pitfalls need to be watched out and avoided.

.. note:: Before going through this tutorial, you should check out `PyTorch Mobile for Android <https://pytorch.org/mobile/android/>`_ and give the PyTorch Android `HelloWorld <https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp>`_ example app a quick try. This tutorial will go beyond the image classification model, usually the first kind of model deployed on mobile. The complete code repo for this tutorial is available `here <https://github.com/pytorch/android-demo-app/ImageSegmentation>`_.

Learning Objectives
-------------------

In this tutorial, you will learn how to:

1. Convert the DeepLabV3 model for Android deployment;

2. Get example input and output of the model in Python for Android app to compare with;

3. Build a new Android app, or reuse an Android example app, to load the converted model;

4. Prepare the model input that can be accepted by the model and process the model output;

5. Complete the UI, refactor, build and run the app to see image segmentation in action.

Pre-requisites
---------------

* PyTorch 1.6 or 1.7

* torchvision 0.7 or 0.8

* Android Studio 3.5.1 or above with NDK installed

Steps
---------

1. Convert the DeepLabV3 model for Android deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step after you find a model that you'd like to deploy on Android is to convert the model to the `TorchScript <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_ format.

.. note::
    Not all PyTorch models can be converted to TorchScript at this time, because a model definition may use the language features not in TorchScript, which is a subset of Python. See the `Script and Optimize Recipe <../recipes/script_optimized.html>`_ for more details.

Simply run the script below to generate the scripted model `deeplabv3_scripted.pt`:

::

    import torch

    # use deeplabv3_resnet50 instead of deeplabv3_resnet101 to reduce the model size
    model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    scriptedm = torch.jit.script(model)
    torch.jit.save(scriptedm, "deeplabv3_scripted.pt")

The size of the generated `deeplabv3_scripted.pt` model file should be around 168MB. Ideally, a model should also be quantized for significant size reduction and faster inference before being deployed on an Android app. But not all models can be successfully or easily quantized at the time of the writing because quantization is still in beta. To have a general understanding of quantization, see the `Quantization Recipe <../recipes/quantization.html>`_ and the resource links there. We will cover in detail how to correctly apply a quantization workflow called Post Training `Static Quantization <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_ to the DeepLabV3 model in a future tutorial or recipe.

2. Get example input and output of the model in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After you have a scripted PyTorch model, you need to test with some example input to make sure it works correctly on Android. To do that, you first need to write or reuse some Python script that uses the model to make inference and examine some example input and output. In the case of DeepLabV3, you can reuse some of the code in the model hub page above. Add the following code snippet to the code above:

::

    from PIL import Image
    from torchvision import transforms
    input_image = Image.open("deeplab.jpg")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        rslt = model(input_batch)
        output = rslt['out'][0]

    print(input_batch.shape)
    print(output.shape)

Download `deeplab.jpg` from `here <https://github.com/jeffxtang/android-demo-app/blob/new_demo_apps/ImageSegmentation/app/src/main/assets/deeplab.jpg>`_, then run the script above and you will see the shapes of the input and output of the model:

::

    torch.Size([1, 3, 400, 400])
    torch.Size([21, 400, 400])

So you will have to provide an input of the exact shape [1, 3, 400, 400] to the model, and then process the output of the size [21, 400, 400]. You should also print out at least the beginning parts of the actual data of the input and output, to be used in Step 4 below to compare with the actual input and output of the model when running in the Android app.

3. Build a new Android app or reuse an example app and load the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, follow Step 3 of the `Model Preparation for Android recipe <../recipes/model_preparation_android.html#add-the-model-and-pytorch-library-on-android>`_ to create a new Android Studio project with PyTorch Mobile enabled. Because both DeepLabV3 used in this tutorial and MobileNet v2 used in the PyTorch HelloWorld Android example are computer vision models, you can also get the `HelloWorld example repo <https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp>`_ to make it easier to modify the code that loads the model and processes the input and output. The main goal in this step and Step 4 is to make sure the model `deeplabv3_scripted.pt` generated in Step 1 can indeed work correctly on Android.

Now add `deeplabv3_scripted.pt` and `deeplab.jpg` used in Step 2 to the Android Studio project, make the `onCreate` method in the `MainActivity` contain the code snippet below:

::

    Module module = null;
    try {
      module = Module.load(assetFilePath(this, "deeplabv3_scripted.pt"));
    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error loading model!", e);
      finish();
    }

Then set a breakpoint at the line `finish()` and build and run the app. If the app does not stop at the breakpoint, you know the scripted model in Step 1 has been successfully loaded on Android. It is a great start, but you need to complete Step 4 before knowing for sure that the model actually works with real input on Android. If the app exits with the error `Error loading model!`, you need to go back to Step 1 to and check out the code and resources listed there to find out why.


4. Process the model input and output for model inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the model loads in the previous step, you need to verify that it works with expected inputs and can generate expected outputs. As the model input for the DeepLabV3 model is an image, the same as that of the MobileNet v2 in the HelloWorld example, you can reuse some of the code in the `MainActivity.java` file from HelloWorld for input processing. Your button click handler code in the `MainActivity.java` should look like this:

::

    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB);
    final float[] inputs = inputTensor.getDataAsFloatArray();

    Map<String, IValue> outTensors =
        module.forward(IValue.from(inputTensor)).toDictStringKey();

    // as documented in https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101,
    // the key "out" of the output tensor contains the semantic masks
    final Tensor outputTensor = outTensors.get("out").toTensor();
    final float[] outputs = outputTensor.getDataAsFloatArray();

    int width = bitmap.getWidth();
    int height = bitmap.getHeight();

.. note::
    The model output is a dictionary for the DeepLabV3 model, so we call `toDictStringKey` after the model `forward` call to correctly extract the result. The model output may also be a single tensor or a tuple of tensors, among other things, for other models.

Set breakpoints at the lines after `final float[] inputs` and `final float[] outputs`, which populate the input tensor and output tensor data to float arrays for easy debugging. Run the app and when it stops at the breakpoints, compare the numbers in `inputs` and `outputs` with the model input and output data you see in Step 2 to see if they match. If they do, you know for sure that the model works successfully on Android. If they do not match perfectly, it does not mean that the model fails to work correctly - as the output is a class probability distribution and as long as the distributions match, you will still get the right segmentation result - the best way to prove if this is the case is to complete the UI and the app to actually see the processed result as a new image.

.. important::
    All we have done so far is to confirm that the model of our interest can be scripted and run correctly in our Android app as in Python. The steps involved so far for using a model in an Android app take a lot, if not most, of our app development time, just like the data pre-processing task involved in a typical machine learning project usually does.


5. Complete the UI, refactor, build and run the app
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The UI for this app is also similar to that for HelloWorld, except that you do not need the `TextView` to show the image classification result. Just change the button text and add another one to show back the original image after the segmentation result is shown. The output processing code should be like this, added to the end of the code snippet in Step 4:

::

    int[] intValues = new int[width * height];
    for (int i = 0; i < intValues.length; i++) {
        intValues[i] = 0xFFFFFFFF;
    }
    for (int j = 0; j < width; j++) {
        for (int k = 0; k < height; k++) {
            int maxj = 0;
            int maxk = 0;
            int maxi = 0;
            double maxnum = -100000.0;
            for (int i=0; i < CLASSNUM; i++) {
                if (scores[i*(width*height) + j*width + k] > maxnum) {
                    maxnum = scores[i*(width*height) + j*width + k];
                    maxj = j; maxk= k; maxi = i;
                }
            }
            if (maxi == PERSON)
                intValues[maxj*width + maxk] = 0xFFFF0000;
            else if (maxi == DOG)
                intValues[maxj*width + maxk] = 0xFF00FF00;
            else if (maxi == SHEEP)
                intValues[maxj*width + maxk] = 0xFF0000FF;
            else
                intValues[maxj*width + maxk] = 0xFF000000;
        }
    }

The constants used in the code above are defined in the beginning of the class `MainActivity`:

::

    private static final int CLASSNUM = 21;
    private static final int DOG = 12;
    private static final int PERSON = 15;
    private static final int SHEEP = 17;


The implementation here is based on the understanding of the DeepLabV3 model, which outputs a tensor of size [21, 400, 400], as shown in Step 2. So for an input image of 400x400, each element in the 400x400 output array is a value between 0 and 20 (for a total of 21 semantic labels described in Introduction) and the value is used to set a specific color.

After the output processing, you will also need to call the code below to render the RGB `intValues` array to a bitmap instance `outputBitmap` before displaying it on an `ImageView`:

::

    Bitmap bmpSegmentation = Bitmap.createScaledBitmap(bitmap, width, height, true);
    Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
    outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0,
        outputBitmap.getWidth(), outputBitmap.getHeight());
    imageView.setImageBitmap(outputBitmap);

Now build and run the app on an Android emulator or an actual device, and you will see the following screens:

.. image:: /_static/img/deeplabv3_android.png
   :width: 300 px
.. image:: /_static/img/deeplabv3_android2.png
   :width: 300 px


Recap
--------

In this tutorial, we described what it takes to convert a pre-trained PyTorch DeepLabV3 model for Android and how to make sure the model can run successfully on Android. Our focus was to help you understand the process of confirming that a model can indeed run on Android. The complete code repo is available `here <https://github.com/pytorch/android-demo-app/ImageSegmentation>`_.

More advanced topics such as quantization and using models via transfer learning or of your own on Android will be covered soon in future demo apps and tutorials.


Learn More
------------

1. `PyTorch Mobile site <https://pytorch.org/mobile>`_
2. `DeepLabV3 model <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_
3. `DeepLabV3 paper <https://arxiv.org/pdf/1706.05587.pdf>`_
