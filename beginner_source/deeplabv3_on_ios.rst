Image Segmentation DeepLabV3 on iOS
==============================================

**Author**: `Jeff Tang <https://github.com/jeffxtang>`_

Introduction
------------

Semantic image segmentation is a computer vision task that uses semantic labels to mark specific regions of an input image. The PyTorch semantic image segmentation `DeepLabV3 model <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_ can be used to label image regions with `20 semantic classes <http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html>`_ including, for example, bicycle, bus, car, dog, and person. It is evident that image segmentation models can be very useful in applications such as autonomous driving and scene understanding.

In this tutorial, we will provide a step-by-step guide on how to prepare and run the PyTorch DeepLabV3 model on iOS, taking you from the beginning of having a model you may want to use on iOS to the end of having a complete iOS app using the model. We will also cover practical and general tips on how to check if your next favorable pre-trained PyTorch models can run on iOS, and what pitfalls need to be watched out and avoided.

.. note:: Before going through this tutorial, you should give `PyTorch Mobile for iOS <https://pytorch.org/mobile/ios/>`_ and its `HelloWorld <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld>`_ sample app a quick try. This tutorial will go beyond the image classification model, usually the first kind of models deployed on mobile. The complete code repo for this tutorial is available `here <https://github.com/pytorch/ios-demo-app/ImageSegmentation>`_.

Learning Objectives
-------------------

In this tutorial, we will learn how to:

1. Convert the DeepLabV3 model for iOS deployment;

2. Get example input and output of the model in Python so later in the iOS app we can prepare the right input for the model and process the model output;

3. Build an iOS Swift app from scratch and add the PyTorch iOS Objective-C template files and code needed for running the model for inference;

4. Prepare the model input that can be accepted by the model and process the model output, which, depends on the model architecture, can be a single tensor, a tuple of tensors, or a dictionary of string and tensor, among others.

5. Complete the UI, refactor, build and run the app to see image segmentation in action.

Pre-requisites
---------------

* PyTorch 1.6 or 1.7

* torchvision 0.7 or 0.8

* Xcode 11 or 12

Steps
---------


1. Convert the DeepLabV3 model for iOS deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step after you have a model that you'd like to deploy on iOS is to convert the model to the TorchScript format.

.. note::
    Not all PyTorch models can be converted for iOS deployment at this time, because a model definition may use the language features not in TorchScript, a subset of Python. See the `Script and Optimize Recipe <../recipes/script_optimized.html>`_ for more details.

Simply run the script below to generate the scripted model `deeplabv3_scripted.pt`:

::

    import torch

    # use deeplabv3_resnet50 instead of deeplabv3_resnet101 to reduce the model size
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    scriptedm = torch.jit.script(model)
    torch.jit.save(scriptedm, "deeplabv3_scripted.pt")

.. warning::
    Ideally, a model should also be quantized for significant size reduction and faster inference before being deployed on an iOS app. But not all models can be successfully quantized at the time of the writing as quantization is still in beta. To have a general understanding of quantization, see the `Quantization Recipe <../recipes/quantization.html>`_ and the resource links in the recipe. We will cover in detail how to correctly apply a quantization workflow called Post Training `Static Quantization <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_ to the DeepLabV3 model in a future tutorial or recipe.

2. Get example input and output of the model in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After you have a scripted PyTorch model, you need to test with some example input to make sure it works correctly on iOS. To do that, we first need to write or reuse some Python script that uses the model to make inference on an input and generate an output. In the case of DeepLabV3, we can reuse most of the code in the model hub page above. Add the following code to the code above:

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

Download `deeplab.jpg` from `here <https://github.com/pytorch/ios-demo-app/blob/master/ImageSegmentation/ImageSegmentation/deeplab.jpg>`_, then run the script above and you will see the shapes of the input and output of the model:

::

    torch.Size([1, 3, 800, 800])
    torch.Size([21, 800, 800])

So you will have to provide an input of the exact shape [1, 3, 800, 800] to the model, and then process the output of the size [21, 800, 800]. You should also print out at least part of the actual data of the input and output, to be used in the next three steps to compare with the actual input and output of the model in the iOS app.

3. Build an iOS app and set up for model inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, follow the step 3 of the Model Preparation for iOS recipe `here <../recipes/model_preparation_ios.html#add-the-model-and-pytorch-library-on-ios>`_ to create a new Xcode project. Because both DeepLabV3 used in this tutorial and MobileNet v2 used in the PyTorch HelloWorld iOS example are computer vision models, you can also just get the HelloWorld example repo and modify the code that loads the model, as well as the input pre-processing and output post-processing code to test the model `deeplabv3_scripted.pt` generated in Step 1.

Either way, after adding `deeplabv3_scripted.pt` and `deeplab.jpg` used in Step 2 to the Xcode project, make your `ViewController.swift` look like this:

::

    class ViewController: UIViewController {
        var image = UIImage(named: "deeplab.jpg")!

        override func viewDidLoad() {
            super.viewDidLoad()
        }

        private lazy var module: TorchModule = {
            if let filePath = Bundle.main.path(forResource: "deeplabv3_scripted", ofType: "pt"),
                let module = TorchModule(fileAtPath: filePath) {
                return module
            } else {
                fatalError("Can't find the model file!")
            }
        }()


        @IBAction func doInfer(_ sender: Any) {
            guard var pixelBuffer = image!.normalized() else {
                return
            }

            let buffer = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer))
        }
    }

Now set a breakpoint at the line `return module` and build and run the app. If the app stops at the breakpoint, you know the scripted model in Step 1 has been successfully loaded on iOS. It is a great start, but you need to complete Step 4 before knowing for sure that the model actually works with real input on iOS. If the app exits with the error `Can't find the model file!`, you need to go back to Step 1 to and check out the code and resources listed there to find out why.


4. Process the model input and output for model inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the model loads in the above step, we need to make sure it works with expected inputs and can generate expected outputs. As the model input for the DeepLabV3 model is the same as that of the MobileNet v2 in the HelloWorld example, we can reuse some of the code in the `TorchModule.mm` file from HelloWorld for processing input. Your `TorchModule.mm` should look like this - the four comments reflect the four places we made changes on the original `TorchModule.mm` used in HelloWorld:

::

    - (unsigned char*)predictImage:(void*)imageBuffer {
        try {
            // 1. the original deeplab.jpg is of size 800x800, and the model uses 21 classes for semantic segmentation
            const int WIDTH = 800;
            const int HEIGHT = 800;
            const int CLASSNUM = 21;

            at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, WIDTH, HEIGHT}, at::kFloat);
            torch::autograd::AutoGradMode guard(false);
            at::AutoNonVariableTypeMode non_var_type_mode(true);

            // 2. convert the input tensor to an NSMutableArray for easy debugging
            float* floatInput = tensor.data_ptr<float>();
            if (!floatInput) {
                return nil;
            }
            NSMutableArray* inputs = [[NSMutableArray alloc] init];
            for (int i = 0; i < 3 * WIDTH * HEIGHT; i++) {
                [inputs addObject:@(floatInput[i])];
            }

            // 3. the output of the model is a dictionary - see https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
            auto outputDict = _impl.forward({tensor}).toGenericDict();

            // 4. convert the output to another NSMutableArray for easy debugging
            auto outputTensor = outputDict.at("out").toTensor();
            float* floatBuffer = outputTensor.data_ptr<float>();
            if (!floatBuffer) {
              return nil;
            }
            NSMutableArray* results = [[NSMutableArray alloc] init];
            for (int i = 0; i < CLASSNUM * WIDTH * HEIGHT; i++) {
              [results addObject:@(floatBuffer[i])];
            }
            unsigned char* buffer = (unsigned char*)malloc(3 * WIDTH * HEIGHT);
        }

.. note::
    We did not bother to change the method name `predictImage` used in HelloWorld although a name like `segment` makes better sense in our use of the DeepLabV3 image segmentation model. All we have done so far is to confirm that the model of our interest can be scripted and run correctly in our iOS app as it does in Python. The steps involved so far for using a model in an iOS app should take most of our app development time, just like the data pre-processing task involved in a typical machine learning project.

With the code changes shown above, you can set breakpoints after the two for loops on `inputs` and `results`, and compare them with the model input and output data you see in Step 2 to see if they match. If they do, you know for sure that the model works successfully on iOS. If they do not match exactly, it does not mean that the model fails to work correctly - as the output is a class probability distribution and if the distributions match, we will still get the right segmentation. The best way to prove if this is the case is to complete the UI and the app to actually show the processed output as a new image.

5. Complete the UI, refactor, build and run the app
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The UI for this app is also similar to that for HelloWorld, except that we do not need the `UITextView` to show the image classification result. We will just change the button text and add another one to show back the original image after the segmentation result is shown. The output processing code should be like this, added to the end of the code snippet in Step 4 in `TorchModule.mm`:

::

    for (int j = 0; j < width; j++) {
        for (int k = 0; k < height; k++) {
            int maxj = 0;
            int maxk = 0;
            int maxi = 0;

            float maxnum = -100000.0;
            for (int i = 0; i < CLASSNUM; i++) {
                if ([results[i * (width * height) + j * width + k] floatValue] > maxnum) {
                    maxnum = [results[i * (width * height) + j * width + k] floatValue];
                    maxj = j;
                    maxk = k;
                    maxi = i;
                }
            }

            if (maxi == PERSON) {
                buffer[3 * (maxj * width + maxk)] = 255;
                buffer[3 * (maxj * width + maxk) + 1] = 0;
                buffer[3 * (maxj * width + maxk) + 2] = 0;
            }
            else if (maxi == DOG) {
                buffer[3 * (maxj * width + maxk)] = 0;
                buffer[3 * (maxj * width + maxk) + 1] = 255;
                buffer[3 * (maxj * width + maxk) + 2] = 0;
            } else if (maxi == SHEEP) {
                buffer[3 * (maxj * width + maxk)] = 0;
                buffer[3 * (maxj * width + maxk) + 1] = 0;
                buffer[3 * (maxj * width + maxk) + 2] = 255;
            } else {
                buffer[3 * (maxj * width + maxk)] = 0;
                buffer[3 * (maxj * width + maxk) + 1] = 0;
                buffer[3 * (maxj * width + maxk) + 2] = 0;
            }
        }
    }

We implement this based on our understanding of the DeepLabV3 model, which outputs a tensor of size [21, 800, 800], as shown in Step 2. So for an input image of 800x800, each element in the 800x800 output array is a value between 0 and 20 (for a total of 21 semantic labels described in Introduction.

Now just run the app on an iOS simulator or an actual iOS device, and you will see the following screens:

.. image:: /_static/img/deeplabv3_ios.png
   :width: 300 px
.. image:: /_static/img/deeplabv3_ios2.png
   :width: 300 px


Recap
--------

In this tutorial, we described in step-by-step detail what it takes to convert a pre-trained PyTorch model to run on iOS. We covered how to correctly process the model input and output, and how to make sure they match what the original model in Python expects and generates.

Learn More
------------

1. `PyTorch Mobile site <https://pytorch.org/mobile>`_
2. `DeepLabV3 model <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_
3. `DeepLabV3 paper <https://arxiv.org/pdf/1706.05587.pdf>`_
