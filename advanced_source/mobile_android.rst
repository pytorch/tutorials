Image Classification from an Android App
========================================

Starting with PyTorch 1.3, you will be able to use PyTorch models that have
been serialized via TorchScript directly from within Android apps. This
tutorial will provide a simple example of:

* Loading a serialized PyTorch model into an Android app from Java
* Use the new PyTorch Java API to define a ``Tensor`` to be fed through the
  model

Note: this is essentially a walkthrough of the "highlights" from the ``vision``
portion of the example Android app we provided with PyTorch 1.3.

* The GitHub repo for the full app is `here <https://github.com/zalandoresearch/fashion-mnist>`__.
* The "vision" files are `here <https://github.com/pytorch/android-demo-app/tree/ik_android_demo_app_init/PyTorchDemoApp/app/src/main/java/org/pytorch/demo/vision>`__.

Setting the scene
~~~~~~~~~~~~~~~~~

A common use case for mobile apps, as opposed to training models from scratch on the device, is
loading a serialized pre-trained model and then passing data through it. This tutorial will walk
through this scenario; suppose you have serialized a model using the following code:

.. code:: python

    import torch
    import torchvision

    model = torchvision.models.resnet18(pretrained=True), torch.rand(1, 3, 224, 224)
    model.eval()
    torch.jit.trace(model, input).save("resnet18.pt")

You can use PyTorch in your project by adding a gradle dependency to the latest snapshot in the
sonatype repository.

.. code:: java

    repositories {
        maven {
            url "https://oss.sonatype.org/content/repositories/snapshots"
        }
    }

    dependencies {
        implementation 'org.pytorch:pytorch_android:0.0.8-SNAPSHOT'
    }

We'll deliver the model, serialized via TorchScript, to the android device as an asset. The
code below provides a function to get the model's absolute file path.

.. code:: java

    import android.content.Context;
    import android.util.Log;

    import java.io.File;
    import java.io.FileOutputStream;
    import java.io.IOException;
    import java.io.InputStream;
    import java.io.OutputStream;

    String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
            if (file.exists() && file.length() > 0) {
                return file.getAbsolutePath();
                }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                    }
                os.flush();
                }
            return file.getAbsolutePath();
            }
        } catch (IOException e) {
            Log.e(TAG, "Error process asset " + assetName + " to file");
        }
        return null;
    }

Now, we can tie these two pieces together, loading the model in as a ``Module``,
just as in regular PyTorch, from ``org.pytorch.Module``, using the ``assetFilePath``
function to do so.

.. code:: java

    import org.pytorch.Module;
    Module module = Module.load(assetFilePath(context, "resnet18.pt"));

Next, we create an the input tensor shape as an array of longs and content of the
Tensor as array of floats...

.. code:: java

    long[] inputTensorDims = new long[] {1, 3, 224, 224};
    float[] inputTensorData = new float[1 * 3 * 224 * 224];

...and use these two arrays to initialize a ``Tensor``:

.. code:: java

    Tensor inputTensor = Tensor.newFloat32Tensor(inputTensorDims, inputTensorData);

IValue
~~~~~~

It is important to note that data fed through a ``Module`` must first be converted to
type ``IValue`` using ``IValue.tensor()``. Then, to perform ``Tensor`` operations on
it once it is fed through the model, it must be converted back to a ``Tensor`` using
``.getTensor()``

.. code:: java

    // convert inputTensor to IValue
    IValue input = IValue.tensor(inputTensor);

    // feed input (of type IValue) through the model
    IValue output = module.forward(input);

    // convert output to Tensor
    Tensor outputTensor = output.getTensor();

[Ivan: high level, why do we have to use the ``IValue`` type here?]

The full code for these functions can be found here (Ivan to share link).
