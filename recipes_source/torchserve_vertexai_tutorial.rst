Deploying a PyTorch Stable Diffusion model as a Vertex AI Endpoint
==================================================================

Deploying large models, like Stable Diffusion, can be challenging and time-consuming.

In this recipe, we will show how you can streamline the deployment of a PyTorch Stable Diffusion
model by leveraging Vertex AI.

PyTorch is the framework used by Stability AI on Stable
Diffusion v1.5.  Vertex AI is a fully-managed machine learning platform with tools and
infrastructure designed to help ML practitioners accelerate and scale ML in production with
the benefit of open-source frameworks like PyTorch.

In four steps you can deploy a PyTorch Stable Diffusion model (v1.5).

Deploying your Stable Diffusion model on a Vertex AI Endpoint can be done in four steps:

* Create a custom TorchServe handler.

* Upload model artifacts to Google Cloud Storage (GCS).

* Create a Vertex AI model with the model artifacts and a prebuilt PyTorch container image.

* Deploy the Vertex AI model onto an endpoint.

Let’s have a look at each step in more detail. You can follow and implement the steps using the
`Notebook example <https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/vertex_endpoints/torchserve/dreambooth_stablediffusion.ipynb>`__.

NOTE: Please keep in mind that this recipe requires a billable Vertex AI as explained in more details in the notebook example.

Create a custom TorchServe handler
----------------------------------

TorchServe is an easy and flexible tool for serving PyTorch models. The model deployed to Vertex AI
uses TorchServe to handle requests and return responses from the model.
You must create a custom TorchServe handler to include in the model artifacts uploaded to Vertex AI. Include the handler file in the
directory with the other model artifacts, like this: `model_artifacts/handler.py`.

After creating the handler file, you must package the handler as a model archiver (MAR) file.
The output file must be named `model.mar`.


.. code:: shell

    !torch-model-archiver \
    -f \
    --model-name <your_model_name> \
    --version 1.0 \
     --handler model_artifacts/handler.py \
    --export-path model_artifacts

Upload model artifacts to Google Cloud Storage (GCS)
----------------------------------------------------

In this step we are uploading
`model artifacts <https://github.com/pytorch/serve/tree/master/model-archiver#artifact-details>`__
to GCS, like the model file or handler. The advantage of storing your artifacts on GCS is that you can
track the artifacts in a central bucket.


.. code:: shell

    BUCKET_NAME = "your-bucket-name-unique"  # @param {type:"string"}
    BUCKET_URI = f"gs://{BUCKET_NAME}/"

    # Will copy the artifacts into the bucket
    !gsutil cp -r model_artifacts $BUCKET_URI

Create a Vertex AI model with the model artifacts and a prebuilt PyTorch container image
----------------------------------------------------------------------------------------

Once you've uploaded the model artifacts into a GCS bucket, you can upload your PyTorch model to
`Vertex AI Model Registry <https://cloud.google.com/vertex-ai/docs/model-registry/introduction>`__.
From the Vertex AI Model Registry, you have an overview of your models
so you can better organize, track, and train new versions. For this you can use the
`Vertex AI SDK <https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk>`__
and this
`pre-built PyTorch container <https://cloud.google.com/blog/products/ai-machine-learning/prebuilt-containers-with-pytorch-and-vertex-ai>`__.


.. code:: shell

    from google.cloud import aiplatform as vertexai
    PYTORCH_PREDICTION_IMAGE_URI = (
        "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest"
    )
    MODEL_DISPLAY_NAME = "stable_diffusion_1_5-unique"
    MODEL_DESCRIPTION = "stable_diffusion_1_5 container"

    vertexai.init(project='your_project', location='us-central1', staging_bucket=BUCKET_NAME)

    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        description=MODEL_DESCRIPTION,
        serving_container_image_uri=PYTORCH_PREDICTION_IMAGE_URI,
        artifact_uri=BUCKET_URI,
    )

Deploy the Vertex AI model onto an endpoint
-------------------------------------------

Once the model has been uploaded to Vertex AI Model Registry you can then take it and deploy
it to an Vertex AI Endpoint. For this you can use the Console or the Vertex AI SDK. In this
example you will deploy the model on a NVIDIA Tesla P100 GPU and n1-standard-8 machine. You can
specify your machine type.


.. code:: shell

    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_P100",
        accelerator_count=1,
        traffic_percentage=100,
        deploy_request_timeout=1200,
        sync=True,
    )

If you follow this
`notebook <https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/vertex_endpoints/torchserve/dreambooth_stablediffusion.ipynb>`__
you can also get online predictions using the Vertex AI SDK as shown in the following snippet.


.. code:: shell

    instances = [{"prompt": "An examplePup dog with a baseball jersey."}]
    response = endpoint.predict(instances=instances)

    with open("img.jpg", "wb") as g:
        g.write(base64.b64decode(response.predictions[0]))

    display.Image("img.jpg")

Create a Vertex AI model with the model artifacts and a prebuilt PyTorch container image

More resources
--------------

This tutorial was created using the vendor documentation. To refer to the original documentation on the vendor site, please see
`torchserve example <https://cloud.google.com/blog/products/ai-machine-learning/get-your-genai-model-going-in-four-easy-steps>`__.
