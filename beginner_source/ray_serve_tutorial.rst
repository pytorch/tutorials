Serving PyTorch Models at Scale with Ray Serve
==============================================
**Author:** `Ricardo Decal <https://github.com/crypdick>`_

This tutorial introduces `Ray Serve <https://docs.ray.io/en/latest/serve/index.html>`_, a scalable framework for serving machine learning models in production. Ray Serve is part of `Ray Distributed <https://pytorch.org/projects/ray/>`_, an open-source PyTorch Foundation project.

Introduction
------------

`Ray Serve <https://docs.ray.io/en/latest/serve/index.html>`_ is an online serving library that helps you deploy machine learning models in production.

Production-ready features
*************************

Ray Serve provides the following production-ready features:

- Handle thousands of concurrent requests efficiently with dynamic request batching
- Autoscale your endpoint to handle variable traffic
- Buffer requests when the endpoint is busy
- Compose multiple models along with business logic into a complete ML application
- Gracefully heal the deployment when nodes are lost
- Handle multi-node/multi-GPU serving
- Flexibly allocate heterogenous compute resources and fractional GPUs
- `LLM-specific features <https://docs.ray.io/en/latest/serve/llm/index.html>`_ such as response streaming, LoRA multiplexing, prefill-decode disaggregation, and more.

Ray Serve also has LLM-specific features such as response streaming, model multiplexing, dynamic request batching, and more.

In this tutorial, you'll learn how to:

1. Deploy a PyTorch model as a web service using Ray Serve
2. Scale the deployment with multiple replicas
3. Use advanced features like autoscaling and request batching
4. Compose multiple deployments into a complete ML application
5. Handle concurrent requests efficiently

Prerequisites
-------------

This tutorial assumes basic familiarity with PyTorch and Python. You'll need to install Ray Serve:

.. code-block:: bash

   pip install "ray[serve]" torch torchvision

Setup
-----

Let's start by importing the necessary libraries:

.. code-block:: python

   import asyncio
   import json
   import time
   from typing import Any, Dict, List

   import aiohttp
   import numpy as np
   import requests
   import torch
   import torch.nn as nn
   from ray import serve
   from starlette.requests import Request
   from torchvision import transforms

Part 1: Deploy a Simple PyTorch Model
--------------------------------------

We'll start with a simple convolutional neural network for MNIST digit classification.
First, let's define our model architecture:

.. code-block:: python

   class MNISTNet(nn.Module):
       """Simple CNN for MNIST digit classification"""
       def __init__(self):
           super(MNISTNet, self).__init__()
           self.conv1 = nn.Conv2d(1, 32, 3, 1)
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.dropout1 = nn.Dropout(0.25)
           self.dropout2 = nn.Dropout(0.5)
           self.fc1 = nn.Linear(9216, 128)
           self.fc2 = nn.Linear(128, 10)

       def forward(self, x):
           x = self.conv1(x)
           x = nn.functional.relu(x)
           x = self.conv2(x)
           x = nn.functional.relu(x)
           x = nn.functional.max_pool2d(x, 2)
           x = self.dropout1(x)
           x = torch.flatten(x, 1)
           x = self.fc1(x)
           x = nn.functional.relu(x)
           x = self.dropout2(x)
           x = self.fc2(x)
           return nn.functional.log_softmax(x, dim=1)

Creating a Ray Serve Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To deploy this model with Ray Serve, we wrap it in a class and add the ``@serve.deployment`` decorator.
The deployment handles incoming HTTP requests and runs inference:

.. code-block:: python

   @serve.deployment
   class MNISTClassifier:
       def __init__(self, model_path: str = None):
           """Initialize the model. If model_path is provided, load weights."""
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self.model = MNISTNet().to(self.device)
           
           if model_path:
               self.model.load_state_dict(torch.load(model_path, map_location=self.device))
           
           self.model.eval()

       async def __call__(self, request: Request) -> Dict[str, Any]:
           """Handle incoming HTTP requests"""
           # Parse the JSON request body
           data = await request.json()
           batch = json.loads(data)
           
           # Run inference
           return await self.predict(batch)
       
       async def predict(self, batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
           """Run inference on a batch of images"""
           # Convert numpy array to tensor
           images = torch.tensor(batch["image"], dtype=torch.float32).to(self.device)
           
           # Run inference
           with torch.no_grad():
               logits = self.model(images)
               predictions = torch.argmax(logits, dim=1).cpu().numpy()
           
           return {
               "predicted_label": predictions.tolist(),
               "logits": logits.cpu().numpy().tolist()
           }

Running the Deployment
~~~~~~~~~~~~~~~~~~~~~~

Now let's deploy and run our model:

.. code-block:: python

   # Create the deployment (but don't run it yet)
   mnist_app = MNISTClassifier.bind()
   
   # Start the Ray Serve application
   handle = serve.run(mnist_app, name="mnist_classifier")

Testing the Deployment
~~~~~~~~~~~~~~~~~~~~~~

Let's test our deployment with some random data:

.. code-block:: python

   # Create a batch of random images (simulating MNIST format: 28x28 grayscale)
   images = np.random.rand(2, 1, 28, 28).tolist()
   json_request = json.dumps({"image": images})
   
   # Send HTTP request
   response = requests.post("http://localhost:8000/", json=json_request)
   print(f"Predictions: {response.json()['predicted_label']}")

Part 2: Scaling with Multiple Replicas
---------------------------------------

One of Ray Serve's key features is the ability to scale your deployment across multiple replicas.
Each replica is an independent instance of your model that can handle requests in parallel.

Configuring Replicas
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create deployment with 4 replicas
   mnist_app = MNISTClassifier.options(
       num_replicas=4,
       ray_actor_options={"num_gpus": 0.25}  # Each replica uses 1/4 of a GPU
   ).bind()
   
   # Update the running deployment
   handle = serve.run(mnist_app, name="mnist_classifier")

This configuration creates 4 replicas, each using 25% of a GPU. This allows you to serve 4 models
on a single GPU, maximizing resource utilization for small models.

Part 3: Autoscaling
-------------------

Ray Serve can automatically scale the number of replicas based on incoming traffic.
This is useful for handling variable workloads without over-provisioning resources.

Configuring Autoscaling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mnist_app = MNISTClassifier.options(
       autoscaling_config={
           "target_ongoing_requests": 10,  # Target 10 requests per replica
           "min_replicas": 0,              # Scale down to 0 when idle
           "max_replicas": 10,             # Scale up to 10 replicas max
           "upscale_delay_s": 5,           # Wait 5s before scaling up
           "downscale_delay_s": 30,        # Wait 30s before scaling down
       },
       ray_actor_options={"num_gpus": 0.1}
   ).bind()
   
   handle = serve.run(mnist_app, name="mnist_classifier")

With this configuration, Ray Serve will:

- Start with 0 replicas (no resources used when idle)
- Scale up when requests arrive (targeting 10 concurrent requests per replica)
- Scale down after 30 seconds of low traffic

Testing Autoscaling with Concurrent Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To see autoscaling in action, we need to send many concurrent requests. Using ``aiohttp``,
we can fire requests asynchronously:

.. code-block:: python

   async def send_request(session, url, data):
       """Send a single async HTTP request"""
       async with session.post(url, json=data) as response:
           return await response.json()

   async def send_concurrent_requests(num_requests=100):
       """Send many requests concurrently"""
       url = "http://localhost:8000/"
       
       # Create sample data
       images = np.random.rand(10, 1, 28, 28).tolist()
       json_request = json.dumps({"image": images})
       
       # Send all requests concurrently
       async with aiohttp.ClientSession() as session:
           tasks = [
               send_request(session, url, json_request)
               for _ in range(num_requests)
           ]
           responses = await asyncio.gather(*tasks)
       
       return responses

   # Run the concurrent requests
   start_time = time.time()
   responses = asyncio.run(send_concurrent_requests(100))
   elapsed = time.time() - start_time
   
   print(f"Processed {len(responses)} requests in {elapsed:.2f} seconds")
   print(f"Throughput: {len(responses)/elapsed:.2f} requests/second")

This approach allows Ray Serve to buffer and batch process the requests efficiently,
automatically scaling replicas as needed.

Part 4: Dynamic Request Batching
---------------------------------

Dynamic request batching is a powerful optimization that groups multiple incoming requests
and processes them together, maximizing GPU utilization.

Implementing Batching
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @serve.deployment
   class BatchedMNISTClassifier:
       def __init__(self, model_path: str = None):
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self.model = MNISTNet().to(self.device)
           
           if model_path:
               self.model.load_state_dict(torch.load(model_path, map_location=self.device))
           
           self.model.eval()

       @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
       async def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
           """Process a batch of images together"""
           print(f"Processing batch of size: {len(images)}")
           
           # Stack all images into a single tensor
           batch_tensor = torch.tensor(
               np.stack(images), 
               dtype=torch.float32
           ).to(self.device)
           
           # Run inference on the entire batch
           with torch.no_grad():
               logits = self.model(batch_tensor)
               predictions = torch.argmax(logits, dim=1).cpu().numpy()
           
           # Return individual results
           return [
               {
                   "predicted_label": int(pred),
                   "logits": logit.cpu().numpy().tolist()
               }
               for pred, logit in zip(predictions, logits)
           ]

       async def __call__(self, request: Request) -> Dict[str, Any]:
           data = await request.json()
           batch = json.loads(data)
           
           # Extract single image and pass to batch handler
           image = np.array(batch["image"])
           result = await self.predict_batch(image)
           
           return result

The ``@serve.batch`` decorator automatically:

- Collects up to ``max_batch_size`` requests
- Waits up to ``batch_wait_timeout_s`` seconds for more requests
- Processes them together in a single forward pass

This can dramatically improve throughput, especially for GPU inference.

Part 5: Composing Multiple Deployments
---------------------------------------

Real-world ML applications often involve multiple steps: preprocessing, inference, and postprocessing.
Ray Serve makes it easy to compose multiple deployments into a pipeline.

Creating a Preprocessing Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @serve.deployment
   class ImagePreprocessor:
       def __init__(self):
           """Initialize preprocessing transforms"""
           self.transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
           ])
       
       async def preprocess(self, images: List[np.ndarray]) -> np.ndarray:
           """Preprocess a batch of images"""
           processed = []
           for img in images:
               # Convert to PIL-compatible format if needed
               if img.dtype != np.uint8:
                   img = (img * 255).astype(np.uint8)
               
               # Apply transforms
               tensor = self.transform(img)
               processed.append(tensor.numpy())
           
           return np.stack(processed)

Creating an Ingress Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ingress deployment orchestrates the pipeline, routing requests through preprocessing
and then to the model:

.. code-block:: python

   @serve.deployment
   class MLPipeline:
       def __init__(self, preprocessor, classifier):
           """Initialize with handles to other deployments"""
           self.preprocessor = preprocessor
           self.classifier = classifier
       
       async def __call__(self, request: Request) -> Dict[str, Any]:
           """Handle end-to-end inference"""
           # Parse request
           data = await request.json()
           batch = json.loads(data)
           images = batch["image"]
           
           # Step 1: Preprocess
           processed_images = await self.preprocessor.preprocess.remote(images)
           
           # Step 2: Run inference
           result = await self.classifier.predict.remote({
               "image": processed_images.tolist()
           })
           
           return result

Deploying the Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Build the application graph
   preprocessor = ImagePreprocessor.bind()
   classifier = MNISTClassifier.options(
       num_replicas=2,
       ray_actor_options={"num_gpus": 0.5}
   ).bind()
   
   pipeline = MLPipeline.bind(
       preprocessor=preprocessor,
       classifier=classifier
   )
   
   # Deploy the entire pipeline
   handle = serve.run(pipeline, name="ml_pipeline")

Now when you send a request to the pipeline, it automatically flows through preprocessing
and inference:

.. code-block:: python

   # Send request to the pipeline
   images = [np.random.rand(28, 28) for _ in range(5)]
   json_request = json.dumps({"image": images})
   
   response = requests.post("http://localhost:8000/", json=json_request)
   print(response.json())

Part 6: Integration with FastAPI
---------------------------------

Ray Serve integrates seamlessly with FastAPI, giving you access to:

- HTTP routing and path parameters
- Request validation with Pydantic models
- Automatic OpenAPI documentation

.. code-block:: python

   from fastapi import FastAPI
   from pydantic import BaseModel

   app = FastAPI()

   class PredictionRequest(BaseModel):
       image: List[List[List[float]]]  # Batch of images

   class PredictionResponse(BaseModel):
       predicted_label: List[int]

   @serve.deployment
   @serve.ingress(app)
   class FastAPIMNISTService:
       def __init__(self):
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self.model = MNISTNet().to(self.device)
           self.model.eval()
       
       @app.post("/predict", response_model=PredictionResponse)
       async def predict(self, request: PredictionRequest):
           """Predict digit from image"""
           images = torch.tensor(
               request.image, 
               dtype=torch.float32
           ).to(self.device)
           
           with torch.no_grad():
               logits = self.model(images)
               predictions = torch.argmax(logits, dim=1).cpu().numpy()
           
           return PredictionResponse(predicted_label=predictions.tolist())
       
       @app.get("/health")
       async def health(self):
           """Health check endpoint"""
           return {"status": "healthy"}

   # Deploy with FastAPI
   fastapi_app = FastAPIMNISTService.bind()
   handle = serve.run(fastapi_app, name="fastapi_mnist")

After deploying, you can:

- Visit ``http://localhost:8000/docs`` for interactive API documentation
- Use the ``/predict`` endpoint for inference
- Use the ``/health`` endpoint for health checks

Cleanup
-------

When you're done, shut down the Ray Serve application:

.. code-block:: python

   serve.shutdown()

Summary
-------

In this tutorial, you learned how to:

- Deploy PyTorch models as web services with Ray Serve
- Scale deployments with multiple replicas and fractional GPU usage
- Configure autoscaling to handle variable workloads
- Use dynamic request batching to maximize throughput
- Compose multiple deployments into ML pipelines
- Send concurrent requests efficiently with async HTTP
- Integrate with FastAPI for production-ready APIs

Ray Serve provides a powerful, flexible framework for serving PyTorch models at scale.
Its Python-first API makes it easy to go from a trained model to a production service.

Next Steps
----------

- For more information on Ray Serve, read the `Ray Serve documentation <https://docs.ray.io/en/latest/serve/index.html>`_.
- Learn about `Ray Distributed <https://docs.ray.io/en/latest/ray-overview.html>`_, the distributed computing framework that powers Ray Serve.
