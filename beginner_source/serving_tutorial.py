"""
Serve PyTorch models at scale with Ray Serve
============================================

**Author:** `Ricardo Decal <https://github.com/crypdick>`__

This tutorial shows how to deploy a PyTorch model using Ray Serve with
production-ready features.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to create a production-ready PyTorch model deployment that can scale to thousands of nodes and GPUs
       * How to configure an HTTP endpoint for the deployment using FastAPI
       * Enable dynamic request batching for higher throughput
       * Configure autoscaling and per-replica CPU/GPU resource allocation
       * Load test the service with concurrent requests and monitor it with the Ray dashboard
       * How Ray Serve deployments can self-heal from failures
       * Ray Serve's advanced features like model multiplexing and model composition.

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.9+ and ``torchvision``
       * Ray Serve (``ray[serve]``) v2.52.1+
       * A GPU is recommended for higher throughput but is not required

`Ray Serve <https://docs.ray.io/en/latest/serve/index.html>`__ is a
scalable framework for serving machine learning models in production.
It’s built on top of `Ray <https://docs.ray.io/en/latest/index.html>`__,
which is a unified framework for scaling AI and Python applications that
simplifies the complexities of distributed computing. Ray is also open
source and part of the PyTorch Foundation.


Setup
-----

To install the dependencies:

"""

# %%bash
# pip install "ray[serve]" torch torchvision

######################################################################
# Start by importing the required libraries:

import asyncio
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
import aiohttp
import numpy as np
import torch
import torch.nn as nn
from ray import serve
from torchvision.transforms import v2

######################################################################
# Define a PyTorch model
# ----------------------
#
# Define a simple convolutional neural network for the MNIST digit
# classification dataset:

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
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

######################################################################
# Define the Ray Serve deployment
# -------------------------------
#
# To deploy this model with Ray Serve, wrap the model in a Python class
# and decorate it with ``@serve.deployment``.
#
# Processing requests in batches is more efficient than processing
# requests one by one, especially when using GPUs. Ray Serve provides
# built-in support for **dynamic request batching**, where individual
# incoming requests are opportunistically batched. Enable dynamic batching
# using the ``@serve.batch`` decorator as shown in the following code:

app = FastAPI()

class ImageRequest(BaseModel):  # Used for request validation and generating API documentation
    image: list[list[float]] | list[list[list[float]]]  # 2D or 3D array

@serve.deployment
@serve.ingress(app)
class MNISTClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTNet().to(self.device)
        # Define the transformation pipeline for the input images.
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # Mean and standard deviation of the MNIST training subset.
            v2.Normalize(mean=[0.1307], std=[0.3013]),
        ])

        self.model.eval()

    # batch_wait_timeout_s is the maximum time to wait for a full batch,
    # trading off latency for throughput.
    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.1)
    async def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        # Stack all images into a single tensor.
        batch_tensor = torch.cat([
            self.transform(img).unsqueeze(0)
            for img in images
        ]).to(self.device).float()

        # Single forward pass on the entire batch at once.
        with torch.no_grad():
            logits = self.model(batch_tensor)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

        # Unbatch the results and preserve their original order.
        return [
            {
                "predicted_label": int(pred),
                "logits": logit.cpu().numpy().tolist()
            }
            for pred, logit in zip(predictions, logits)
        ]

    @app.post("/")
    async def handle_request(self, request: ImageRequest):
        """Handle an incoming HTTP request using FastAPI.

        Inputs are automatically validated using the Pydantic model.
        """
        # Process the single request.
        image_array = np.array(request.image)

        # Ray Serve's @serve.batch automatically batches requests.
        result = await self.predict_batch(image_array)

        return result


######################################################################
# This is a FastAPI app, which extends Ray Serve with features like
# automatic request validation with Pydantic, auto-generated OpenAPI-style
# API documentation, and more.
#
# Configure autoscaling and resource allocation
# ---------------------------------------------
#
# In production, traffic can vary significantly. Ray Serve’s
# **autoscaling** feature automatically adjusts the number of replicas
# based on traffic load, ensuring you have enough capacity during peaks
# while saving resources during quiet periods. Ray Serve scales to very
# large deployments with thousands of nodes and replicas.
#
# You can also specify **resource allocation** per replica, such as the
# number of CPUs or GPUs. Ray Serve handles the orchestration of these
# resources across your cluster. Ray also supports `fractional
# GPUs <https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#fractional-accelerators>`__,
# allowing multiple replicas to share a single GPU when models are small
# enough to fit in memory together.
#
# The following is a sample configuration with autoscaling and resource
# allocation:

num_cpus_per_replica = 1
num_gpus_per_replica = 1  # Set to 0 to run the model on CPUs instead of GPUs.
mnist_app = MNISTClassifier.options(
    autoscaling_config={
        "target_ongoing_requests": 50,  # Target 50 ongoing requests per replica.
        "min_replicas": 1,              # Keep at least 1 replica alive.
        "max_replicas": 80,             # Scale up to 80 replicas to maintain target_ongoing_requests.
        "upscale_delay_s": 5,           # Wait 5s before scaling up.
        "downscale_delay_s": 30,        # Wait 30s before scaling down.
    },
    # Max invocations to handle_request per replica to process simultaneously.
    # Requests exceeding this limit are queued by the router until capacity is available.
    max_ongoing_requests=200,
    # Max queue size for requests that exceed max_ongoing_requests.
    # If the queue is full, future requests are backpressured with errors until space is available.
    # -1 means the queue can grow until cluster memory is exhausted.
    max_queued_requests=-1,
    # Set the resources per replica.
    ray_actor_options={"num_cpus": num_cpus_per_replica, "num_gpus": num_gpus_per_replica}
).bind()

######################################################################
# The app is ready to deploy. Suppose you ran this on a cluster of 10
# machines, each with 4 GPUs. With ``num_gpus=0.5``, Ray schedules 2
# replicas per GPU, giving you 80 replicas across the cluster. This
# configuration permits the deployment to elastically scale up to 80
# replicas as needed to handle traffic spikes and scale back down to 1
# replica when traffic subsides.
#
# Test the endpoint with concurrent requests
# ------------------------------------------
#
# To deploy the app, use the ``serve.run`` function:

# Start the Ray Serve application.
handle = serve.run(mnist_app, name="mnist_classifier")

######################################################################
# You will see output similar to:

# %%bash
# Started Serve in namespace "serve".
# Registering autoscaling state for deployment Deployment(name='MNISTClassifier', app='mnist_classifier')
# Deploying new version of Deployment(name='MNISTClassifier', app='mnist_classifier') (initial target replicas: 1).
# Proxy starting on node ... (HTTP port: 8000).
# Got updated endpoints: {}.
# Got updated endpoints: {Deployment(name='MNISTClassifier', app='mnist_classifier'): EndpointInfo(route='/', app_is_cross_language=False, route_patterns=None)}.
# Started <ray.serve._private.router.SharedRouterLongPollClient object at 0x73a53c52c250>.
# Adding 1 replica to Deployment(name='MNISTClassifier', app='mnist_classifier').
# Got updated endpoints: {Deployment(name='MNISTClassifier', app='mnist_classifier'): EndpointInfo(route='/', app_is_cross_language=False, route_patterns=['/', '/docs', '/docs/oauth2-redirect', '/openapi.json', '/redoc'])}.
# Application 'mnist_classifier' is ready at http://127.0.0.1:8000/.

######################################################################
# The app is now listening for requests on port 8000.
#
# To test the deployment, you can send many requests concurrently using
# ``aiohttp``. The following code demonstrates how to send 1000 concurrent
# requests to the app:

async def send_single_request(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()

async def send_concurrent_requests(num_requests):
    image = np.random.rand(28, 28).tolist()

    print(f"Sending {num_requests} concurrent requests...")
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_single_request(session, url="http://localhost:8000/", data={"image": image})
            for _ in range(num_requests)
        ]
        responses = await asyncio.gather(*tasks)

    return responses

# Run the concurrent requests.
start_time = time.time()
responses = asyncio.run(send_concurrent_requests(1000))
elapsed = time.time() - start_time

print(f"Processed {len(responses)} requests in {elapsed:.2f} seconds")
print(f"Throughput: {len(responses)/elapsed:.2f} requests/second")

######################################################################
# Ray Serve automatically buffers and load balances requests across the
# replicas.
#
# Fault tolerance
# ---------------
#
# In production, process and machine failures are inevitable. Ray Serve is designed
# so that each major component in the Serve stack (the controller, replicas, and proxies) can fail
# and recover while your application continues to handle traffic.
#
# Serve can also recover from larger infrastructure failures, such as entire nodes or pods
# failing. Serve can even recover from head node failures, or the entire head pod if
# deploying on KubeRay.
#
# For more information about Ray Serve's fault tolerance, see the
# `Ray Serve fault-tolerance guide <https://docs.ray.io/en/master/serve/production-guide/fault-tolerance.html>`__
#
# Monitor the deployment
# ----------------------
#
# Monitoring is critical when running large-scale deployments. The `Ray
# dashboard <https://docs.ray.io/en/latest/ray-observability/getting-started.html>`__
# displays Serve metrics like request throughput, latency, and error
# rates. It also shows cluster resource usage, replica status, and overall
# deployment health in real time. The dashboard also lets you inspect logs
# from individual replicas across the cluster.
#
# For debugging, Ray offers `distributed debugging
# tools <https://docs.ray.io/en/latest/ray-observability/index.html>`__
# that let you attach a debugger to running replicas across the cluster.
# For more information, see the `Ray Serve monitoring
# documentation <https://docs.ray.io/en/latest/serve/monitoring.html>`__.
#
# Summary
# -------
#
# In this tutorial, you:
#
# - Deployed a PyTorch model using Ray Serve with production best
#   practices.
# - Enabled **dynamic request batching** to optimize performance.
# - Configured **autoscaling** and **fractional GPU allocation** to
#   efficiently scale across a cluster.
# - Tested the service with concurrent asynchronous requests.
#
# Further reading
# ---------------
#
# Ray Serve has more production features that are out of scope for this
# tutorial but are worth checking out:
#
# - Specialized `large language model (LLM) serving
#   APIs <https://docs.ray.io/en/latest/serve/llm/index.html>`__ that
#   handle complexities like managing key-value (KV) caches and continuous
#   batching.
# - `Model
#   multiplexing <https://docs.ray.io/en/latest/serve/model-multiplexing.html>`__
#   to dynamically load and serve many different models on the same
#   deployment. This is useful for serving per-user fine-tuned models, for
#   example.
# - `Composed
#   deployments <https://docs.ray.io/en/latest/serve/model_composition.html>`__
#   to orchestrate multiple deployments into a single app.
#
# For more information, see the `Ray Serve
# documentation <https://docs.ray.io/en/latest/serve/index.html>`__ and
# `Ray Serve
# examples <https://docs.ray.io/en/latest/serve/examples.html>`__.
