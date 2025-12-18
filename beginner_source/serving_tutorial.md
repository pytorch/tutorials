---
jupyter:
  jupytext:
    default_lexer: ipython3
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Serve PyTorch models at scale with Ray Serve

**Author:** [Ricardo Decal](https://github.com/crypdick)

This tutorial shows how to deploy a PyTorch model using Ray Serve with production-ready features.

[Ray Serve](https://docs.ray.io/en/latest/serve/index.html) is a scalable framework for serving machine learning models in production built on top of Ray. [Ray](https://docs.ray.io/en/latest/index.html), a project of the PyTorch Foundation, is an open-source unified framework for scaling AI and Python applications. Ray simplifies distributed workloads by handling the complexity of distributed computing.

In this tutorial, you'll learn how to deploy a PyTorch model with Ray Serve and use its production-ready features. Ray Serve allows you to easily scale your model inference across multiple nodes and GPUs, providing features like dynamic batching, autoscaling, fault tolerance, and observability out of the box.

## Setup

Install the dependencies:

```bash
pip install "ray[serve]" torch torchvision
```

Start by importing the required libraries:

```python
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
```

## Define a PyTorch model

Define a simple convolutional neural network for MNIST digit classification:

```python
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

```

## Define the Ray Serve deployment

To deploy this model with Ray Serve, wrap the model in a Python class and decorate it with `@serve.deployment`.

Processing requests in batches is more efficient than processing requests one by one, especially when using GPUs. Ray Serve provides built-in support for **dynamic request batching**, where individual incoming requests are opportunistically batched. The `@serve.batch` decorator on the `predict_batch` method below enables this.

```python
app = FastAPI()

class ImageRequest(BaseModel):  # Used for request validation and documentation
    image: list[list[float]] | list[list[list[float]]]

@serve.deployment
@serve.ingress(app)
class MNISTClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTNet().to(self.device)
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.1307], std=[0.3013]),
        ])

        self.model.eval()

    # batch_wait_timeout_s is the maximum time to wait for a full batch.
    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.1)
    async def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        # Stack all images into a single tensor.
        batch_tensor = torch.cat([
            self.transform(img).unsqueeze(0) 
            for img in images
        ]).to(self.device).float()
        
        # Run inference on the entire batch.
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

        # Ray Serve's @serve.batch will automatically batch requests.
        result = await self.predict_batch(image_array)
        
        return result

```

This is a FastAPI app, which gives us batteries-included features like automatic request validation (via Pydantic), OpenAPI-style API documentation, and more.

### Configure autoscaling and resource allocation

In production, traffic can vary significantly. Ray Serve's **autoscaling** feature automatically adjusts the number of replicas based on traffic load, ensuring you have enough capacity during peaks while saving resources during quiet periods.

You can also specify **resource allocation** per replica, such as the number of CPUs or GPUs. Ray Serve handles the orchestration of these resources across your cluster.

Below is a sample configuration with autoscaling and resource allocation:

```python
mnist_app = MNISTClassifier.options(
    autoscaling_config={
        "target_ongoing_requests": 50,  # Target 50 ongoing requests per replica.
        "min_replicas": 1,              # Keep at least 1 replica alive.
        "max_replicas": 5,              # Scale up to 5 replicas to maintain target_ongoing_requests.
        "upscale_delay_s": 5,           # Wait 5s before scaling up.
        "downscale_delay_s": 30,        # Wait 30s before scaling down.
    },
    # Max concurrent requests per replica before queueing.
    # If the queue fills the shared cluster memory, future requests are backpressured until memory is freed.
    max_ongoing_requests=100,
    ray_actor_options={"num_cpus": 1, "num_gpus": 1} 
).bind()
```

The app is now ready to be deployed.

## Testing the endpoint with with concurrent requests

To deploy the app, use the `serve.run` function:

```python
# Start the Ray Serve application.
handle = serve.run(mnist_app, name="mnist_classifier")
```

You should see an output similar to:

```bash
Started Serve in namespace "serve".
Registering autoscaling state for deployment Deployment(name='MNISTClassifier', app='mnist_classifier')
Deploying new version of Deployment(name='MNISTClassifier', app='mnist_classifier') (initial target replicas: 1).
Proxy starting on node ... (HTTP port: 8000).
Got updated endpoints: {}.
Got updated endpoints: {Deployment(name='MNISTClassifier', app='mnist_classifier'): EndpointInfo(route='/', app_is_cross_language=False, route_patterns=None)}.
Started <ray.serve._private.router.SharedRouterLongPollClient object at 0x73a53c52c250>.
Adding 1 replica to Deployment(name='MNISTClassifier', app='mnist_classifier').
Got updated endpoints: {Deployment(name='MNISTClassifier', app='mnist_classifier'): EndpointInfo(route='/', app_is_cross_language=False, route_patterns=['/', '/docs', '/docs/oauth2-redirect', '/openapi.json', '/redoc'])}.
Application 'mnist_classifier' is ready at http://127.0.0.1:8000/.
```

The app is now listening for requests on port 8000.

To test the batching, you can send many requests concurrently using `aiohttp`. Below is a sample function that sends 2000 concurrent requests to the app:

```python
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
responses = asyncio.run(send_concurrent_requests(2000))
elapsed = time.time() - start_time

print(f"Processed {len(responses)} requests in {elapsed:.2f} seconds")
print(f"Throughput: {len(responses)/elapsed:.2f} requests/second")
```

You should see high throughput numbers, confirming that requests are being batched and processed in parallel across the replicas.

## Monitoring the deployment

Ray Serve provides built-in monitoring tools to help you track the status and performance of your deployment.
This dashboard lets you view Serving metrics like request throughput, latency, and error rates, as well as cluster status and resource utilization. For more information, see the [Ray Serve monitoring documentation](https://docs.ray.io/en/latest/serve/monitoring.html).

## Summary

In this tutorial, you learned how to:

- Deploy PyTorch models using Ray Serve with production best practices.
- Enable **dynamic request batching** to optimize performance.
- Configure **autoscaling** to handle traffic spikes.
- Test the service with concurrent asynchronous requests.

## Further reading

Ray Serve has more production features that are out of scope for this tutorial, but are worth checking out:

- Specialized **LLM serving APIs** that handles complexities like managing KV caches and continuous batching.
- **Model multiplexing** to dynamically load and serve many different models (e.g., per-user fine-tuned models) on a single deployment.
- **Composed Deployments** to orchestrate multiple deployments into a single application.

For more information, see the [Ray Serve documentation](https://docs.ray.io/en/latest/serve/index.html) and [Ray Serve examples](https://docs.ray.io/en/latest/serve/examples/index.html).
