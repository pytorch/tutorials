Note

Go to the end
to download the full example code.

# Serve PyTorch models at scale with Ray Serve

**Author:** [Ricardo Decal](https://github.com/crypdick)

This tutorial shows how to deploy a PyTorch model using Ray Serve with
production-ready features.

 What you will learn

- How to create a production-ready PyTorch model deployment that can scale to thousands of nodes and GPUs
- How to configure an HTTP endpoint for the deployment using FastAPI
- Enable dynamic request batching for higher throughput
- Configure autoscaling and per-replica CPU/GPU resource allocation
- Load test the service with concurrent requests and monitor it with the Ray dashboard
- How Ray Serve deployments can self-heal from failures
- Ray Serve's advanced features like model multiplexing and model composition.

 Prerequisites

- PyTorch v2.9+ and `torchvision`
- Ray Serve (`ray[serve]`) v2.52.1+
- A GPU is recommended for higher throughput but is not required

[Ray Serve](https://docs.ray.io/en/latest/serve/index.html) is a
scalable framework for serving machine learning models in production.
It's built on top of [Ray](https://docs.ray.io/en/latest/index.html),
which is a unified framework for scaling AI and Python applications that
simplifies the complexities of distributed computing. Ray is also open
source and part of the PyTorch Foundation.

## Setup

To install the dependencies, run:

```
pip install "ray[serve]" torch torchvision
```

Start by importing the required libraries:

## Define a PyTorch model

Define a simple convolutional neural network for the MNIST digit
classification dataset:

## Define the Ray Serve deployment

To deploy this model with Ray Serve, wrap the model in a Python class
and decorate it with `@serve.deployment`.

Processing requests in batches is more efficient than processing
requests one by one, especially when using GPUs. Ray Serve provides
built-in support for **dynamic request batching**, where individual
incoming requests are opportunistically batched. Enable dynamic batching
using the `@serve.batch` decorator as shown in the following code:

This is a FastAPI app, which extends Ray Serve with features like
automatic request validation with Pydantic, auto-generated OpenAPI-style
API documentation, and more.

## Configure autoscaling and resource allocation

In production, traffic can vary significantly. Ray Serve's
**autoscaling** feature automatically adjusts the number of replicas
based on traffic load, ensuring you have enough capacity during peaks
while saving resources during quiet periods. Ray Serve scales to very
large deployments with thousands of nodes and replicas.

You can also specify **resource allocation** per replica, such as the
number of CPUs or GPUs. Ray Serve handles the orchestration of these
resources across your cluster. Ray also supports [fractional
GPUs](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#fractional-accelerators),
allowing multiple replicas to share a single GPU when models are small
enough to fit in memory together.

The following is a sample configuration with autoscaling and resource
allocation:

The app is ready to deploy. Suppose you ran this on a cluster of 10
machines, each with 4 GPUs. With `num_gpus=0.5`, Ray schedules 2
replicas per GPU, giving you 80 replicas across the cluster. This
configuration permits the deployment to elastically scale up to 80
replicas as needed to handle traffic spikes and scale back down to 1
replica when traffic subsides.

## Test the endpoint with concurrent requests

To deploy the app, use the `serve.run` function:

```
# Start the Ray Serve application.
```

You will see output similar to:

```
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

To test the deployment, you can send many requests concurrently using
`aiohttp`. The following code demonstrates how to send 1000 concurrent
requests to the app:

```
# Run the concurrent requests.
```

Ray Serve automatically buffers and load balances requests across the
replicas.

## Fault tolerance

In production, process and machine failures are inevitable. Ray Serve is designed
so that each major component in the Serve stack (the controller, replicas, and proxies) can fail
and recover while your application continues to handle traffic.

Serve can also recover from larger infrastructure failures, such as entire nodes or pods
failing. Serve can even recover from head node failures, or the entire head pod if
deploying on KubeRay.

For more information about Ray Serve's fault tolerance, see the
[Ray Serve fault-tolerance guide](https://docs.ray.io/en/master/serve/production-guide/fault-tolerance.html).

## Monitor the deployment

Monitoring is critical when running large-scale deployments. The [Ray
dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
displays Serve metrics like request throughput, latency, and error
rates. It also shows cluster resource usage, replica status, and overall
deployment health in real time. The dashboard also lets you inspect logs
from individual replicas across the cluster.

For debugging, Ray offers [distributed debugging
tools](https://docs.ray.io/en/latest/ray-observability/index.html)
that let you attach a debugger to running replicas across the cluster.
For more information, see the [Ray Serve monitoring
documentation](https://docs.ray.io/en/latest/serve/monitoring.html).

## Conclusion

In this tutorial, you:

- Deployed a PyTorch model using Ray Serve with production best
practices.
- Enabled **dynamic request batching** to optimize performance.
- Configured **autoscaling** and **fractional GPU allocation** to
efficiently scale across a cluster.
- Tested the service with concurrent asynchronous requests.

## Further reading

Ray Serve has more production features that are out of scope for this
tutorial but are worth checking out:

- Specialized [large language model (LLM) serving
APIs](https://docs.ray.io/en/latest/serve/llm/index.html) that
handle complexities like managing key-value (KV) caches and continuous
batching.
- [Model
multiplexing](https://docs.ray.io/en/latest/serve/model-multiplexing.html)
to dynamically load and serve many different models on the same
deployment. This is useful for serving per-user fine-tuned models, for
example.
- [Composed
deployments](https://docs.ray.io/en/latest/serve/model_composition.html)
to orchestrate multiple deployments into a single app.

For more information, see the [Ray Serve
documentation](https://docs.ray.io/en/latest/serve/index.html) and
[Ray Serve
examples](https://docs.ray.io/en/latest/serve/examples.html).

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.003 seconds)

[`Download Jupyter notebook: serving_tutorial.ipynb`](../_downloads/095f6a9fedd89af43ae07761382ef458/serving_tutorial.ipynb)

[`Download Python source code: serving_tutorial.py`](../_downloads/d2b155dc9cc96275bea8c0d67032bca1/serving_tutorial.py)

[`Download zipped: serving_tutorial.zip`](../_downloads/c7aeacfb336810ce37328988c0da8606/serving_tutorial.zip)