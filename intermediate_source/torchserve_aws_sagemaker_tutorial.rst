TorchServe on AWS SageMaker
============================

In this tutorial you will learn how you can efficiently serve PyTorch models using Torchserve and AWS Sagemaker

Why TorchServe?
^^^^^^^^^^^^^^^^

| TorchServe is the recommended model server for PyTorch, preinstalled in the AWS PyTorch Deep Learning Container (DLC). This powerful tool offers customers a consistent and user-friendly experience, delivering high performance in deploying multiple PyTorch models across various AWS instances, including CPU, GPU, Neuron, and Graviton, regardless of the model size or distribution.
| TorchServe is easy to use. It comes with a convenient CLI to deploy locally and is easy to package into a container and scale out with Amazon SageMaker or Amazon EKS. With default handlers for common problems such as image classification, object detection, image segmentation, and text classification, you can deploy with just a few lines of code—no more writing lengthy service handlers for initialization, preprocessing, and post-processing. TorchServe is open-source, which means it's fully open and extensible to fit your deployment needs.

To get started on how to use TorchServe you can refer to this tutorial: `TorchServe QuickStart  <https://pytorch.org/serve/getting_started.html>`_

The following table lists the AWS PyTorch DLCs supported by TorchServe
````````````````````````````````````````````````````````````````````````

.. list-table::
  :header-rows: 1

  * - Instance type
    - SageMaker PyTorch DLC link
  * - CPU and GPU
    - `SageMaker PyTorch containers <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#sagemaker-framework-containers-sm-support-only>`_
  * - Neuron
    - `PyTorch Neuron containers <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-containers>`_
  * - Graviton
    - `SageMaker PyTorch Graviton containers <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#sagemaker-framework-graviton-containers-sm-support-only>`_



You can follow along with this tutorial through an Amazon EC2 instance, or your laptop or desktop. If you're using a local laptop or desktop, make sure you download and install the `AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html>`_ and configure it, `AWS SDK for Python (boto3) <https://aws.amazon.com/sdk-for-python/>`_, and `Amazon SageMaker Python SDK <https://github.com/aws/sagemaker-python-sdk#installing-the-sagemaker-python-sdk>`_. After you deploy, the models are hosted on Amazon SageMaker fully managed deployment instances.

The code, configuration files, Jupyter notebooks, and Dockerfiles used in this post are available on `GitHub <https://github.com/shashankprasanna/torchserve-examples.git>`_. The steps in the following example are from the ``deploy_torchserve.ipynb`` Jupyter notebook.

Cloning the example repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To clone the example repository, enter the following code:

    .. code:: shell

        git clone https://github.com/shashankprasanna/torchserve-examples.git
        cd torchserve-examples

Clone the TorchServe repository a nd install torch-model-archiver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``torch-model-archiver`` tool to create a model archive file. The .mar model archive file contains model checkpoints along with it's ``state_dict`` (dictionary object that maps each layer to its parameter tensor).

    .. code:: shell

        git clone https://github.com/pytorch/serve.git
        pip install serve/model-archiver

To download a PyTorch model and create a TorchServe archive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One can customise the handler by passing the `<custom_handler.py> <https://github.com/pytorch/serve/blob/master/docs/custom_service.md>`_ instead of ``image_classifier``

    .. code:: shell

        wget -q https://download.pytorch.org/models/densenet161-8d451a50.pth

        export model_file_name='densenet161'
        
        torch-model-archiver --model-name $model_file_name \
            --version 1.0 --model-file serve/examples/image_classifier/densenet_161/model.py \
            --serialized-file densenet161-8d451a50.pth \
            --extra-files serve/examples/image_classifier/index_to_name.json \
            --handler image_classifier

Uploading the model to Amazon S3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To upload the model to Amazon S3, complete the following steps:

    #. Create a boto3 session and get the Region and account information

        .. code:: python3 

            import boto3, time, json
            sess    = boto3.Session()
            sm      = sess.client('sagemaker')
            region  = sess.region_name
            account = boto3.client('sts').get_caller_identity().get('Account')

            import sagemaker
            role = sagemaker.get_execution_role()
            sagemaker_session = sagemaker.Session(boto_session=sess)

            Get the default Amazon SageMaker S3 bucket name

            bucket_name = sagemaker_session.default_bucket()
            prefix = 'torchserve'

    #. Create a compressed tar.gz file out of the densenet161.mar file, because Amazon SageMaker expects models to be in a tar.gz file.

        .. code:: shell

            tar cvfz $model_file_name.tar.gz densenet161.mar

    #. Upload the model to your S3 bucket under the models directory.

        .. code:: shell

            aws s3 cp $model_file_name.tar.gz s3://{bucket_name}/{prefix}/model

Creating an Amazon ECR registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Create a new Docker container registry for your TorchServe container images. Amazon SageMaker pulls the TorchServe container from this registry. See the following code:

        .. code:: python3 

            registry_name = 'torchserve'

        .. code:: shell

            aws ecr create-repository --repository-name torchserve

Building a TorchServe Docker container and pushing it to Amazon ECR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository for this post already contains a Dockerfile for building a TorchServe container. Build a Docker container image locally and push it to your Amazon ECR repository you created in the previous step. See the following code:

        .. code:: python3 

            image_label = 'v1'
            image = f'{account}.dkr.ecr.{region}.amazonaws.com/{registry_name}:{image_label}'

        .. code:: shell

            docker build -t {registry_name}:{image_label} .
            $(aws ecr get-login --no-include-email --region {region})
            docker tag {registry_name}:{image_label} {image}
            docker push {image}

    You get the following output confirming that the container was built and pushed to Amazon ECR successfully:
    
        .. image:: static/torchserve_container_amazonECR.png
            :alt: output when docker container was successfully built and pushed to Amazon ECR

Hosting an inference endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    There are multiple ways to host an inference endpoint and make predictions. The quickest approach is to use the Amazon SageMaker Python SDK. However, if you're going to invoke the endpoint from a client application, you should use `Amazon SDK <https://aws.amazon.com/tools/>`_ for the language of your choice.
        
    Hosting an inference endpoint and making predictions with Amazon SageMaker Python SDK

    To host an inference endpoint and make predictions using Amazon SageMaker Python SDK, complete the following steps:

    #. Create a model. The model function expects the name of the TorchServe container image and the location of your trained models. See the following code:

        .. code:: python3 

            import sagemaker
            from sagemaker.model import Model
            from sagemaker.predictor import RealTimePredictor
            role = sagemaker.get_execution_role()

            model_data = f's3://{bucket_name}/models/$model_file_name.tar.gz'
            sm_model_name = 'torchserve-densenet161'

            torchserve_model = Model(model_data = model_data, 
                                    image = image,
                                    role = role,
                                    predictor_cls=RealTimePredictor,
                                    name = sm_model_name)

        For more information about the model function, see `Model <https://sagemaker.readthedocs.io/en/stable/model.html>`_
    
    #. On the Amazon SageMaker console, to see the model details, choose Models.

        .. image:: static/torchserve_model_hosting_aws_sagemaker.png 
            :alt: image of aws sagemaker console showing model details
    
    #. Deploy the model endpoint. Specify the instance type and number of instances you want Amazon SageMaker to run the container on. See the following code:

        .. code:: python3 

            endpoint_name = 'torchserve-endpoint-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
            predictor = torchserve_model.deploy(instance_type='ml.m4.xlarge',
                initial_instance_count=1,
                endpoint_name = endpoint_name)

        You can also set it up to automatically scale based on metrics, such as the total number of invocations. For more information, see `Automatically Scale Amazon SageMaker Models <https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html>`_
    
    #. On the Amazon SageMaker console, to see the hosted endpoint, choose Endpoints.
    
        .. image:: static/torchserve_endpoint_aws_sagemaker.png
            :alt: detail about endpoint on aws sagemaker console

    #. Test the model with the following code:

        .. code:: shell

            wget -q https://s3.amazonaws.com/model-server/inputs/kitten.jpg 
    
        .. code:: python3 

            file_name = 'kitten.jpg'
            with open(file_name, 'rb') as f:
            payload = f.read()
            payload = payload

            response = predictor.predict(data=payload)
            print(*json.loads(response), sep = '\n')
        
        The following screenshot shows the output of invoking the model hosted by TorchServe. The model thinks the kitten in the image is either a tiger cat or a tabby cat.

        .. image:: static/torchserve_model_output_aws_sagemaker.png
            :alt: model's response corresponding to the payload image


    If you're building applications such as mobile apps or webpages that need to invoke the TorchServe endpoint for getting predictions on new data, you can use Amazon API rather than the Amazon SageMaker SDK. For example, if you're using Python on the client side, use the Amazon SDK for Python (boto3). For an example of how to use boto3 to create a model, configure an endpoint, create an endpoint, and finally run inferences on the inference endpoint, refer to this example `Jupyter notebook on GitHub. <https://github.com/shashankprasanna/torchserve-examples/blob/master/deploy_torchserve.ipynb>`_


Metrics
~~~~~~~~

TorchServe supports both system level and model level metrics. You can enable metrics in either log format mode or Prometheus mode through the environment variable TS_METRICS_MODE. You can use the TorchServe central metrics config file metrics.yaml to specify the types of metrics to be tracked, such as request counts, latency, memory usage, GPU utilization, and more. By referring to this file, you can gain insights into the performance and health of the deployed models and effectively monitor the TorchServe server's behavior in real-time. For more detailed information, see the `TorchServe metrics documentation <https://github.com/pytorch/serve/blob/master/docs/metrics.md#torchserve-metrics>`_. You can access TorchServe metrics logs that are similar to the StatsD format through the Amazon CloudWatch log filter. The following is an example of a TorchServe metrics log:

    .. code:: shell

        CPUUtilization.Percent:0.0|#Level:Host|#hostname:my_machine_name,timestamp:1682098185
        DiskAvailable.Gigabytes:318.0416717529297|#Level:Host|#hostname:my_machine_name,timestamp:1682098185

Reference
~~~~~~~~~~

- `Deploying PyTorch models for inference at scale using TorchServe <https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-models-for-inference-at-scale-using-torchserve/>`_
- `Deploy models with TorchServe <https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-models-frameworks-torchserve.html>`_
- `Running TorchServe <https://pytorch.org/serve/server.html>`_