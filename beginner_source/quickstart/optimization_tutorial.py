"""
`Quickstart <quickstart_tutorial.html>`_ >
`Tensors <tensor_tutorial.html>`_ > 
`DataSets & DataLoaders <dataquickstart_tutorial.html>`_ >
`Transforms <transforms_tutorial.html>`_ >
`Build Model <buildmodel_tutorial.html>`_ >
`Autograd <autograd_tutorial.html>`_ >
**Optimization** >
`Save & Load Model <saveloadrun_tutorial.html>`_

Optimizing Model Parameters
===========================

Now that we have a model and data it's time to train, validate and test our model by optimizing it's parameters on 
our data. Training a model is an iterative process; in each iteration (called an *epoch*) the model makes a guess about the output, calculates the error in its guess (*loss*), collects the derivatives of the error with respect to its parameters (as we saw in the `previous section  <autograd_tutorial.html>`_), and **optimizes** these parameters using gradient descent. For a more detailed walkthrough of this process, check out this video on `backpropagation from 3Blue1Brown <https://www.youtube.com/watch?v=tIeHLnjs5U8>`__.

Hyperparameters
-----------------

Hyperparameters are adjustable parameters that let you control the model optimization process. 
Different hyperparameter values can impact model training and convergence rates (`read more <https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html>`__ about hyperparameter tuning)

In our case, we need to define the following hyperparameters:

 - **Number of Epochs**- the number times to iterate over the dataset
 - **Batch Size** - the number of data samples seen by the model in each epoch
 - **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

.. code-block:: Python

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

We also need to create the model class instance (defined in the previous section):

.. code-block:: Python

    model = NeuralNetwork()

Optimization Loop
-----------------

.. figure:: /_static/img/quickstart/optimizationloops.png
   :alt:

Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each 
iteration of the optimization loop is called an **epoch**. Each epoch consists of two main parts:

  1. **The Train Loop** - main loop that iterates over all dataset and performs training
  2. **The Validation/Test Loop** - goes through the validation / test dataset to evaluate model performance on the test data. 

Here is a high-level view of optimization loop:

.. code-block:: Python

  for epoch in range(num_epochs):  # Iterate over all epochs

    # Training loop:
    for train_features, train_labels in train_dataloader: # Go over all minibatches
        out = model(train_features) # Compute network output
        loss = loss_function(out,train_labels) # Compute loss function
        # optimize weights to minimize loss
        ...

    # Evaluation loop
    model.eval() # set to evaluation mode not to compute gradients
    for test_features, test_labels:
        out = model(test_features)
        loss = loss_function(out,train_labels)
        # store / display the loss and/or other metrics
        ...

Complete code for optimization loop will be presented at the end of this section.

Loss Function
-------------

When presented with some training data, our untrained network is likely not to give the correct 
answer. **Loss function** measures the degree of dissimilarity of obtained result to the target value, 
and it is the loss function that we want to minimize during training. To calculate the loss we make a 
prediction using the inputs of our given data sample and compare it against the true data label value.

Common loss functions include `Mean Square Error <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`_ (for regression tasks), `Negative Log Likelihood <https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss>`_, and `CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss>`_ (for classification tasks).

In our example, we will use the built-in Cross Entropy Loss function:

.. code-block:: Python

    # Initialize the loss function
    loss_function = nn.CrossEntropyLoss()

Optimizer
---------

Optimization is the process of adjusting model paramters on each training step. **Optimization algorithm** defines 
how this process is performed. In this example we use Stochastic Gradient Descent.
All optimization logic is encapsulated in ``optimizer`` object. In our case, we will instantiate the stochastic gradient descent optimizer:

.. code-block:: Python

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

In addition to SGD there are many `different optimizers <https://pytorch.org/docs/stable/optim.html>`_ available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models.

Inside the training loop, optimization happens in three steps:

 * Call ``optimizer.zero_grad()`` function to zero the gradients. As you have seen in the previous section on automatic differentiation, gradients by default add up, so we need to explicitly zero them on each step.
 * Calculate the loss using loss function. This builds a computation graph, which PyTorch uses to automatically update parameters with respect to our model's loss during training. This is done with one call to ``loss.backwards()``. 
 * Once we have our gradients, we call ``optimizer.step()`` to adjust the parameters by the gradients collected in the backward pass.  

.. figure:: https://discuss.pytorch.org/uploads/default/original/1X/c7e0a44b7bcebfb41315b56f8418ce37f0adbfeb.png
   :alt: tensor graph

Putting it all together
-----------------------

Below is the complete code for the optimization loop. If you want a complete runnable example of training the model, refer to the `main quickstart page <quickstart_tutorial.html>`_. The code below is commented to explain what goes on, but essentially it is put together from concepts that we have described above.  

.. code-block:: Python

  for epoch in range(num_epochs): # Do training for each epoch

    # Training loop over all data in minibatches
    for train_batch, (train_inputs, train_labels) in enumerate(train_dataloader):
        model.train()  # Set model to train mode
        # we need to move the data to the devices used for training
        train_inputs, train_labels = 
          train_inputs.to(device), train_labels.to(device)
        optimizer.zero_grad()       # zero out gradients
        pred = model(train_inputs)  # make a prediction on the current batch
        loss = cost_function(pred, train_labels)  # compute loss function
        loss.backward()  # compute gradients of loss function
        optimizer.step()  # update parameters

    # Test loop: go over test dataset
    for test_batch, (test_inputs, test_labels) in enumerate(test_dataloader):
        # move data to the device we use for computations
        test_inputs, test_labels = 
           test_inputs.to(device), test_labels.to(device)
        pred = model(test_inputs)   # evaluate model on test minibatch
        test_loss += cost_function(pred, test_labels).item() # compute loss
        # compute the metrics for classification: 
        # how many classes were guessed correctly
        correct += 
          (pred.argmax(1) == test_labels.argmax(1))
          .type(torch.float).sum().item()

    test_loss /= len(test_dataloader.dataset)
    correct /= len(test_dataloader.dataset)
    print('Epoch {} test Error:'.format(epoch))
    print('acc: {:>0.1f}%, avg loss: {:>8f}'.format(100*correct, test_loss))

Creating Custom Cost Functions
------------------------------

In addition to the included PyTorch cost functions you can create your own custom cost functions as long as they are differentiable. Here is an example of custom Cross Entropy Loss implementation from the `Stanford CS230 <https://cs230.stanford.edu/blog/pytorch/#loss-function>`_ course:

.. code-block:: Python

    def myCrossEntropyLoss(outputs, labels):
        batch_size = outputs.size()[0]
        # compute the log of softmax values
        outputs = F.log_softmax(outputs, dim=1)
        # pick the values corresponding to the labels
        outputs = outputs[range(batch_size), labels]
        return -torch.sum(outputs)/num_examples

It can be called just like the out of the box implementation above.

.. code-block:: Python

    loss = myCrossEntropyLoss(model_prediction, true_value)

A more in depth explanation of PyTorch cost functions is outside the scope of the quickstart but you can learn more
about the different common cost functions for deep learning in the PyTorch `documentation <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.

Using Train/Validation/Test Split to Optimize Hyperparameters
-------------------------------------------------------------

In our example, we have split the data between train and test datasets. However, as we mentioned above, different hyperparameters can yield different model performance. Thus, it makes sense to use a part of the dataset for **hyperparameter optimization**. In this case, we split the dataset into three parts:

 * Training data
 * Validation data, which is used inside optimization loop to determine the accuracy of the current model and the optimal number of epochs. After a certain number of epochs, validation accuracy typically starts to decrease, which means that we have reached optimal performance for given hyperparameters.
 * Test data, which is used to measure the performance of the model for given hyperparameters. It is important that test data are independent from validation data, i.e. the same dataset cannot be used for both validation and test purposes.

We will not consider hyperparameter optimization further in this quickstart. 

Next learn how to `save our trained model <saveloadrun_tutorial.html>`_.

"""
