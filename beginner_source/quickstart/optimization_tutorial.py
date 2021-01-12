"""
.. include:: /beginner_source/quickstart/qs_toc.txt

6. Optimizing Model Parameters
===========================
Now that we have a model and data it's time to train, validate and test our model by optimizating it's paramerters on our data! 
To get started lets take a look at some example model optimization code:
"""

# Initilize hyper parameters
learning_rate = 0.01
num_epochs = 100

# Initilize model, optimizer and example cost function
model = NeuralNework()  # From Previous Model Section
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # optimizer
cost_function = nn.CrossEntropyLoss()

# For loop to iterate over epoch
for epoch in range(num_epochs):
    # Train loop over batches
    for train_batch, (train_inputs, train_labels) in enumerate(train_dataloader):
        model.train()  # Set model to train mode
        train_inputs, train_labels = train_inputs.to(
            device), train_labels.to(device)
        optimizer.zero_grad()  # zero out gradient
        pred = model(train_inputs)  # make a prediction on this batch!
        loss = cost_function(pred, train_labels)  # how bad is it?
        loss.backward()  # compute gradients
        optimizer.step()  # update parameters

        # validation loop
        model.eval()  # Set model to evaluate mode and start validation loop
        for val_batch, (val_inputs, val_labels) in enumerate(val_dataloader):
            val_inputs, val_labels = val_inputs.to(
                device), val_labels.to(device)
            pred = model(val_inputs)
            test_loss += cost_function(pred, val_labels).item()
            correct += (pred.argmax(1) == val_labels.argmax(1)
                        ).type(torch.float).sum().item()
        val_loss /= len(val_dataloader.dataset)
        correct /= len(val_dataloader.dataset)
        print('\nValidation Error:')
        print('acc: {:>0.1f}%, avg loss: {:>8f}'.format(100*correct, val_loss))
        # Make any additonal hyperparameter modifications here

    # Test loop
    for test_batch, (test_inputs, test_labels) in enumerate(test_dataloader):
        test_inputs, test_labels = test_inputs.to(
            device), test_labels.to(device)
        pred = model(test_inputs)
        test_loss += cost_function(pred, test_labels).item()
        correct += (pred.argmax(1) == test_labels.argmax(1)
                    ).type(torch.float).sum().item()
    test_loss /= len(test_dataloader.dataset)
    correct /= len(test_dataloader.dataset)
    print('\nTest Error:')
    print('acc: {:>0.1f}%, avg loss: {:>8f}'.format(100*correct, test_loss))

######################################################
# To understand this code need to understand a how to handle 4 core deep learning concepts in PyTorch:
#
# 1. Hyperparameters (learning rates, batch sizes, epochs etc)
# 2. Optimization Loops
# 3. Loss
# 4. Optimizers
#
# Let's dissect these core concepts one by one by the time we end every line the code above will make sense.
#
# Hyperparameters
# -----------------


######################################################
# Hyperparameters are adjustable parameters that let you control the model optimization process. For example, with neural networks, you can configure:
#
# - **Number of Epochs**- the number times iterate over the dataset to update model parameters
# - **Batch Size** - the number of samples in the dataset to evaluate before you update model parameters
# - **Learning Rate** - how much to update models parameters at each batch/epoch. Set this value too large and your model won't learn optimally if you set it too small and it will learn really slowly.

learning_rate = 1e-3
batch_size = 64
epochs = 5

######################################################
# Optimizaton Loops
# -----------------
#
# Once we set our hyperparameters, we can then train and optimize our model with an optimization loop.
#
# Each iteration of the optimiziation loop is called an Epoch. Each epoch is comprized of three main subloops in PyTorch.
#

############################################################
# .. figure:: /_static/img/quickstart/optimizationloops.png
#    :alt:
#

#############################################################
#  1. **The Train Loop** -  Core loop iterates over all the epochs
#  2. **The Validation Loop** - Validate  loss after each weight parameter update and can be used to gauge hyper parameter performance and update them for the next batch.
#  3. **The Test Loop** - is used to evaluate our models performance after each epoch on traditional metrics to show how much our model is generalizing from the train and validation dataset to the test dataset it's never seen before.
#

for epoch in range(num_epochs):  # Optimization Loop
    # Train loop over batches
    model.train()  # set model to train
    # Model Update Code
    model.eval()  # After exiting batch loop set model to eval to speed up evaluation and not track gradients (this is explained below)
    # Validation Loop
    # - Validation metric logging and hyperparameter update happens here
    # After exiting train loop set model to eval to speed up evaluation and not track gradients (this is explained below)
    # Test Loop
    # - Test preformance happens here

######################################################
# Loss and Cost Function
# ----------------------
#
# The loss is the value used to update our parameters. To calculate the loss we make a prediction using the inputs of our given data sample and compare it with a cost function against the true data label value.
#

preds = model(inputs)
loss = cost_function(preds, labels)

######################################################
# Common loss functions include `Mean Square Error <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`_,
# `Negative Log Likelihood <https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss>`_,
# and `CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss>`_.
# Here is an example built in Cross Entropy Loss cost function call from the PyTorch nn module.
#

cost_function = nn.CrossEntropyLoss()
loss = cost_function(model_prediction, true_value)

######################################################
# In addition to the included PyTorch cost functions you can create your own custom cost functions as long as they are differentiable.
#
# See this example custom Cross Entropy Loss implementation from the `Stanford CS230 <https://cs230.stanford.edu/blog/pytorch/#loss-function>`_ course below.
#


def myCrossEntropyLoss(outputs, labels):
    batch_size = outputs.size()[0]               # batch_size
    # compute the log of softmax values
    outputs = F.log_softmax(outputs, dim=1)
    # pick the values corresponding to the labels
    outputs = outputs[range(batch_size), labels]
    return -torch.sum(outputs)/num_examples

######################################################
# It can be called just like the out of the box implementation above.
#


loss = myCrossEntropyLoss(model_prediction, true_value)

######################################################
# A more in depth explanation of PyTorch cost functions is outside the scope of the blitz but you can learn more
# about the different common cost functions for deep learning in the PyTorch `documentation <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.
#
# Optimizer
# ---------
# Using the loss, we can then optimize our models parameters. By default, each tensor maintains
# a graph of every operation applied on it unless otherwise specified using the torch.no_grad() command.

############################################################
# .. figure:: https://discuss.pytorch.org/uploads/default/original/1X/c7e0a44b7bcebfb41315b56f8418ce37f0adbfeb.png
#    :alt: tensor graph
#
# PyTorch uses this graph to automatically update parameters with respect to our model's loss during training. This is done with one
# line ``loss.backwards()``. Once we have our gradients the optimizer is used to propgate the gradients from the backwards command
# to update all the parameters in our model.

optimizer.zero_grad()  # make sure previous gradients are cleared
loss.backward()  # calculates gradients with respect to loss
optimizer.step()

######################################################
# The standard method for optimization is called Stochastic Gradient Descent, to learn more check out this awesome
# video by `3blue1brown <https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi>`_.
#
# An Optimizer can be initalized with the Pytorch optim module, as an example lets initialize an SGD optimizer.
# The PyTorch SGD optimizer takes our model and our learning rate hyperparameter as input.

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

######################################################
# In addition to SGD there are many different optimizers and variations of this method in PyTorch such
# as ADAM and RMSProp, that work better for different kinds of models. They are outside the scope
# of this Blitz, but can check out the full list of optimizers `here <https://pytorch.org/docs/stable/optim.html>`_.
#
# With this we have all we need to know to train, validate and test PyTorch deep learning models.

##################################################################
# Next: Learn more about `Automatic Differentiation with AutoGrad <autograd_tutorial.html>`_.
#

