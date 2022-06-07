"""
Linear Regression in PyTorch Tutorial
================
**Author:** Jack Snowdon

In this tutorial, we will generate a linear regression with PyTorch to find
the best-fitting line in a linear relationship between an independent and 
dependent variable. Linear regressions are popular among ML projects used
for predictive analysis.
"""

from torch import nn
import torch
from torch import tensor

# x_data is the independent variable, y_data the target variable to be learned.
x_data = tensor([[1.0], [7.0], [3.0], [5.0]])
y_data = tensor([[2.0], [3.0], [7.0], [7.0]])

# We need to set up a model class
class Model(nn.Module):
    def __init__(self):
        """
        Initialize the model type. We also initialize with the 'Linear' layer -- set
        our input and output size to 1, since each dataset the x/y values are 1 to 1.
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        Define our forward pass function. This function uses our input data as defined
        above with x as its input -- it returns the predicted value of Y.
        Read more regarding forward and backward functions at https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
        """
        y_pred = self.linear(x)
        return y_pred


# Instantiate model class from above
model = Model()

# Now we need a Loss Function (i.e. Criterion) and Optimizer.
# A Loss Function will be calculated from our original y_data target, as well
# as the prediction y_pred returned from our model. The function will update
# our weights as to preference the best model selection (optimize). 
# We use the Mean Square Error function. 
criterion = torch.nn.MSELoss(reduction='sum')

# Update our model parameters using SGD -- Read more below:
#https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
# LR is our learning rate, which is a hyperparameter for gradient descent.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# We can now begin to train our model.
# First set an epoch size, which represents an individual pass through the
# entire training set. We can vary this given our data set size.

for epoch in range(100):
    # Complete forward pass, set our gradient to zero since it will accumulate otherwise
    y_pred = model(x_data)
    optimizer.zero_grad()
    
    # Compute loss
    loss = criterion(y_pred, y_data)

    # Backward pass, update optimizer parameter given current gradient.
    loss.backward()
    optimizer.step()


# After training, test with a novel x_value
x = tensor([[5.0]])
y_pred = model(x)
print("Prediction:", y_pred.data[0][0])