Note

Go to the end
to download the full example code.

# PyTorch: Custom nn Modules

A third order polynomial, trained to predict \(y=\sin(x)\) from \(-\pi\)
to \(\pi\) by minimizing squared Euclidean distance.

This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.

```
import torch
import math

class Polynomial3(torch.nn.Module):
 def __init__(self):
 """
 In the constructor we instantiate four parameters and assign them as
 member parameters.
 """
 super().__init__()
 self.a = torch.nn.Parameter(torch.randn(()))
 self.b = torch.nn.Parameter(torch.randn(()))
 self.c = torch.nn.Parameter(torch.randn(()))
 self.d = torch.nn.Parameter(torch.randn(()))

 def forward(self, x):
 """
 In the forward function we accept a Tensor of input data and we must return
 a Tensor of output data. We can use Modules defined in the constructor as
 well as arbitrary operators on Tensors.
 """
 return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

 def string(self):
 """
 Just like any class in Python, you can also define custom method on PyTorch modules
 """
 return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Construct our model by instantiating the class defined above
model = Polynomial3()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
 # Forward pass: Compute predicted y by passing x to the model
 y_pred = model(x)

 # Compute and print loss
 loss = criterion(y_pred, y)
 if t % 100 == 99:
 print(t, loss.item())

 # Zero gradients, perform a backward pass, and update the weights.
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

print(f'Result: {model.string()}')
```

[`Download Jupyter notebook: polynomial_module.ipynb`](../../_downloads/fdb76f84e688e2ecc24fa38edfa41aea/polynomial_module.ipynb)

[`Download Python source code: polynomial_module.py`](../../_downloads/4dbaf9210d9de48b066fe57085912ccf/polynomial_module.py)

[`Download zipped: polynomial_module.zip`](../../_downloads/50b31e19a63de93f4e85a4f9cc45b844/polynomial_module.zip)