Autograd in C++ Frontend
========================

The ``autograd`` package is crucial for building highly flexible and dynamic neural
networks in PyTorch. Most of the autograd APIs in PyTorch Python frontend are also available
in C++ frontend, allowing easy translation of autograd code from Python to C++.

In this tutorial explore several examples of doing autograd in PyTorch C++ frontend.
Note that this tutorial assumes that you already have a basic understanding of
autograd in Python frontend. If that's not the case, please first read
`Autograd: Automatic Differentiation <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_.

Basic autograd operations
-------------------------

(Adapted from `this tutorial <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#autograd-automatic-differentiation>`_)

Create a tensor and set ``torch::requires_grad()`` to track computation with it

.. code-block:: cpp

  auto x = torch::ones({2, 2}, torch::requires_grad());
  std::cout << x << std::endl;

Out:

.. code-block:: shell

  1 1
  1 1
  [ CPUFloatType{2,2} ]


Do a tensor operation:

.. code-block:: cpp

  auto y = x + 2;
  std::cout << y << std::endl;

Out:

.. code-block:: shell

   3  3
   3  3
  [ CPUFloatType{2,2} ]

``y`` was created as a result of an operation, so it has a ``grad_fn``.

.. code-block:: cpp

  std::cout << y.grad_fn()->name() << std::endl;

Out:

.. code-block:: shell

  AddBackward1

Do more operations on ``y``

.. code-block:: cpp

  auto z = y * y * 3;
  auto out = z.mean();
  
  std::cout << z << std::endl;
  std::cout << z.grad_fn()->name() << std::endl;
  std::cout << out << std::endl;
  std::cout << out.grad_fn()->name() << std::endl;

Out:

.. code-block:: shell

   27  27
   27  27
  [ CPUFloatType{2,2} ]
  MulBackward1
  27
  [ CPUFloatType{} ]
  MeanBackward0


``.requires_grad_( ... )`` changes an existing tensor's ``requires_grad`` flag in-place.

.. code-block:: cpp

  auto a = torch::randn({2, 2});
  a = ((a * 3) / (a - 1));
  std::cout << a.requires_grad() << std::endl;
  
  a.requires_grad_(true);
  std::cout << a.requires_grad() << std::endl;
  
  auto b = (a * a).sum();
  std::cout << b.grad_fn()->name() << std::endl;

Out:

.. code-block:: shell

  false
  true
  SumBackward0

Let's backprop now. Because ``out`` contains a single scalar, ``out.backward()``
is equivalent to ``out.backward(torch::tensor(1.))``.

.. code-block:: cpp

  out.backward();

Print gradients d(out)/dx

.. code-block:: cpp

  std::cout << x.grad() << std::endl;

Out:

.. code-block:: shell

   4.5000  4.5000
   4.5000  4.5000
  [ CPUFloatType{2,2} ]

You should have got a matrix of ``4.5``. For explanations on how we arrive at this value,
please see `the corresponding section in this tutorial <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients>`_.

Now let's take a look at an example of vector-Jacobian product:

.. code-block:: cpp

  x = torch::randn(3, torch::requires_grad());
  
  y = x * 2;
  while (y.norm().item<double>() < 1000) {
    y = y * 2;
  }
    
  std::cout << y << std::endl;
  std::cout << y.grad_fn()->name() << std::endl;

Out:

.. code-block:: shell

  -1021.4020
    314.6695
   -613.4944
  [ CPUFloatType{3} ]
  MulBackward1

If we want the vector-Jacobian product, pass the vector to ``backward`` as argument:

.. code-block:: cpp

  auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
  y.backward(v);
  
  std::cout << x.grad() << std::endl;

Out:

.. code-block:: shell

    102.4000
   1024.0000
      0.1024
  [ CPUFloatType{3} ]

You can also stop autograd from tracking history on tensors that require gradients
either by putting ``torch::NoGradGuard`` in a code block

.. code-block:: cpp

  std::cout << x.requires_grad() << std::endl;
  std::cout << x.pow(2).requires_grad() << std::endl;
  
  {
    torch::NoGradGuard no_grad;
    std::cout << x.pow(2).requires_grad() << std::endl;
  }


Out:

.. code-block:: shell

  true
  true
  false

Or by using ``.detach()`` to get a new tensor with the same content but that does
not require gradients:

.. code-block:: cpp

  std::cout << x.requires_grad() << std::endl;
  y = x.detach();
  std::cout << y.requires_grad() << std::endl;
  std::cout << x.eq(y).all().item<bool>() << std::endl;

Out:

.. code-block:: shell

  true
  false
  true

For more information on C++ tensor autograd APIs such as ``grad`` / ``requires_grad`` /
``is_leaf`` / ``backward`` / ``detach`` / ``detach_`` / ``register_hook`` / ``retain_grad``,
please see `the corresponding C++ API docs <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html>`_.

Computing higher-order gradients in C++
---------------------------------------

One of the applications of higher-order gradients is calculating gradient penalty.
Let's see an example of it using ``torch::autograd::grad``:

.. code-block:: cpp

  #include <torch/torch.h>
  
  auto model = torch::nn::Linear(4, 3);
  
  auto input = torch::randn({3, 4}).requires_grad_(true);
  auto output = model(input);
  
  // Calculate loss
  auto target = torch::randn({3, 3});
  auto loss = torch::nn::MSELoss()(output, target);
  
  // Use norm of gradients as penalty
  auto grad_output = torch::ones_like(output);
  auto gradient = torch::autograd::grad({output}, {input}, /*grad_outputs=*/{grad_output}, /*create_graph=*/true)[0];
  auto gradient_penalty = torch::pow((gradient.norm(2, /*dim=*/1) - 1), 2).mean();
  
  // Add gradient penalty to loss
  auto combined_loss = loss + gradient_penalty;
  combined_loss.backward();
  
  std::cout << input.grad() << std::endl;

Out:

.. code-block:: shell

  -0.1042 -0.0638  0.0103  0.0723
  -0.2543 -0.1222  0.0071  0.0814
  -0.1683 -0.1052  0.0355  0.1024
  [ CPUFloatType{3,4} ]

Please see the documentation for ``torch::autograd::backward``
(`link <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1afa9b5d4329085df4b6b3d4b4be48914b.html>`_)
and ``torch::autograd::grad``
(`link <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1e03c42b14b40c306f9eb947ef842d9c.html>`_)
for more information on how to use them.

Using custom autograd function in C++
-------------------------------------

(Adapted from `this tutorial <https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd>`_)

Adding a new elementary operation to ``torch::autograd`` requires implementing a new ``torch::autograd::Function``
subclass for each operation. ``torch::autograd::Function`` s are what ``torch::autograd``
uses to compute the results and gradients, and encode the operation history. Every
new function requires you to implement 2 methods: ``forward`` and ``backward``, and
please see `this link <https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html>`_
for the detailed requirements.

Below you can find code for a ``Linear`` function from ``torch::nn``:

.. code-block:: cpp

  #include <torch/torch.h>
  
  using namespace torch::autograd;
  
  // Inherit from Function
  class LinearFunction : public Function<LinearFunction> {
   public:
    // Note that both forward and backward are static functions
  
    // bias is an optional argument
    static torch::Tensor forward(
        AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
      ctx->save_for_backward({input, weight, bias});
      auto output = input.mm(weight.t());
      if (bias.defined()) {
        output += bias.unsqueeze(0).expand_as(output);
      }
      return output;
    }
  
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
      auto saved = ctx->get_saved_variables();
      auto input = saved[0];
      auto weight = saved[1];
      auto bias = saved[2];
  
      auto grad_output = grad_outputs[0];
      auto grad_input = grad_output.mm(weight);
      auto grad_weight = grad_output.t().mm(input);
      auto grad_bias = torch::Tensor();
      if (bias.defined()) {
        grad_bias = grad_output.sum(0);
      }
  
      return {grad_input, grad_weight, grad_bias};
    }
  };

Then, we can use the ``LinearFunction`` in the following way:

.. code-block:: cpp

  auto x = torch::randn({2, 3}).requires_grad_();
  auto weight = torch::randn({4, 3}).requires_grad_();
  auto y = LinearFunction::apply(x, weight);
  y.sum().backward();
  
  std::cout << x.grad() << std::endl;
  std::cout << weight.grad() << std::endl;

Out:

.. code-block:: shell

   0.5314  1.2807  1.4864
   0.5314  1.2807  1.4864
  [ CPUFloatType{2,3} ]
   3.7608  0.9101  0.0073
   3.7608  0.9101  0.0073
   3.7608  0.9101  0.0073
   3.7608  0.9101  0.0073
  [ CPUFloatType{4,3} ]

Here, we give an additional example of a function that is parametrized by non-tensor arguments:

.. code-block:: cpp

  #include <torch/torch.h>
  
  using namespace torch::autograd;
  
  class MulConstant : public Function<MulConstant> {
   public:
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor tensor, double constant) {
      // ctx is a context object that can be used to stash information
      // for backward computation
      ctx->saved_data["constant"] = constant;
      return tensor * constant;
    }
  
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
      // We return as many input gradients as there were arguments.
      // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
      return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
    }
  };

Then, we can use the ``MulConstant`` in the following way:

.. code-block:: cpp

  auto x = torch::randn({2}).requires_grad_();
  auto y = MulConstant::apply(x, 5.5);
  y.sum().backward();

  std::cout << x.grad() << std::endl;

Out:

.. code-block:: shell

   5.5000
   5.5000
  [ CPUFloatType{2} ]

For more information on ``torch::autograd::Function``, please see
`its documentation <https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html>`_.

Translating autograd code from Python to C++
--------------------------------------------

On a high level, the easiest way to use autograd in C++ is to have working
autograd code in Python first, and then translate your autograd code from Python to
C++ using the following table:

+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Python                         | C++                                                                                                                                                                    |
+================================+========================================================================================================================================================================+
| ``torch.autograd.backward``    | ``torch::autograd::backward`` (`link <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1afa9b5d4329085df4b6b3d4b4be48914b.html>`_)                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.autograd.grad``        | ``torch::autograd::grad`` (`link <https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1e03c42b14b40c306f9eb947ef842d9c.html>`_)                      |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.detach``        | ``torch::Tensor::detach`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor6detachEv>`_)                                              |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.detach_``       | ``torch::Tensor::detach_`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7detach_Ev>`_)                                            |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.backward``      | ``torch::Tensor::backward`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8backwardERK6Tensorbb>`_)                                |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.register_hook`` | ``torch::Tensor::register_hook`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4I0ENK2at6Tensor13register_hookE18hook_return_void_tI1TERR1T>`_) |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.requires_grad`` | ``torch::Tensor::requires_grad_`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor14requires_grad_Eb>`_)                             |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.retain_grad``   | ``torch::Tensor::retain_grad`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor11retain_gradEv>`_)                                   |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.grad``          | ``torch::Tensor::grad`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4gradEv>`_)                                                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.grad_fn``       | ``torch::Tensor::grad_fn`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7grad_fnEv>`_)                                            |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.set_data``      | ``torch::Tensor::set_data`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8set_dataERK6Tensor>`_)                                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.data``          | ``torch::Tensor::data`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4dataEv>`_)                                                  |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.output_nr``     | ``torch::Tensor::output_nr`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor9output_nrEv>`_)                                        |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``torch.Tensor.is_leaf``       | ``torch::Tensor::is_leaf`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7is_leafEv>`_)                                            |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

After translation, most of your Python autograd code should just work in C++.
If that's not the case, please file a bug report at `GitHub issues <https://github.com/pytorch/pytorch/issues>`_
and we will fix it as soon as possible.

Conclusion
----------

You should now have a good overview of PyTorch's C++ autograd API.
You can find the code examples displayed in this note `here
<https://github.com/pytorch/examples/tree/master/cpp/autograd>`_. As always, if you run into any
problems or have questions, you can use our `forum <https://discuss.pytorch.org/>`_
or `GitHub issues <https://github.com/pytorch/pytorch/issues>`_ to get in touch.
