Compiled Autograd: Capturing a larger backward graph for ``torch.compile``
==========================================================================
**Author:** `Simon Fan <https://github.com/xmfan>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How compiled autograd interacts with ``torch.compile``
       * How to use the compiled autograd API
       * How to inspect logs using ``TORCH_LOGS``

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch 2.4
       * Complete the `Introduction to torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_

Overview
--------
Compiled Autograd is a ``torch.compile`` extension introduced in PyTorch 2.4
that allows the capture of a larger backward graph.

While ``torch.compile`` does capture the backward graph, it does so **partially**. The AOTAutograd component captures the backward graph ahead-of-time, with certain limitations:

* Graph breaks in the forward lead to graph breaks in the backward
* `Backward hooks <https://pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution>`_ are not captured

Compiled Autograd addresses these limitations by directly integrating with the autograd engine, allowing
it to capture the full backward graph at runtime. Models with these two characteristics should try
Compiled Autograd, and potentially observe better performance.

However, Compiled Autograd introduces its own limitations:

* Added runtime overhead at the start of the backward for cache lookup
* More prone to recompiles and graph breaks in dynamo due to the larger capture

.. note:: Compiled Autograd is under active development and is not yet compatible with all existing PyTorch features. For the latest status on a particular feature, refer to `Compiled Autograd Landing Page <https://docs.google.com/document/d/11VucFBEewzqgkABIjebZIzMvrXr3BtcY1aGKpX61pJY>`_.

Setup
-----
In this tutorial, we will base our examples on this simple neural network model.
It takes a a 10-dimensional input vector, processes it through a single linear layer, and outputs another 10-dimensional vector.

.. code:: python

   import torch

   class Model(torch.nn.Module):
      def __init__(self):
         super().__init__()
         self.linear = torch.nn.Linear(10, 10)

      def forward(self, x):
         return self.linear(x)

Basic usage
------------
Before calling the torch.compile API, make sure to set ``torch._dynamo.config.compiled_autograd`` to ``True``:

.. code:: python

   model = Model()
   x = torch.randn(10)

   torch._dynamo.config.compiled_autograd = True
   @torch.compile
   def train(model, x):
      loss = model(x).sum()
      loss.backward()

   train(model, x) 

In the code above, we create an instance of the ``Model`` class and generate a random 10-dimensional tensor ``x`` by using torch.randn(10).
We define the training loop function ``train`` and decorate it with @torch.compile to optimize its execution.
When ``train(model, x)`` is called:

* Python Interpreter calls Dynamo, since this call was decorated with ``@torch.compile``
* Dynamo intercepts the python bytecode, simulates their execution and records the operations into a graph
* AOTDispatcher disables hooks and calls the autograd engine to compute gradients for ``model.linear.weight`` and ``model.linear.bias``, and records the operations into a graph. Using ``torch.autograd.Function``, AOTDispatcher rewrites the forward and backward implementation of ``train``.
* Inductor generates a function corresponding to an optimized implementation of the AOTDispatcher forward and backward
* Dynamo sets the optimized function to be evaluated next by Python Interpreter
* Python Interpreter executes the optimized function, which basically executes ``loss = model(x).sum()``
* Python Interpreter executes ``loss.backward()``, calling into the autograd engine, which routes to the Compiled Autograd engine since we enabled the config: ``torch._dynamo.config.compiled_autograd = True``
* Compiled Autograd computes the gradients for ``model.linear.weight`` and ``model.linear.bias``, and records the operations into a graph, including any hooks it encounters. During this, it will record the backward previously rewritten by AOTDispatcher. Compiled Autograd then generates a new function which corresponds to a fully traced implementation of ``loss.backward()``, and executes it with ``torch.compile`` in inference mode
* The same steps recursively apply to the Compiled Autograd graph, but this time AOTDispatcher does not need to partition this graph into a forward and backward

Inspecting the compiled autograd logs
-------------------------------------
Run the script with the ``TORCH_LOGS`` environment variables:

* To only print the compiled autograd graph, use ``TORCH_LOGS="compiled_autograd" python example.py``
* To print the graph with more tensor metadata and recompile reasons, at the cost of performance, use ``TORCH_LOGS="compiled_autograd_verbose" python example.py``

Rerun the snippet above, the compiled autograd graph should now be logged to ``stderr``. Certain graph nodes will have names that are prefixed by ``aot0_``,
these correspond to the nodes previously compiled ahead of time in AOTAutograd backward graph 0, for example, ``aot0_view_2`` corresponds to ``view_2`` of the AOT backward graph with id=0.


.. code:: python

   stderr_output = """
   DEBUG:torch._dynamo.compiled_autograd.__compiled_autograd_verbose:Cache miss due to new autograd node: torch::autograd::GraphRoot (NodeCall 0) with key size 39, previous key sizes=[]
   DEBUG:torch._dynamo.compiled_autograd.__compiled_autograd_verbose:TRACED GRAPH
   ===== Compiled autograd graph =====
   <eval_with_key>.4 class CompiledAutograd(torch.nn.Module):
      def forward(self, inputs, sizes, scalars, hooks):
         # No stacktrace found for following nodes
         aot0_tangents_1: "f32[][]cpu" = inputs[0]
         aot0_primals_3: "f32[10][1]cpu" = inputs[1]
         getitem_2: "f32[10][1]cpu" = inputs[2]
         getitem_3: "f32[10, 10][10, 1]cpu" = inputs[3];  inputs = None
         
            # File: /data/users/xmfan/a/pytorch/torch/_dynamo/compiled_autograd.py:483 in set_node_origin, code: CompiledFunctionBackward0 (NodeCall 1)
         aot0_expand: "f32[10][0]cpu" = torch.ops.aten.expand.default(aot0_tangents_1, [10]);  aot0_tangents_1 = None
         aot0_view_2: "f32[1, 10][0, 0]cpu" = torch.ops.aten.view.default(aot0_expand, [1, 10]);  aot0_expand = None
         aot0_permute_2: "f32[10, 1][0, 0]cpu" = torch.ops.aten.permute.default(aot0_view_2, [1, 0])
         aot0_select: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 0)
         aot0_view: "f32[1, 10][10, 1]cpu" = torch.ops.aten.view.default(aot0_primals_3, [1, 10]);  aot0_primals_3 = None
         aot0_mul_3: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select, aot0_view);  aot0_select = None
         aot0_select_1: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 1)
         aot0_mul_4: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select_1, aot0_view);  aot0_select_1 = None
         aot0_select_2: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 2)
         aot0_mul_5: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select_2, aot0_view);  aot0_select_2 = None
         aot0_select_3: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 3)
         aot0_mul_6: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select_3, aot0_view);  aot0_select_3 = None
         aot0_select_4: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 4)
         aot0_mul_7: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select_4, aot0_view);  aot0_select_4 = None
         aot0_select_5: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 5)
         aot0_mul_8: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select_5, aot0_view);  aot0_select_5 = None
         aot0_select_6: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 6)
         aot0_mul_9: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select_6, aot0_view);  aot0_select_6 = None
         aot0_select_7: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 7)
         aot0_mul_10: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select_7, aot0_view);  aot0_select_7 = None
         aot0_select_8: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 8)
         aot0_mul_11: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select_8, aot0_view);  aot0_select_8 = None
         aot0_select_9: "f32[1][0]cpu" = torch.ops.aten.select.int(aot0_permute_2, 0, 9);  aot0_permute_2 = None
         aot0_mul_12: "f32[1, 10][10, 1]cpu" = torch.ops.aten.mul.Tensor(aot0_select_9, aot0_view);  aot0_select_9 = aot0_view = None
         aot0_cat: "f32[10, 10][10, 1]cpu" = torch.ops.aten.cat.default([aot0_mul_3, aot0_mul_4, aot0_mul_5, aot0_mul_6, aot0_mul_7, aot0_mul_8, aot0_mul_9, aot0_mul_10, aot0_mul_11, aot0_mul_12]);  aot0_mul_3 = aot0_mul_4 = aot0_mul_5 = aot0_mul_6 = aot0_mul_7 = aot0_mul_8 = aot0_mul_9 = aot0_mul_10 = aot0_mul_11 = aot0_mul_12 = None
         aot0_permute_3: "f32[10, 10][1, 10]cpu" = torch.ops.aten.permute.default(aot0_cat, [1, 0]);  aot0_cat = None
         aot0_sum_3: "f32[1, 10][10, 1]cpu" = torch.ops.aten.sum.dim_IntList(aot0_view_2, [0], True);  aot0_view_2 = None
         aot0_view_3: "f32[10][1]cpu" = torch.ops.aten.view.default(aot0_sum_3, [10]);  aot0_sum_3 = None
         
            # File: /data/users/xmfan/a/pytorch/torch/_dynamo/compiled_autograd.py:483 in set_node_origin, code: torch::autograd::AccumulateGrad (NodeCall 2)
         accumulate_grad_ = torch.ops.inductor.accumulate_grad_.default(getitem_2, aot0_view_3);  getitem_2 = aot0_view_3 = accumulate_grad_ = None
         
            # File: /data/users/xmfan/a/pytorch/torch/_dynamo/compiled_autograd.py:483 in set_node_origin, code: CompiledFunctionBackward0 (NodeCall 1)
         aot0_permute_4: "f32[10, 10][10, 1]cpu" = torch.ops.aten.permute.default(aot0_permute_3, [1, 0]);  aot0_permute_3 = None
         
            # File: /data/users/xmfan/a/pytorch/torch/_dynamo/compiled_autograd.py:483 in set_node_origin, code: torch::autograd::AccumulateGrad (NodeCall 3)
         accumulate_grad__1 = torch.ops.inductor.accumulate_grad_.default(getitem_3, aot0_permute_4);  getitem_3 = aot0_permute_4 = accumulate_grad__1 = None
         _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
         return []
   """

.. note:: This is the graph on which we will call ``torch.compile``, **NOT** the optimized graph. Compiled Autograd essentially generates some unoptimized Python code to represent the entire C++ autograd execution.

Compiling the forward and backward pass using different flags
-------------------------------------------------------------
You can use different compiler configs for the two compilations, for example, the backward may be a fullgraph even if there are graph breaks in the forward.

.. code:: python

def train(model, x):
    model = torch.compile(model)
    loss = model(x).sum()
    torch._dynamo.config.compiled_autograd = True
    torch.compile(lambda: loss.backward(), fullgraph=True)()

Or you can use the context manager, which will apply to all autograd calls within its scope.

.. code:: python

   def train(model, x):
      model = torch.compile(model)
      loss = model(x).sum()
      with torch._dynamo.compiled_autograd.enable(torch.compile(fullgraph=True)):
         loss.backward()


Compiled Autograd addresses certain limitations of AOTAutograd
--------------------------------------------------------------
1. Graph breaks in the forward lead to graph breaks in the backward

.. code:: python

   @torch.compile(backend="aot_eager")
   def fn(x):
      # 1st graph
      temp = x + 10
      torch._dynamo.graph_break()
      # 2nd graph
      temp = temp + 10
      torch._dynamo.graph_break()
      # 3rd graph
      return temp.sum()

   x = torch.randn(10, 10, requires_grad=True)
   torch._dynamo.utils.counters.clear()
   loss = fn(x)

   # 1. base torch.compile 
   loss.backward(retain_graph=True)
   assert(torch._dynamo.utils.counters["stats"]["unique_graphs"] == 3)
   torch._dynamo.utils.counters.clear()

   # 2. torch.compile with compiled autograd
   with torch._dynamo.compiled_autograd.enable(torch.compile(backend="aot_eager")):
      loss.backward()

   # single graph for the backward
   assert(torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1)


In the ``1. base torch.compile`` case, we see that 3 backward graphs were produced due to the 2 graph breaks in the compiled function ``fn``. 
Whereas in ``2. torch.compile with compiled autograd``, we see that a full backward graph was traced despite the graph breaks.

2. Backward hooks are not captured

.. code:: python

   @torch.compile(backend="aot_eager")
   def fn(x):
      return x.sum()

   x = torch.randn(10, 10, requires_grad=True)
   x.register_hook(lambda grad: grad+10)
   loss = fn(x)

   with torch._dynamo.compiled_autograd.enable(torch.compile(backend="aot_eager")):
      loss.backward()

There should be a ``call_hook`` node in the graph, which dynamo will later inline into

.. code:: python

   stderr_output = """
   DEBUG:torch._dynamo.compiled_autograd.__compiled_autograd_verbose:Cache miss due to new autograd node: torch::autograd::GraphRoot (NodeCall 0) with key size 39, previous key sizes=[]
   DEBUG:torch._dynamo.compiled_autograd.__compiled_autograd_verbose:TRACED GRAPH
   ===== Compiled autograd graph =====
   <eval_with_key>.2 class CompiledAutograd(torch.nn.Module):
      def forward(self, inputs, sizes, scalars, hooks):
         ...
         getitem_2 = hooks[0];  hooks = None
         call_hook: "f32[10, 10][0, 0]cpu" = torch__dynamo_external_utils_call_hook(getitem_2, aot0_expand, hook_type = 'tensor_pre_hook');  getitem_2 = aot0_expand = None
         ...
   """

Common recompilation reasons for Compiled Autograd
--------------------------------------------------
1. Due to change in autograd structure 

.. code:: python

   torch._dynamo.config.compiled_autograd = True
   x = torch.randn(10, requires_grad=True)
   for op in [torch.add, torch.sub, torch.mul, torch.div]:
      loss = op(x, x).sum()
      torch.compile(lambda: loss.backward(), backend="eager")()

You should see some recompile messages: **Cache miss due to new autograd node**.

.. code:: python

   stderr_output = """
   Cache miss due to new autograd node: torch::autograd::GraphRoot (NodeCall 0) with key size 39, previous key sizes=[] 
   ...
   Cache miss due to new autograd node: SubBackward0 (NodeCall 2) with key size 56, previous key sizes=[]
   ...
   Cache miss due to new autograd node: MulBackward0 (NodeCall 2) with key size 71, previous key sizes=[]
   ...
   Cache miss due to new autograd node: DivBackward0 (NodeCall 2) with key size 70, previous key sizes=[]
   ...
   """

2. Due to dynamic shapes

.. code:: python

   torch._dynamo.config.compiled_autograd = True
   for i in [10, 100, 10]:
      x = torch.randn(i, i, requires_grad=True)
      loss = x.sum()
      torch.compile(lambda: loss.backward(), backend="eager")()

You should see some recompiles messages: **Cache miss due to changed shapes**.

.. code:: python

   stderr_output = """
   ...
   Cache miss due to changed shapes: marking size idx 0 of torch::autograd::GraphRoot (NodeCall 0) as dynamic
   Cache miss due to changed shapes: marking size idx 1 of torch::autograd::AccumulateGrad (NodeCall 2) as dynamic
   Cache miss due to changed shapes: marking size idx 2 of torch::autograd::AccumulateGrad (NodeCall 2) as dynamic
   Cache miss due to changed shapes: marking size idx 3 of torch::autograd::AccumulateGrad (NodeCall 2) as dynamic
   ...
   """

Conclusion
----------
In this tutorial, we went over the high-level ecosystem of ``torch.compile`` with compiled autograd, the basics of compiled autograd and a few common recompilation reasons.
