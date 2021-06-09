Dynamic Parallelism in TorchScript
==================================

In this tutorial, we introduce the syntax for doing *dynamic inter-op parallelism*
in TorchScript. This parallelism has the following properties:

* dynamic - The number of parallel tasks created and their workload can depend on the control flow of the program.
* inter-op - The parallelism is concerned with running TorchScript program fragments in parallel. This is distinct from *intra-op parallelism*, which is concerned with splitting up individual operators and running subsets of the operator's work in parallel.
Basic Syntax
------------

The two important APIs for dynamic parallelism are:

* ``torch.jit.fork(fn : Callable[..., T], *args, **kwargs) -> torch.jit.Future[T]``
* ``torch.jit.wait(fut : torch.jit.Future[T]) -> T``

A good way to demonstrate how these work is by way of an example:

.. code-block:: python

    import torch

    def foo(x):
        return torch.neg(x)

    @torch.jit.script
    def example(x):
        # Call `foo` using parallelism:
        # First, we "fork" off a task. This task will run `foo` with argument `x`
        future = torch.jit.fork(foo, x)

        # Call `foo` normally
        x_normal = foo(x)

        # Second, we "wait" on the task. Since the task may be running in
        # parallel, we have to "wait" for its result to become available.
        # Notice that by having lines of code between the "fork()" and "wait()"
        # call for a given Future, we can overlap computations so that they
        # run in parallel.
        x_parallel = torch.jit.wait(future)

        return x_normal, x_parallel

    print(example(torch.ones(1))) # (-1., -1.)


``fork()`` takes the callable ``fn`` and arguments to that callable ``args``
and ``kwargs`` and creates an asynchronous task for the execution of ``fn``.
``fn`` can be a function, method, or Module instance. ``fork()`` returns a
reference to the value of the result of this execution, called a ``Future``.
Because ``fork`` returns immediately after creating the async task, ``fn`` may
not have been executed by the time the line of code after the ``fork()`` call
is executed. Thus, ``wait()`` is used to wait for the async task to complete
and return the value.

These constructs can be used to overlap the execution of statements within a
function (shown in the worked example section) or be composed with other language
constructs like loops:

.. code-block:: python

    import torch
    from typing import List

    def foo(x):
        return torch.neg(x)

    @torch.jit.script
    def example(x):
        futures : List[torch.jit.Future[torch.Tensor]] = []
        for _ in range(100):
            futures.append(torch.jit.fork(foo, x))

        results = []
        for future in futures:
            results.append(torch.jit.wait(future))

        return torch.sum(torch.stack(results))

    print(example(torch.ones([])))

.. note::

    When we initialized an empty list of Futures, we needed to add an explicit
    type annotation to ``futures``. In TorchScript, empty containers default
    to assuming they contain Tensor values, so we annotate the list constructor
    # as being of type ``List[torch.jit.Future[torch.Tensor]]``

This example uses ``fork()`` to launch 100 instances of the function ``foo``,
waits on the 100 tasks to complete, then sums the results, returning ``-100.0``.

Applied Example: Ensemble of Bidirectional LSTMs
------------------------------------------------

Let's try to apply parallelism to a more realistic example and see what sort
of performance we can get out of it. First, let's define the baseline model: an
ensemble of bidirectional LSTM layers.

.. code-block:: python

    import torch, time

    # In RNN parlance, the dimensions we care about are:
    # # of time-steps (T)
    # Batch size (B)
    # Hidden size/number of "channels" (C)
    T, B, C = 50, 50, 1024

    # A module that defines a single "bidirectional LSTM". This is simply two
    # LSTMs applied to the same sequence, but one in reverse
    class BidirectionalRecurrentLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell_f = torch.nn.LSTM(input_size=C, hidden_size=C)
            self.cell_b = torch.nn.LSTM(input_size=C, hidden_size=C)

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            # Forward layer
            output_f, _ = self.cell_f(x)

            # Backward layer. Flip input in the time dimension (dim 0), apply the
            # layer, then flip the outputs in the time dimension
            x_rev = torch.flip(x, dims=[0])
            output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
            output_b_rev = torch.flip(output_b, dims=[0])

            return torch.cat((output_f, output_b_rev), dim=2)


    # An "ensemble" of `BidirectionalRecurrentLSTM` modules. The modules in the
    # ensemble are run one-by-one on the same input then their results are
    # stacked and summed together, returning the combined result.
    class LSTMEnsemble(torch.nn.Module):
        def __init__(self, n_models):
            super().__init__()
            self.n_models = n_models
            self.models = torch.nn.ModuleList([
                BidirectionalRecurrentLSTM() for _ in range(self.n_models)])

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            results = []
            for model in self.models:
                results.append(model(x))
            return torch.stack(results).sum(dim=0)

    # For a head-to-head comparison to what we're going to do with fork/wait, let's
    # instantiate the model and compile it with TorchScript
    ens = torch.jit.script(LSTMEnsemble(n_models=4))

    # Normally you would pull this input out of an embedding table, but for the
    # purpose of this demo let's just use random data.
    x = torch.rand(T, B, C)

    # Let's run the model once to warm up things like the memory allocator
    ens(x)

    x = torch.rand(T, B, C)

    # Let's see how fast it runs!
    s = time.time()
    ens(x)
    print('Inference took', time.time() - s, ' seconds')

On my machine, this network runs in ``2.05`` seconds. We can do a lot better!

Parallelizing Forward and Backward Layers
-----------------------------------------

A very simple thing we can do is parallelize the forward and backward layers
within ``BidirectionalRecurrentLSTM``. For this, the structure of the computation
is static, so we don't actually even need any loops. Let's rewrite the ``forward``
method of ``BidirectionalRecurrentLSTM`` like so:

.. code-block:: python

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            # Forward layer - fork() so this can run in parallel to the backward
            # layer
            future_f = torch.jit.fork(self.cell_f, x)

            # Backward layer. Flip input in the time dimension (dim 0), apply the
            # layer, then flip the outputs in the time dimension
            x_rev = torch.flip(x, dims=[0])
            output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
            output_b_rev = torch.flip(output_b, dims=[0])

            # Retrieve the output from the forward layer. Note this needs to happen
            # *after* the stuff we want to parallelize with
            output_f, _ = torch.jit.wait(future_f)

            return torch.cat((output_f, output_b_rev), dim=2)

In this example, ``forward()`` delegates execution of ``cell_f`` to another thread,
while it continues to execute ``cell_b``. This causes the execution of both the
cells to be overlapped with each other.

Running the script again with this simple modification yields a runtime of
``1.71`` seconds for an improvement of ``17%``!

Aside: Visualizing Parallelism
------------------------------

We're not done optimizing our model but it's worth introducing the tooling we
have for visualizing performance. One important tool is the `PyTorch profiler <https://pytorch.org/docs/stable/autograd.html#profiler>`_.

Let's use the profiler along with the Chrome trace export functionality to
visualize the performance of our parallelized model:

.. code-block:: python

    with torch.autograd.profiler.profile() as prof:
        ens(x)
    prof.export_chrome_trace('parallel.json')

This snippet of code will write out a file named ``parallel.json``. If you
navigate Google Chrome to ``chrome://tracing``, click the ``Load`` button, and
load in that JSON file, you should see a timeline like the following:

.. image:: https://i.imgur.com/rm5hdG9.png

The horizontal axis of the timeline represents time and the vertical axis
represents threads of execution. As we can see, we are running two ``lstm``
instances at a time. This is the result of our hard work parallelizing the
bidirectional layers!

Parallelizing Models in the Ensemble
------------------------------------

You may have noticed that there is a further parallelization opportunity in our
code: we can also run the models contained in ``LSTMEnsemble`` in parallel with
each other. The way to do that is simple enough, this is how we should change
the ``forward`` method of ``LSTMEnsemble``:

.. code-block:: python

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            # Launch tasks for each model
            futures : List[torch.jit.Future[torch.Tensor]] = []
            for model in self.models:
                futures.append(torch.jit.fork(model, x))

            # Collect the results from the launched tasks
            results : List[torch.Tensor] = []
            for future in futures:
                results.append(torch.jit.wait(future))

            return torch.stack(results).sum(dim=0)

Or, if you value brevity, we can use list comprehensions:

.. code-block:: python

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            futures = [torch.jit.fork(model, x) for model in self.models]
            results = [torch.jit.wait(fut) for fut in futures]
            return torch.stack(results).sum(dim=0)

Like described in the intro, we've used loops to fork off tasks for each of the
models in our ensemble. We've then used another loop to wait for all of the
tasks to be completed. This provides even more overlap of computation.

With this small update, the script runs in ``1.4`` seconds, for a total speedup
of ``32%``! Pretty good for two lines of code.

We can also use the Chrome tracer again to see where's going on:

.. image:: https://i.imgur.com/kA0gyQm.png

We can now see that all ``LSTM`` instances are being run fully in parallel.

Conclusion
----------

In this tutorial, we learned about ``fork()`` and ``wait()``, the basic APIs
for doing dynamic, inter-op parallelism in TorchScript. We saw a few typical
usage patterns for using these functions to parallelize the execution of
functions, methods, or ``Modules`` in TorchScript code. Finally, we worked through
an example of optimizing a model using this technique and explored the performance
measurement and visualization tooling available in PyTorch.
