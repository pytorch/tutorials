Supporting Custom C++ Classes in PyTorch 2
==========================================

This tutorial is a follow-on to the
:doc:`custom C++ classes <torch_script_custom_classes>` tutorial, and
introduces additional steps that are needed to support custom C++ classes in
PyTorch 2.

Concretely, there are a few steps:

1. Implement an ``__obj_flatten__`` method to the C++ custom class
    implementation to allow us to inspect its states and guard the changes. The
    method should return a tuple of tuple of attribute_name, value
    (``tuple[tuple[str, value] * n]``).
2. Register a python fake class using ``@torch._library.register_fake_class``
    a. Implement “fake methods” of each of the class’s c++ methods, which should
        have the same schema as the C++ implementation.
    b. Additionally, implement an ``__obj_unflatten__`` classmethod in the Python
        fake class to tell us how to create a fake class from the flattened
        states returned by ``__obj_flatten__``.

Here is a breakdown of the diff. Following the guide in
:doc:`Extending TorchScript with Custom C++ Classes <torch_script_custom_classes>`,
we can create a thread-safe tensor queue and build it.

.. code-block:: cpp

    // Thread-safe Tensor Queue
    struct TensorQueue : torch::CustomClassHolder {
    ...
    private:
    std::deque<at::Tensor> queue_;
    std::mutex mutex_;
    at::Tensor init_tensor_;
    };
    // The torch binding code
    TORCH_LIBRARY(MyCustomClass, m) {
    m.class_<TensorQueue>("TensorQueue")
        .def(torch::init<at::Tensor>())
        .def("push", &TensorQueue::push)
        .def("pop", &TensorQueue::pop)
        .def("top", &TensorQueue::top)
        .def("size", &TensorQueue::size)
        .def("clone_queue", &TensorQueue::clone_queue)
        .def("get_raw_queue", &TensorQueue::get_raw_queue)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<TensorQueue>& self)
                -> c10::Dict<std::string, at::Tensor> {
                return self->serialize();
            },
            // __setstate__
            [](c10::Dict<std::string, at::Tensor> data)
                -> c10::intrusive_ptr<TensorQueue> {
                return c10::make_intrusive<TensorQueue>(std::move(data));
            });
    }

**Step 1**: Add an ``__obj_flatten__`` method to the C++ custom class implementation:

.. code-block:: cpp

    // Thread-safe Tensor Queue
    struct TensorQueue : torch::CustomClassHolder {
    ...
    std::tuple<std::tuple<std::string, std::vector<at::Tensor>>, std::tuple<std::string, at::Tensor>> __obj_flatten__() {
        return std::tuple(std::tuple("queue", this->get_raw_queue()), std::tuple("init_tensor_", this->init_tensor_.clone()));
    }
    ...
    }

    TORCH_LIBRARY(MyCustomClass, m) {
    m.class_<TensorQueue>("TensorQueue")
        .def(torch::init<at::Tensor>())
        .def("__obj_flatten__", &TensorQueue::__obj_flatten__)
        ...
    }

**Step 2a**: Register a fake class in Python that implements each method.

.. code-block:: python

    # namespace::class_name
    @torch._library.register_fake_class("MyCustomClass::TensorQueue")
    class FakeTensorQueue:
        def __init__(
            self,
            queue: List[torch.Tensor],
            init_tensor_: torch.Tensor
        ) -> None:
            self.queue = queue
            self.init_tensor_ = init_tensor_

        def push(self, tensor: torch.Tensor) -> None:
            self.queue.append(tensor)

        def pop(self) -> torch.Tensor:
            if len(self.queue) > 0:
                return self.queue.pop(0)
            return self.init_tensor_

        def size(self) -> int:
        return len(self.queue)

**Step 2b**: Implement an ``__obj_unflatten__`` classmethod in Python.

.. code-block:: python

    # namespace::class_name
    @torch._library.register_fake_class("MyCustomClass::TensorQueue")
    class FakeTensorQueue:
    ...
        @classmethod
        def __obj_unflatten__(cls, flattened_tq):
            return cls(**dict(flattened_tq))

    ...

That’s it! Now we can create a module that uses this object and run it with ``torch.compile`` or ``torch.export``:

.. code-block::python

    import torch

    torch.ops.load_library("//caffe2/test:test_torchbind_cpp_impl")
    tq = torch.classes.MyCustomClass.TensorQueue(torch.empty(0).fill_(-1))

    class Mod(torch.nn.Module):
        def forward(self, tq, x):
            tq.push(x.sin())
            tq.push(x.cos())
            poped_t = tq.pop()
            assert torch.allclose(poped_t, x.sin())
            return tq, poped_t

    tq, poped_t = torch.compile(Mod(), backend="eager", fullgraph=True)(tq, torch.randn(2, 3))
    assert tq.size() == 1

    exported_program = torch.export.export(Mod(), (tq, torch.randn(2, 3),), strict=False)
    exported_program.module()(tq, torch.randn(2, 3))

We can also implement custom ops that take custom classes as inputs. For
example, we could register a custom op ``for_each_add_(tq, tensor)``

.. code-block:: cpp

    struct TensorQueue : torch::CustomClassHolder {
    ...
    void for_each_add_(at::Tensor inc) {
        for (auto& t : queue_) {
        t.add_(inc);
        }
    }
    ...
    }


    TORCH_LIBRARY_FRAGMENT(MyCustomClass, m) {
    m.class_<TensorQueue>("TensorQueue")
    .def("for_each_add_", &TensorQueue::for_each_add_);

    m.def(
        "for_each_add_(__torch__.torch.classes.MyCustomClass.TensorQueue foo, Tensor inc) -> ()");

    }

    void for_each_add_(c10::intrusive_ptr<TensorQueue> tq, at::Tensor inc) {
    tq->for_each_add_(inc);
    }

    TORCH_LIBRARY_IMPL(MyCustomClass, CPU, m) {
    m.impl("for_each_add_", for_each_add_);
    }


Since the fake class is implemented in python, we require the fake
implementation of custom op must also be registered in python:

.. code-block:: python

    @torch.library.register_fake("MyCustomClass::for_each_add_")
    def fake_for_each_add_(tq, inc):
        tq.for_each_add_(inc)

After re-compilation, we can export the custom op with:

.. code-block:: python

    class ForEachAdd(torch.nn.Module):
        def forward(self, tq: torch.ScriptObject, a: torch.Tensor) -> torch.ScriptObject:
            torch.ops.MyCustomClass.for_each_add_(tq, a)
            return tq

    mod = ForEachAdd()
    tq = empty_tensor_queue()
    qlen = 10
    for i in range(qlen):
    tq.push(torch.zeros(1))

    ep = torch.export.export(mod, (tq, torch.ones(1)), strict=False)

Why do we need to make a Fake Class?
------------------------------------

Tracing with real custom object has several major downsides:

1. Operators on real objects can be time consuming e.g. the custom object
    might be reading from the network or loading data from the disk.
2. We don’t want to mutate the real custom object or create side-effects to the environment while tracing.
3. It cannot support dynamic shapes.

However, it may be difficult for users to write a fake class: the original class
uses some third-party library that determines the output shape of the methods,
or is complicated and written by others. Besides, users may not care about the
limitations listed above. In this case, please reach out to us!
