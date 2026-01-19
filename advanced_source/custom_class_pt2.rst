Supporting Custom C++ Classes in torch.compile/torch.export
===========================================================


This tutorial is a follow-on to the
:doc:`custom C++ classes <custom_classes>` tutorial, and
introduces additional steps that are needed to support custom C++ classes in
torch.compile/torch.export.

.. warning::

    This feature is in prototype status and is subject to backwards compatibility
    breaking changes. This tutorial provides a snapshot as of PyTorch 2.8. If
    you run into any issues, please reach out to us on Github!

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
:doc:`Extending TorchScript with Custom C++ Classes <custom_classes>`,
we can create a thread-safe tensor queue and build it.

.. code-block:: cpp

    // Thread-safe Tensor Queue

    #include <torch/custom_class.h>
    #include <torch/script.h>

    #include <iostream>
    #include <string>
    #include <vector>

    using namespace torch::jit;

    // Thread-safe Tensor Queue
    struct TensorQueue : torch::CustomClassHolder {
    explicit TensorQueue(at::Tensor t) : init_tensor_(t) {}

    explicit TensorQueue(c10::Dict<std::string, at::Tensor> dict) {
        init_tensor_ = dict.at(std::string("init_tensor"));
        const std::string key = "queue";
        at::Tensor size_tensor;
        size_tensor = dict.at(std::string(key + "/size")).cpu();
        const auto* size_tensor_acc = size_tensor.const_data_ptr<int64_t>();
        int64_t queue_size = size_tensor_acc[0];

        for (const auto index : c10::irange(queue_size)) {
            at::Tensor val;
            queue_[index] = dict.at(key + "/" + std::to_string(index));
            queue_.push_back(val);
        }
    }

    // Push the element to the rear of queue.
    // Lock is added for thread safe.
    void push(at::Tensor x) {
        std::lock_guard<std::mutex> guard(mutex_);
        queue_.push_back(x);
    }
    // Pop the front element of queue and return it.
    // If empty, return init_tensor_.
    // Lock is added for thread safe.
    at::Tensor pop() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!queue_.empty()) {
            auto val = queue_.front();
            queue_.pop_front();
            return val;
        } else {
            return init_tensor_;
        }
    }

    std::vector<at::Tensor> get_raw_queue() {
        std::vector<at::Tensor> raw_queue(queue_.begin(), queue_.end());
        return raw_queue;
    }

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
            .def("get_raw_queue", &TensorQueue::get_raw_queue);
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
    };

    TORCH_LIBRARY(MyCustomClass, m) {
        m.class_<TensorQueue>("TensorQueue")
            .def(torch::init<at::Tensor>())
            ...
            .def("__obj_flatten__", &TensorQueue::__obj_flatten__);
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

**Step 2b**: Implement an ``__obj_unflatten__`` classmethod in Python.

.. code-block:: python

    # namespace::class_name
    @torch._library.register_fake_class("MyCustomClass::TensorQueue")
    class FakeTensorQueue:
        ...
        @classmethod
        def __obj_unflatten__(cls, flattened_tq):
            return cls(**dict(flattened_tq))


That’s it! Now we can create a module that uses this object and run it with ``torch.compile`` or ``torch.export``.

.. code-block:: python

    import torch

    torch.classes.load_library("build/libcustom_class.so")
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
            ...
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

:term:`Tracing` with real custom object has several major downsides:

1. Operators on real objects can be time consuming e.g. the custom object
   might be reading from the network or loading data from the disk.

2. We don’t want to mutate the real custom object or create side-effects to the environment while tracing.

3. It cannot support dynamic shapes.

However, it may be difficult for users to write a fake class, e.g. if the
original class uses some third-party library that determines the output shape of
the methods, or is complicated and written by others. In such cases, users can
disable the fakification requirement by defining a ``tracing_mode`` method to
return ``"real"``:

.. code-block:: cpp

    std::string tracing_mode() {
        return "real";
    }


A caveat of fakification is regarding **tensor aliasing.** We assume that no
tensors within a torchbind object aliases a tensor outside of the torchbind
object. Therefore, mutating one of these tensors will result in undefined
behavior.
