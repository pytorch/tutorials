Customize Process Group Backends Using Cpp Extensions
=====================================================

**Author**: `Howard Huang <https://github.com/H-Huang>`, `Feng Tian <https://github.com/ftian1>`__, `Shen Li <https://mrshenli.github.io/>`__, `Min Si <https://minsii.github.io/>`__

.. note::
   |edit| View and edit this tutorial in `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/process_group_cpp_extension_tutorial.rst>`__.

Prerequisites:

-  `PyTorch Distributed Overview <../beginner/dist_overview.html>`__
-  `PyTorch Collective Communication Package <https://pytorch.org/docs/stable/distributed.html>`__
-  `PyTorch Cpp Extension <https://pytorch.org/docs/stable/cpp_extension.html>`__
-  `Writing Distributed Applications with PyTorch <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`__

This tutorial demonstrates how to implement a custom ``Backend`` and plug that into
`PyTorch distributed package <https://pytorch.org/docs/stable/distributed.html>`__ using
`cpp extensions <https://pytorch.org/docs/stable/cpp_extension.html>`__. This is helpful when you need a specialized software
stack for your hardware, or when you would like to experiment with new
collective communication algorithms.


Basics
------

PyTorch collective communications power several widely adopted distributed
training features, including
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__,
`ZeroRedundancyOptimizer <https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer>`__,
`FullyShardedDataParallel <https://github.com/pytorch/pytorch/blob/master/torch/distributed/_fsdp/fully_sharded_data_parallel.py>`__.
In order to make the same collective communication API work with
different communication backends, the distributed package abstracts collective
communication operations into a
`Backend <https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp>`__
class. Different backends can
then be implemented as subclasses of ``Backend`` using preferred
third-party libraries. PyTorch distributed comes with three default backends,
``ProcessGroupNCCL``, ``ProcessGroupGloo``, and ``ProcessGroupMPI``. However,
beyond these three backends, there are also other communication libraries
(e.g., `UCC <https://github.com/openucx/ucc>`__,
`OneCCL <https://github.com/oneapi-src/oneCCL>`__), different types of hardware
(e.g., `TPU <https://cloud.google.com/tpu>`__,
`Trainum <https://aws.amazon.com/machine-learning/trainium/>`__), and emerging
communication algorithms (e.g.,
`Herring <https://www.amazon.science/publications/herring-rethinking-the-parameter-server-at-scale-for-the-cloud>`__,
`Reduction Server <https://cloud.google.com/blog/topics/developers-practitioners/optimize-training-performance-reduction-server-vertex-ai>`__).
Therefore, the distributed package exposes extension APIs to allow customizing
collective communication backends.


The 4 steps below show how to implement a dummy ``Backend`` backend
and use that in Python application code. Please note that this tutorial focuses
on demonstrating the extension APIs, instead of developing a functioning
communication backend. Hence, the ``dummy`` backend just covers a subset of the
APIs (``all_reduce`` and ``all_gather``), and simply sets the values of tensors
to 0.


Step 1: Implement a Subclass of ``Backend``
------------------------------------------------

This first step is to implement a ``Backend`` subclass that overrides
target collective communication APIs and runs the custom communication algorithm.
The extension also needs to implement a ``Work`` subclass, which
serves as a future of communication results and allows asynchronous execution in
application code. If the extension uses third-party libraries, it can
include the headers and call into the library APIs from the ``BackendDummy``
subclass. The two code snippets below present the implementation of ``dummy.h`` and
``dummy.cpp``. See the `dummy collectives <https://github.com/H-Huang/torch_collective_extension>`__
repository for the full implementation.

.. code-block:: cpp

    // file name: dummy.hpp
    #include <torch/python.h>

    #include <torch/csrc/distributed/c10d/Backend.hpp>
    #include <torch/csrc/distributed/c10d/Work.hpp>
    #include <torch/csrc/distributed/c10d/Store.hpp>
    #include <torch/csrc/distributed/c10d/Types.hpp>
    #include <torch/csrc/distributed/c10d/Utils.hpp>

    #include <pybind11/chrono.h>

    namespace c10d {

    class BackendDummy : public Backend {
      public:
        BackendDummy(int rank, int size);

        c10::intrusive_ptr<Work> allgather(
            std::vector<std::vector<at::Tensor>>& outputTensors,
            std::vector<at::Tensor>& inputTensors,
            const AllgatherOptions& opts = AllgatherOptions()) override;

        c10::intrusive_ptr<Work> allreduce(
            std::vector<at::Tensor>& tensors,
            const AllreduceOptions& opts = AllreduceOptions()) override;

        // The collective communication APIs without a custom implementation
        // will error out if invoked by application code.
    };
    
    class WorkDummy : public Work {
      public:
        WorkDummy(
          OpType opType,
          c10::intrusive_ptr<c10::ivalue::Future> future) // future of the output
          : Work(
              -1, // rank, only used by recvAnySource, irrelevant in this demo
              opType),
          future_(std::move(future)) {}
        bool isCompleted() override;
        bool isSuccess() const override;
        bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
        virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

      private:
        c10::intrusive_ptr<c10::ivalue::Future> future_;
    };
    } // namespace c10d


.. code-block:: cpp

    // file name: dummy.cpp
    #include "dummy.hpp"

    namespace c10d {

    // This is a dummy allgather that sets all output tensors to zero
    // Modify the implementation to conduct real communication asynchronously
    c10::intrusive_ptr<Work> BackendDummy::allgather(
            std::vector<std::vector<at::Tensor>>& outputTensors,
            std::vector<at::Tensor>& inputTensors,
            const AllgatherOptions& /* unused */) {
        for (auto& outputTensorVec : outputTensors) {
            for (auto& outputTensor : outputTensorVec) {
                outputTensor.zero_();
            }
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
        future->markCompleted(c10::IValue(outputTensors));
        return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
    }

    // This is a dummy allreduce that sets all output tensors to zero
    // Modify the implementation to conduct real communication asynchronously
    c10::intrusive_ptr<Work> BackendDummy::allreduce(
            std::vector<at::Tensor>& tensors,
            const AllreduceOptions& opts) {
        for (auto& tensor : tensors) {
            tensor.zero_();
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
    }
    } // namespace c10d

Step 2: Expose The Extension Python APIs
----------------------------------------

The backend constructors are called
`from Python side <https://github.com/pytorch/pytorch/blob/v1.9.0/torch/distributed/distributed_c10d.py#L643-L650>`__,
so the extension also needs to expose the constructor APIs to Python. This can
be done by adding the following methods. In this example, ``store`` and
``timeout`` are ignored by the ``BackendDummy`` instantiation method, as
those are not used in this dummy implementation. However, real-world extensions
should consider using the ``store`` to perform rendezvous and supporting the
``timeout`` argument.

.. code-block:: cpp

    // file name: dummy.hpp
    class BackendDummy : public Backend {
        ...
        <Step 1 code>
        ...

        static c10::intrusive_ptr<Backend> createBackendDummy(
            const c10::intrusive_ptr<::c10d::Store>& store,
            int rank,
            int size,
            const std::chrono::duration<float>& timeout);

        static void BackendDummyConstructor() __attribute__((constructor)) {
            py::object module = py::module::import("torch.distributed");
            py::object register_backend =
                module.attr("Backend").attr("register_backend");
            // torch.distributed.Backend.register_backend will add `dummy` as a
            // new valid backend.
            register_backend("dummy", py::cpp_function(createBackendDummy));
        }
    }

.. code-block:: cpp

    // file name: dummy.cpp
    c10::intrusive_ptr<Backend> BackendDummy::createBackendDummy(
            const c10::intrusive_ptr<::c10d::Store>& /* unused */,
            int rank,
            int size,
            const std::chrono::duration<float>& /* unused */) {
        return c10::make_intrusive<BackendDummy>(rank, size);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("createBackendDummy", &BackendDummy::createBackendDummy);
    }


Step 3: Build The Custom Extension
----------------------------------

Now, the extension source code files are ready. We can then use
`cpp extensions <https://pytorch.org/docs/stable/cpp_extension.html>`__
to build it. To do that, create a ``setup.py`` file that prepares the paths and
commands. Then call ``python setup.py develop`` to install the extension.

If the extension depends on third-party libraries, you can also specify
``libraries_dirs`` and ``libraries`` to the cpp extension APIs. See the
`torch ucc <https://github.com/openucx/torch-ucc>`__
project as a real-world example.

.. code-block:: python

    # file name: setup.py
    import os
    import sys
    import torch
    from setuptools import setup
    from torch.utils import cpp_extension

    sources = ["src/dummy.cpp"]
    include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/"]

    if torch.cuda.is_available():
        module = cpp_extension.CUDAExtension(
            name = "dummy_collectives",
            sources = sources,
            include_dirs = include_dirs,
        )
    else:
        module = cpp_extension.CppExtension(
            name = "dummy_collectives",
            sources = sources,
            include_dirs = include_dirs,
        )

    setup(
        name = "Dummy-Collectives",
        version = "0.0.1",
        ext_modules = [module],
        cmdclass={'build_ext': cpp_extension.BuildExtension}
    )

Step 4: Use The Extension in Application
----------------------------------------

After installation, you can conveniently use the ``dummy`` backend when calling
`init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__
as if it is an builtin backend.

We can specify dispatching based on backend by changing the ``backend`` argument of ``init_process_group``. We 
can dispatch collective with CPU tensor to ``gloo`` backend and dispatch collective with CUDA tensor to ``dummy`` backend by 
specifying ``cpu:gloo,cuda:dummy`` as the backend argument.

To send all tensors to ``dummy`` backend, we can simply specify ``dummy`` as the backend argument.

.. code-block:: python

    import os

    import torch
    # importing dummy_collectives makes torch.distributed recognize `dummy`
    # as a valid backend.
    import dummy_collectives

    import torch.distributed as dist

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Alternatively:
    # dist.init_process_group("dummy", rank=0, world_size=1)
    dist.init_process_group("cpu:gloo,cuda:dummy", rank=0, world_size=1)

    # this goes through gloo
    x = torch.ones(6)
    dist.all_reduce(x)
    print(f"cpu allreduce: {x}")

    # this goes through dummy
    if torch.cuda.is_available():
        y = x.cuda()
        dist.all_reduce(y)
        print(f"cuda allreduce: {y}")

        try:
            dist.broadcast(y, 0)
        except RuntimeError:
            print("got RuntimeError when calling broadcast")
