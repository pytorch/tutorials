Using CommDebugMode
=====================================================

**Author**: `Anshul Sinha <https://github.com/sinhaanshul>`__

Prerequisites:

- `Distributed Communication Package - torch.distributed <https://pytorch.org/docs/stable/distributed.html>`__
- Python 3.8 - 3.11
- PyTorch 2.2


What is CommDebugMode and why is it useful
------------------
As the size of models continues to increase, users are seeking to leverage various combinations of parallel strategies to scale up distributed training. However, the lack of interoperability between existing solutions poses a significant challenge, primarily due to the absence of a unified abstraction that can bridge these different parallelism strategies. To address this issue, PyTorch has proposed DistributedTensor (DTensor)which abstracts away the complexities of tensor communication in distributed training, providing a seamless user experience. However, this abstraction creates a lack of transparency that can make it challenging for users to identify and resolve issues. To address this challenge, my internship project aims to develop and enhance CommDebugMode, a Python context manager that will serve as one of the primary debugging tools for DTensors. CommDebugMode is a python context manager that enables users to view when and why collective operations are happening when using DTensors, addressing this problem.


How to use CommDebugMode
------------------------
Using CommDebugMode and getting its output is very simple.

.. code-block:: python

    comm_mode = CommDebugMode()
        with comm_mode:
            output = model(inp)

        # print the operation level collective tracing information
        print(comm_mode.generate_comm_debug_tracing_table(noise_level=2))

        # log the operation level collective tracing information to a file
        comm_mode.log_comm_debug_tracing_table_to_file(
            noise_level=1, file_name="transformer_operation_log.txt"
        )

        # dump the operation level collective tracing information to json file,
        # used in the visual browser below
        comm_mode.generate_json_dump(noise_level=2)
.. code-block:: python
