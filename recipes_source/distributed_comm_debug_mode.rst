Getting Started with ``CommDebugMode``
=====================================================

**Author**: `Anshul Sinha <https://github.com/sinhaanshul>`__


In this tutorial, we will explore how to use ``CommDebugMode`` with PyTorch's
DistributedTensor (DTensor) for debugging by tracking collective operations in distributed training environments.

Prerequisites
---------------------

* Python 3.8 - 3.11
* PyTorch 2.2 or later


What is ``CommDebugMode`` and why is it useful
----------------------------------------------------
As the size of models continues to increase, users are seeking to leverage various combinations
of parallel strategies to scale up distributed training. However, the lack of interoperability
between existing solutions poses a significant challenge, primarily due to the absence of a
unified abstraction that can bridge these different parallelism strategies. To address this
issue, PyTorch has proposed `DistributedTensor(DTensor)
<https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/examples/comm_mode_features_example.py>`_
which abstracts away the complexities of tensor communication in distributed training,
providing a seamless user experience. However, when dealing with existing parallelism solutions and
developing parallelism solutions using the unified abstraction like DTensor, the lack of transparency
about what and when the collective communications happens under the hood could make it challenging
for advanced users to identify and resolve issues. To address this challenge, ``CommDebugMode``, a
Python context manager will serve as one of the primary debugging tools for DTensors, enabling
users to view when and why collective operations are happening when using DTensors, effectively
addressing this issue.


Using ``CommDebugMode``
------------------------

Here is how you can use ``CommDebugMode``:

.. code-block:: python

    # The model used in this example is a MLPModule applying Tensor Parallel
    comm_mode = CommDebugMode()
        with comm_mode:
            output = model(inp)

    # print the operation level collective tracing information
    print(comm_mode.generate_comm_debug_tracing_table(noise_level=0))

    # log the operation level collective tracing information to a file
    comm_mode.log_comm_debug_tracing_table_to_file(
        noise_level=1, file_name="transformer_operation_log.txt"
    )

    # dump the operation level collective tracing information to json file,
    # used in the visual browser below
    comm_mode.generate_json_dump(noise_level=2)

This is what the output looks like for a MLPModule at noise level 0:

.. code-block:: python

    Expected Output:
        Global
          FORWARD PASS
            *c10d_functional.all_reduce: 1
            MLPModule
              FORWARD PASS
                *c10d_functional.all_reduce: 1
                MLPModule.net1
                MLPModule.relu
                MLPModule.net2
                  FORWARD PASS
                    *c10d_functional.all_reduce: 1

To use ``CommDebugMode``, you must wrap the code running the model in ``CommDebugMode`` and call the API that
you want to use to display the data. You can also use a ``noise_level`` argument to control the verbosity
level of displayed information. Here is what each noise level displays:

| 0. Prints module-level collective counts
| 1. Prints DTensor operations (not including trivial operations), module sharding information
| 2. Prints tensor operations (not including trivial operations)
| 3. Prints all operations

In the example above, you can see that the collective operation, all_reduce, occurs once in the forward pass
of the ``MLPModule``. Furthermore, you can use ``CommDebugMode`` to pinpoint that the all-reduce operation happens
in the second linear layer of the ``MLPModule``.


Below is the interactive module tree visualization that you can use to upload your own JSON dump:

.. raw:: html

    <!DOCTYPE html>
    <html lang ="en">
    <head>
        <meta charset="UTF-8">
        <meta name = "viewport" content="width=device-width, initial-scale=1.0">
        <title>CommDebugMode Module Tree</title>
        <style>
            ul, #tree-container {
                list-style-type: none;
                margin: 0;
                padding: 0;
            }
            .caret {
                cursor: pointer;
                user-select: none;
            }
            .caret::before {
                content: "\25B6";
                color:black;
                display: inline-block;
                margin-right: 6px;
            }
            .caret-down::before {
                transform: rotate(90deg);
            }
            .tree {
                padding-left: 20px;
            }
            .tree ul {
                padding-left: 20px;
            }
            .nested {
                display: none;
            }
            .active {
                display: block;
            }
            .forward-pass,
            .backward-pass {
                margin-left: 40px;
            }
            .forward-pass table {
                margin-left: 40px;
                width: auto;
            }
            .forward-pass table td, .forward-pass table th {
                padding: 8px;
            }
            .forward-pass ul {
                display: none;
            }
            table {
                font-family: arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            td, th {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            tr:nth-child(even) {
                background-color: #dddddd;
            }
            #drop-area {
                position: relative;
                width: 25%;
                height: 100px;
                border: 2px dashed #ccc;
                border-radius: 5px;
                padding: 0px;
                text-align: center;
            }
            .drag-drop-block {
                display: inline-block;
                width: 200px;
                height: 50px;
                background-color: #f7f7f7;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                color: #666;
                cursor: pointer;
            }
            #file-input {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0;
            }
        </style>
    </head>
    <body>
        <div id="drop-area">
            <div class="drag-drop-block">
              <span>Drag file here</span>
            </div>
            <input type="file" id="file-input" accept=".json">
          </div>
        <div id="tree-container"></div>
        <script src="https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/distributed/tensor/debug/comm_mode_broswer_visual.js"></script>
    </body>
    </html>

Conclusion
------------------------------------------

In this recipe, we have learned how to use ``CommDebugMode`` to debug Distributed Tensors and
parallelism solutions that uses communication collectives with PyTorch. You can use your own
JSON outputs in the embedded visual browser.

For more detailed information about ``CommDebugMode``, see
`comm_mode_features_example.py
<https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/examples/comm_mode_features_example.py>`_
