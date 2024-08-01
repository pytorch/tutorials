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

All users have to do is wrap the code running the model in CommDebugMode and call the API that they want to use to display the data.
Documentation Title
===================

Introduction to the Module
--------------------------

Below is the interactive module tree visualization:

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
        <script src="https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/distributed/_tensor/debug/comm_mode_broswer_visual.js"></script>
    </body>
    </html>

.. raw:: html
