Using CommDebugMode
=====================================================

**Author**: `Anshul Sinha <https://github.com/sinhaanshul>`__

Prerequisites:

- `Distributed Communication Package - torch.distributed <https://pytorch.org/docs/stable/distributed.html>`__
- Python 3.8 - 3.11
- PyTorch 2.2


What is CommDebugMode and why is it useful
------------------------------------------
As the size of models continues to increase, users are seeking to leverage various combinations
of parallel strategies to scale up distributed training. However, the lack of interoperability
between existing solutions poses a significant challenge, primarily due to the absence of a
unified abstraction that can bridge these different parallelism strategies. To address this
issue, PyTorch has proposed DistributedTensor(DTensor) which abstracts away the complexities of
tensor communication in distributed training, providing a seamless user experience. However,
this abstraction creates a lack of transparency that can make it challenging for users to
identify and resolve issues. To address this challenge, CommDebugMode, a Python context manager
will serve as one of the primary debugging tools for DTensors, enabling users to view when and
why collective operations are happening when using DTensors, addressing this problem.


How to use CommDebugMode
------------------------
Using CommDebugMode and getting its output is very simple.

.. code-block:: python

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

    """
    This is what the output looks like for a MLPModule at noise level 0
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
    """

All users have to do is wrap the code running the model in CommDebugMode and call the API that
they want to use to display the data. One important thing to note is that the users can use a noise_level
arguement to control how much information is displayed to the user. The information below shows what each
noise level displays

| 0. prints module-level collective counts
| 1. prints dTensor operations not included in trivial operations, module information
| 2. prints operations not included in trivial operations
| 3. prints all operations

In the example above, users can see in the first picture that the collective operation, all_reduce, occurs
once in the forward pass of the MLPModule. The second picture provides a greater level of detail, allowing
users to pinpoint that the all-reduce operation happens in the second linear layer of the MLPModule.


Below is the interactive module tree visualization that users can upload their JSON dump to:

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

Conclusion
------------------------------------------
In conclusion, we have learned how to use CommDebugMode in order to debug Distributed Tensors
and can use future json dumps in the embedded visual browser.

For more detailed information about CommDebugMode, please see
https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/examples/comm_mode_features_example.py
