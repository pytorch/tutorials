# Getting Started with `CommDebugMode`

**Author**: [Anshul Sinha](https://github.com/sinhaanshul)

In this tutorial, we will explore how to use `CommDebugMode` with PyTorch's
DistributedTensor (DTensor) for debugging by tracking collective operations in distributed training environments.

## Prerequisites

- Python 3.8 - 3.11
- PyTorch 2.2 or later

## What is `CommDebugMode` and why is it useful

As the size of models continues to increase, users are seeking to leverage various combinations
of parallel strategies to scale up distributed training. However, the lack of interoperability
between existing solutions poses a significant challenge, primarily due to the absence of a
unified abstraction that can bridge these different parallelism strategies. To address this
issue, PyTorch has proposed [DistributedTensor(DTensor)](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/examples/comm_mode_features_example.py)
which abstracts away the complexities of tensor communication in distributed training,
providing a seamless user experience. However, when dealing with existing parallelism solutions and
developing parallelism solutions using the unified abstraction like DTensor, the lack of transparency
about what and when the collective communications happens under the hood could make it challenging
for advanced users to identify and resolve issues. To address this challenge, `CommDebugMode`, a
Python context manager will serve as one of the primary debugging tools for DTensors, enabling
users to view when and why collective operations are happening when using DTensors, effectively
addressing this issue.

## Using `CommDebugMode`

Here is how you can use `CommDebugMode`:

```
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
```

This is what the output looks like for a MLPModule at noise level 0:

```
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
```

To use `CommDebugMode`, you must wrap the code running the model in `CommDebugMode` and call the API that
you want to use to display the data. You can also use a `noise_level` argument to control the verbosity
level of displayed information. Here is what each noise level displays:

0. Prints module-level collective counts
1. Prints DTensor operations (not including trivial operations), module sharding information
2. Prints tensor operations (not including trivial operations)
3. Prints all operations

In the example above, you can see that the collective operation, all_reduce, occurs once in the forward pass
of the `MLPModule`. Furthermore, you can use `CommDebugMode` to pinpoint that the all-reduce operation happens
in the second linear layer of the `MLPModule`.

Below is the interactive module tree visualization that you can use to upload your own JSON dump:

html

CommDebugMode Module Tree

Drag file here

## Conclusion

In this recipe, we have learned how to use `CommDebugMode` to debug Distributed Tensors and
parallelism solutions that uses communication collectives with PyTorch. You can use your own
JSON outputs in the embedded visual browser.

For more detailed information about `CommDebugMode`, see
[comm_mode_features_example.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/examples/comm_mode_features_example.py)