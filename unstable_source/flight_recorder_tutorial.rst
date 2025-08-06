Flight Recorder for Debugging Stuck Jobs
====================================================
**Author**: `Chirag Pandya <https://github.com/c-p-i-o>`_, `Junjie Wang <https://github.com/fduwjj>`_

What you will learn
-------------------
* Learn about a new tool for debugging stuck jobs during distributed training.
* Learn how you can enable the tool and use the collected data for analyzing stuck jobs.

Prerequisites
-------------

- PyTorch version 2.5 or later.
- `tabulate <https://pypi.org/project/tabulate/>`__. You can install by running ``pip install tabulate``.


Overview
--------
An AI distributed training job refers to the process of training a machine learning model using multiple devices, such
as GPUs or CPUs, connected in a network. This approach allows for faster and more efficient training of large models
that require significant computational resources.
An engineer’s goal is to complete an AI training job as quickly as possible and make continuous improvements so that
subsequent training can be done faster. A trained, usable model is the final desired outcome.
One of the biggest impediment to completing training is the concept of a *stuck job*.

A distributed AI training job is considered `stuck` when it stops making meaningful progress for an extended period of
time.

A job can get stuck for various reasons:

- **Data Starvation:** This occurs when the training job is not receiving data at the expected rate, possibly due to issues with the data pipeline or the data source.

- **Resource Constraints:** If the system running the job does not have enough computational resources (such as CPU, GPU, or memory), the job might not be able to proceed.

- **Network Issues:** In a distributed training setup, different parts of the model or data may be processed on different devices. If there are network issues, communication between these devices may be disrupted, causing the job to get stuck.

- **Software Bugs or Errors:** Errors in the training code or the underlying libraries and frameworks can also cause a job to get stuck.

- **Synchronization Issues:** In distributed training, different parts of the computation are often run in parallel and need to be synchronized at certain points. If this synchronization fails, the job can get stuck. For example, a deadlock can occur if one or more ranks fail to join a collective while the remaining ranks have joined. This results in an indefinite wait for the job to progress.

Flight Recorder, as the name suggests, captures diagnostics information as collectives run. The captured diagnostic
information is used to help identify the root causes of issues when jobs become stuck.
Flight Recorder consists of two core parts:

- The collection portion: when enabled, information about collectives is recorded in an in-memory circular buffer. Upon job timeout, or on demand, the in-memory buffer can be retrieved or dumped to file.

- An analyzer script is available in the `tools/flight_recorder <https://github.com/pytorch/pytorch/tree/main/tools/flight_recorder>`__ directory (details below).
   The analyzer script runs known heuristics using the collected data and attempts to automatically identify the underlying issue that caused the job to stall.

Enabling Flight Recorder
------------------------
There are three required environment variables to get the initial version of Flight Recorder working.

- ``TORCH_NCCL_TRACE_BUFFER_SIZE = (0, N)``: Setting ``N`` to a positive number enables collection.
  ``N`` represents the number of entries that will be kept internally in a circular buffer.
  We recommended to set this value at *2000*. The default value is ``2000``.
- ``TORCH_NCCL_DUMP_ON_TIMEOUT = (true, false)``: Setting this to ``true`` will write out diagnostic files to disk on job timeout.
  If enabled, there will be one file per rank output in the job's running directory. The default value is ``false``.
- ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE``: Setting the path where the flight recorder will be dumped with file prefix. One file per
  rank. The default value is ``/tmp/nccl_trace_rank_``.

**Optional settings:**

- ``TORCH_NCCL_TRACE_CPP_STACK = (true, false)``: Setting this to true enables C++ stack traces to be captured in Flight Recorder.
  C++ stack traces can be useful in providing the exact code path from a PyTorch Python call down to the primitive
  C++ implementation. Also see ``TORCH_SYMBOLIZE_MODE`` in additional settings.
- ``TORCH_NCCL_ENABLE_TIMING = (true, false)``: Setting this to ``true`` will enable additional cuda events at the start of each collective and
  records the *duration* of each collective. This may incur some CPU overhead. In the collected data, the
  *duration* field indicates how long each collective took to execute.

Additional Settings
-------------------

- ``TORCH_SYMBOLIZE_MODE = (dladdr, addr2line, fast)``: This setting determines the program used to retrieve C++ traces from a running program.
     The default setting is ``addr2line``.

     ``fast`` is a new experimental mode that is shown to be much faster than the traditional ``addr2line``.
     Use this setting in conjunction with ``TORCH_NCCL_TRACE_CPP_STACK`` to collect C++ traces in the Flight Recorder data.
- If you prefer not to have the flight recorder data dumped into the local disk but rather onto your own storage, you can define your own writer class.
  This class should inherit from class ``::c10d::DebugInfoWriter`` `(code) <https://github.com/pytorch/pytorch/blob/release/2.5/torch/csrc/distributed/c10d/NCCLUtils.hpp#L237>`__
  and then register the new writer using ``::c10d::DebugInfoWriter::registerWriter`` `(code) <https://github.com/pytorch/pytorch/blob/release/2.5/torch/csrc/distributed/c10d/NCCLUtils.hpp#L242>`__
  before we initiate PyTorch distributed.

Retrieving Flight Recorder Data via an API
------------------------------------------

You can also retrieve Flight Recorder data with an API call.
The API with the default arguments is shown below:

.. code:: python

  torch._C._distributed_c10d._dump_nccl_trace(includeCollectives=True, includeStackTraces=True, onlyActive=False)

To view the data, you can ``unpickle`` it as shown below:

.. code:: python

  t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
  print(t)

Flight Recorder File Formats
----------------------------

Flight Recorder files are dumped in ``pickle`` format. Files are written to local disks or mounted shared NFS
folders.

The contents of a Flight Recorder ``unpickled`` file are shown below:

.. code-block:: json

  {
    "version": "2.5",
    "pg_config": {
      "0": {
      "name": "0",
      "desc": "default_pg",
      "ranks": "[0, 1]"
      }
    },
    "pg_status": {
      "0": {
      "last_enqueued_collective": 2,
      "last_started_collective": -1,
      "last_completed_collective": 2
      }
    },
    "entries": [
    {
      "frames": [
      {
      "name": "test_short_pickle",
      "filename": "pytorch/test/distributed/test_c10d_nccl.py",
      "line": 3647
      },
      {
      "name": "spawn_main",
      "filename": ".conda/envs/pytorch-3.10/lib/python3.10/multiprocessing/spawn.py",
      "line": 116
      },
      {
      "name": "<module>",
      "filename": "<string>",
      "line": 1
      }
      ],
      "record_id": 0,
      "pg_id": 0,
      "process_group": ("0", "default_pg"),
      "collective_seq_id": 1,
      "p2p_seq_id": 0,
      "op_id": 1,
      "profiling_name": "nccl:all_reduce",
      "time_created_ns": 1724779239936775119,
      "input_sizes": [[3, 4]],
      "input_dtypes": ["Float"],
      "output_sizes": [[3, 4]],
      "output_dtypes": ["Float"],
      "state": "completed",
      "time_discovered_started_ns": null,
      "time_discovered_completed_ns": 1724779239975811724,
      "retired": true,
      "timeout_ms": 600000,
      "is_p2p": false
      },
      ...
      ]
  }

Analyzing Flight Recorder Dumps
-------------------------------

We have convenient scripts available in `pytorch/tools/flight_recorder` directory for analyzing captured
data.

To run the convenience script, follow these steps:

1. Copy all files from a rank into a single directory.

2. To run the script, use this command:

.. code:: shell

  python fr_trace.py <dump dir containing trace files> [-o <output file>]

If you install the PyTorch nightly build or build from scratch with ``USE_DISTRIBUTED=1``, you can directly use the following
command directly:

.. code:: shell

  torchfrtrace <dump dir containing trace files> [-o <output file>]


Currently, we support two modes for the analyzer script. The first mode allows the script to apply some heuristics to the parsed flight
recorder dumps to generate a report identifying potential culprits for the timeout. The second mode is simply outputs the raw dumps.
By default, the script prints flight recoder dumps for all ranks and all ``ProcessGroups``(PGs). This can be narrowed down to certain
ranks and PGs using the *--selected-ranks* argument for ranks and *--pg-filters* argument for PGs. An example command is:

Caveat: tabulate module is needed, so you might need pip install it first.

.. code:: shell

  python fr_trace.py <dump dir containing trace files> -j [--selected-ranks i j k ...] [--pg-filters tp dp]
  torchfrtrace <dump dir containing trace files> -j [--selected-ranks i j k ...] [--pg-filters 0 2]

An End-to-End Example
------------------------------------
To demonstrate the use of Flight Recorder, we will use a small program where we induce mismatched collectives.
In this example, ``rank0`` is programmed to do an additional collective.
The Flight Recorder dump files are saved to the ``/tmp`` directory.
For demonstration purposes, we named this program ``crash.py``.

.. note::
   Please note that this is a simplified example. In real-world scenarios, the process would involve more
   complexities.

.. code:: python

  import torch
  import torch.distributed as dist
  import os
  from datetime import timedelta

  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert world_size <= 8, "world size must be less than or equal to 8"
  os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = "/tmp/trace_"
  os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
  os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "2000"
  device = torch.device(f"cuda:{local_rank}")
  print(f"{local_rank=} {world_size=} master addr: {os.environ['MASTER_ADDR']} master port: {os.environ['MASTER_PORT']} {device=}")

  # Initialize the process group with a small timeout so that jobs fail quickly
  dist.init_process_group("nccl", world_size=world_size, rank=local_rank, timeout=timedelta(seconds=1))

  a = torch.full((3, 4), float(local_rank), device=device)
  # Write some collectives to populate Flight Recorder data
  for i in range(2):
    print(f"calling allreduce on {local_rank=}")
    f = dist.all_reduce(a)

  # rank0 is doing an additional collective
  if local_rank == 0:
    print("rank0 is doing an allreduce on tensor b, but other ranks forgot")
    b = torch.full((4,5), float(local_rank), device=device)
    f = dist.all_reduce(b)

  for i in range(2):
    print(f"calling allreduce on {local_rank=}")
    f = dist.all_reduce(a)

  torch.cuda.synchronize(device=device)
  print(f"{local_rank=} exiting")


To run this program, use ``torchrun``:


.. code:: python

  torchrun --nnodes=1 --nproc_per_node=2 crash.py

You should see two files in the ``/tmp`` directory:

.. code:: bash

  $ls /tmp/trace*
  # Expected output
  /tmp/trace_0 /tmp/trace_1

Finally, to analyze these two files, we use the ``torchfrtrace`` command:

.. code:: bash

  torchfrtrace --prefix "trace_" /tmp/

The output from the trace command is meant to be human-readable. It includes information about the
set of collectives that caused a failure.
The output for the command above is shown below.
We can clearly see that rank 1 did not join the "all_reduce" collective.

.. code-block:: bash
  $torchfrtrace --prefix "trace_" /tmp/
  Not all ranks joining collective 5 at entry 4
  group info: 0:default_pg
  collective: nccl:all_reduce
  missing ranks: {1}
  input sizes: [[3, 4]]
  output sizes: [[3, 4]]
  expected ranks: 2
  collective state: scheduled
  collective stack trace:
    all_reduce at /home/cpio/local/pytorch/torch/distributed/distributed_c10d.py:2696
    wrapper at /home/cpio/local/pytorch/torch/distributed/c10d_logger.py:83
    <module> at /home/cpio/test/crash.py:44



Conclusion
----------
In this tutorial, we have learned about a new PyTorch diagnostic tool called Flight Recorder.
We have discussed how to enable Flight Recorder to collect diagnostic data from a machine.
Additionally, we explored how to analyze the data captured from the Flight Recorder using a
convenience script located in the `tools/flight_recorder <https://github.com/pytorch/pytorch/tree/main/tools/flight_recorder>`__
directory of the PyTorch repository.
