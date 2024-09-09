(prototype) Flight Recorder for Debugging
=========================================
**Author**: `Chirag Pandya <https://github.com/c-p-i-o>`_, `Junjie Wang <https://github.com/fduwjj>`_

What you will learn
-------------------
* Learn about a new tool for debugging stuck jobs during distributed training.
* Learn how you can enable the tool and use the collected data for analyzing stuck jobs.

Prerequisites
-------------
- PyTorch version 2.5 or later.


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
There are two required environment variables to get the initial version of Flight Recorder working.

- ``TORCH_NCCL_TRACE_BUFFER_SIZE = (0, N)``: Setting ``N`` to a positive number enables collection.
     ``N`` represents the number of entries that will be kept internally in a circular buffer.
     We recommended to set this value at *2000*.
- ``TORCH_NCCL_DUMP_ON_TIMEOUT = (true, false)``: Setting this to ``true`` will write out diagnostic files to disk on job timeout.
     If enabled, there will be one file per rank output in the job's running directory.

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

.. code:: python

  python fr_trace.py -d <dump dir containing trace files> [-o <output file>]


Conclusion
----------
In this tutorial, we have learned about a new PyTorch diagnostic tool called Flight Recorder.
We have discussed how to enable Flight Recorder to collect diagnostic data from a machine.
Additionally, we explored how to analyze the data captured from the Flight Recorder using a
convenience script located in the `tools/flight_recorder <https://github.com/pytorch/pytorch/tree/main/tools/flight_recorder>`__
directory of the PyTorch repository.
