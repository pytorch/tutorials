(prototype) Flight Recorder for Debugging
=========================================
**Author**: `Chirag Pandya <https://github.com/c-p-i-o>`, `Junjie Wang <https://github.com/fduwjj>`

What you will learn
-------------------
This tutorial introduces a new tool for debugging stuck jobs during distributed training. The tutorial explains how this
new tool can be enabled and how to use the collected data for analyzing stuck jobs.

Overview, Background and Motivation
-----------------------------------
An AI distributed training job refers to the process of training a machine learning model using multiple devices, such
as GPUs or CPUs, connected in a network. This approach allows for faster and more efficient training of large models
that require significant computational resources.
An engineer’s goal is to complete an AI training job as fast as possible and make continuous improvements such that
subsequent training can be done faster. A trained usable model is the final desired outcome.
One of the biggest impediment to completing training is the concept of a "stuck job".

A distributed AI training job is considered "stuck" when it stops making meaningful progress for an extended period of
time.

A job can get stuck for various reasons:
    - Data Starvation: This happens when the training job is not receiving data at the expected rate. This could be due to
issues with the data pipeline or the data source.
    - Resource Constraints: If the system running the job does not have enough computational resources (like CPU, GPU, or
memory), the job might not be able to proceed.
    - Network Issues: In a distributed training setup, different parts of the model or data may be processed on different
devices. If there are network issues, communication between these devices may be disrupted, causing the job to get
stuck.
    - Software Bugs or Errors: Errors in the training code or the underlying libraries and frameworks can also cause a job to
get stuck.
    - Synchronization Issues: In distributed training, different parts of the computation are often run in parallel and need
to be synchronized at certain points. If this synchronization fails, the job can get stuck. For example, a deadlock can
occur if one or ranks fail to join a collective while the remaining ranks have joined. This results in an
indefinite wait for the job to progress.

Flight Recorder, as the name suggests, captures diagnostics information as collectives run. The captured diagnostic
information can be used to help root cause the underlying issue when jobs get stuck.
There are 2 core parts to flight recorder.
- The collection portion. When enabled, information about collectives are recorded in an in-memory circular buffer.
Upon job timeout, or on demand, the in-memory buffer can be retrieved or dumped to file.
- An analyzer script is available in the `pytorch/tools/flight_recorder` directory (details below).

Prerequisites
-------------
None. This is a new debugging tool that is available in PyTorch version 2.5.

Enabling Flight Recorder
------------------------
There are two required environment variables to get the initial version of flight recorder working.
   - TORCH_NCCL_TRACE_BUFFER_SIZE (0, N where N is a positive number) N = collection enabled. N represents the number of
     entries that will be kept internally in a circular buffer. Recommended to set this at 2000.
   - TORCH_NCCL_DUMP_ON_TIMEOUT = (true, false) true = write out diagnostic files to disk on job timeout. If set,
     there will be one file per rank output in the jobs running directory.
Optional settings:
   - TORCH_NCCL_TRACE_CPP_STACK (true, false) true = enable cpp stack trace captures in flight recorder (for slow
     addr2line - see additinal settings)
   - TORCH_NCCL_ENABLE_TIMING (true, false) true = enable additional cuda events at the start of each collective and
     record the ‘duration’ of each collective. May incur some CPU overhead.

Additional settings
-------------------
TORCH_SYMBOLIZE_MODE: {dladdr, addr2line, fast}: This setting controls the program that is used to retrieve C++ traces
from a running program. The default setting is `addr2line`. `fast` is a new experimental mode that is shown to be much
faster than the traditional `addr2line`.

Retrieving Flight Recorder Data via an API
------------------------------------------
Flight recorder data can also be retrieved via an API call.
The API is shown below with the default arguments.
.. code:: python
  torch._C._distributed_c10d._dump_nccl_trace(includeCollectives=True, includeStackTraces=True, onlyActive=False)

To view the data, you can unpickle the data
.. code:: python
  t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
  print(t)

Flight Recorder File Formats
----------------------------
Flight recorder files are dumped out in `pickle` format. Files are written out to local disks or mounted shared NFS
folders.
Contents of a flight recorder `unpickled` file is shown below.
.. code-block: JSON
  {
    "version": "2.3",
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
          ...
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
    ...]
  }

Analyzing Flight Recorder Dumps
-------------------------------
We have convenient scripts available in `pytorch/tools/flight_recorder` directory that can be used to analyze captured
data.

1. In order to run the convenience script, all files from a rank must first be copied over into a single directory.

2. To run it, one can use command line:
.. code:: python
  python fr_trace.py -d <dump dir containing trace files> [-o <output file>]


Conclusion
----------
This tutorial introduces a new PyTorch diagnostic tool called `flight recorder`. The tutorial talks about how flight
recorder can be enabled to collect diagnostic data from a machine.
Data captured from flight recorder can be analyzed using a convenience script in the `tools/flight_recorder` directory
in the PyTorch repository.
