(prototype) Flight Recorder for Debugging
=========================================
**Author**: `Chirag Pandya <https://github.com/c-p-i-o>`, `Junjie Wang <https://github.com/fduwjj>`

This tutorial introduces a new tool for debugging stuck jobs during distributed training.

Background and Motivation
--------------------------
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

Flight Recorder captures diagnostics information as collectives run. The diagnostic information can be used to help root
cause the underlying issue. There are 2 core parts to flight recorder.
- The collection portion. When enabled, information about collectives are recorded in an in-memory circular buffer.
Upon job timeout, or on demand, the in-memory buffer can be retrieved or dumped to file.
- An analyzer script is available in the `pytorch/tools/flight_recorder` directory (details below). T

Enabling Flight Recorder
------------------------
There are two required environment variables to get the initial version of flight recorder working.
   - TORCH_NCCL_TRACE_BUFFER_SIZE (0, N where N is a postitive number) N = collection enabled. Recommended to set this
     to 2000)
   - TORCH_NCCL_DUMP_ON_TIMEOUT = (true, false) true = write out diagnostic files to disk on job timeout.
Optional settings:
   - TORCH_NCCL_TRACE_CPP_STACK (true, false) true = enable cpp stack trace captures in flight recorder (for slow
     addr2line - see additinal settings)
   - TORCH_NCCL_ENABLE_TIMING (true, false) true = enable additional cuda events at the start of each collective and
     record the ‘duration’ of each collective. May incur some CPU overhead.

Flight Recorder File Formats
----------------------------
Flight recorder files are dumped out in `pickle` format.
```
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
```
Analyzing Flight Recorder Dumps
-------------------------------
We have convenient scripts available in `pytorch/tools/flight_recorder` directory that can be used to analyze captured
data.

To run it, one can use command line: 
```
python fr_trace.py -d <dump dir containing trace files> [-o <output file>]
```
