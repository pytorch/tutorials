Debugging Hangs with Flight Recorder Using TorchComms and Debug Server
===================================================================

**TorchComms** is a Python library that provides a high-level
communicator abstraction over PyTorch's distributed backends (NCCL,
Gloo, etc.). It wraps collective operations with a hook system that
lets you attach instrumentation, such as logging, profiling, or recording,
without modifying application code. For more information, see
`TorchComms Documentation <https://meta-pytorch.org/torchcomms/main/index.html>`_.

The **TorchComms Flight Recorder** is one such hook
(``FlightRecorderHook``). It maintains a fixed-size ring buffer that
silently records every collective operation issued through a TorchComms
communicator that captures the operation type, sequence number, tensor
shapes, execution state, and the Python stack trace at the call site. For a reference on the Flight Recorder, see
`Flight Recorder Hook <https://meta-pytorch.org/torchcomms/main/hooks.html#flightrecorderhook>`_
in TorchComms.

The **Debug Server** (``torch.distributed.debug``) runs an HTTP server
on each rank that can periodically dump Flight Recorder snapshots and
Python stack traces to disk. For full API documentation on the debug server, its
endpoints and periodic dumping, see the
`torch.distributed debug HTTP server docs <https://pytorch.org/docs/main/distributed.html#torch-distributed-debug-http-server>`_.

This tutorial walks you through a concrete example: a two-rank job where
one rank hangs before a collective, and shows how to use the Flight
Recorder and Debug Server to diagnose it step by step.

.. note::

   This tutorial covers the **TorchComms Debug Server** approach to
   flight recorder dumps. For the older environment-variable-based
   Flight Recorder configuration, see
   :doc:`/unstable/flight_recorder_tutorial`.


Prerequisites
-------------

* PyTorch 2.5 or later with ``torch.distributed``
* ``torchcomms`` installed (``pip install torchcomms``)
* A CUDA host with 2 or more GPUs, or use ``TEST_BACKEND=gloo TEST_DEVICE=cpu`` for CPU-only testing
* Familiarity with `distributed PyTorch concepts <https://pytorch.org/tutorials/beginner/dist_overview.html>`_


What You Will Learn
-------------------

* How to attach the ``FlightRecorderHook`` to a TorchComms communicator
* How to start the Debug Server with periodic Flight Recorder dumps
* How to read aggregated text dumps to identify missing ranks and mismatched collectives
* How to use per-rank pickle traces with the FR CLI for cross-rank analysis
* How to interpret stack trace snapshots to pinpoint the exact line of a hang


Flight Recorder Overview
------------------------

Each Flight Recorder entry captures:

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - ``collective_seq_id``
     - Monotonically increasing sequence number (same across all ranks for a given collective)
   * - ``profiling_name``
     - e.g. ``nccl:all_reduce``, ``nccl:broadcast``
   * - ``state``
     - ``scheduled`` → ``started`` → ``completed``
   * - ``input_dims`` / ``output_dims``
     - Tensor shapes
   * - ``traceback``
     - Python stack trace at the call site

When periodic dumping is enabled on the debug server, each dump cycle
produces two kinds of output:

* **Aggregated text files** (``torchcomms_fr_trace_<ts>.txt``) — the
  frontend on rank 0 fetches FR data from all ranks and writes a
  human-readable table.
* **Per-rank pickle files** (``per_rank/rank_<N>``) — each rank's worker
  server writes its own pickle trace. These can be fed to the
  **FR CLI** (``python -m torch.distributed.flight_recorder.fr_trace``)
  for automated cross-rank mismatch detection.


The Scenario
------------

The demo script below creates a two-phase workload:

* **Phase 1** (all ranks): 3 ``all_reduce`` + 1 ``broadcast`` operations complete normally.
* **Phase 2**:

  * The hanging rank enters ``time.sleep``.
  * Other ranks issue another ``all_reduce`` that times out waiting for the hanging rank.

When the timeout fires, the dump directory contains:

::

   FR_DUMP_DIR/
   ├── torchcomms_fr_trace_<ts>.txt   ← aggregated text
   └── per_rank/                      ← per-rank pickle files
       ├── rank_0
       └── rank_1


Demo Script
-----------

Save the following as ``verify_flight_recorder.py``:

.. code-block:: python

   import os
   import time
   from datetime import timedelta

   import torch
   from torch.distributed.debug import start_debug_server
   from torchcomms import new_comm, ReduceOp
   from torchcomms.hooks import FlightRecorderHook


   def main():
       backend = os.environ.get("TEST_BACKEND", "gloo")
       device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

       dump_dir = os.environ.get("FR_DUMP_DIR", "/tmp/fr_hang_debug")
       dump_interval = float(os.environ.get("FR_DUMP_INTERVAL", "5"))
       timeout_seconds = int(os.environ.get("COMM_TIMEOUT", "30"))
       hanging_rank = int(os.environ.get("HANGING_RANK", "-1"))

       os.makedirs(dump_dir, exist_ok=True)

       per_rank_dir = os.path.join(dump_dir, "per_rank")
       os.makedirs(per_rank_dir, exist_ok=True)
       dump_prefix = os.path.join(per_rank_dir, "rank_")
       os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"] = dump_prefix

       comm = new_comm(
           backend=backend,
           device=device,
           name="main_comm",
           timeout=timedelta(seconds=timeout_seconds),
           abort_process_on_timeout_or_error=False,
       )

       rank = comm.get_rank()
       world_size = comm.get_size()

       if hanging_rank < 0:
           hanging_rank = world_size - 1

       num_devices = torch.cuda.device_count()
       device_id = rank % num_devices
       target_device = torch.device(f"cuda:{device_id}")

       print(
           f"[Rank {rank}/{world_size}] device={device_id}, "
           f"hanging_rank={hanging_rank}, timeout={timeout_seconds}s"
       )

       # ── Debug Server with Periodic Dumps ──
       start_debug_server(
           port=25999,
           dump_dir=dump_dir,
           dump_interval=dump_interval,
           enabled_dumps={"torchcomms_fr_trace", "stacks"},
       )
       if rank == 0:
           print(f"[Rank {rank}] Debug server: http://localhost:25999")
           print(f"[Rank {rank}] Periodic dumps every {dump_interval}s → {dump_dir}")
           print(f"[Rank {rank}] Per-rank pickles → {per_rank_dir}")

       # ── Flight Recorder Hook ──
       recorder = FlightRecorderHook(max_entries=100)
       recorder.register_with_comm(comm)

       tensor = torch.full(
           (1024,),
           float(rank + 1),
           dtype=torch.float32,
           device=target_device,
       )

       # ── Phase 1: Successful collectives (all ranks) ──
       print(f"[Rank {rank}] Phase 1: Running 3 all_reduce + 1 broadcast")
       for _i in range(3):
           comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)
       comm.broadcast(tensor, root=0, async_op=False)
       torch.cuda.current_stream().synchronize()
       print(f"[Rank {rank}] Phase 1 complete")

       # ── Phase 2: One rank hangs ──
       if rank == hanging_rank:
           print(f"[Rank {rank}] >>> HANGING – entering infinite sleep <<<")
           while True:
               time.sleep(1)

       print(
           f"[Rank {rank}] Phase 2: all_reduce "
           f"(rank {hanging_rank} will NOT participate)"
       )
       print(f"[Rank {rank}] Expecting timeout in ~{timeout_seconds}s ...")

       try:
           comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)
       except Exception as e:
           print(f"[Rank {rank}] Caught timeout: {type(e).__name__}: {e}")
           recorder.dump_file(rank)
           print(f"[Rank {rank}] Pickle trace written to {dump_prefix}{rank}")

       recorder.unregister()
       comm.finalize()


   if __name__ == "__main__":
       main()

.. note::

   This script requires ``torchcomms`` (``pip install torchcomms``) and
   ``torch.distributed.debug``. The ``torchcomms`` package depends on
   ``tabulate``, ``jinja2``, and ``aiohttp``.


Running the Demo
----------------

Launch
^^^^^^

.. code-block:: bash

   FR_DUMP_DIR=/tmp/fr_hang_debug \
   FR_DUMP_INTERVAL=3 \
   COMM_TIMEOUT=15 \
   TEST_BACKEND=gloo \
   TEST_DEVICE=cpu \
   torchrun --nproc_per_node=2 verify_flight_recorder.py

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``FR_DUMP_DIR``
     - ``/tmp/fr_hang_debug``
     - Root dump directory
   * - ``FR_DUMP_INTERVAL``
     - ``5``
     - Seconds between periodic dumps
   * - ``COMM_TIMEOUT``
     - ``30``
     - Communicator timeout (seconds)
   * - ``HANGING_RANK``
     - ``-1`` (last rank)
     - Which rank to hang
   * - ``TEST_BACKEND``
     - ``gloo``
     - Communication backend
   * - ``TEST_DEVICE``
     - ``cuda``
     - Tensor device

Expected output
^^^^^^^^^^^^^^^

::

   [Rank 0/2] device=0, hanging_rank=1, timeout=15s
   [Rank 1/2] device=1, hanging_rank=1, timeout=15s
   [Rank 0] Debug server: http://localhost:25999
   [Rank 0] Periodic dumps every 3.0s → /tmp/fr_hang_debug
   [Rank 0] Per-rank pickles → /tmp/fr_hang_debug/per_rank
   [Rank 0] Phase 1: Running 3 all_reduce + 1 broadcast
   [Rank 0] Phase 1 complete
   [Rank 0] Phase 2: all_reduce (rank 1 will NOT participate)
   [Rank 0] Expecting timeout in ~15s ...
   [Rank 1] Phase 1 complete
   [Rank 1] >>> HANGING – entering infinite sleep <<<

   ... periodic mismatch warnings every 3 seconds ...

   Not all ranks joining collective, sequence number: 4
   collective: nccl:all_reduce
   missing ranks: {1}
   collective state: scheduled

   ... ~15 seconds pass ...

   [Rank 0] Caught timeout: RuntimeError: Timed out waiting 15000ms for recv operation
   [Rank 0] Pickle trace written to /tmp/fr_hang_debug/per_rank/rank_0


Reading the Aggregated Text Dumps
---------------------------------

The debug server writes periodic text snapshots aggregating data from
all ranks:

.. code-block:: bash

   $ ls /tmp/fr_hang_debug/torchcomms_fr_trace_*.txt
   torchcomms_fr_trace_20260401_192058.txt
   torchcomms_fr_trace_20260401_192101.txt
   torchcomms_fr_trace_20260401_192104.txt
   ...

Open one of the snapshots written during the hang:

.. code-block:: bash

   cat /tmp/fr_hang_debug/torchcomms_fr_trace_20260401_192104.txt

The **Collectives** table shows every recorded operation:

::

   --- Collectives ---
     id  group_id    pass_check  collective_seq_id  collective_name    collective_state  missing_ranks
      0  main_comm   True        0                  nccl:all_reduce    scheduled
      1  main_comm   True        1                  nccl:all_reduce    scheduled
      2  main_comm   True        2                  nccl:all_reduce    scheduled
      3  main_comm   True        3                  nccl:broadcast     scheduled
      4  main_comm   True        4                  nccl:all_reduce    scheduled         {1}    ← MISMATCH

The **NCCL Calls** table shows which ranks participated:

::

   --- NCCL Calls ---
     id  collective_id  group_id   global_rank  collective_type
      0              0  main_comm            0  nccl:all_reduce
      1              0  main_comm            1  nccl:all_reduce
      ...
      6              3  main_comm            0  nccl:broadcast
      7              3  main_comm            1  nccl:broadcast
      8                 main_comm            0  nccl:all_reduce   ← Only rank 0!

The **Dump File** section confirms per-rank pickle files were written:

::

   === TorchComms FR Dump File ===
   Rank 0: OK - Flight Recorder debug info written to /tmp/fr_hang_debug/per_rank/rank_0
   Rank 1: OK - Flight Recorder debug info written to /tmp/fr_hang_debug/per_rank/rank_1

The ``stacks_*.txt`` files show Python tracebacks, pinpointing the
exact line each rank is stuck at:

.. code-block:: bash

   $ cat /tmp/fr_hang_debug/stacks_20260401_192104.txt

   === Rank 0 ===
     File "verify_flight_recorder.py", line 148 in main    ← all_reduce (waiting)

   === Rank 1 ===
     File "verify_flight_recorder.py", line 140 in main    ← time.sleep (the hang!)

Rank 1 never issued ``collective_seq_id=4``. The stacks dump confirms
it is stuck in ``time.sleep``, not in a collective.


Running the FR CLI on Per-Rank Pickle Dumps
-------------------------------------------

The periodic dump also triggers each rank's worker server to write a
pickle trace file into the ``per_rank/`` subdirectory:

.. code-block:: bash

   $ ls /tmp/fr_hang_debug/per_rank/
   rank_0  rank_1

Cross-rank mismatch analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m torch.distributed.flight_recorder.fr_trace \
     /tmp/fr_hang_debug/per_rank -p rank_

Output:

::

   Not all ranks joining collective, sequence number: 4
   internal record id: 4
   group info: main_comm:gloo
   collective: nccl:all_reduce
   missing ranks: {1}
   input sizes: [[1024]]
   output sizes: [[1024]]
   world size: 2
   expected ranks: {0, 1}
   collective state: scheduled

The CLI detected that rank 1 never issued ``collective_seq_id=4``.

Side-by-side raw entry view
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m torch.distributed.flight_recorder.fr_trace \
     /tmp/fr_hang_debug/per_rank -p rank_ -j

Output:

::

   Rank 0                                             Rank 1
   -------------------------------------------------  -------------------------------------------------
   all_reduce(input_sizes=[[1024]], state=scheduled)   all_reduce(input_sizes=[[1024]], state=scheduled)
   all_reduce(input_sizes=[[1024]], state=scheduled)   all_reduce(input_sizes=[[1024]], state=scheduled)
   all_reduce(input_sizes=[[1024]], state=scheduled)   all_reduce(input_sizes=[[1024]], state=scheduled)
   broadcast(input_sizes=[[1024]], state=scheduled)    broadcast(input_sizes=[[1024]], state=scheduled)
   all_reduce(input_sizes=[[1024]], state=scheduled)

Rank 0 has 5 entries (3 ``all_reduce`` + 1 ``broadcast`` + the stuck
``all_reduce``). Rank 1 has only 4 — the 5th ``all_reduce`` is missing
because rank 1 hung before issuing it.

With stack traces
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m torch.distributed.flight_recorder.fr_trace \
     /tmp/fr_hang_debug/per_rank -p rank_ -j --print_stack_trace

This adds Python stack traces to each entry, showing exactly where in
user code each collective was called.


What to Look For
----------------

.. list-table::
   :header-rows: 1

   * - Symptom
     - Likely cause
   * - ``missing_ranks: {N}`` in the Collectives table
     - Rank N hung or crashed before issuing the next collective
   * - Rank X's last entry is ``state=started``, others are ``completed``
     - Rank X issued the collective but is waiting for a peer that never joined
   * - Mismatched ``collective_name`` at the same ``collective_seq_id``
     - Code-path divergence — ranks are calling different collectives
   * - Mismatched ``input_sizes`` / ``output_sizes``
     - Tensor shape inconsistency across ranks
   * - Stacks dump shows ``time.sleep`` or user code (not a collective)
     - The rank is stuck in compute, not in a collective


FR CLI Quick Reference
----------------------

.. code-block:: bash

   # Cross-rank mismatch analysis:
   python -m torch.distributed.flight_recorder.fr_trace <dir> -p <prefix>

   # Side-by-side raw entries per rank:
   python -m torch.distributed.flight_recorder.fr_trace <dir> -p <prefix> -j

   # With stack traces:
   python -m torch.distributed.flight_recorder.fr_trace <dir> -p <prefix> -j --print_stack_trace

   # Best-effort when some rank dumps are missing:
   python -m torch.distributed.flight_recorder.fr_trace <dir> -p <prefix> --allow-incomplete-ranks


Conclusion
----------

In this tutorial, you have learned how to use the TorchComms Flight
Recorder and the Debug Server to diagnose a single-rank hang in a
distributed PyTorch job. By examining the aggregated text dumps, per-rank
pickle traces, and stack trace snapshots, you identified which collective
was stuck, which rank failed to participate, and the exact line of code
responsible for the hang. You can apply this same workflow to debug real-world
distributed training hangs — replace the simulated ``time.sleep`` with
whatever your job is actually stuck on, and the Flight Recorder will show
you where ranks diverged.

See Also
^^^^^^^^

* `TorchComms Documentation <https://meta-pytorch.org/torchcomms/main/index.html>`_
* `Flight Recorder Hook API <https://meta-pytorch.org/torchcomms/main/hooks.html#flightrecorderhook>`_
* `torch.distributed Debug HTTP Server <https://pytorch.org/docs/main/distributed.html#torch-distributed-debug-http-server>`_
* :doc:`/unstable/flight_recorder_tutorial` — environment-variable-based Flight Recorder configuration
