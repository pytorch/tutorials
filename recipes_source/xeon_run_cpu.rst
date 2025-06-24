Optimizing CPU Performance on Intel® Xeon® with run_cpu Script
======================================================================

There are several configuration options that can impact the performance of PyTorch inference when executed on Intel® Xeon® Scalable Processors.
To get peak performance, the ``torch.backends.xeon.run_cpu`` script is provided that optimizes the configuration of thread and memory management.
For thread management, the script configures thread affinity and the preload of Intel® OMP library.
For memory management, it configures NUMA binding and preloads optimized memory allocation libraries, such as TCMalloc and JeMalloc.
In addition, the script provides tunable parameters for compute resource allocation in both single instance and multiple instance scenarios,
helping the users try out an optimal coordination of resource utilization for the specific workloads.

What You Will Learn
-------------------

* How to utilize tools like ``numactl``, ``taskset``, Intel® OpenMP Runtime Library and optimized memory
  allocators such as ``TCMalloc`` and ``JeMalloc`` for enhanced performance.
* How to configure CPU resources and memory management to maximize PyTorch inference performance on Intel® Xeon® processors.

Introduction of the Optimizations
---------------------------------

Applying NUMA Access Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is beneficial that an increasing number of CPU cores are being provided to users within a single socket, as this offers greater computational resources.
However, this also leads to competition for memory access, which can cause programs to stall due to busy memory.
To address this problem, Non-Uniform Memory Access (NUMA) was introduced.
Unlike Uniform Memory Access (UMA), where all memories are equally accessible to all cores,
NUMA organizes memory into multiple groups. Certain number of memories are directly attached to one socket's integrated memory controller to become local memory of this socket.
Local memory access is much faster than remote memory access.

Users can get CPU information with ``lscpu`` command on Linux to learn how many cores and sockets are there on the machine.
Additionally, this command provides NUMA information, such as the distribution of CPU cores.
Below is an example of executing  ``lscpu`` on a machine equipped with an Intel® Xeon® CPU Max 9480:

.. code-block:: console

   $ lscpu
   ...
   CPU(s):                  224
     On-line CPU(s) list:   0-223
   Vendor ID:               GenuineIntel
     Model name:            Intel (R) Xeon (R) CPU Max 9480
       CPU family:          6
       Model:               143
       Thread(s) per core:  2
       Core(s) per socket:  56
       Socket(s):           2
   ...
   NUMA:
     NUMA node(s):          2
     NUMA node0 CPU(s):     0-55,112-167
     NUMA node1 CPU(s):     56-111,168-223
   ...

* Two sockets were detected, each containing 56 physical cores. With Hyper-Threading enabled, each core can handle 2 threads, resulting in 56 logical cores per socket. Therefore, the machine has a total of 224 CPU cores in service.
* Typically, physical cores are indexed before logical cores. In this scenario, cores 0-55 are the physical cores on the first NUMA node, and cores 56-111 are the physical cores on the second NUMA node.
* Logical cores are indexed subsequently: cores 112-167 correspond to the logical cores on the first NUMA node, and cores 168-223 to those on the second NUMA node.

Typically, running PyTorch programs with compute intense workloads should avoid using logical cores to get good performance.

Linux provides a tool called ``numactl`` that allows user control of NUMA policy for processes or shared memory.
It runs processes with a specific NUMA scheduling or memory placement policy.
As described above, cores share high-speed cache in one socket, thus it is a good idea to avoid cross socket computations.
From a memory access perspective, bounding memory access locally is much faster than accessing remote memories.
``numactl`` command should have been installed in recent Linux distributions. In case it is missing, you can install it manually with the installation command, like on Ubuntu:

.. code-block:: console

   $ apt-get install numactl

on CentOS you can run the following command:

.. code-block:: console

   $ yum install numactl

The ``taskset`` command in Linux is another powerful utility that allows you to set or retrieve the CPU affinity of a running process.
``taskset`` are pre-installed in most Linux distributions and in case it's not, on Ubuntu you can install it with the command:

.. code-block:: console

   $ apt-get install util-linux

on CentOS you can run the following command:

.. code-block:: console

   $ yum install util-linux

Using Intel® OpenMP Runtime Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMP is an implementation of multithreading, a method of parallelizing where a primary thread (a series of instructions executed consecutively) forks a specified number of sub-threads and the system divides a task among them. The threads then run concurrently, with the runtime environment allocating threads to different processors.
Users can control OpenMP behaviors with some environment variable settings to fit for their workloads, the settings are read and executed by OMP libraries. By default, PyTorch uses GNU OpenMP Library (GNU libgomp) for parallel computation. On Intel® platforms, Intel® OpenMP Runtime Library (libiomp) provides OpenMP API specification support. It usually brings more performance benefits compared to libgomp.

The Intel® OpenMP Runtime Library can be installed using one of these commands:

.. code-block:: console

   $ pip install intel-openmp

or

.. code-block:: console

   $ conda install mkl

Choosing an Optimized Memory Allocator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory allocator plays an important role from performance perspective as well. A more efficient memory usage reduces overhead on unnecessary memory allocations or destructions, and thus results in a faster execution. From practical experiences, for deep learning workloads, ``TCMalloc`` or ``JeMalloc`` can get better performance by reusing memory as much as possible than default malloc operations.

You can install ``TCMalloc`` by running the following command on Ubuntu:

.. code-block:: console

   $ apt-get install google-perftools

On CentOS, you can install it by running:

.. code-block:: console

   $ yum install gperftools

In a conda environment, it can also be installed by running:

.. code-block:: console

   $ conda install conda-forge::gperftools

On Ubuntu ``JeMalloc`` can be installed by this command:

.. code-block:: console

   $ apt-get install libjemalloc2

On CentOS it can be installed by running:

.. code-block:: console

   $ yum install jemalloc

In a conda environment, it can also be installed by running:

.. code-block:: console

   $ conda install conda-forge::jemalloc

Quick Start Example Commands
----------------------------

1. To run single-instance inference with 1 thread on 1 CPU core (only Core #0 would be used):

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu --ninstances 1 --ncores-per-instance 1 <program.py> [program_args]

2. To run single-instance inference on a single CPU node (NUMA socket):

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu --node-id 0 <program.py> [program_args]

3. To run multi-instance inference, 8 instances with 14 cores per instance on a 112-core CPU:

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu --ninstances 8 --ncores-per-instance 14 <program.py> [program_args]

4. To run inference in throughput mode, in which all the cores in each CPU node set up an instance:

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu --throughput-mode <program.py> [program_args]

.. note::

   Term "instance" here doesn't refer to a cloud instance. This script is executed as a single process which invokes multiple "instances" which are formed from multiple threads. "Instance" is kind of group of threads in this context.

Using ``torch.backends.xeon.run_cpu``
-------------------------------------

The argument list and usage guidance can be shown with the following command:

.. code-block:: console

   $ python -m torch.backends.xeon.run_cpu –h
   usage: run_cpu.py [-h] [--multi-instance] [-m] [--no-python] [--enable-tcmalloc] [--enable-jemalloc] [--use-default-allocator] [--disable-iomp] [--ncores-per-instance] [--ninstances] [--skip-cross-node-cores] [--rank] [--latency-mode] [--throughput-mode] [--node-id] [--use-logical-core] [--disable-numactl] [--disable-taskset] [--core-list] [--log-path] [--log-file-prefix] <program> [program_args]

The command above has the following positional arguments:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - knob
     - help
   * - ``program``
     - The full path of the program/script to be launched.
   * - ``program_args``
     - The input arguments for the program/script to be launched.

Explanation of the options
~~~~~~~~~~~~~~~~~~~~~~~~~~

The generic option settings (knobs) include the following:

.. list-table::
   :widths: 25 10 15 50
   :header-rows: 1

   * - knob
     - type
     - default value
     - help
   * - ``-h``, ``--help``
     - 
     - 
     - To show the help message and exit.
   * - ``-m``, ``--module``
     - 
     - 
     - To change each process to interpret the launch script as a python module, executing with the same behavior as "python -m".
   * - ``--no-python``
     - bool
     - False
     - To avoid prepending the program with "python" - just execute it directly. Useful when the script is not a Python script.
   * - ``--log-path``
     - str
     - ``''``
     - To specify the log file directory. Default path is ``''``, which means disable logging to files.
   * - ``--log-file-prefix``
     - str
     - "run"
     - Prefix of the log file name.

Knobs for applying or disabling optimizations are:

.. list-table::
   :widths: 25 10 15 50
   :header-rows: 1

   * - knob
     - type
     - default value
     - help
   * - ``--enable-tcmalloc``
     - bool
     - False
     - To enable ``TCMalloc`` memory allocator.
   * - ``--enable-jemalloc``
     - bool
     - False
     - To enable ``JeMalloc`` memory allocator.
   * - ``--use-default-allocator``
     - bool
     - False
     - To use default memory allocator. Neither ``TCMalloc`` nor ``JeMalloc`` would be used.
   * - ``--disable-iomp``
     - bool
     - False
     - By default, Intel® OpenMP lib will be used if installed. Setting this flag would disable the usage of Intel® OpenMP.

.. note::

   Memory allocators influence performance. If the user does not specify a desired memory allocator, the ``run_cpu`` script will search if any of them is installed in the order of TCMalloc > JeMalloc > PyTorch default memory allocator, and takes the first matched one.

Knobs for controlling instance number and compute resource allocation are:

.. list-table::
   :widths: 25 10 15 50
   :header-rows: 1

   * - knob
     - type
     - default value
     - help
   * - ``--ninstances``
     - int
     - 0
     - Number of instances.
   * - ``--ncores-per-instance``
     - int
     - 0
     - Number of cores used by each instance.
   * - ``--node-id``
     - int
     - -1
     - The node ID to be used for multi-instance, by default all nodes will be used.
   * - ``--core-list``
     - str
     - ``''``
     - To specify the core list as ``'core_id, core_id, ....'`` or core range as ``'core_id-core_id'``. By dafault all the cores will be used.
   * - ``--use-logical-core``
     - bool
     - False
     - By default only physical cores are used. Specifying this flag enables logical cores usage.
   * - ``--skip-cross-node-cores``
     - bool
     - False
     - To prevent the workload to be executed on cores across NUMA nodes.
   * - ``--rank``
     - int
     - -1
     - To specify instance index to assign ncores_per_instance for rank; otherwise ncores_per_instance will be assigned sequentially to the instances.
   * - ``--multi-instance``
     - bool
     - False
     - A quick set to invoke multiple instances of the workload on multi-socket CPU servers.
   * - ``--latency-mode``
     - bool
     - False
     - A quick set to invoke benchmarking with latency mode, in which all physical cores are used and 4 cores per instance.
   * - ``--throughput-mode``
     - bool
     - False
     - A quick set to invoke benchmarking with throughput mode, in which all physical cores are used and 1 numa node per instance.
   * - ``--disable-numactl``
     - bool
     - False
     - By default ``numactl`` command is used to control NUMA access. Setting this flag will disable it.
   * - ``--disable-taskset``
     - bool
     - False
     - To disable the usage of ``taskset`` command.
	 
.. note::

   Environment variables that will be set by this script include the following:

   .. list-table::
      :widths: 25 50
      :header-rows: 1

      * - Environment Variable
        - Value
      * - LD_PRELOAD
        - Depending on knobs you set, <lib>/libiomp5.so, <lib>/libjemalloc.so, <lib>/libtcmalloc.so might be appended to LD_PRELOAD.
      * - KMP_AFFINITY
        - If libiomp5.so is preloaded, KMP_AFFINITY could be set to ``"granularity=fine,compact,1,0"``.
      * - KMP_BLOCKTIME
        - If libiomp5.so is preloaded, KMP_BLOCKTIME is set to "1".
      * - OMP_NUM_THREADS
        - Value of ``ncores_per_instance``
      * - MALLOC_CONF
        - If libjemalloc.so is preloaded, MALLOC_CONF will be set to ``"oversize_threshold:1,background_thread:true,metadata_thp:auto"``.
		
   Please note that the script respects environment variables set preliminarily. For example, if you have set the environment variables mentioned above before running the script, the values of the variables will not be overwritten by the script.

Conclusion
----------

In this tutorial, we explored a variety of advanced configurations and tools designed to optimize PyTorch inference performance on Intel® Xeon® Scalable Processors. 
By leveraging the ``torch.backends.xeon.run_cpu`` script, we demonstrated how to fine-tune thread and memory management to achieve peak performance.
We covered essential concepts such as NUMA access control, optimized memory allocators like ``TCMalloc`` and ``JeMalloc``, and the use of Intel® OpenMP for efficient multithreading.

Additionally, we provided practical command-line examples to guide you through setting up single and multiple instance scenarios, ensuring optimal resource utilization tailored to specific workloads.
By understanding and applying these techniques, users can significantly enhance the efficiency and speed of their PyTorch applications on Intel® Xeon® platforms.

See also:

* `PyTorch Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#cpu-specific-optimizations>`__
* `PyTorch Multiprocessing Best Practices <https://pytorch.org/docs/stable/notes/multiprocessing.html#cpu-in-multiprocessing>`__
