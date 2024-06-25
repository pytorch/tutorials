Compile Time Caching in ``torch.compile``
=========================================================
**Authors:** `Oguz Ulgen <https://github.com/oulgen>`_ and `Sam Larsen <https://github.com/masnesral>`_

Introduction
------------------

PyTorch Inductor implements several caches to reduce compilation latency.
This recipe demonstrates how you can configure various parts of the caching in ``torch.compile``.

Prerequisites
-------------------

Before starting this recipe, make sure that you have the following:

* Basic understanding of ``torch.compile``. See:

  * `torch.compiler API documentation <https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler>`__
  * `Introduction to torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__

* PyTorch 2.4 or later

Inductor Cache Settings
----------------------------

Most of these caches are in-memory, only used within the same process, and are transparent to the user. An exception is the FX graph cache that stores compiled FX graphs. This cache allows Inductor to avoid recompilation across process boundaries when it encounters the same graph with the same Tensor input shapes (and the same configuration). The default implementation stores compiled artifacts in the system temp directory. An optional feature also supports sharing those artifacts within a cluster by storing them in a Redis database.

There are a few settings relevant to caching and to FX graph caching in particular.
The settings are accessible via environment variables listed below or can be hard-coded in Inductor’s config file.

TORCHINDUCTOR_FX_GRAPH_CACHE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This setting enables the local FX graph cache feature, i.e., by storing artifacts in the host’s temp directory. ``1`` enables, and any other value disables it. By default, the disk location is per username, but users can enable sharing across usernames by specifying ``TORCHINDUCTOR_CACHE_DIR`` (below).

TORCHINDUCTOR_CACHE_DIR
~~~~~~~~~~~~~~~~~~~~~~~~
This setting specifies the location of all on-disk caches. By default, the location is in the system temp directory under ``torchinductor_<username>``, for example, ``/tmp/torchinductor_myusername``.

Note that if ``TRITON_CACHE_DIR`` is not set in the environment, Inductor sets the Triton cache directory to this same temp location, under the Triton subdirectory.

TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This setting enables the remote FX graph cache feature. The current implementation uses Redis. ``1`` enables caching, and any other value disables it. The following environment variables configure the host and port of the Redis server:

``TORCHINDUCTOR_REDIS_HOST`` (defaults to ``localhost``)
``TORCHINDUCTOR_REDIS_PORT`` (defaults to ``6379``)

Note that if Inductor locates a remote cache entry, it stores the compiled artifact in the local on-disk cache; that local artifact would be served on subsequent runs on the same machine.

TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This setting enables a remote cache for Inductor’s autotuner. As with the remote FX graph cache, the current implementation uses Redis. ``1`` enables caching, and any other value disables it. The same host / port environment variables listed above apply to this cache.

TORCHINDUCTOR_FORCE_DISABLE_CACHES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set this value to ``1`` to disable all Inductor caching. This setting is useful for tasks like experimenting with cold-start compile times or forcing recompilation for debugging purposes.

Conclusion
-------------
In this recipe, we have learned that PyTorch Inductor's caching mechanisms significantly reduce compilation latency by utilizing both local and remote caches, which operate seamlessly in the background without requiring user intervention.
Additionally, we explored the various settings and environment variables that allow users to configure and optimize these caching features according to their specific needs.
