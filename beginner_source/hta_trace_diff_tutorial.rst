Trace Diff using Holistic Trace Analysis
========================================

**Author:** `Anupam Bhatnagar <https://github.com/anupambhatnagar>`_

Occasionally, users need to identify the changes in PyTorch operators and CUDA
kernels resulting from a code change. To support this requirement, HTA
provides a trace comparison feature. This feature allows the user to input two
sets of trace files where the first can be thought of as the *control group*
and the second as the *test group*, similar to an A/B test. The ``TraceDiff`` class
provides functions to compare the differences between traces and functionality
to visualize these differences. In particular, users can find operators and
kernels that were added and removed from each group, along with the frequency
of each operator/kernel and the cumulative time taken by the operator/kernel.

The `TraceDiff <https://hta.readthedocs.io/en/latest/source/api/trace_diff_api.html>`_ class 
has the following methods:

* `compare_traces <https://hta.readthedocs.io/en/latest/source/api/trace_diff_api.html#hta.trace_diff.TraceDiff.compare_traces>`_:
  Compare the frequency and total duration of CPU operators and GPU kernels from
  two sets of traces.

* `ops_diff <https://hta.readthedocs.io/en/latest/source/api/trace_diff_api.html#hta.trace_diff.TraceDiff.ops_diff>`_:
  Get the operators and kernels which have been:

    #. **added** to the test trace and are absent in the control trace
    #. **deleted** from the test trace and are present in the control trace
    #. **increased** in frequency in the test trace and exist in the control trace
    #. **decreased** in frequency in the test trace and exist in the control trace
    #. **unchanged** between the two sets of traces

* `visualize_counts_diff <https://hta.readthedocs.io/en/latest/source/api/trace_diff_api.html#hta.trace_diff.TraceDiff.visualize_counts_diff>`_

* `visualize_duration_diff <https://hta.readthedocs.io/en/latest/source/api/trace_diff_api.html#hta.trace_diff.TraceDiff.visualize_duration_diff>`_

The last two methods can be used to visualize various changes in frequency and
duration of CPU operators and GPU kernels, using the output of the
``compare_traces`` method.

For example, the top ten operators with increase in frequency can be computed as
follows:

.. code-block:: python

    df = compare_traces_output.sort_values(by="diff_counts", ascending=False).head(10)
    TraceDiff.visualize_counts_diff(df)

.. image:: ../_static/img/hta/counts_diff.png

Similarly, the top ten operators with the largest change in duration can be computed as
follows:

.. code-block:: python

    df = compare_traces_output.sort_values(by="diff_duration", ascending=False)
    # The duration differerence can be overshadowed by the "ProfilerStep",
    # so we can filter it out to show the trend of other operators.
    df = df.loc[~df.index.str.startswith("ProfilerStep")].head(10)
    TraceDiff.visualize_duration_diff(df)

.. image:: ../_static/img/hta/duration_diff.png

For a detailed example of this feature see the `trace_diff_demo notebook
<https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/examples/trace_diff_demo.ipynb>`_
in the examples folder of the repository.

