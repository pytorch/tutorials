Profiling PyTorch workloads with The Instrumentation and Tracing Technology (ITT) API
=====================================================================================

In this recipe, you will learn:

* An overview of Intel® VTune™ Profiler
* An overview of the Instrumentation and Tracing Technology (ITT) API
* How to visualize PyTorch model hierarchy in Intel® VTune™ Profiler
* A short sample code showcasing how to use PyTorch ITT APIs


Requirements
------------

* PyTorch 1.13+
* Intel® VTune™ Profiler

The instructions for installing PyTorch are available at `pytorch.org <https://pytorch.org/>`_.


Overview of Intel® VTune™ Profiler
----------------------------------

Intel® VTune™ Profiler is a performance analysis tool for serial and 
multithreaded applications. Users can use VTune Profiler to analyze their applications. Intel® VTune™ Profiler helps to identify potential benefits for applications from available hardware resources.

For those who are familiar with Intel Architecture, Intel® VTune™ Profiler provides a rich set of metrics to help users understand how the application executed on Intel platforms, and thus have an idea where the performance bottleneck is.

More detailed information, including getting started guide, are available `here <https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html>`_.

Overview of the Instrumentation and Tracing Technology (ITT) API
----------------------------------------------------------------

`The Instrumentation and Tracing Technology API (ITT API) <https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/api-support/instrumentation-and-tracing-technology-apis.html>`_ provided by the Intel® VTune™ Profiler enables target application to generate and control the collection of trace data during its execution.

ITT API has the following features:

* Controls application performance overhead based on the amount of traces that you collect.
* Enables trace collection without recompiling your application.
* Supports applications in C/C++ and Fortran environments on Windows*, Linux*, FreeBSD*, or Android* systems.
* Supports instrumentation for tracing application code.

The advantage of ITT feature is to label time span of individual PyTorch operators, as well as customized regions, on Intel® VTune™ Profiler GUI. When users find anything abnormal, it will be very helptful to locate which operator behaved unexpected.

**Note:** The ITT API had been integrated into PyTorch since 1.13. Users don't need to invoke the original ITT C/C++ APIs, but only need to invoke the Python APIs in PyTorch. More detailed information can be found at `PyTorch Docs <https://pytorch.org/docs/stable/profiler.html#intel-instrumentation-and-tracing-technology-apis>`_.

How to visualize PyTorch model hierarchy in Intel® VTune™ Profiler
------------------------------------------------------------------

Two types of usage are provided in PyTorch:

1. Implicit invocation: By default, all operators registered following PyTorch operator registration mechanism will be labled by ITT feature automatically when is feature is enabled.

2. Explicit invocation: If customized labeling is needed, users can use APIs mentioned at `PyTorch Docs <https://pytorch.org/docs/stable/profiler.html#intel-instrumentation-and-tracing-technology-apis>`_ explicitly to label a desired range.


To enable this feature, codes which are expected to be labeled should be invoked under a `torch.autograd.profiler.emit_itt()` scope, as below.

.. code:: python3

   with torch.autograd.profiler.emit_itt():
     <codes...>


To verify functionality, you need to start an Intel® VTune™ Profiler instance. Please check the `Intel® VTune™ Profiler user guide <https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/launch.html>`_ for steps of launch Intel® VTune™ Profiler.

Once you get Intel® VTune™ Profiler GUI launched, you should see a user interface as below.

.. figure:: /_static/img/itt_tutorial/vtune_start.png
   :width: 100%
   :align: center

Three sample results are available in the left side navigation bar under `sample (matrix)` project. If you don't want profiling results appear in this default sample project, you can create a new project via the button `New Project...` under the blue `Configure Analysis...` button. To start a new profiling, click the blue `Configure Analysis...` button to initiate configuration of the profiling.

.. figure:: /_static/img/itt_tutorial/vtune_config.png
   :width: 100%
   :align: center

Right side of the windows is split into 3 parts: `WHERE` (top left), `WHAT` (bottom left), and `HOW` (right). With `WHERE`, you can assign a machine where you want to run the profiling on. With `WHAT`, you can set path of the application that you want to profile. To profile a PyTorch script, it is recommended to wrap all manual steps, including activate a conda environment and setting required environment variable, into a bash script, then profile this bash script. In the screenshot above, we wrapped all steps into the `launch.sh` bash script and profile `bash` with parameter to be `<path_of_launch.sh>`. In the right side `HOW`, you can choose whatever type that you would like to profile. Details can be found at `Intel® VTune™ Profiler user guide <https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance.html>`_.

With a successful profiling with ITT, you can open `Platform` tab of the profiling result to see labels in Intel® VTune™ Profiler timeline. All operators starting with `aten::` are operators labeled implicitly by the ITT feature in PyTorch. Labels `iteration_N` are explicitly labeled with specific APIs `torch.profiler.itt.range_push()`, `torch.profiler.itt.range_pop()` or `torch.profiler.itt.range()` scope. Please check the sample code in next section for details.

.. figure:: /_static/img/itt_tutorial/vtune_timeline.png
   :width: 100%
   :align: center

A short sample code showcasing how to use PyTorch ITT APIs
----------------------------------------------------------

Sample code below is the script that was used for profiling in the screenshots above.

The topology is formed by 2 operators, `Conv2d` and `Linear`. Three iterations of inference were performed. Each iteration was labled by PyTorch ITT APIs as text string `iteration_N`. Either pair of `torch.profile.itt.range_push` and `torch.profile.itt.range_pop` or `torch.profile.itt.range` scope does the customized labeling feature.

.. code:: python3

   # sample.py

   import torch
   import torch.nn as nn
   
   class ITTSample(nn.Module):
     def __init__(self):
       super(ITTSample, self).__init__()
       self.conv = nn.Conv2d(3, 5, 3)
       self.linear = nn.Linear(292820, 1000)
   
     def forward(self, x):
       x = self.conv(x)
       x = x.view(x.shape[0], -1)
       x = self.linear(x)
       return x
   
   def main():
     m = ITTSample()
     x = torch.rand(10, 3, 244, 244)
     with torch.autograd.profiler.emit_itt():
       for i in range(3)
         # Labeling a region with pair of range_push and range_pop
         #torch.profiler.itt.range_push(f'iteration_{i}')
         #m(x)
         #torch.profiler.itt.range_pop()
   
         # Labeling a region with range scope
         with torch.profiler.itt.range(f'iteration_{i}'):
           m(x)
   
   if __name__ == '__main__':
     main()


The `launch.sh` bash script to wrap all manual steps is shown below.

.. code:: bash

   # launch.sh

   #!/bin/bash
   
   # Retrive the directory path where contains both the sample.py and launch.sh so that this script can be invoked from any directory
   BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
   source ~/miniconda3/bin/activate
   conda activate ipex_py38
   cd ${BASEFOLDER}
   python sample.py
