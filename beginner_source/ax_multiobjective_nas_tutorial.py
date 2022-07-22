# -*- coding: utf-8 -*-
"""
Multi-Objective NAS with Ax
==================================================

**Authors:** `David Eriksson <https://github.com/dme65>`__,
`Max Balandat <https://github.com/Balandat>`__,
and the Adaptive Experimentation team at Meta.

In this tutorial we show how to use Ax to run multi-objective neural
architecture search for a simple neural network model on the popular
MNIST dataset.

While the underlying methodology can be used for more complicated models
and larger datasets, we opt for a tutorial that is easily runnable
end-to-end on a laptop in less than an hour.

In our example, we will tune the widths of two hidden layers, the
learning rate, the dropout probability, the batch size, and the
number of training epochs. The goal is to trade off performance
(accuracy on the validation set) and model size (the number of
model parameters).

This tutorial makes use of the following PyTorch libraries:

- `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`__ (specifying the model and training loop)
- `TorchX <https://github.com/pytorch/torchx>`__ (for running training jobs remotely / asynchronously)
- `BoTorch <https://github.com/pytorch/botorch>`__ (the Bayesian Optimization library powering Ax's algorithms)

"""


######################################################################
# Defining the TorchX App
# -----------------------
#
# Our goal is to optimize the PyTorch Lightning training job defined
# in `mnist_train_nas.py <https://github.com/pytorch/tutorials/tree/master/beginner_source/mnist_train_nas.py`__.
# To do this using TorchX, we write a helper function that takes in
# the values of the architcture and hyperparameters of the training
# job and creates a TorchX AppDef (see the
# `TorchX docs <https://pytorch.org/torchx/latest/basics.html>`__
# for more detail) with the appropriate settings.
#


from pathlib import Path

from torchx import specs
from torchx.version import TORCHX_IMAGE


def trainer(
    log_path: str,
    hidden_size_1: int,
    hidden_size_2: int,
    learning_rate: float,
    epochs: int,
    dropout: float,
    batch_size: int,
    trial_idx: int = -1,
) -> specs.AppDef:

    # load in the training code
    train_source = Path(__file__).parent.joinpath("mnist_train_nas.py")
    with train_source.open("rt") as f:
        source = f.read()

    # define the log path so we can pass it to the TorchX AppDef
    if trial_idx >= 0:
        log_path = Path(log_path).joinpath(str(trial_idx)).absolute().as_posix()

    return specs.AppDef(
        name="trainer",
        roles=[
            specs.Role(
                name="trainer",
                image=TORCHX_IMAGE,
                entrypoint="python",
                args=[
                    "-c",
                    source,
                    "--log_path",
                    log_path,
                    "--hidden_size_1",
                    str(hidden_size_1),
                    "--hidden_size_2",
                    str(hidden_size_2),
                    "--learning_rate",
                    str(learning_rate),
                    "--epochs",
                    str(epochs),
                    "--dropout",
                    str(dropout),
                    "--batch_size",
                    str(batch_size),
                ],
            )
        ],
    )


######################################################################
# Setting up the Runner
# ---------------------
#
# Ax’s Runner abstraction allows writing interfaces to various backends.
# Ax already comes with Runner for TorchX, and so we just need to
# configure it. For the purpose of this tutorial we run jobs locally
# in a fully asynchronous fashion.
#
# In order to launch them on a cluster, you can instead specify a
# different TorchX scheduler and adjust the configuration appropriately
# (e.g. if you have a Kubernetes cluster,  you just need to change the
# scheduler from local_cwd to kubernetes).
#


import tempfile
from ax.runners.torchx import TorchXRunner

# Make a temporary dir to log our results into
log_dir = tempfile.mkdtemp()

ax_runner = TorchXRunner(
    tracker_base="/tmp/",
    component=trainer,
    # NOTE: To launch this job on a cluster instead of locally you can
    # specify a different scheduler and adjust args appropriately.
    scheduler="local_cwd",
    component_const_params={"log_path": log_dir},
    cfg={},
)

######################################################################
# Setting up the SearchSpace
# --------------------------
#
# First, we define our search space. Ax supports both range parameters
# of type integer and float as well as choice parameters which can have
# non-numerical types such as strings.
#
# We will tune the hidden sizes, learning rate, dropout, and number of
# epochs as range parameters and tune the batch size as an ordered choice
# parameter to enforce it to be a power of 2.
#

from ax.core import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    SearchSpace,
)

parameters = [
    # NOTE: hidden_size_1 and hidden_size_2 should probably be powers of 2,
    # but in that case num_params can't take on that many values which makes the
    # Pareto frontier look pretty weird. This seems fine for a tutorial.
    RangeParameter(
        name="hidden_size_1",
        lower=16,
        upper=128,
        parameter_type=ParameterType.INT,
        log_scale=True,
    ),
    RangeParameter(
        name="hidden_size_2",
        lower=16,
        upper=128,
        parameter_type=ParameterType.INT,
        log_scale=True,
    ),
    RangeParameter(
        name="learning_rate",
        lower=1e-4,
        upper=1e-2,
        parameter_type=ParameterType.FLOAT,
        log_scale=True,
    ),
    RangeParameter(
        name="epochs",
        lower=1,
        upper=4,
        parameter_type=ParameterType.INT,
    ),
    RangeParameter(
        name="dropout",
        lower=0.0,
        upper=0.5,
        parameter_type=ParameterType.FLOAT,
    ),
    ChoiceParameter(  # NOTE: ChoiceParameters don't require log-scale
        name="batch_size",
        values=[32, 64, 128, 256],
        parameter_type=ParameterType.INT,
        is_ordered=True,
        sort_values=True,
    ),
]

search_space = SearchSpace(
    parameters=parameters,
    parameter_constraints=[],  # It may make sense to add a constraint hidden_size_2 <= hidden_size_1
)


######################################################################
# Setting up Metrics
# ------------------
#
# Ax has the concept of a Metric that defines properties of outcomes
# and how observations are obtained for these outcomes. This allows
# e.g. encodig how data is fetched from some distributed execution
# backend and post-processed before being passed as input to Ax.
#
# In this tutorial we will use multi-objective optimization with the
# goal of maximizing the validation accuracy and minimizing the number
# of model parameters. The latter represents a simple proxy of model
# latency, which is hard to estimate accurately for small ML models
# (in an actual application we would benchmark the latency while
# running the model on-device).
#
# In our example TorchX will run the training jobs in a fully async
# fashion locally and write the results to disk based on the trial
# index (see trainer() function above). We'll define a metric class
# that is a ware of that logging directory. By subclassing
# `TensorboardCurveMetric` we get the logic to read and parse the
# tensorboard logs for free.
#

from ax.metrics.tensorboard import TensorboardCurveMetric


class MyTensorboardMetric(TensorboardCurveMetric):

    # We need to tell the new Tensorboard metric how to get the id /
    # file handle for the tensorboard logs from a trial. In this case
    # our convention is to just save a separate file per trial in
    # the pre-specified log dir.
    @classmethod
    def get_ids_from_trials(cls, trials):
        return {
            trial.index: Path(log_dir).joinpath(str(trial.index)).as_posix()
            for trial in trials
        }

    # This indicates whether the metric is queryable while the trial is
    # still running. We don't use this in the current tutorial, but Ax
    # utilizes this to implement early-stopping capabilities.
    @classmethod
    def is_available_while_running(cls):
        return False


######################################################################
# Now we can instatiate the Metrics for accuracy and the number of
# model parameters (TODO: Show signature of TensorboardCurveMetric
# constructor). We specify `lower_is_better` to indicate the
# favorable direction of a metric.
#

val_acc = MyTensorboardMetric("val_acc", "val_acc", lower_is_better=False)
model_num_params = MyTensorboardMetric("num_params", "num_params", lower_is_better=True)


######################################################################
# Setting up the OptimizationConfig
# ----------------------------------
#
# The way to tell Ax what it should optimize is by means of an
# `OptimizatioConfig`. Here we use a `MultiObjectiveOptimizationConfig`
# as we will be performing multi-objective optimization.
#
# Additionally, Ax supports placing constraints on the different
# metrics by specifying objective thresholds, which bound the region
# of interest in the outcome space that we want to explore. For this
# ecample we will constrain the validation accuracy to be at least
# 0.94 (94%) and the number of model parameters to be at most 80,000.
#

from ax.core import MultiObjective, Objective, ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig


opt_config = MultiObjectiveOptimizationConfig(
    objective=MultiObjective(
        objectives=[
            Objective(metric=val_acc, minimize=False),
            Objective(metric=model_num_params, minimize=True),
        ],
    ),
    objective_thresholds=[
        ObjectiveThreshold(metric=val_acc, bound=0.94, relative=False),
        ObjectiveThreshold(metric=model_num_params, bound=80_000, relative=False),
    ],
)


######################################################################
# Creating the Ax Experiment
# --------------------------
#
# In Ax, the `Experiment` object is the object that stores all the
# information about the problem setup (`Experiment`s can be serialized
# to JSON or stored to a databse backend such as MySQL in order to
# persist and be available to load on different machines - we won't
# be using this functionality in this tutorial though).
#

from ax.core import Experiment

experiment = Experiment(
    name="torchx_mnist",
    search_space=search_space,
    optimization_config=opt_config,
    runner=ax_runner,
)

######################################################################
# Choosing the GenerationStrategy
# -------------------------------
#
# A `GenerationStrategy` is the abstract representation of how we
# would like to perform the optimization. While this can be customized
# (if you’d like to do so, see
# `this tutorial <https://ax.dev/tutorials/generation_str ategy.html>`___),
# in most cases Ax can automatically determine an appropriate strategy
# based on the search space, optimization config, and the total number
# of trials we want to run.
#
# Typically, Ax will choose to evauate a number of random configurations
# before starting a model-based Bayesian Optimization strategy.
#


total_trials = 48  # total evaluation budget

from ax.modelbridge.dispatch_utils import choose_generation_strategy

gs = choose_generation_strategy(
    search_space=experiment.search_space,
    optimization_config=experiment.optimization_config,
    num_trials=total_trials,
  )


######################################################################
# Configuring the Scheduler
# -------------------------
#
# The `Scheduler` acts as the loop control for the optimization.
# It communicates with the backend to launch trials, check their status,
# and retrieve results (in the case of this tutorial it is simply reading
# and parsing the locally saved logs, bug in a remote execution setting
# it would call APIs).
#
# The following illustration from the Ax Scheduler tutorial summarizes how the
# Scheduler interacts with any external system used to run trial evaluations:
#
# .. image:: ../../_static/img/ax_scheduler_illustration.png
#
#
# The Scheduler requires the `Experiment` and the `GenerationStrategy`.
# A set of options can be passed in via `SchedulerOptions`. Here, we
# configure the number of total evaluations as well as `max_pending_trials`,
# the maximum number of trials that should run concurrently (in our
# local setting this is the number of training jobs running as individual
# processes, while in a remote execution setting this would be the number
# of machines you want to use in parallel).
#


from ax.service.scheduler import Scheduler, SchedulerOptions

scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=gs,
    options=SchedulerOptions(
        total_trials=total_trials, max_pending_trials=4
    ),
)


######################################################################
# Running the optimization
# ------------------------
#
# Everything is configured, we can now let Ax run the optimization
# in a fully automated fashion. The Scheduler will periodially check
# the logs for the status of all currenty running trials, and if a
# trial completes it will update its status on the experiment and
# fetch the observations needed for the Bayesian Optimization
# algorithm.
#

scheduler.run_all_trials()


######################################################################
# Evaluating the results
# ----------------------
#
# We can now inspect the result of the optimization using helper
# functions and visualizations integrated with Ax.

######################################################################
# First, we generate a dataframe with a summary of the results
# of the experiment.
# TODO: What do we see?
#

from ax.service.utils.report_utils import exp_to_df

df = exp_to_df(experiment)
print(df)


######################################################################
# We can also visualize the Pareto frontier of tradeoffs between the
# validation accuracy and the number of model parameters.
#
# The final results can be seen in the Figure below where the color
# corresponds to the iteration number for each trial. We see that our
# method was able to successfully explore the trade-offs  and found
# both large models with high validation accuracy as well as small
# models with low validation accuracy.
#

from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly
from ax.utils.notebook.plotting import init_notebook_plotting, render

fig = _pareto_frontier_scatter_2d_plotly(experiment)
init_notebook_plotting()
render(fig)


######################################################################
# To better understand what our surrogate models have learned about
# the black box objectives, let's take a look at the LOO cross
# validation results. Since our models are Gaussian Processes, they
# not only provide point predictions but also uncertainty estimates
# about these predictions.
#
# Ax makes it easy to do leave-one-out cross-validation. As the figures
# below show, the model size (num_params) metric is much easier to model
# than the validation accuracy (val_acc) metric.
#

from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate
from ax.plot.diagnostic import interact_cross_validation

# The surrogate model is stored on the GenerationStrategy
cv = cross_validate(gs.model)
diagnostics = compute_diagnostics(cv)
render(interact_cross_validation(cv))


# TODO: Contour plots?


# Suggest removing SAASBO below - seems somewhat orthogonal to the tutorial
# and could confuse the reader more than it may help.

# ######################################################################
# # We can also see how a SAASBO (TODO: link) model compares against our
# # standard models;
# #
# # TODO: What do we see?
# #

# from ax.modelbridge.registry import Cont_X_trans, Models, Y_trans
# from ax.modelbridge.transforms.winsorize import Winsorize
# from ax.models.torch.botorch_moo_defaults import get_NEHVI

# model_kwargs = {
#     "objective_thresholds": opt_config.objective_thresholds,
#     "acqf_constructor": get_NEHVI,
#     "transforms": [Winsorize] + Y_trans + Cont_X_trans,
#     "transform_configs": {
#         "Winsorize": {
#             "optimization_config": opt_config
#         },
#     },
# }

# model_saas = Models.FULLYBAYESIANMOO(
#     experiment=experiment,
#     data=experiment.fetch_data(),
#     use_saas=True,
#     num_samples=256,
#     warmup_steps=512,
#     gp_kernel="matern",
#     disable_progbar=False,
#     verbose=False,
#     **model_kwargs,
# )

# cv = cross_validate(model_saas)
# diagnostics = compute_diagnostics(cv)
# render(interact_cross_validation(cv))


######################################################################
# Learn more
# ----------
#
# TODO: Pointers to stuff
#


######################################################################
# Acknowledgements
# ----------------
#
# We thank the TorchX team (in particular Kiuk Chung and Tristan Rice)
# for their help with integrating TorchX with Ax.
#
