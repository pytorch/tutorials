"""
Introducing ``torch.export``
========================

This tutorial introduces steps to export a PyTorch program using the PyTorch 2.0 features.

The following prerequisites must be met to get an exported
program that satisfies the users’ requirements:

* Users should provide example inputs and have an good understanding of the
shape dynamism of all inputs, including whether they are dynamic or not, as well
as which dimensions are dynamic.

* Users should have the ability to modify all of the code utilized in the
model, including both 1st and 3rd-party libraries.
"""

from typing import Callable

import torch
from functorch.experimental.control_flow import cond
from torch._export import dynamic_dim, export, ExportedProgram
from torch._export.constraints import constrain_as_size

# Util to check soundness
def check_soundness(f: Callable, exported_program: ExportedProgram, *inputs):
    try:
        print(
            "Sound!"
            if torch.equal(f(*inputs), exported_program(*inputs))
            else "Not Sound!"
        )
    except Exception as e:
        print("Raise runtime error!")
        raise e


######################################################################
# Demo Program
# --------------------
#

# We will use the following demo program to demonstrate how ``torch_export`` works:

def user_fn(x):
    if x.shape[0] > 5 and x.shape[0] < 10:
        return x + 10
    else:
        return x * 2


######################################################################
# 1. Constraint Discovery
# -----------------------
#
# Given a program and rough dynamism requirements, how can we fine-tune the
# input constraints for exportability?
#
# - The compiler will discover reasonable input constraints based on user’s specifications
# - The discovered constraints are user consumeable directly
#


######################################################################
# Case 1.1. User provides a constraint
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# By default, all dimensions are assumed to be static during tracing
# unless one is explicitly specified as dynamic using ``dynamic_dim()``.
# So there is no constraint discovery.
#
# All dimensions are exported statically
# Expect to see the exported program is specialized at dim 0.
t_1 = torch.zeros([8])
exported = export(user_fn, (t_1,), constraints=None, _add_runtime_assertions=False)
print(exported)

# Feed an error input to the exported program, the result is unsound!
error_input = torch.randn([100])
check_soundness(user_fn, exported, error_input)

# Turn constraints into runtime assertions
sound_exported = export(user_fn, (t_1,), constraints=None)
print(sound_exported)

# If input.shape[0] != 8, error out.
error_input = torch.randn([100])
check_soundness(user_fn, sound_exported, error_input)


######################################################################
# Case 1.2. Rely on the compiler to determine constraints
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Debug logging can be enabled by setting the environment variable
# ``TORCH_LOGS="dynamic"`` to track where the suggested constraints
# originate
#
# We will rely on the compiler to discover reasonable constraints.
# All we know is that dim 0 should be dynamic.
#
t_1 = torch.zeros([8])


def specify_constraints(x):
    return [
        dynamic_dim(x, 0) > 1,
    ]


constraints = specify_constraints(t_1)

# Get a sound exported program
sound_exported = export(user_fn, (t_1,), constraints=constraints)
print(sound_exported)

# Should expect to see the graph can produce sound result for 5 < new_input.shape[0] < 10,
new_input = torch.randn([6])
check_soundness(user_fn, sound_exported, new_input)

# Expect to see error if new_input.shape[0] <= 5
error_input = torch.randn([5])
check_soundness(user_fn, sound_exported, error_input)


######################################################################
# Case 1.3. Refine what the compiler discovered
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# We will make the input constraints tighter:
#
t_1 = torch.zeros([8])

constraints = [
    7 <= dynamic_dim(t_1, 0),
    dynamic_dim(t_1, 0) <= 8,
]

sound_exported = export(user_fn, (t_1,), constraints=constraints)
print(sound_exported)

# Expect to see error if input.shape[0] is not int [7, 8]
error_input = torch.randn([6])
check_soundness(user_fn, sound_exported, error_input)


######################################################################
# Case 1.4. Make a relaxed constraint
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# We want to make a relexed constraint
#
t_1 = torch.zeros([8])

constraints = [
    6 <= dynamic_dim(t_1, 0),
    dynamic_dim(t_1, 0) <= 16,
]

# Try export will end up with ConstrainViolation error!
sound_exported = export(user_fn, (t_1,), constraints=constraints)
print(sound_exported)


######################################################################
# 2. Constraint Violation and Resolution
# --------------------------------------
#
# Next, we will discuss how to make the program exportable and satisfy input constraints as
# requirements.
#


######################################################################
# Case 2.1. User code does not satisfy the user-specified input constraints
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Users will have to either adjust the constraints or rewrite the program.
#
# The following examples will focus on demonstrating how to rewrite code by
# using other export APIs, for example ``cond()``, ``constrain_as_*()``, and other.
#
# The following code demonstrates how you can rewrite our demo example by using the control flow op.
#
# .. note:: Although, the example is not an ideal use case for ``cond()``, it 
#               demonstrates how you can rewrite the code.
#
# Original code:
#
# def user_fn(x):
#     if x.shape[0] > 5 and x.shape[0] < 10:
#         return x + 10
#     else:
#         return x * 2
# Rewritten code:

def user_fn_mod(x):
    def true_branch(x):
        return x + 10

    def false_branch(val):
        return x * 2

    return cond(x.shape[0] > 5 and x.shape[0] < 10, true_branch, false_branch, [x])

# Re-apply the same relexed constraints:
#
t_1 = torch.zeros([8])

constraints = [
    6 <= dynamic_dim(t_1, 0),
    dynamic_dim(t_1, 0) <= 16,
]

sound_exported = export(user_fn_mod, (t_1,), constraints=constraints)
print(sound_exported)

# Should expect to see the graph can produce sound result for 6 <= new_input.shape[0] <= 16,
new_input = torch.randn([16])
check_soundness(user_fn_mod, sound_exported, new_input)


######################################################################
# Case 2.2. Example inputs trigger constraint violation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Case 2.2.1 Example inputs trigger input constraint violation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# One of the examples when a constraint violation happen is when the user mistakenly provides an input that conflicts with input constraints. For example: 
#
t_1 = torch.zeros([3])

constraints = [
    6 <= dynamic_dim(t_1, 0),
    dynamic_dim(t_1, 0) <= 16,
]

exported = export(user_fn_mod, (t_1,), constraints=constraints)
print(exported)


######################################################################
# Case 2.2.2 Example inputs trigger inline constraint violation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 1. What is inline constraint?
# 2. When and how to use it?
#

# User's program contains data-dependent operations
#
def user_fn_ddo(x):
    y = x.nonzero()
    num_nonzeros = y.shape[0]
    if num_nonzeros > 4:
        return x.cos()
    else:
        return x.sin()


t_1 = torch.rand(4, 4)

exported = export(user_fn_ddo, (t_1,), constraints=None)
print(exported)

# Here is an example of how you can re-write this code with an inline constraint: 
# .. note:: There are different ways of rewriting to make the program exportable.
#       For example, you can also use ``torch.nonzero_static()``.
#
def user_fn_ddo_mod(x):
    y = x.nonzero()
    num_nonzeros = y.shape[0]
    # Specify the inline constraint here
    constrain_as_size(num_nonzeros, min=4, max=64)
    if num_nonzeros >= 4:
        return x.cos()
    else:
        return x.sin()


t_1 = torch.rand(4, 4)


def specify_constraints(x):
    return [
        4 <= dynamic_dim(x, 0),
        4 <= dynamic_dim(x, 1),
    ]


sound_exported = export(
    user_fn_ddo_mod, (t_1,), constraints=specify_constraints(t_1)
)
print(sound_exported)

# Good!
new_input = torch.rand(8, 8)
check_soundness(user_fn_ddo_mod, sound_exported, new_input)

# Input constraint violation caught!
#
new_input = torch.rand(2, 8)
check_soundness(user_fn_ddo_mod, sound_exported, new_input)

# Inline constraint violation caught!
#
new_input = torch.ones(10, 10)
check_soundness(user_fn_ddo_mod, sound_exported, new_input)
