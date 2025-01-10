# -*- coding: utf-8 -*-

"""
``torch.compiler.set_stance`` Tutorial
=================================
**Author:** William Wen
"""

######################################################################
# ``torch.compiler.set_stance`` is a ``torch.compiler`` API that
# enables you to change the behavior of ``torch.compile`` across different
# calls to your model without having to reapply ``torch.compile`` to your model.
#
# This recipe provides some examples on how to use ``torch.compiler.set_stance``.
#
# **Contents**
#
# .. contents::
#     :local:
#
# **Requirements**
#
# - ``torch >= 2.6``

######################################################################
# Description
# -----------
# ``torch.compile.set_stance`` can be used as a decorator, context manager, or raw function
# to change the behavior of ``torch.compile`` across different calls to your model.
#
# In the example below, the ``"force_eager"`` stance ignores all ``torch.compile`` directives.

import torch


@torch.compile
def foo(x):
    if torch.compiler.is_compiling():
        # torch.compile is active
        return x + 1
    else:
        # torch.compile is not active
        return x - 1


inp = torch.zeros(3)

print(foo(inp))  # compiled, prints 1

######################################################################
# Sample decorator usage


@torch.compiler.set_stance("force_eager")
def bar(x):
    # force disable the compiler
    return foo(x)


print(bar(inp))  # not compiled, prints -1

######################################################################
# Sample context manager usage

with torch.compiler.set_stance("force_eager"):
    print(foo(inp))  # not compiled, prints -1

######################################################################
# Sample raw function usage

torch.compiler.set_stance("force_eager")
print(foo(inp))  # not compiled, prints -1
torch.compiler.set_stance("default")

print(foo(inp))  # compiled, prints 1

######################################################################
# ``torch.compile`` stance can only be changed _outside_ of any ``torch.compile`` region. Attempts
# to do otherwise will result in an error.


@torch.compile
def baz(x):
    # error!
    with torch.compiler.set_stance("force_eager"):
        return x + 1


try:
    baz(inp)
except Exception as e:
    print(e)


@torch.compiler.set_stance("force_eager")
def inner(x):
    return x + 1


@torch.compile
def outer(x):
    # error!
    return inner(x)


try:
    outer(inp)
except Exception as e:
    print(e)

######################################################################
# Other stances include:
#  - ``"default"``: The default stance, used for normal compilation.
#  - ``"eager_on_recompile"``: Run code eagerly when a recompile is necessary. If there is cached compiled code valid for the input, it will still be used.
#  - ``"fail_on_recompile"``: Raise an error when recompiling a function.
#
# See the ``torch.compiler.set_stance`` `doc page <https://pytorch.org/docs/main/generated/torch.compiler.set_stance.html#torch.compiler.set_stance>`__
# for more stances and options. More stances/options may also be added in the future.

######################################################################
# Examples
# --------

######################################################################
# Preventing recompilation
# ========================
#
# Some models do not expect any recompilations - for example, you may always inputs to be the same shape.
# Since recompilations may be expensive, we may wish to error out when we attempt to recompile so we can detect and fix recompilation cases.
# The ``"fail_on_recompilation"`` stance can be used for this.


@torch.compile
def my_big_model(x):
    return torch.relu(x)


# first compilation
my_big_model(torch.randn(3))

with torch.compiler.set_stance("fail_on_recompile"):
    my_big_model(torch.randn(3))  # no recompilation - OK
    try:
        my_big_model(torch.randn(4))  # recompilation - error
    except Exception as e:
        print(e)

######################################################################
# If erroring out is too disruptive, we can use ``"eager_on_recompile"`` instead,
# which will cause ``torch.compile`` to fall back to eager instead of erroring out.
# This may be useful if we don't expect recompilations to happen frequently, but
# when one is required, we'd rather pay the cost of running eagerly over the cost of recompilation.


@torch.compile
def my_huge_model(x):
    if torch.compiler.is_compiling():
        return x + 1
    else:
        return x - 1


# first compilation
print(my_huge_model(torch.zeros(3)))  # 1

with torch.compiler.set_stance("eager_on_recompile"):
    print(my_huge_model(torch.zeros(3)))  # 1
    print(my_huge_model(torch.zeros(4)))  # -1
    print(my_huge_model(torch.zeros(3)))  # 1


######################################################################
# Measuring performance gains
# ===========================
#
# ``torch.compiler.set_stance`` can be used to compare eager vs. compiled performance
# without having to define a separate eager model.


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


@torch.compile
def my_gigantic_model(x, y):
    x = x @ y
    x = x @ y
    x = x @ y
    return x


inps = torch.randn(5, 5), torch.randn(5, 5)

with torch.compiler.set_stance("force_eager"):
    print("eager:", timed(lambda: my_gigantic_model(*inps))[1])

# warmups
for _ in range(3):
    my_gigantic_model(*inps)

print("compiled:", timed(lambda: my_gigantic_model(*inps))[1])


######################################################################
# Crashing sooner
# ===============
#
# Running an eager iteration first before a compiled iteration using the ``"force_eager"`` stance
# can help us to catch errors unrelated to ``torch.compile`` before attempting a very long compile.


@torch.compile
def my_humongous_model(x):
    return torch.sin(x, x)


try:
    with torch.compiler.set_stance("force_eager"):
        print(my_humongous_model(torch.randn(3)))
    # this call to the compiled model won't run
    print(my_humongous_model(torch.randn(3)))
except Exception as e:
    print(e)
