Note

Go to the end
to download the full example code.

[Introduction](introyt1_tutorial.html) ||
**Tensors** ||
[Autograd](autogradyt_tutorial.html) ||
[Building Models](modelsyt_tutorial.html) ||
[TensorBoard Support](tensorboardyt_tutorial.html) ||
[Training Models](trainingyt.html) ||
[Model Understanding](captumyt.html)

# Introduction to PyTorch Tensors

Follow along with the video below or on [youtube](https://www.youtube.com/watch?v=r7QDUPb2dCM).

Tensors are the central data abstraction in PyTorch. This interactive
notebook provides an in-depth introduction to the `torch.Tensor`
class.

First things first, let's import the PyTorch module. We'll also add
Python's math module to facilitate some of the examples.

## Creating Tensors

The simplest way to create a tensor is with the `torch.empty()` call:

Let's upack what we just did:

- We created a tensor using one of the numerous factory methods
attached to the `torch` module.
- The tensor itself is 2-dimensional, having 3 rows and 4 columns.
- The type of the object returned is `torch.Tensor`, which is an
alias for `torch.FloatTensor`; by default, PyTorch tensors are
populated with 32-bit floating point numbers. (More on data types
below.)
- You will probably see some random-looking values when printing your
tensor. The `torch.empty()` call allocates memory for the tensor,
but does not initialize it with any values - so what you're seeing is
whatever was in memory at the time of allocation.

A brief note about tensors and their number of dimensions, and
terminology:

- You will sometimes see a 1-dimensional tensor called a
*vector.*
- Likewise, a 2-dimensional tensor is often referred to as a
*matrix.*
- Anything with more than two dimensions is generally just
called a tensor.

More often than not, you'll want to initialize your tensor with some
value. Common cases are all zeros, all ones, or random values, and the
`torch` module provides factory methods for all of these:

The factory methods all do just what you'd expect - we have a tensor
full of zeros, another full of ones, and another with random values
between 0 and 1.

### Random Tensors and Seeding

Speaking of the random tensor, did you notice the call to
`torch.manual_seed()` immediately preceding it? Initializing tensors,
such as a model's learning weights, with random values is common but
there are times - especially in research settings - where you'll want
some assurance of the reproducibility of your results. Manually setting
your random number generator's seed is the way to do this. Let's look
more closely:

What you should see above is that `random1` and `random3` carry
identical values, as do `random2` and `random4`. Manually setting
the RNG's seed resets it, so that identical computations depending on
random number should, in most settings, provide identical results.

For more information, see the [PyTorch documentation on
reproducibility](https://pytorch.org/docs/stable/notes/randomness.html).

### Tensor Shapes

Often, when you're performing operations on two or more tensors, they
will need to be of the same *shape* - that is, having the same number of
dimensions and the same number of cells in each dimension. For that, we
have the `torch.*_like()` methods:

The first new thing in the code cell above is the use of the `.shape`
property on a tensor. This property contains a list of the extent of
each dimension of a tensor - in our case, `x` is a three-dimensional
tensor with shape 2 x 2 x 3.

Below that, we call the `.empty_like()`, `.zeros_like()`,
`.ones_like()`, and `.rand_like()` methods. Using the `.shape`
property, we can verify that each of these methods returns a tensor of
identical dimensionality and extent.

The last way to create a tensor that will cover is to specify its data
directly from a PyTorch collection:

Using `torch.tensor()` is the most straightforward way to create a
tensor if you already have data in a Python tuple or list. As shown
above, nesting the collections will result in a multi-dimensional
tensor.

Note

`torch.tensor()` creates a copy of the data.

### Tensor Data Types

Setting the datatype of a tensor is possible a couple of ways:

The simplest way to set the underlying data type of a tensor is with an
optional argument at creation time. In the first line of the cell above,
we set `dtype=torch.int16` for the tensor `a`. When we print `a`,
we can see that it's full of `1` rather than `1.` - Python's subtle
cue that this is an integer type rather than floating point.

Another thing to notice about printing `a` is that, unlike when we
left `dtype` as the default (32-bit floating point), printing the
tensor also specifies its `dtype`.

You may have also spotted that we went from specifying the tensor's
shape as a series of integer arguments, to grouping those arguments in a
tuple. This is not strictly necessary - PyTorch will take a series of
initial, unlabeled integer arguments as a tensor shape - but when adding
the optional arguments, it can make your intent more readable.

The other way to set the datatype is with the `.to()` method. In the
cell above, we create a random floating point tensor `b` in the usual
way. Following that, we create `c` by converting `b` to a 32-bit
integer with the `.to()` method. Note that `c` contains all the same
values as `b`, but truncated to integers.

For more information, see the [data types documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype).

## Math & Logic with PyTorch Tensors

Now that you know some of the ways to create a tensor... what can you do
with them?

Let's look at basic arithmetic first, and how tensors interact with
simple scalars:

As you can see above, arithmetic operations between tensors and scalars,
such as addition, subtraction, multiplication, division, and
exponentiation are distributed over every element of the tensor. Because
the output of such an operation will be a tensor, you can chain them
together with the usual operator precedence rules, as in the line where
we create `threes`.

Similar operations between two tensors also behave like you'd
intuitively expect:

It's important to note here that all of the tensors in the previous code
cell were of identical shape. What happens when we try to perform a
binary operation on tensors if dissimilar shape?

Note

The following cell throws a run-time error. This is intentional.

```
a = torch.rand(2, 3)
b = torch.rand(3, 2)

print(a * b)
```

In the general case, you cannot operate on tensors of different shape
this way, even in a case like the cell above, where the tensors have an
identical number of elements.

### In Brief: Tensor Broadcasting

Note

If you are familiar with broadcasting semantics in NumPy
ndarrays, you'll find the same rules apply here.

The exception to the same-shapes rule is *tensor broadcasting.* Here's
an example:

What's the trick here? How is it we got to multiply a 2x4 tensor by a
1x4 tensor?

Broadcasting is a way to perform an operation between tensors that have
similarities in their shapes. In the example above, the one-row,
four-column tensor is multiplied by *both rows* of the two-row,
four-column tensor.

This is an important operation in Deep Learning. The common example is
multiplying a tensor of learning weights by a *batch* of input tensors,
applying the operation to each instance in the batch separately, and
returning a tensor of identical shape - just like our (2, 4) * (1, 4)
example above returned a tensor of shape (2, 4).

The rules for broadcasting are:

- Each tensor must have at least one dimension - no empty tensors.
- Comparing the dimension sizes of the two tensors, *going from last to
first:*

- Each dimension must be equal, *or*
- One of the dimensions must be of size 1, *or*
- The dimension does not exist in one of the tensors

Tensors of identical shape, of course, are trivially "broadcastable", as
you saw earlier.

Here are some examples of situations that honor the above rules and
allow broadcasting:

Look closely at the values of each tensor above:

- The multiplication operation that created `b` was
broadcast over every "layer" of `a`.
- For `c`, the operation was broadcast over every layer and row of
`a` - every 3-element column is identical.
- For `d`, we switched it around - now every *row* is identical,
across layers and columns.

For more information on broadcasting, see the [PyTorch
documentation](https://pytorch.org/docs/stable/notes/broadcasting.html)
on the topic.

Here are some examples of attempts at broadcasting that will fail:

Note

The following cell throws a run-time error. This is intentional.

```
a = torch.ones(4, 3, 2)

b = a * torch.rand(4, 3) # dimensions must match last-to-first

c = a * torch.rand( 2, 3) # both 3rd & 2nd dims different

d = a * torch.rand((0, )) # can't broadcast with an empty tensor
```

### More Math with Tensors

PyTorch tensors have over three hundred operations that can be performed
on them.

Here is a small sample from some of the major categories of operations:

```
# common functions

# trigonometric functions and their inverses

# bitwise operations

# comparisons:

# reductions:

# vector and linear algebra operations
```

This is a small sample of operations. For more details and the full inventory of
math functions, have a look at the
[documentation](https://pytorch.org/docs/stable/torch.html#math-operations).
For more details and the full inventory of linear algebra operations, have a
look at this [documentation](https://pytorch.org/docs/stable/linalg.html).

### Altering Tensors in Place

Most binary operations on tensors will return a third, new tensor. When
we say `c = a * b` (where `a` and `b` are tensors), the new tensor
`c` will occupy a region of memory distinct from the other tensors.

There are times, though, that you may wish to alter a tensor in place -
for example, if you're doing an element-wise computation where you can
discard intermediate values. For this, most of the math functions have a
version with an appended underscore (`_`) that will alter a tensor in
place.

For example:

For arithmetic operations, there are functions that behave similarly:

Note that these in-place arithmetic functions are methods on the
`torch.Tensor` object, not attached to the `torch` module like many
other functions (e.g., `torch.sin()`). As you can see from
`a.add_(b)`, *the calling tensor is the one that gets changed in
place.*

There is another option for placing the result of a computation in an
existing, allocated tensor. Many of the methods and functions we've seen
so far - including creation methods! - have an `out` argument that
lets you specify a tensor to receive the output. If the `out` tensor
is the correct shape and `dtype`, this can happen without a new memory
allocation:

## Copying Tensors

As with any object in Python, assigning a tensor to a variable makes the
variable a *label* of the tensor, and does not copy it. For example:

But what if you want a separate copy of the data to work on? The
`clone()` method is there for you:

**There is an important thing to be aware of when using ``clone()``.**
If your source tensor has autograd, enabled then so will the clone.
**This will be covered more deeply in the video on autograd,** but if
you want the light version of the details, continue on.

*In many cases, this will be what you want.* For example, if your model
has multiple computation paths in its `forward()` method, and *both*
the original tensor and its clone contribute to the model's output, then
to enable model learning you want autograd turned on for both tensors.
If your source tensor has autograd enabled (which it generally will if
it's a set of learning weights or derived from a computation involving
the weights), then you'll get the result you want.

On the other hand, if you're doing a computation where *neither* the
original tensor nor its clone need to track gradients, then as long as
the source tensor has autograd turned off, you're good to go.

*There is a third case,* though: Imagine you're performing a computation
in your model's `forward()` function, where gradients are turned on
for everything by default, but you want to pull out some values
mid-stream to generate some metrics. In this case, you *don't* want the
cloned copy of your source tensor to track gradients - performance is
improved with autograd's history tracking turned off. For this, you can
use the `.detach()` method on the source tensor:

What's happening here?

- We create `a` with `requires_grad=True` turned on. **We haven't
covered this optional argument yet, but will during the unit on
autograd.**
- When we print `a`, it informs us that the property
`requires_grad=True` - this means that autograd and computation
history tracking are turned on.
- We clone `a` and label it `b`. When we print `b`, we can see
that it's tracking its computation history - it has inherited
`a`'s autograd settings, and added to the computation history.
- We clone `a` into `c`, but we call `detach()` first.
- Printing `c`, we see no computation history, and no
`requires_grad=True`.

The `detach()` method *detaches the tensor from its computation
history.* It says, "do whatever comes next as if autograd was off." It
does this *without* changing `a` - you can see that when we print
`a` again at the end, it retains its `requires_grad=True` property.

## Moving to [Accelerator](https://pytorch.org/docs/stable/torch.html#accelerators)

One of the major advantages of PyTorch is its robust acceleration on an
[accelerator](https://pytorch.org/docs/stable/torch.html#accelerators)
such as CUDA, MPS, MTIA, or XPU.
So far, everything we've done has been on CPU. How do we move to the faster
hardware?

First, we should check whether an accelerator is available, with the
`is_available()` method.

Note

If you do not have an accelerator, the executable cells in this section will not execute any
accelerator-related code.

Once we've determined that one or more accelerators is available, we need to put
our data someplace where the accelerator can see it. Your CPU does computation
on data in your computer's RAM. Your accelerator has dedicated memory attached
to it. Whenever you want to perform a computation on a device, you must
move *all* the data needed for that computation to memory accessible by
that device. (Colloquially, "moving the data to memory accessible by the
GPU" is shorted to, "moving the data to the GPU".)

There are multiple ways to get your data onto your target device. You
may do it at creation time:

By default, new tensors are created on the CPU, so we have to specify
when we want to create our tensor on the accelerator with the optional
`device` argument. You can see when we print the new tensor, PyTorch
informs us which device it's on (if it's not on CPU).

You can query the number of accelerators with `torch.accelerator.device_count()`. If
you have more than one accelerator, you can specify them by index, take CUDA for example:
`device='cuda:0'`, `device='cuda:1'`, etc.

As a coding practice, specifying our devices everywhere with string
constants is pretty fragile. In an ideal world, your code would perform
robustly whether you're on CPU or accelerator hardware. You can do this by
creating a device handle that can be passed to your tensors instead of a
string:

If you have an existing tensor living on one device, you can move it to
another with the `to()` method. The following line of code creates a
tensor on CPU, and moves it to whichever device handle you acquired in
the previous cell.

It is important to know that in order to do computation involving two or
more tensors, *all of the tensors must be on the same device*. The
following code will throw a runtime error, regardless of whether you
have an accelerator device available, take CUDA for example:

```
x = torch.rand(2, 2)
y = torch.rand(2, 2, device='cuda')
z = x + y # exception will be thrown
```

## Manipulating Tensor Shapes

Sometimes, you'll need to change the shape of your tensor. Below, we'll
look at a few common cases, and how to handle them.

### Changing the Number of Dimensions

One case where you might need to change the number of dimensions is
passing a single instance of input to your model. PyTorch models
generally expect *batches* of input.

For example, imagine having a model that works on 3 x 226 x 226 images -
a 226-pixel square with 3 color channels. When you load and transform
it, you'll get a tensor of shape `(3, 226, 226)`. Your model, though,
is expecting input of shape `(N, 3, 226, 226)`, where `N` is the
number of images in the batch. So how do you make a batch of one?

The `unsqueeze()` method adds a dimension of extent 1.
`unsqueeze(0)` adds it as a new zeroth dimension - now you have a
batch of one!

So if that's *un*squeezing? What do we mean by squeezing? We're taking
advantage of the fact that any dimension of extent 1 *does not* change
the number of elements in the tensor.

Continuing the example above, let's say the model's output is a
20-element vector for each input. You would then expect the output to
have shape `(N, 20)`, where `N` is the number of instances in the
input batch. That means that for our single-input batch, we'll get an
output of shape `(1, 20)`.

What if you want to do some *non-batched* computation with that output -
something that's just expecting a 20-element vector?

You can see from the shapes that our 2-dimensional tensor is now
1-dimensional, and if you look closely at the output of the cell above
you'll see that printing `a` shows an "extra" set of square brackets
`[]` due to having an extra dimension.

You may only `squeeze()` dimensions of extent 1. See above where we
try to squeeze a dimension of size 2 in `c`, and get back the same
shape we started with. Calls to `squeeze()` and `unsqueeze()` can
only act on dimensions of extent 1 because to do otherwise would change
the number of elements in the tensor.

Another place you might use `unsqueeze()` is to ease broadcasting.
Recall the example above where we had the following code:

```
a = torch.ones(4, 3, 2)

c = a * torch.rand( 3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)
```

The net effect of that was to broadcast the operation over dimensions 0
and 2, causing the random, 3 x 1 tensor to be multiplied element-wise by
every 3-element column in `a`.

What if the random vector had just been 3-element vector? We'd lose the
ability to do the broadcast, because the final dimensions would not
match up according to the broadcasting rules. `unsqueeze()` comes to
the rescue:

The `squeeze()` and `unsqueeze()` methods also have in-place
versions, `squeeze_()` and `unsqueeze_()`:

Sometimes you'll want to change the shape of a tensor more radically,
while still preserving the number of elements and their contents. One
case where this happens is at the interface between a convolutional
layer of a model and a linear layer of the model - this is common in
image classification models. A convolution kernel will yield an output
tensor of shape *features x width x height,* but the following linear
layer expects a 1-dimensional input. `reshape()` will do this for you,
provided that the dimensions you request yield the same number of
elements as the input tensor has:

```
# can also call it as a method on the torch module:
```

Note

The `(6 * 20 * 20,)` argument in the final line of the cell
above is because PyTorch expects a **tuple** when specifying a
tensor shape - but when the shape is the first argument of a method, it
lets us cheat and just use a series of integers. Here, we had to add the
parentheses and comma to convince the method that this is really a
one-element tuple.

When it can, `reshape()` will return a *view* on the tensor to be
changed - that is, a separate tensor object looking at the same
underlying region of memory. *This is important:* That means any change
made to the source tensor will be reflected in the view on that tensor,
unless you `clone()` it.

There *are* conditions, beyond the scope of this introduction, where
`reshape()` has to return a tensor carrying a copy of the data. For
more information, see the
[docs](https://pytorch.org/docs/stable/torch.html#torch.reshape).

## NumPy Bridge

In the section above on broadcasting, it was mentioned that PyTorch's
broadcast semantics are compatible with NumPy's - but the kinship
between PyTorch and NumPy goes even deeper than that.

If you have existing ML or scientific code with data stored in NumPy
ndarrays, you may wish to express that same data as PyTorch tensors,
whether to take advantage of PyTorch's GPU acceleration, or its
efficient abstractions for building ML models. It's easy to switch
between ndarrays and PyTorch tensors:

PyTorch creates a tensor of the same shape and containing the same data
as the NumPy array, going so far as to keep NumPy's default 64-bit float
data type.

The conversion can just as easily go the other way:

It is important to know that these converted objects are using *the same
underlying memory* as their source objects, meaning that changes to one
are reflected in the other:

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

**Total running time of the script:** (0 minutes 0.002 seconds)

[`Download Jupyter notebook: tensors_deeper_tutorial.ipynb`](../../_downloads/63a0f0fc7b3ffb15d3a5ac8db3d521ee/tensors_deeper_tutorial.ipynb)

[`Download Python source code: tensors_deeper_tutorial.py`](../../_downloads/be017e7b39198fdf668c138fd8d57abe/tensors_deeper_tutorial.py)

[`Download zipped: tensors_deeper_tutorial.zip`](../../_downloads/3661507963f5283da25e44c8ac84d1a4/tensors_deeper_tutorial.zip)