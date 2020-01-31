Custom Dataset and Dataloader in PyTorch C++
============================================

One of the most powerful components of PyTorch are the dataset utilities which
help users optimize and parallelize reading and serving data into models.  

In this tutorial, we are going to create a custom dataset that contains 2
inputs and a label.  This dataset will be inspired by the `MNIST dataset
<https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/datasets/mnist.h>`_.
These datasets can be used for applications that require more than a single
input and single label.

.. tip::

  We recommend reading the `C++ Frontend Tutorial <cpp_frontend.html>`_ if 
  you have not done so already.  There you will learn how to setup a C++ PyTorch 
  project.

Motivation
----------

We want to leverage the raw speed of C++ for reading data and take advantage of
C++'s native multithreading with a custom dataset.  Additionally, the ability to
customize the outputs of our dataset allows us to easily use a wider range of
PyTorch models.  For example, the current batch of transformer models have
multiple inputs for the various masks, token type ids, and position ids that are
used as inputs into the network.  The base PyTorch C++ dataset only supports two
different tensors ("data" and "target") for each example.  Here we will create a
dataset that supports 3 different tensors, and we will create the necessary
files to use the powerful PyTorch dataloader utilities.

Setting Up the Project
----------------------

As in the frontend tutorial above, we will be using ``cmake`` to compile our
project.

.. code-block:: cmake
  :caption: CMakeLists.txt
  :name: cmakelists-txt

  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

  # allows us to use Torch_ROOT environmental variable
  cmake_policy(SET CMP0074 NEW)

  # alternative to set_property(...)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  project(custom-dataset 
          VERSION 0.0.1 
          DESCRIPTION "Custom Dataset with PyTorch's C++ Frontend")

  # add torch
  find_package(Torch REQUIRED)

  # put all sources files into a single variable
  set(SOURCES
      custom_dataset.cpp
      main.cpp
     )

  add_executable(custom-dataset ${SOURCES})
  target_link_libraries(custom-dataset "${TORCH_LIBRARIES}")

.. note::

  If you need ``CMAKE_PREFIX_PATH`` to be something other than the 
  location to the ``libtorch`` directory, you can use the environmental 
  variable ``Torch_DIR`` as an alternative.  To do so, you export 
  it in your shell as follows:

.. code-block:: shell

  export Torch_DIR=/full/path/to/libtorch/share/cmake/Torch
  # or
  export Torch_DIR=$(realpath rel/path/to/libtorch/share/cmake/Torch)

Now we are ready to create the files required for our project.  As seen in the
``CMakeLists.txt`` file above, the two source files that we are going to use
will be called ``main.cpp`` and ``custom_dataset.cpp``.  In addition to these
files, we will also create header files for the custom dataset, the custom input
type, and the custom stacking dataset transform in the files
``custom_dataset.h``, ``custom_input_type.h``, and ``custom_stack.h``,
respectively.  


Custom Input Type
^^^^^^^^^^^^^^^^^

The default dataset uses a templated struct called ``Example<>`` with two
members, ``data`` and ``target``.  We are going to create a struct with three
members, ``inputone``, ``inputtwo``, and ``label``.  Ultimately, we will be able
to use this in our dataloader as well to create batches for each of these three
members regardless of relative shapes.

.. note::

  ``Example<>`` is shorthand for ``Example<torch::Tensor, torch::Tensor>``, 
  we will be using something similar.  PyTorch's C++ library makes extensive 
  use of templates and we need to create some custom version of the transforms 
  for our custom input type.

We'll begin by creating a ``struct`` with three members named ``inputone``,
``inputtwo``, and ``label`` with three types that default to a torch tensor
type.  We are also going to create a default constructor for this struct.  

.. code-block:: cpp
  :caption: custom_input_type.h
  :name: custom-input-type-h

  #pragma once
  
  #include <torch/types.h>
  template < typename InputOne = torch::Tensor, 
             typename InputTwo = torch::Tensor,
             typename Label = torch::Tensor >
  struct ThreeTensorInput {
    ThreeTensorInput() = default;
    ThreeTensorInput(InputOne inputone, InputTwo inputtwo, Label label)
        : inputone(std::move(inputone)), 
          inputtwo(std::move(inputtwo)),
          label(std::move(label)) {}

    InputOne inputone;
    InputTwo inputtwo;
    Label label;
  };  // don't forget the semi-colon here

This could be useful if ``inputone`` and ``inputtwo`` have different types or
dimensions or just to organize your data types.

Dataset Class
^^^^^^^^^^^^^

Next, we will create our dataset from the standard ``Dataset`` included with
PyTorch's data utilities.  This is a minimal dataset that holds our examples in
a vector and returns one example each time the ``get`` method is called.  It is
very similar to the MNIST dataset example except that we are simplifying it a
bit by removing the train / test type of the MNIST dataset.  

.. code-block:: cpp
  :caption: custom_dataset.h
  :name: custom-dataset-h

  #pragma once

  #include <torch/data/datasets/base.h>
  #include <torch/types.h>

  #include "custom_input_type.h"

  template <typename CustomSingleExample = ThreeTensorInput<>>
  class CustomDataset
      : public torch::data::datasets::Dataset<CustomDataset<CustomSingleExample>,
                                              CustomSingleExample> {
  public:
    using CustomExampleType = CustomSingleExample;
    // constructor
    explicit CustomDataset(const std::vector<CustomExampleType> &examples);
    // get item
    virtual CustomExampleType get(std::size_t index) override;
    // dataset size
    torch::optional<std::size_t> size() const override;
    // get all examples
    const std::vector<CustomExampleType> &examples() const;

  private:
    std::vector<CustomExampleType> examples_;
  };

Dataset Implementation
**********************

The other big departure from the MNIST dataset is that we've replaced
``Example<>`` with ``ThreeTensorInput<>`` and ``ExampleType`` with
``CustomExampleType``.  What is important here is that we using the dataset
template with our custom dataset and custom type
``Dataset<CustomDataset<CustomSingleExample>, CustomSingleExample>``.  In the
MNIST dataset the example type is implicit because it's the default parameter,
but since we are changing it, we need to explicitly put our type here.  Next we
want to define all of the methods that we need to override in 
``custom_dataset.cpp``.

.. code-block:: cpp
  :caption: custom_dataset.cpp
  :name: custom-dataset-cpp

  #include "custom_dataset.h"

  using namespace std;

  template <typename T>
  CustomDataset<T>::CustomDataset(const vector<T> &examples)
      : examples_(examples) {}

  template <typename T> T CustomDataset<T>::get(size_t index) {
    T ex = examples_[index];
    return std::move(ex);
  }

  template <typename T> torch::optional<size_t> CustomDataset<T>::size() const {
    torch::optional<size_t> sz(examples_.size());
    return sz;
  }

  // the following line is required for the linker to work correctly
  template class CustomDataset<>; // add our custom example with default argument

.. note::

  We are using ``T`` as our typename here for convenience, because we need to define 
  our methods for this templated class.  Lastly, we need to explicitly 
  instantiate our custom class with each type that we are going to use in our 
  template.  For this example, we are only using the default type ``ThreeTensorInput``, 
  which is the default type.  Thus we can instatiate it no template parameters.  

Run the Custom Dataset
^^^^^^^^^^^^^^^^^^^^^^

At this point we have a functional dataset class.  We can load tensors into this
dataset and retrieve them by using the ``get`` method.  To use our dataset, we
will create a minimal main class that loads our dataset with 10 examples where
the inputs and labels are all different sizes.

.. code-block:: cpp
  :caption: main.cpp
  :name: main-no-dataloader-cpp

  #include <iostream>
  #include <torch/torch.h>

  #include "custom_dataset.h"
  #include "custom_input_type.h"

  using namespace std;
  using namespace torch;

  int main() {
    int dataset_sz = 10;

    vector<Tensor> ones, twos, labels;
    ones.reserve(dataset_sz);
    twos.reserve(dataset_sz);
    labels.reserve(dataset_sz);

    vector<ThreeTensorInput<>> examples_;
    for (int i = 0; i < dataset_sz; ++i) {
      ones.push_back(torch::rand({2, 3}));  // size = (2, 3)
      twos.push_back(torch::rand({3, 2}));  // size = (3, 2)
      labels.push_back(torch::randint(5, 1));  // size = (1)
      examples_.emplace_back(ones[i], twos[i], labels[i]);
    }

    CustomDataset<> ds(examples_);
    assert((static_cast<size_t>(dataset_sz), ds.size().value()));
    
    auto ex = ds.get(0);

    cout << ex.inputone << "\n"
         << ex.inputtwo << "\n"
         << ex.label << endl;

    return 1;
  }

Dataloader
^^^^^^^^^^

Ok, if we want to iterate through our dataset a single example at a time then
we are done.  Of course, we probably want to process our data in minibatches.
As in the python frontend we also have a ``Dataloader`` utility class in the
C++ frontend.  However, in the python frontend there is a parameter in the
``Dataloader`` class called ``collate_fn``.  The default collation function
basically converts a list of basic python types, numpy arrays, or torch tensors
stacks them into a minibatch.  One normally doesn't need to write a custom
collation function except for special cases such as sequence data when you need
to pad a sequence to the length of the longest sequence in the batch.  In the
C++ frontent, the equivalent to the collation function are mapping transforms.
These transforms take a batch from the ``get_batch`` function and apply
themselves to the batch.  By default, the ``get_batch`` function returns a
vector of the type returned by the ``get`` function.  In the MNIST dataset,
they use the ``map`` method to transform our ``BatchDataset`` into a
``MapDataset``.  The code looks something like:

.. code-block:: cpp

  int batch_size = 3;
  auto ds(examples_).map(data::transforms::Stack<>());
  auto dl = data::make_data_loader<data::samplers::SequentialSampler>(
      move(ds), batch_size);
  
Custom Stack Transform
**********************

But here we see that the ``Stack<>`` transform is a default template.  Spoiler
alert, the default type parameter here is ``Example<>`` and this transform
explicitly stacks the ``data`` and ``target`` members.  So we are going to have
to write our own stack transform for our custom type.  The original
implementation of ``Stack`` is `here
<https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/transforms/stack.h>`_.
So let's create a file called ``custom_stack.h`` and do that.

.. code-block:: cpp
  :caption: custom_stack.h
  :name: custom-stack-h

  #pragma once

  #include <torch/data/transforms.h>
  #include <vector>

  #include "custom_input_type.h"

  template <>
  struct torch::data::transforms::Stack<ThreeTensorInput<>>
      : public torch::data::transforms::Collation<ThreeTensorInput<>> {
    ThreeTensorInput<>
    apply_batch(std::vector<ThreeTensorInput<>> examples) override {
      std::vector<torch::Tensor> inputone, inputtwo, label;
      inputone.reserve(examples.size());
      inputtwo.reserve(examples.size());
      label.reserve(examples.size());
      for (auto &example : examples) {
        inputone.push_back(std::move(example.inputone));
        inputtwo.push_back(std::move(example.inputtwo));
        label.push_back(std::move(example.label));
      }
      return {torch::stack(inputone), torch::stack(inputtwo),
              torch::stack(label)};
    }
  };

As stated earlier, the default ``get_batch`` method creates a vector of our
custom type.  Then ``torch::stack`` is applied to each of the three members of
this custom type, which stacks the members with an added dimension for the
batch.  Since our custom type is comprised of only tensors, the return type of
this transform is also a ``ThreeTensorInput<>``, but with the extra dimension.
Now we can go back to the ``main.cpp``, include this header file, map our
dataset with this custom transform, and create the dataloader.  Feel free to
try to do this yourself.  Don't forget to add our custom input type where it's
needed.  

Main with Dataloader
^^^^^^^^^^^^^^^^^^^^

Ok, your final code should look like this:

.. code-block:: cpp
  :caption: main.cpp
  :name: main-cpp

  #include <iostream>
  #include <torch/torch.h>

  #include "custom_dataset.h"
  #include "custom_input_type.h"
  #include "custom_stack.h"  // our custom stack transform

  using namespace std;
  using namespace torch;

  int main() {
    int dataset_sz = 10;
    int batch_size = 3;  // added batch size variable

    vector<Tensor> ones, twos, labels;
    ones.reserve(dataset_sz);
    twos.reserve(dataset_sz);
    labels.reserve(dataset_sz);

    vector<ThreeTensorInput<>> examples_;
    for (int i = 0; i < dataset_sz; ++i) {
      ones.push_back(torch::rand({2, 3}));
      twos.push_back(torch::rand({3, 2}));
      labels.push_back(torch::randint(5, 1));
      examples_.emplace_back(ones[i], twos[i], labels[i]);
    }

    CustomDataset<> ds(examples_);
    assert((static_cast<size_t>(dataset_sz), ds.size().value()));
    // Stack takes our custom type which is also a templated class
    auto ds_map = ds.map(data::transforms::Stack<ThreeTensorInput<>>());
    auto dl = data::make_data_loader<data::samplers::SequentialSampler>(
        move(ds_map), batch_size);
    for (auto &mb : *dl) {
      cout << mb.inputone << "\n" << mb.inputtwo << "\n" << mb.label << endl;
    }

    return 1;
  }

Now let's build and run the code.

.. code-block:: shell

  export Torch_DIR=$(realpath rel/path/to/libtorch/share/cmake/Torch)
  mkdir build && cd build
  cmake ..
  make
  ./custom-dataset

There we have it.  A minimal example of a PyTorch dataset and dataloader in C++.

