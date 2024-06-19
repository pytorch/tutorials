# Contributing to tutorials

We want to make contributing to this project as easy and transparent as
possible. This file covers information on flagging issues, contributing
updates to existing tutorials--and also submitting new tutorials.

NOTE: This guide assumes that you have your GitHub account properly
configured, such as having an SSH key. If this is your first time
contributing on GitHub, see the [GitHub
Documentation](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)
on contributing to projects.


# Issues

We use [GitHub Issues](https://github.com/pytorch/tutorials/issues) to
track public bugs. Please ensure your description is clear and has
sufficient instructions to be able to reproduce the issue.


# Security Bugs

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for
the safe disclosure of security bugs. For these types of issues, please
go through the process outlined on that page and do not file a public
issue.

# Contributor License Agreement ("CLA")

In order to accept a pull request, you need to submit a CLA. You only
need to do this once and you will be able to work on all of Facebook's
open source projects, not just PyTorch.

Complete your CLA here: <https://code.facebook.com/cla>


# License

By contributing to the tutorials, you agree that your contributions will
be licensed as described in the `LICENSE` file in the root directory of
this source tree.


# Updates to existing tutorials

We welcome your pull requests (PR) for updates and fixes.

1. If you haven't already, complete the Contributor License Agreement
   ("CLA").
1. Fork the repo and create a branch from
   [`main`](https://github.com/pytorch/tutorials).
1. Test your code.
1. Lint your code with a tool such as
   [Pylint](https://pylint.pycqa.org/en/latest/).
1. Submit your PR for review.


# New Tutorials

There are three types of tutorial content that we host on
[`pytorch.org/tutorials`](https://github.com/pytorch/tutorials):

* **Interactive tutorials** are authored and submitted as Python files.
  The build system  converts these into Jupyter notebooks and HTML. The
  code in these tutorials is run every time they are built. To keep
  these tutorials up and running all their package dependencies need to
  be resolved--which makes it more challenging to maintain this type of
  tutorial.

* **Non-interactive tutorials** are authored and submitted as
  reStructuredText files. The build system only converts them into HTML;
  the code in them does not run on build. These tutorials are easier to
  create and maintain but they do not provide an interactive experience.
  An example is the [Dynamic Quantization
  tutorial](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html).

* **Recipes** are tutorials that provide bite-sized, actionable
  examples of how to use specific features, which differentiates them
  from full-length tutorials. Recipes can be interactive or
  non-interactive.


# Managing data that is used by your tutorial

Your tutorial might depend on external data, such as pre-trained models,
training data, or test data. We recommend storing this data in a
commonly-used storage service, such as Amazon S3, and instructing your
users to download the data at the beginning of your tutorial.

To download your data add a function to the [download.py](https://github.com/pytorch/tutorials/blob/main/.jenkins/download_data.py)
script. Follow the same pattern as other download functions.
Please do not add download logic to `Makefile` as it will incur download overhead for all CI shards.

# Python packages used by your tutorial

If your tutorial has dependencies that are not already defined in
`requirements.txt`, you should add them to that file. We recommend that
you use only mature, well-supported packages in your tutorial. Packages
that are obscure or not well-maintained may break as a result of, for
example, updates to Python or PyTorch or other packages. If your
tutorial fails to build in our Continuous Integration (CI) system, we
might contact you in order to resolve the issue.


# Deprecation of tutorials

Under some circumstances, we might deprecate--and subsequently
archive--a tutorial removing it from the site. For example, if the
tutorial breaks in our CI and we are not able to resolve the issue and
are also not able to reach you, we might archive the tutorial. In these
situations, resolving the breaking issue would normally be sufficient to
make the tutorial available again.

Another situation in which a tutorial might be deprecated is if it
consistently receives low ratings--or low usage--by the community. Again,
if this occurs, we will attempt to contact you.

If we identify, or suspect, that your tutorial--or a package that your
tutorial uses--has a **security or privacy** issue, we will immediately
take the tutorial off the site.


# Guidance for authoring tutorials and recipes

In this section, we describe the process for creating tutorials and
recipes for Pytorch.

The first step is to decide which type of tutorial you want to create,
taking into account how much support you can provide to keep the
tutorial up-to-date. Ideally, your tutorial should demonstrate PyTorch
functionality that is not duplicated in other tutorials.

As described earlier, tutorials are resources that provide a holistic
end-to-end understanding of how to use PyTorch. Recipes are scoped
examples of how to use specific features; the goal of a recipe is to
teach readers how to easily leverage features of PyTorch for their
needs. Tutorials and recipes are always _actionable_. If the material is
purely informative, consider adding it to the API docs instead.

View our current [full-length tutorials](https://pytorch.org/tutorials/).

To create actionable tutorials, start by identifying _learning
objectives_, which are the end goals. Working backwards from these
objectives will help to eliminate extraneous information.


## Learning objectives ##

To create the learning objectives, focus on what the user will
implement. Set expectations by explicitly stating what the recipe will
cover and what users will implement by the end. Here are some examples:

- Create a custom dataset
- Integrate a dataset using a library
- Iterate over samples in the dataset
- Apply a transform to the dataset


## Voice and writing style ##

Write for a global audience with an instructive and directive voice.

- PyTorch has a global audience; use clear, easy to understand
  language. Avoid idioms or other figures of speech.
- To keep your instructions concise, use
  [active voice](https://writing.wisc.edu/handbook/style/ccs_activevoice/) as much as possible.
- For a short guide on the essentials of writing style,
  [The Elements of Style](https://www.gutenberg.org/files/37134/37134-h/37134-h.htm)
  is invaluable.
- For extensive guidance on technical-writing style, the Google developer documentation
  [google style](https://developers.google.com/style)
  is a great resource.
- Think of the process as similar to creating a (really practical)
  Medium post.


## Structure ##

We recommend that tutorials use the following structure which guides users through the learning experience and provides appropriate context:

1. Introduction
1. Motivation: Why is this topic important?
1. Link to relevant research papers or other background material.
1. Learning objectives: Clearly state what the tutorial covers and what
   users will implement by the end. For example: Provide a summary of
   how the Integrated Gradients feature works and how to implement it
   using Captum. The
   [TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
   tutorial provides a good example of how to specify learning
   objectives.
1. Setup and requirements. Call out any required setup or data
   downloads.
1. Step-by-step instructions. Ideally, the steps in the tutorial should
   map back to the learning objectives. Consider adding comments in the
   code that correspond to these steps and that help to clarify what
   each section of the code is doing.
1. Link to relevant [PyTorch
   documentation](https://pytorch.org/docs/stable/index.html). This
   helps readers have context for the tutorial source code and better
   understand how and why it implements the technique you’re
   demonstrating.
1. Recap/Conclusion: Summarize the steps and concepts covered. Highlight
   key takeaways.
1. (Optional) Additional practice exercises for users to test their
   knowledge. An example is [NLP From Scratch: Generating Names with a
   Character-Level RNN tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#exercises).
1. Additional resources for more learning, such as documentation, other
   tutorials, or relevant research.


## Example Tutorials ##

The following tutorials do a good job of demonstrating the ideas
described in the preceding sections:

- [Chatbot Tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
- [Tensorboard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [NLP From Scratch: Generating Names with a Character-Level RNN
Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)

If you are creating a recipe, we recommend that you use [this
template](https://github.com/pytorch/tutorials/blob/main/beginner_source/template_tutorial.py)
as a guide.


# Submission Process #

Submit your tutorial as either a Python (`.py`) file or a
reStructuredText (`.rst`) file. For Python files, the filename for your
tutorial should end in "`_tutorial.py`"; for example,
"`cool_pytorch_feature_tutorial.py`".

Do not submit a Jupyter notebook. If you develop your tutorial in
Jupyter, you'll need to convert it to Python. This
[script](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe)
is one option for performing this conversion.

For Python files, our CI system runs your code during each build.


## Add Your Tutorial Code ##

1. [Fork and
   clone](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)
   the repo:
   [https://github.com/pytorch/tutorials](https://github.com/pytorch/tutorials)

1. Put the tutorial in one of the
   [`beginner_source`](https://github.com/pytorch/tutorials/tree/main/beginner_source),
   [`intermediate_source`](https://github.com/pytorch/tutorials/tree/main/intermediate_source),
   [`advanced_source`](https://github.com/pytorch/tutorials/tree/main/advanced_source)
   based on the technical level of the content. For recipes, put the
   recipe in
   [`recipes_source`](https://github.com/pytorch/tutorials/tree/main/recipes_source).
   In addition, for recipes, add the recipe in the recipes
   [README.txt](https://github.com/pytorch/tutorials/blob/main/recipes_source/recipes/README.txt)
   file.


## Include Your Tutorial in `index.rst`#

In order for your tutorial to appear on the website, and through tag
search, you need to include it in `index.rst`, or for recipes, in
`recipes_index.rst`.

1. Open the relevant file
   [`index.rst`](https://github.com/pytorch/tutorials/blob/main/index.rst)
   or
   [`recipes_index.rst`](https://github.com/pytorch/tutorials/blob/main/recipes_source/recipes_index.rst)
1. Add a _card_ in reStructuredText format similar to the following:

```
.. customcarditem::
   :header: Learn the Basics # Tutorial title
   :card_description: A step-by-step guide to building a complete ML workflow with PyTorch.  # Short description
   :image: _static/img/thumbnails/cropped/60-min-blitz.png  # Image that appears with the card
   :link: beginner/basics/intro.html
   :tags: Getting-Started
```


### Link ###

The `link` should be the path to your tutorial in the source tree. For
example, if the tutorial is in `beginner_source`, the link will be
`beginner_source/rest/of/the/path.html`


### Tags ###

Choose tags from the existing tags in the file. Reach out to a project
maintainer to create a new tag. The list of tags should not have any
white space between the words. Multi-word tags, such as “Getting
Started”, should be hyphenated: Getting-Started. Otherwise, the tutorial
might fail to build, and the cards will not display properly.


### Image ###

Add a thumbnail to the
[`_static/img/thumbnails/cropped`](https://github.com/pytorch/tutorials/tree/main/_static/img/thumbnails/cropped)
directory. Images that render the best are square--that is, they have
equal `x` and `y` dimensions--and also have high resolution. [Here is an
example](https://github.com/pytorch/tutorials/blob/main/_static/img/thumbnails/cropped/loading-data.PNG).

## `toctree` ##

1. Add your tutorial under the corresponding toctree (also in
   `index.rst`). For example, if you are adding a tutorial that
   demonstrates the PyTorch ability to process images or video, add it
   under `Image and Video`:

```
.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Image and Video

   intermediate/torchvision_tutorial
   beginner/my-new-tutorial
```


## Test Your Tutorial Locally ##

The following command builds an HTML version of the tutorial website.

```
make html-noplot
```

This command does not run your tutorial code. To build the tutorial in a
way that executes the code, use `make docs`. However, unless you have a
GPU-powered machine and a proper PyTorch CUDA setup, running this `make`
command locally won't work. The continuous integration (CI) system will
test your tutorial when you submit your PR.


## Submit the PR ##

NOTE: Please do not use [ghstack](https://github.com/ezyang/ghstack). We
do not support ghstack in the [`pytorch/tutorials`](https://github.com/pytorch/tutorials) repo.

Submit the changes as a PR to the main branch of
[`pytorch/tutorials`](https://github.com/pytorch/tutorials).

1. Add your changes, commit, and push:

    ```
    git add -A
    git commit -m "Add <mytutorial>"
    git push --set-upstream mybranch
    ```

1. Submit the PR and tag individuals on the PyTorch project who can review
   your PR.
1. Address all feedback comments from your reviewers.
1. Make sure all CI checks are passing.

Once you submit your PR, you can see a generated Netlify preview of your
build. You can see an example Netlify preview at the following URL:

>  <https://deploy-preview-954--pytorch-tutorials-preview.netlify.app/>


## Do not merge the PR yourself ##

Please **DO NOT MERGE** your own PR; the tutorial won't be published. In order to avoid potential build breaks with the tutorials site, only certain maintainers can authorize publishing.

