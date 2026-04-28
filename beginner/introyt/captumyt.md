Note

Go to the end
to download the full example code.

[Introduction](introyt1_tutorial.html) ||
[Tensors](tensors_deeper_tutorial.html) ||
[Autograd](autogradyt_tutorial.html) ||
[Building Models](modelsyt_tutorial.html) ||
[TensorBoard Support](tensorboardyt_tutorial.html) ||
[Training Models](trainingyt.html) ||
**Model Understanding**

# Model Understanding with Captum

Follow along with the video below or on [youtube](https://www.youtube.com/watch?v=Am2EF9CLu-g). Download the notebook and corresponding files
[here](https://pytorch-tutorial-assets.s3.amazonaws.com/youtube-series/video7.zip).

[Captum](https://captum.ai/) ("comprehension" in Latin) is an open
source, extensible library for model interpretability built on PyTorch.

With the increase in model complexity and the resulting lack of
transparency, model interpretability methods have become increasingly
important. Model understanding is both an active area of research as
well as an area of focus for practical applications across industries
using machine learning. Captum provides state-of-the-art algorithms,
including Integrated Gradients, to provide researchers and developers
with an easy way to understand which features are contributing to a
model's output.

Full documentation, an API reference, and a suite of tutorials on
specific topics are available at the [captum.ai](https://captum.ai/)
website.

## Introduction

Captum's approach to model interpretability is in terms of
*attributions.* There are three kinds of attributions available in
Captum:

- **Feature Attribution** seeks to explain a particular output in terms
of features of the input that generated it. Explaining whether a
movie review was positive or negative in terms of certain words in
the review is an example of feature attribution.
- **Layer Attribution** examines the activity of a model's hidden layer
subsequent to a particular input. Examining the spatially-mapped
output of a convolutional layer in response to an input image in an
example of layer attribution.
- **Neuron Attribution** is analagous to layer attribution, but focuses
on the activity of a single neuron.

In this interactive notebook, we'll look at Feature Attribution and
Layer Attribution.

Each of the three attribution types has multiple **attribution
algorithms** associated with it. Many attribution algorithms fall into
two broad categories:

- **Gradient-based algorithms** calculate the backward gradients of a
model output, layer output, or neuron activation with respect to the
input. **Integrated Gradients** (for features), **Layer Gradient *
Activation**, and **Neuron Conductance** are all gradient-based
algorithms.
- **Perturbation-based algorithms** examine the changes in the output
of a model, layer, or neuron in response to changes in the input. The
input perturbations may be directed or random. **Occlusion,**
**Feature Ablation,** and **Feature Permutation** are all
perturbation-based algorithms.

We'll be examining algorithms of both types below.

Especially where large models are involved, it can be valuable to
visualize attribution data in ways that relate it easily to the input
features being examined. While it is certainly possible to create your
own visualizations with Matplotlib, Plotly, or similar tools, Captum
offers enhanced tools specific to its attributions:

- The `captum.attr.visualization` module (imported below as `viz`)
provides helpful functions for visualizing attributions related to
images.
- **Captum Insights** is an easy-to-use API on top of Captum that
provides a visualization widget with ready-made visualizations for
image, text, and arbitrary model types.

Both of these visualization toolsets will be demonstrated in this
notebook. The first few examples will focus on computer vision use
cases, but the Captum Insights section at the end will demonstrate
visualization of attributions in a multi-model, visual
question-and-answer model.

## Installation

Before you get started, you need to have a Python environment with:

- Python version 3.6 or higher
- For the Captum Insights example, Flask 1.1 or higher and Flask-Compress
(the latest version is recommended)
- PyTorch version 1.2 or higher (the latest version is recommended)
- TorchVision version 0.6 or higher (the latest version is recommended)
- Captum (the latest version is recommended)
- Matplotlib version 3.3.4, since Captum currently uses a Matplotlib
function whose arguments have been renamed in later versions

To install Captum in an Anaconda or pip virtual environment, use the
appropriate command for your environment below:

With `conda`:

```
conda install pytorch torchvision captum flask-compress matplotlib=3.3.4 -c pytorch
```

With `pip`:

```
pip install torch torchvision captum matplotlib==3.3.4 Flask-Compress
```

Restart this notebook in the environment you set up, and you're ready to
go!

## A First Example

To start, let's take a simple, visual example. We'll start with a ResNet
model pretrained on the ImageNet dataset. We'll get a test input, and
use different **Feature Attribution** algorithms to examine how the
input images affect the output, and see a helpful visualization of this
input attribution map for some test images.

First, some imports:

Now we'll use the TorchVision model library to download a pretrained
ResNet. Since we're not training, we'll place it in evaluation mode for
now.

The place where you got this interactive notebook should also have an
`img` folder with a file `cat.jpg` in it.

Our ResNet model was trained on the ImageNet dataset, and expects images
to be of a certain size, with the channel data normalized to a specific
range of values. We'll also pull in the list of human-readable labels
for the categories our model recognizes - that should be in the `img`
folder as well.

```
# model expects 224x224 3-color image

# standard ImageNet normalization
```

Now, we can ask the question: What does our model think this image
represents?

We've confirmed that ResNet thinks our image of a cat is, in fact, a
cat. But *why* does the model think this is an image of a cat?

For the answer to that, we turn to Captum.

## Feature Attribution with Integrated Gradients

**Feature attribution** attributes a particular output to features of
the input. It uses a specific input - here, our test image - to generate
a map of the relative importance of each input feature to a particular
output feature.

[Integrated
Gradients](https://captum.ai/api/integrated_gradients.html) is one of
the feature attribution algorithms available in Captum. Integrated
Gradients assigns an importance score to each input feature by
approximating the integral of the gradients of the model's output with
respect to the inputs.

In our case, we're going to be taking a specific element of the output
vector - that is, the one indicating the model's confidence in its
chosen category - and use Integrated Gradients to understand what parts
of the input image contributed to this output.

Once we have the importance map from Integrated Gradients, we'll use the
visualization tools in Captum to give a helpful representation of the
importance map. Captum's `visualize_image_attr()` function provides a
variety of options for customizing display of your attribution data.
Here, we pass in a custom Matplotlib color map.

Running the cell with the `integrated_gradients.attribute()` call will
usually take a minute or two.

```
# Initialize the attribution algorithm with the model

# Ask the algorithm to attribute our output target to

# Show the original image for comparison
```

In the image above, you should see that Integrated Gradients gives us
the strongest signal around the cat's location in the image.

## Feature Attribution with Occlusion

Gradient-based attribution methods help to understand the model in terms
of directly computing out the output changes with respect to the input.
*Perturbation-based attribution* methods approach this more directly, by
introducing changes to the input to measure the effect on the output.
[Occlusion](https://captum.ai/api/occlusion.html) is one such method.
It involves replacing sections of the input image, and examining the
effect on the output signal.

Below, we set up Occlusion attribution. Similarly to configuring a
convolutional neural network, you can specify the size of the target
region, and a stride length to determine the spacing of individual
measurements. We'll visualize the output of our Occlusion attribution
with `visualize_image_attr_multiple()`, showing heat maps of both
positive and negative attribution by region, and by masking the original
image with the positive attribution regions. The masking gives a very
instructive view of what regions of our cat photo the model found to be
most "cat-like".

Again, we see greater significance placed on the region of the image
that contains the cat.

## Layer Attribution with Layer GradCAM

**Layer Attribution** allows you to attribute the activity of hidden
layers within your model to features of your input. Below, we'll use a
layer attribution algorithm to examine the activity of one of the
convolutional layers within our model.

GradCAM computes the gradients of the target output with respect to the
given layer, averages for each output channel (dimension 2 of output),
and multiplies the average gradient for each channel by the layer
activations. The results are summed over all channels. GradCAM is
designed for convnets; since the activity of convolutional layers often
maps spatially to the input, GradCAM attributions are often upsampled
and used to mask the input.

Layer attribution is set up similarly to input attribution, except that
in addition to the model, you must specify a hidden layer within the
model that you wish to examine. As above, when we call `attribute()`,
we specify the target class of interest.

We'll use the convenience method `interpolate()` in the
[LayerAttribution](https://captum.ai/api/base_classes.html?highlight=layerattribution#captum.attr.LayerAttribution)
base class to upsample this attribution data for comparison to the input
image.

Visualizations such as this can give you novel insights into how your
hidden layers respond to your input.

## Visualization with Captum Insights

Captum Insights is an interpretability visualization widget built on top
of Captum to facilitate model understanding. Captum Insights works
across images, text, and other features to help users understand feature
attribution. It allows you to visualize attribution for multiple
input/output pairs, and provides visualization tools for image, text,
and arbitrary data.

In this section of the notebook, we'll visualize multiple image
classification inferences with Captum Insights.

First, let's gather some image and see what the model thinks of them.
For variety, we'll take our cat, a teapot, and a trilobite fossil:

...and it looks like our model is identifying them all correctly - but of
course, we want to dig deeper. For that we'll use the Captum Insights
widget, which we configure with an `AttributionVisualizer` object,
imported below. The `AttributionVisualizer` expects batches of data,
so we'll bring in Captum's `Batch` helper class. And we'll be looking
at images specifically, so well also import `ImageFeature`.

We configure the `AttributionVisualizer` with the following arguments:

- An array of models to be examined (in our case, just the one)
- A scoring function, which allows Captum Insights to pull out the
top-k predictions from a model
- An ordered, human-readable list of classes our model is trained on
- A list of features to look for - in our case, an `ImageFeature`
- A dataset, which is an iterable object returning batches of inputs
and labels - just like you'd use for training

```
# Baseline is all-zeros input - this may differ depending on your data

# merging our image transforms from above
```

Note that running the cell above didn't take much time at all, unlike
our attributions above. That's because Captum Insights lets you
configure different attribution algorithms in a visual widget, after
which it will compute and display the attributions. *That* process will
take a few minutes.

Running the cell below will render the Captum Insights widget. You can
then choose attributions methods and their arguments, filter model
responses based on predicted class or prediction correctness, see the
model's predictions with associated probabilities, and view heatmaps of
the attribution compared with the original image.

```
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

[`Download Jupyter notebook: captumyt.ipynb`](../../_downloads/c28f42852d456daf9af72da6c6909556/captumyt.ipynb)

[`Download Python source code: captumyt.py`](../../_downloads/d2274535fcc404633095941d2fbbe536/captumyt.py)

[`Download zipped: captumyt.zip`](../../_downloads/5b0771853c144f4574c7298d8784c1c2/captumyt.zip)