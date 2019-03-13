"""
Audio Classifier Tutorial
=========================
**Author**: `Winston Herring <https://github.com/winston6>`_

This tutorial will show you how to correctly format an audio dataset and
then train/test an audio classifier network on the dataset. First, let’s
import the common torch packages as well as ``torchaudio``, ``pandas``,
and ``numpy``. ``torchaudio`` is available `here <https://github.com/pytorch/audio>`_
and can be installed by following the
instructions on the website.

"""












######################################################################
# Let’s check if a CUDA GPU is available and select our device. Running
# the network on a GPU will greatly decrease the training/testing runtime.
#





######################################################################
# Importing the Dataset
# ---------------------
#
# We will use the UrbanSound8K dataset to train our network. It is
# available for free `here <https://urbansounddataset.weebly.com/>`_ and contains
# 10 audio classes with over 8000 audio samples! Once you have downloaded
# the compressed dataset, extract it to your current working directory.
# First, we will look at the csv file that provides information about the
# individual sound files. ``pandas`` allows us to open the csv file and
# use ``.iloc()`` to access the data within it.
#





######################################################################
# The 10 audio classes in the UrbanSound8K dataset are air_conditioner,
# car_horn, children_playing, dog_bark, drilling, enginge_idling,
# gun_shot, jackhammer, siren, and street_music. Let’s play a couple files
# and see what they sound like. The first file is street music and the
# second is an air conditioner.
#







######################################################################
# Formatting the Data
# -------------------
#
# Now that we know the format of the csv file entries, we can construct
# our dataset. We will create a rapper class for our dataset using
# ``torch.utils.data.Dataset`` that will handle loading the files and
# performing some formatting steps. The UrbanSound8K dataset is separated
# into 10 folders. We will use the data from 9 of these folders to train
# our network and then use the 10th folder to test the network. The rapper
# class will store the file names, labels, and folder numbers of the audio
# files in the inputted folder list when initialized. The actual loading
# and formatting steps will happen in the access function ``__getitem__``.
#
# In ``__getitem__``, we use ``torchaudio.load()`` to convert the wav
# files to tensors. ``torchaudio.load()`` returns a tuple containing the
# newly created tensor along with the sampling frequency of the audio file
# (44.1kHz for UrbanSound8K). The dataset uses two channels for audio so
# we will use ``torchaudio.transforms.DownmixMono()`` to convert the audio
# data to one channel. Next, we need to format the audio data. The network
# we will make takes an input size of 32,000, while most of the audio
# files have well over 100,000 samples. The UrbanSound8K audio is sampled
# at 44.1kHz, so 32,000 samples only covers around 700 milliseconds. By
# downsampling the audio to aproximately 8kHz, we can represent 4 seconds
# with the 32,000 samples. This downsampling is achieved by taking every
# fifth sample of the original audio tensor. Not every audio tensor is
# long enough to handle the downsampling so these tensors will need to be
# padded with zeros. The minimum length that won’t require padding is
# 160,000 samples.
#


#rapper for the UrbanSound8K dataset



























































######################################################################
# Define the Network
# ------------------
#
# For this tutorial we will use a convolutional neural network to process
# the raw audio data. Usually more advanced transforms are applied to the
# audio data, however CNNs can be used to accurately process the raw data.
# The specific architecture is modeled after the M5 network architecture
# described in https://arxiv.org/pdf/1610.00087.pdf. An important aspect
# of models processing raw audio data is the receptive field of their
# first layer’s filters. Our model’s first filter is length 80 so when
# processing audio sampled at 8kHz the receptive field is around 10ms.
# This size is similar to speech processing applications that often use
# receptive fields ranging from 20ms to 40ms.
#










































######################################################################
# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first, we will train with
# a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
# to 0.001 during training.
#





######################################################################
# Training and Testing the Network
# --------------------------------
#
# Now let’s define a training function that will feed our training data
# into the model and perform the backward pass and optimization steps.
#



















######################################################################
# Now that we have a training function, we need to make one for testing
# the networks accuracy. We will set the model to ``eval()`` mode and then
# run inference on the test dataset. Calling ``eval()`` sets the training
# variable in all modules in the network to false. Certain layers like
# batch normalization and dropout layers behave differently during
# training so this step is crucial for getting correct results.
#
















######################################################################
# Finally, we can train and test the network. We will train the network
# for ten epochs then reduce the learn rate and train for ten more epochs.
# The network will be tested after each epoch to see how the accuracy
# varies during the training.
#
# .. note:: Due to a build issue, we've reduced the number of epochs to 10.
#           Run this sample with 40 locally to get the proper values.
#










######################################################################
# Conclusion
# ----------
#
# If trained on 9 folders, the network should be more than 50% accurate by
# the end of the training process. Training on less folders will result in
# a lower overall accuracy but may be necessary if long runtimes are a
# problem. Greater accuracies can be achieved using deeper CNNs at the
# expense of a larger memory footprint.
#
# For more advanced audio applications, such as speech recognition,
# recurrent neural networks (RNNs) are commonly used. There are also other
# data preprocessing methods, such as finding the mel frequency cepstral
# coefficients (MFCC), that can reduce the size of the dataset.
#


# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%