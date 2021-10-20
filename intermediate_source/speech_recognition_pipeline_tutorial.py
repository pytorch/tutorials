"""
Speech Recognition with Torchaudio
==================================

**Author**: `Moto Hira <moto@fb.com>`__

This tutorial shows how to perform speech recognition using using
pre-trained models from wav2vec 2.0
[`paper <https://arxiv.org/abs/2006.11477>`__].

"""


######################################################################
# Overview
# --------
# 
# The process of speech recognition looks like the following.
# 
# 1. Extract the acoustic features from audio waveform
# 
# 2. Estimate the class of the acoustic features frame-by-frame
# 
# 3. Generate hypothesis from the sequence of the class probabilities
# 
# Torchaudio provides easy access to the pre-trained weights and
# associated information, such as the expected sample rate and class
# labels. They are bundled together and available under
# ``torchaudio.pipelines`` module.
# 


######################################################################
# Preparation
# -----------
# 
# First we import the necessary packages, and fetch data that we work on.
# 

# %matplotlib inline

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


import matplotlib
import matplotlib.pyplot as plt

[width, height] = matplotlib.rcParams['figure.figsize']
if width < 10:
  matplotlib.rcParams['figure.figsize'] = [width * 2.5, height]

import IPython



######################################################################
# Creating a pipeline
# -------------------
# 
# First, we will create a Wav2Vec2 model that performs the feature
# extraction and the classification.
# 
# There are two types of Wav2Vec2 pre-trained weights available in
# torchaudio. The ones fine-tuned for ASR task, and the ones not
# fine-tuned.
# 
# Wav2Vec2 (and HuBERT) models are trained in self-supervised manner. They
# are firstly trained with audio only for representation learning, then
# fine-tuned for a specific task with additional labels.
# 
# The pre-trained weights without fine-tuning can be used not only ASR but
# also for other downstream tasks as well, but this tutorial does not
# cover them.
# 
# We will use fine-tuned weights. There are multiple models available in
# ``torchaudio.pipelines``. Please check the
# `documentation <https://pytorch.org/audio/stable/pipelines.html>`__ for
# the detail of how they are trained.
# 
# We will use ``torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`` here.
# 
# The bundle object provides the interface to instantiate model and other
# information. Sampling rate and the class labels are found as follow.
# 

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())


######################################################################
# Model can be constructed as following. This process will automatically
# fetch the pre-trained weights and load it into the model.
# 

model = bundle.get_model().to(device)

print(type(model))


######################################################################
# Loading data
# ------------
# 
# We will use the speech data from `VOiCES
# dataset <https://iqtlabs.github.io/voices/>`__, which is licensed under
# Creative Commos BY 4.0.
# 

# In Colab, uncomment the following line to download the audio.
# !wget -O speech.wav https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav

SPEECH_FILE = "speech.wav"

IPython.display.Audio(SPEECH_FILE)


######################################################################
# To load data, we use ``torchaudio.load``.
# 
# If the sampling rate is different from what the pipeline expects, then
# we can use ``torchaudio.functional.resample`` for resampling.
# 
# **Note** -
# ```torchaudio.functional.resample`` <https://pytorch.org/audio/stable/functional.html#resample>`__
# works on CUDA tensors as well. - When performing resampling multiple
# times on the same set of sample rates, using
# ```torchaudio.transforms.Resample`` <https://pytorch.org/audio/stable/transforms.html#resample>`__
# might improve the performace.
# 

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
  waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


######################################################################
# Extracting acoustic features
# ----------------------------
# 
# The next step is to extract acoustic features from the audio.
# 
# Note that Wav2Vec2 models fine-tuned for ASR task can perform feature
# extraction and classification with one step, but for the sake of the
# tutorial, we also show how to perform feature extraction here.
# 

with torch.inference_mode():
  features, _ = model.extract_features(waveform)


######################################################################
# The returned features is a list of tensors. Each tensor is the output of
# a transformer layer.
# 

for i, feats in enumerate(features):
  plt.imshow(feats[0].cpu())
  plt.title(f"Feature from transformer layer {i+1}")
  plt.xlabel("Feature dimension")
  plt.ylabel("Frame (time-axis)")
  plt.show()


######################################################################
# Feature classification
# ----------------------
# 
# Once the acoustic features are extracted, the next step is to classify
# them into a set of categories.
# 
# Wav2Vec2 model provides method to perform the feature extraction and
# classification in one step.
# 

with torch.inference_mode():
  emission, _ = model(waveform)


######################################################################
# The output is in the form of logits. It is not in the form of
# probability.
# 
# Let’s visualize this.
# 

plt.imshow(emission[0].cpu().T)
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.colorbar()
plt.show()
print("Class labels:", bundle.get_labels())


######################################################################
# We can see that there are strong indications to certain labels across
# the time line.
# 
# Note that the class 1 to 3, (``<pad>``, ``</s>`` and ``<unk>``) have
# mostly huge negative values, this is an artifact from the original
# ``fairseq`` implementation where these labels are added by default but
# not used during the training.
# 


######################################################################
# Generating transcripts
# ----------------------
# 
# From the sequence of label probabilities, now we want to generate
# transcripts. The process to generate hypotheses is often called
# “decoding”.
# 
# Decoding is more elaborate than simple classification because
# classification at certain time step can be affected by surrounding
# observations.
# 
# For example, take a word like ``night`` and ``knight``. Even if their
# prior probability distribution are differnt (in typical conversations,
# ``night`` would occur way more often than ``knight``), to accurately
# generate transcripts with ``knight``, such as ``a knight with a sword``,
# the decoding process has to postpone the final decision until it sees
# enough context.
# 
# There are many decoding technicque proposed, and they require external
# resources, such as word dictionary and language models.
# 
# In this tutorial, for the sake of simplicity, we will perform greeding
# decoding which does not depend on such external components, and simply
# pick up the best hypothesis at each time step. Therefore, the context
# information are not used, and only one transcript can be generated.
# 
# We start by defining greedy decoding algorithm.
# 

class GreedyCTCDecoder(torch.nn.Module):
  def __init__(self, labels, ignore):
    super().__init__()
    self.labels = labels
    self.ignore = ignore

  def forward(self, emission: torch.Tensor) -> str:
    """Given a sequence emission over labels, get the best path string
    Args:
      emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
    Returns:
      str: The resulting transcript
    """
    indices = torch.argmax(emission, dim=-1)  # [num_seq,]
    indices = torch.unique_consecutive(indices, dim=-1)
    indices = [i for i in indices if i not in self.ignore]
    return ''.join([self.labels[i] for i in indices])


######################################################################
# Now create the decoder object and decode the transcript.
# 

decoder = GreedyCTCDecoder(
    labels=bundle.get_labels(), 
    ignore=(0, 1, 2, 3),
)
transcript = decoder(emission[0])


######################################################################
# Let’s check the result and listen again the audio.
# 

print(transcript)
IPython.display.Audio(SPEECH_FILE)


######################################################################
# There are few remarks in decoding.
# 
# Firstly, the ASR model is fine-tuned using a loss function calle CTC.
# The detail of CTC loss is explained
# `here <https://distill.pub/2017/ctc/>`__. In CTC a blank token (ϵ) is a
# special token which represents a repetation of the previous symbol. In
# decoding, these are simply ignored.
# 
# Secondly, as is explained in the feature extraction section, the
# Wav2Vec2 model originated from ``fairseq`` has labels that are not used.
# These also has to be ignored.
# 


######################################################################
# Conclusion
# ----------
# 
# In this tutorial, we looked at how to use ``torchaudio.pipeline`` to
# perform acoustic feature extraction and speech recognition. Constructing
# a model and getting the emission is as short as two lines.
# 
# ::
# 
#    model = buWAV2VEC2_ASR_BASE_960Hndle.get_model()
#    emission = model(waveforms, ...)
# 
