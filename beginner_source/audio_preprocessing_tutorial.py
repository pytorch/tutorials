"""
Audio I/O and Pre-Processing with torchaudio
============================================

PyTorch is an open source deep learning platform that provides a
seamless path from research prototyping to production deployment with
GPU support.

Significant effort in solving machine learning problems goes into data
preparation. ``torchaudio`` leverages PyTorch’s GPU support, and provides
many tools to make data loading easy and more readable. In this
tutorial, we will see how to load and preprocess data from a simple
dataset. Please visit
`Audio I/O and Pre-Processing with torchaudio <https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html>`__ to learn more.

For this tutorial, please make sure the ``matplotlib`` package is
installed for easier visualization.

"""

# Uncomment the following line to run in Google Colab
# !pip install torchaudio 
import torch
import torchaudio
import requests
import matplotlib.pyplot as plt

######################################################################
# Opening a file
# -----------------
# 
# ``torchaudio`` also supports loading sound files in the wav and mp3 format. We
# call waveform the resulting raw audio signal.
# 

url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
r = requests.get(url)

with open('steam-train-whistle-daniel_simon-converted-from-mp3.wav', 'wb') as f:
    f.write(r.content)

filename = "steam-train-whistle-daniel_simon-converted-from-mp3.wav"
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())

######################################################################
# When you load a file in ``torchaudio``, you can optionally specify the backend to use either 
# `SoX <https://pypi.org/project/sox/>`_ or `SoundFile <https://pypi.org/project/SoundFile/>`_ 
# via ``torchaudio.set_audio_backend``. These backends are loaded lazily when needed.
# 
# ``torchaudio`` also makes JIT compilation optional for functions, and uses ``nn.Module`` where possible.

######################################################################
# Transformations
# ---------------
# 
# ``torchaudio`` supports a growing list of
# `transformations <https://pytorch.org/audio/stable/transforms.html>`_.
# 
# -  **Resample**: Resample waveform to a different sample rate.
# -  **Spectrogram**: Create a spectrogram from a waveform.
# -  **GriffinLim**: Compute waveform from a linear scale magnitude spectrogram using 
#    the Griffin-Lim transformation.
# -  **ComputeDeltas**: Compute delta coefficients of a tensor, usually a spectrogram.
# -  **ComplexNorm**: Compute the norm of a complex tensor.
# -  **MelScale**: This turns a normal STFT into a Mel-frequency STFT,
#    using a conversion matrix.
# -  **AmplitudeToDB**: This turns a spectrogram from the
#    power/amplitude scale to the decibel scale.
# -  **MFCC**: Create the Mel-frequency cepstrum coefficients from a
#    waveform.
# -  **MelSpectrogram**: Create MEL Spectrograms from a waveform using the
#    STFT function in PyTorch.
# -  **MuLawEncoding**: Encode waveform based on mu-law companding.
# -  **MuLawDecoding**: Decode mu-law encoded waveform.
# -  **TimeStretch**: Stretch a spectrogram in time without modifying pitch for a given rate.
# -  **FrequencyMasking**: Apply masking to a spectrogram in the frequency domain.
# -  **TimeMasking**: Apply masking to a spectrogram in the time domain.
#
# Each transform supports batching: you can perform a transform on a single raw 
# audio signal or spectrogram, or many of the same shape.
# 
# Since all transforms are ``nn.Modules`` or ``jit.ScriptModules``, they can be
# used as part of a neural network at any point.
# 


######################################################################
# To start, we can look at the log of the spectrogram on a log scale.
# 

specgram = torchaudio.transforms.Spectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')


######################################################################
# Or we can look at the Mel Spectrogram on a log scale.
# 

specgram = torchaudio.transforms.MelSpectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')


######################################################################
# We can resample the waveform, one channel at a time.
# 

new_sample_rate = sample_rate/10

# Since Resample applies to a single channel, we resample first channel here
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())


######################################################################
# As another example of transformations, we can encode the signal based on
# Mu-Law enconding. But to do so, we need the signal to be between -1 and
# 1. Since the tensor is just a regular PyTorch tensor, we can apply
# standard operators on it.
# 

# Let's check if the tensor is in the interval [-1,1]
print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))


######################################################################
# Since the waveform is already between -1 and 1, we do not need to
# normalize it.
# 

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

# Let's normalize to the full interval [-1,1]
# waveform = normalize(waveform)


######################################################################
# Let’s apply encode the waveform.
# 

transformed = torchaudio.transforms.MuLawEncoding()(waveform)

print("Shape of transformed waveform: {}".format(transformed.size()))

plt.figure()
plt.plot(transformed[0,:].numpy())


######################################################################
# And now decode.
# 

reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)

print("Shape of recovered waveform: {}".format(reconstructed.size()))

plt.figure()
plt.plot(reconstructed[0,:].numpy())


######################################################################
# We can finally compare the original waveform with its reconstructed
# version.
# 

# Compute median relative difference
err = ((waveform-reconstructed).abs() / waveform.abs()).median()

print("Median relative difference between original and MuLaw reconstucted signals: {:.2%}".format(err))


######################################################################
# Functional
# ---------------
# 
# The transformations seen above rely on lower level stateless functions for their computations. 
# These functions are available under ``torchaudio.functional``. The complete list is available 
# `here <https://pytorch.org/audio/functional.html>`_ and includes:
#
# -  **istft**: Inverse short time Fourier Transform.
# -  **gain**: Applies amplification or attenuation to the whole waveform.
# -  **dither**: Increases the perceived dynamic range of audio stored at a
#    particular bit-depth.
# -  **compute_deltas**: Compute delta coefficients of a tensor.
# -  **equalizer_biquad**: Design biquad peaking equalizer filter and perform filtering.
# -  **lowpass_biquad**: Design biquad lowpass filter and perform filtering.
# -  **highpass_biquad**:Design biquad highpass filter and perform filtering.
# 
# For example, let's try the `mu_law_encoding` functional:

mu_law_encoding_waveform = torchaudio.functional.mu_law_encoding(waveform, quantization_channels=256)

print("Shape of transformed waveform: {}".format(mu_law_encoding_waveform.size()))

plt.figure()
plt.plot(mu_law_encoding_waveform[0,:].numpy())

######################################################################
# You can see how the output from ``torchaudio.functional.mu_law_encoding`` is the same as 
# the output from ``torchaudio.transforms.MuLawEncoding``.
#
# Now let's experiment with a few of the other functionals and visualize their output. Taking our 
# spectogram, we can compute it's deltas:

computed = torchaudio.functional.compute_deltas(specgram.contiguous(), win_length=3)
print("Shape of computed deltas: {}".format(computed.shape))

plt.figure()
plt.imshow(computed.log2()[0,:,:].detach().numpy(), cmap='gray')

######################################################################
# We can take the original waveform and apply different effects to it.
#

gain_waveform = torchaudio.functional.gain(waveform, gain_db=5.0)
print("Min of gain_waveform: {}\nMax of gain_waveform: {}\nMean of gain_waveform: {}".format(gain_waveform.min(), gain_waveform.max(), gain_waveform.mean()))

dither_waveform = torchaudio.functional.dither(waveform)
print("Min of dither_waveform: {}\nMax of dither_waveform: {}\nMean of dither_waveform: {}".format(dither_waveform.min(), dither_waveform.max(), dither_waveform.mean()))

######################################################################
# Another example of the capabilities in ``torchaudio.functional`` are applying filters to our
# waveform. Applying the lowpass biquad filter to our waveform will output a new waveform with 
# the signal of the frequency modified.

lowpass_waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=3000)

print("Min of lowpass_waveform: {}\nMax of lowpass_waveform: {}\nMean of lowpass_waveform: {}".format(lowpass_waveform.min(), lowpass_waveform.max(), lowpass_waveform.mean()))

plt.figure()
plt.plot(lowpass_waveform.t().numpy())

######################################################################
# We can also visualize a waveform with the highpass biquad filter.
# 

highpass_waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq=2000)

print("Min of highpass_waveform: {}\nMax of highpass_waveform: {}\nMean of highpass_waveform: {}".format(highpass_waveform.min(), highpass_waveform.max(), highpass_waveform.mean()))

plt.figure()
plt.plot(highpass_waveform.t().numpy())


######################################################################
# Migrating to torchaudio from Kaldi
# ----------------------------------
# 
# Users may be familiar with
# `Kaldi <http://github.com/kaldi-asr/kaldi>`_, a toolkit for speech
# recognition. ``torchaudio`` offers compatibility with it in
# ``torchaudio.kaldi_io``. It can indeed read from kaldi scp, or ark file
# or streams with:
# 
# -  read_vec_int_ark
# -  read_vec_flt_scp
# -  read_vec_flt_arkfile/stream
# -  read_mat_scp
# -  read_mat_ark
# 
# ``torchaudio`` provides Kaldi-compatible transforms for ``spectrogram``,
# ``fbank``, ``mfcc``, and ``resample_waveform with the benefit of GPU support, see
# `here <compliance.kaldi.html>`__ for more information.
# 

n_fft = 400.0
frame_length = n_fft / sample_rate * 1000.0
frame_shift = frame_length / 2.0

params = {
    "channel": 0,
    "dither": 0.0,
    "window_type": "hanning",
    "frame_length": frame_length,
    "frame_shift": frame_shift,
    "remove_dc_offset": False,
    "round_to_power_of_two": False,
    "sample_frequency": sample_rate,
}

specgram = torchaudio.compliance.kaldi.spectrogram(waveform, **params)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.t().numpy(), cmap='gray')


######################################################################
# We also support computing the filterbank features from waveforms,
# matching Kaldi’s implementation.
# 

fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)

print("Shape of fbank: {}".format(fbank.size()))

plt.figure()
plt.imshow(fbank.t().numpy(), cmap='gray')


######################################################################
# You can create mel frequency cepstral coefficients from a raw audio signal
# This matches the input/output of Kaldi’s compute-mfcc-feats.
# 

mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **params)

print("Shape of mfcc: {}".format(mfcc.size()))

plt.figure()
plt.imshow(mfcc.t().numpy(), cmap='gray')


######################################################################
# Available Datasets
# -----------------
# 
# If you do not want to create your own dataset to train your model, ``torchaudio`` offers a
# unified dataset interface. This interface supports lazy-loading of files to memory, download 
# and extract functions, and datasets to build models.
# 
# The datasets ``torchaudio`` currently supports are:
#
# -  **VCTK**: Speech data uttered by 109 native speakers of English with various accents
#    (`Read more here <https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`_).
# -  **Yesno**: Sixty recordings of one individual saying yes or no in Hebrew; each
#    recording is eight words long (`Read more here <https://www.openslr.org/1/>`_).
# -  **Common Voice**: An open source, multi-language dataset of voices that anyone can use
#    to train speech-enabled applications (`Read more here <https://voice.mozilla.org/en/datasets>`_).
# -  **LibriSpeech**: Large-scale (1000 hours) corpus of read English speech (`Read more here <http://www.openslr.org/12>`_).
# 

yesno_data = torchaudio.datasets.YESNO('./', download=True)

# A data point in Yesno is a tuple (waveform, sample_rate, labels) where labels is a list of integers with 1 for yes and 0 for no.

# Pick data point number 3 to see an example of the the yesno_data:
n = 3
waveform, sample_rate, labels = yesno_data[n]

print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))

plt.figure()
plt.plot(waveform.t().numpy())


######################################################################
# Now, whenever you ask for a sound file from the dataset, it is loaded in memory only when you ask for it.
# Meaning, the dataset only loads and keeps in memory the items that you want and use, saving on memory.
#

######################################################################
# Conclusion
# ----------
# 
# We used an example raw audio signal, or waveform, to illustrate how to
# open an audio file using ``torchaudio``, and how to pre-process,
# transform, and apply functions to such waveform. We also demonstrated how
# to use familiar Kaldi functions, as well as utilize built-in datasets to 
# construct our models. Given that ``torchaudio`` is built on PyTorch,
# these techniques can be used as building blocks for more advanced audio
# applications, such as speech recognition, while leveraging GPUs.
# 
