"""
Speech Command Recognition with torchaudio
==========================================

This tutorial will show you how to correctly format an audio dataset and
then train/test an audio classifier network on the dataset. First, let’s
import the common torch packages such as
``torchaudio <https://github.com/pytorch/audio>``\ \_ and can be
installed by following the instructions on the website.

"""

# !pip install torch
# !pip install torchaudio

import os

import IPython.display as ipd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

######################################################################
# Let’s check if a CUDA GPU is available and select our device. Running
# the network on a GPU will greatly decrease the training/testing runtime.
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


######################################################################
# Importing the Dataset
# ---------------------
#
# We use torchaudio to download and represent the dataset. Here we use
# SpeechCommands, which is a datasets of 35 commands spoken by different
# people. The dataset ``SPEECHCOMMANDS`` is a ``torch.utils.data.Dataset``
# version of the dataset.
#
# The actual loading and formatting steps happen in the access function
# ``__getitem__``. In ``__getitem__``, we use ``torchaudio.load()`` to
# convert the audio files to tensors. ``torchaudio.load()`` returns a
# tuple containing the newly created tensor along with the sampling
# frequency of the audio file (16kHz for SpeechCommands). In this dataset,
# all audio files are about 1 second long (and so about 16000 time frames
# long).
#
# Here we wrap it to split it into standard training, validation, testing
# subsets.
#


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        if subset in ["training", "validation"]:
            filepath = os.path.join(self._path, "validation_list.txt")
            with open(filepath) as f:
                validation_list = [
                    os.path.join(self._path, line.strip()) for line in f.readlines()
                ]

        if subset in ["training", "testing"]:
            filepath = os.path.join(self._path, "testing_list.txt")
            with open(filepath) as f:
                testing_list = [
                    os.path.join(self._path, line.strip()) for line in f.readlines()
                ]

        if subset == "validation":
            walker = validation_list
        elif subset == "testing":
            walker = testing_list
        elif subset in ["training", None]:
            walker = self._walker  # defined by SPEECHCOMMANDS parent class
        else:
            raise ValueError(
                "When `subset` not None, it must take a value from {'training', 'validation', 'testing'}."
            )

        if subset == "training":
            walker = filter(
                lambda w: not (w in validation_list or w in testing_list), walker
            )

        self._walker = list(walker)


train_set = SubsetSC("training")
# valid_set = SubsetSC("validation")
test_set = SubsetSC("testing")


waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]


######################################################################
# A data point in the SPEECHCOMMANDS dataset is a tuple made of a waveform
# (the audio signal), the sample rate, the utterance (label), the ID of
# the speaker, the number of the utterance.
#

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())


######################################################################
# Let’s find the list of labels available in the dataset.
#

labels = list(set(datapoint[2] for datapoint in train_set))
labels


######################################################################
# The 35 audio labels are commands that are said by users. The first few
# files are people saying “marvin”.
#

waveform_first, *_ = train_set[0]
ipd.Audio(waveform_first.numpy(), rate=sample_rate)

waveform_second, *_ = train_set[1]
ipd.Audio(waveform_second.numpy(), rate=sample_rate)


######################################################################
# The last file is someone saying “visual”.
#

waveform_last, *_ = train_set[-1]
ipd.Audio(waveform_last.numpy(), rate=sample_rate)


######################################################################
# Formatting the Data
# -------------------
#
# The dataset uses a single channel for audio. We do not need to down mix
# the audio channels (which we could do for instance by either taking the
# mean along the channel dimension, or simply keeping only one of the
# channels).
#


######################################################################
# We downsample the audio for faster processing without losing too much of
# the classification power.
#

new_sample_rate = 8000
transform = torchaudio.transforms.Resample(
    orig_freq=sample_rate, new_freq=new_sample_rate
)
transformed = transform(waveform)

ipd.Audio(transformed.numpy(), rate=new_sample_rate)


######################################################################
# To encode each word, we use a simple language model where we represent
# each fo the 35 words by its corresponding position of the command in the
# list above.
#


def encode(word):
    return torch.tensor(labels.index(word))


encode("yes")


######################################################################
# We now define a collate function that assembles a list of audio
# recordings and a list of utterances into two batched tensors. In this
# function, we also apply the resampling, and the encoding. The collate
# function is used in the pytroch data loader that allow us to iterate
# over a dataset by batches.
#


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    # and so we are only interested in item 0 and 2

    # Apply transforms to waveforms
    tensors = [transform(b[0]) for b in batch]
    tensors = pad_sequence(tensors)

    # Apply transform to target utterance
    targets = [encode(b[2]) for b in batch]
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 128

kwargs = (
    {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}
)  # needed for using datasets on gpu
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **kwargs
)


######################################################################
# Define the Network
# ------------------
#
# For this tutorial we will use a convolutional neural network to process
# the raw audio data. Usually more advanced transforms are applied to the
# audio data, however CNNs can be used to accurately process the raw data.
# The specific architecture is modeled after the M5 network architecture
# described in ``this paper <https://arxiv.org/pdf/1610.00087.pdf>``\ \_.
# An important aspect of models processing raw audio data is the receptive
# field of their first layer’s filters. Our model’s first filter is length
# 80 so when processing audio sampled at 8kHz the receptive field is
# around 10ms (and at 4kHz, around 20 ms). This size is similar to speech
# processing applications that often use receptive fields ranging from
# 20ms to 40ms.
#


class M5(nn.Module):
    def __init__(self, stride=16, n_channel=32, n_output=35):
        super().__init__()
        self.conv1 = nn.Conv1d(1, n_channel, 80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, 3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, 3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, 3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


model = M5(n_output=len(labels))
model.to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


n = count_parameters(model)
print("Number of parameters: %s" % n)


######################################################################
# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first, we will train with
# a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
# to 0.001 during training.
#

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.1
)  # reduce the learning after 20 epochs by a factor of 10


######################################################################
# Training and Testing the Network
# --------------------------------
#
# Now let’s define a training function that will feed our training data
# into the model and perform the backward pass and optimization steps.
#
# Finally, we can train and test the network. We will train the network
# for ten epochs then reduce the learn rate and train for ten more epochs.
# The network will be tested after each epoch to see how the accuracy
# varies during the training.
#


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss:.6f}"
            )

        if "pbar" in globals():
            pbar.update()


######################################################################
# Now that we have a training function, we need to make one for testing
# the networks accuracy. We will set the model to ``eval()`` mode and then
# run inference on the test dataset. Calling ``eval()`` sets the training
# variable in all modules in the network to false. Certain layers like
# batch normalization and dropout layers behave differently during
# training so this step is crucial for getting correct results.
#


def argmax(tensor):
    # index of the max log-probability
    return tensor.max(-1)[1]


def number_of_correct(pred, target):
    # compute number of correct predictions
    return pred.squeeze().eq(target).cpu().sum().item()


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        pred = argmax(output)
        correct += number_of_correct(pred, target)

        if "pbar" in globals():
            pbar.update()

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )


######################################################################
# Finally, we can train and test the network. We will train the network
# for ten epochs then reduce the learn rate and train for ten more epochs.
# The network will be tested after each epoch to see how the accuracy
# varies during the training.
#

log_interval = 20
n_epoch = 2

with tqdm(total=n_epoch * (len(train_loader) + len(test_loader))) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()


######################################################################
# Let’s try looking at one of the last words in the train and test set.
#

waveform, sample_rate, utterance, *_ = train_set[-1]
ipd.Audio(waveform.numpy(), rate=sample_rate)

waveform = transform(waveform)
output = model(waveform.unsqueeze(0))
output = argmax(output).squeeze()
print(f"Expected: {utterance}. Predicted: {labels[output]}.")

waveform, sample_rate, utterance, *_ = test_set[-1]
ipd.Audio(waveform.numpy(), rate=sample_rate)

waveform = transform(waveform)
output = model(waveform.unsqueeze(0))
output = argmax(output).squeeze()
print(f"Expected: {utterance}. Predicted: {labels[output]}.")


######################################################################
# Conclusion
# ----------
#
# After two epochs, the network should be more than 70% accurate.
#
# In this tutorial, we used torchaudio to load a dataset and resample the
# signal. We have then defined a neural network that we trained to
# recognize a given command. There are also other data preprocessing
# methods, such as finding the mel frequency cepstral coefficients (MFCC),
# that can reduce the size of the dataset. This transform is also
# available in torchaudio as ``torchaudio.transforms.MFCC``.
#
