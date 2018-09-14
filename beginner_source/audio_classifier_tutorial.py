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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np


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
# We will use the UrbanSound8K dataset to train our network. It is
# available for free `here <https://urbansounddataset.weebly.com/>`_ and contains
# 10 audio classes with over 8000 audio samples! Once you have downloaded
# the compressed dataset, extract it to your current working directory.
# First, we will look at the csv file that provides information about the
# individual sound files. ``pandas`` allows us to open the csv file and
# use ``.iloc()`` to access the data within it.
# 

csvData = pd.read_csv('./UrbanSound8K/metadata/UrbanSound8K.csv')
print(csvData.iloc[0, :])


######################################################################
# The 10 audio classes in the UrbanSound8K dataset are air_conditioner,
# car_horn, children_playing, dog_bark, drilling, enginge_idling,
# gun_shot, jackhammer, siren, and street_music. Let’s play a couple files
# and see what they sound like. The first file is street music and the
# second is an air conditioner.
# 

import IPython.display as ipd
ipd.Audio('./UrbanSound8K/audio/fold1/108041-9-0-5.wav')

ipd.Audio('./UrbanSound8K/audio/fold5/100852-0-0-19.wav')


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

class UrbanSoundDataset(Dataset):
#rapper for the UrbanSound8K dataset
    # Argument List
    #  path to the UrbanSound8K csv file
    #  path to the UrbanSound8K audio files
    #  list of folders to use in the dataset
    
    def __init__(self, csv_path, file_path, folderList):
        csvData = pd.read_csv(csv_path)
        #initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        #loop through the csv entries and only add entries from folders in the folder list
        for i in range(0,len(csvData)):
            if csvData.iloc[i, 5] in folderList:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 6])
                self.folders.append(csvData.iloc[i, 5])
                
        self.file_path = file_path
        self.mixer = torchaudio.transforms.DownmixMono() #UrbanSound8K uses two channels, this will convert them to one
        self.folderList = folderList
        
    def __getitem__(self, index):
        #format the file path and load the file
        path = self.file_path + "fold" + str(self.folders[index]) + "/" + self.file_names[index]
        sound = torchaudio.load(path, out = None, normalization = True)
        #load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        soundData = self.mixer(sound[0])
        #downsample the audio to ~8kHz
        tempData = torch.zeros([160000, 1]) #tempData accounts for audio clips that are too short
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]
        
        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        soundFormatted[:32000] = soundData[::5] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        return soundFormatted, self.labels[index]
    
    def __len__(self):
        return len(self.file_names)

    
csv_path = './UrbanSound8K/metadata/UrbanSound8K.csv'
file_path = './UrbanSound8K/audio/'

train_set = UrbanSoundDataset(csv_path, file_path, range(1,10))
test_set = UrbanSoundDataset(csv_path, file_path, [10])
print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True, **kwargs)


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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 10)
        
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
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2)

model = Net()
model.to(device)
print(model)


######################################################################
# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first, we will train with
# a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
# to 0.001 during training.
# 

optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)


######################################################################
# Training and Testing the Network
# --------------------------------
# 
# Now let’s define a training function that will feed our training data
# into the model and perform the backward pass and optimization steps.
# 

def train(model, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        data = data.requires_grad_() #set requires_grad to True for training
        output = model(data)
        output = output.permute(1, 0, 2) #original output dimensions are batchSizex1x10 
        loss = F.nll_loss(output[0], target) #the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


######################################################################
# Now that we have a training function, we need to make one for testing
# the networks accuracy. We will set the model to ``eval()`` mode and then
# run inference on the test dataset. Calling ``eval()`` sets the training
# variable in all modules in the network to false. Certain layers like
# batch normalization and dropout layers behave differently during
# training so this step is crucial for getting correct results.
# 

def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1] # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


######################################################################
# Finally, we can train and test the network. We will train the network
# for ten epochs then reduce the learn rate and train for ten more epochs.
# The network will be tested after each epoch to see how the accuracy
# varies during the training.
# 

log_interval = 20
for epoch in range(1, 41):
    if epoch == 31:
        print("First round of training complete. Setting learn rate to 0.001.")
    scheduler.step()
    train(model, epoch)
    test(model, epoch)


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

