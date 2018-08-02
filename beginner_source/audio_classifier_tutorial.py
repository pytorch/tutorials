"""
Audio Classifier Tutorial
=========================
**Author**: `Winston Herring <https://github.com/winston6>`_

This tutorial will show you how to correctly format an audio dataset and
then train/test an audio classifier network on the dataset. First, let’s
import the common torch packages as well as ``torchaudio`` and
``pandas``. ``Torchaudio`` is available here
https://github.com/pytorch/audio and can be installed by following the
instructions on the website.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchaudio
import pandas as pd
import numpy as np


######################################################################
# Let’s check if a CUDA GPU is available. Running the network on a GPU
# will greatly decrease the training/testing runtime.
# 

cuda_gpu = torch.cuda.is_available()
if cuda_gpu:
    print("GPU available for use")
else:
    print("GPU unavailable")


######################################################################
# Importing the Dataset
# ---------------------
# 
# We will use the UrbanSound8K dataset to train our network. It is
# available for free `here <https://urbansounddataset.weebly.com/>`_ and contains
# 10 audio classes with over 8000 audio samples! Once you have downloaded
# the compressed dataset, extract it to your current working directory.
# First, we will look at the csv file that provides information about the
# individual sound files. ``Pandas`` allows us to open the csv file and
# use ``.iloc()`` to access the data within it.
# 

csvData = pd.read_csv('./UrbanSound8K/metadata/UrbanSound8K.csv')
print(csvData.iloc[0,:])


######################################################################
# Now that we know the format of the csv file entries, we can begin
# constructing our dataset. The UrbanSound8K dataset is separated into 10
# folders. We will use the data from 3 of these folders to train our
# network and then use the 10th folder to test the network. Here we use
# the ``torchaudio.load()`` to convert the wav files to tensors.
# ``Torchaudio.load()`` returns a tuple containing the newly created
# tensor along with the sampling frequency of the audio file (44.1kHz for
# UrbanSound8K). The dataset uses two channels for audio so we will use
# ``torchaudio.transforms.DownmixMono()`` to convert the audio data to one
# channel.
# 

#initialize label and data lists
trainLabels = []
trainData = []
testData = []
testLabels = []
prefix = "./UrbanSound8K/audio/fold" # first part of file name   ex. ./UrbanSound8K/audio/fold3/<fileName>
mixer = torchaudio.transforms.DownmixMono() #used to convert stereo to mono
for i in range(0,len(csvData)):
    if csvData.iloc[i,5] < 4: #ignore files from test data folder(s)
        fileName = prefix + str(csvData.iloc[i,5]) +"/" + csvData.iloc[i,0]
        temp = torchaudio.load(fileName, out = None, normalization = True)
        tempData = mixer(temp[0])
        trainData.append(tempData)
        trainLabels.append(csvData.iloc[i,6])
    elif csvData.iloc[i,5] == 10:
        fileName = prefix + str(csvData.iloc[i,5]) +"/" + csvData.iloc[i,0]
        temp = torchaudio.load(fileName, out = None, normalization = True)
        tempData = mixer(temp[0])
        testData.append(tempData)
        testLabels.append(csvData.iloc[i,6])

#convert label lists to tensors
trainLabels = np.asarray(trainLabels)
trainLabels = torch.from_numpy(trainLabels)
testLabels = np.asarray(testLabels)
testLabels = torch.from_numpy(testLabels)
        
print("Size of train dataset: " + str(len(trainData)))
print("Size of test dataset: " + str(len(testData)))


######################################################################
# Before we begin formatting the data, let’s check to see what the average
# length of each audio tensor is.
# 

averageLen = 0
for i in range(0,len(trainData)):
    averageLen += len(trainData[i])
averageLen /= len(trainData)
print("Average sample length: " + str(round(averageLen)))


######################################################################
# Formatting the Data
# -------------------
# 
# Next, we need to format the audio data. The network we will make takes
# an input size of 32,000, while most of the audio files have well over
# 100,000 samples. The UrbanSound8K audio is sampled at 44.1kHz, so 32,000
# samples only covers around 700 milliseconds. By downsampling the audio
# to aproximately 8kHz, we can represent 4 seconds with the 32,000
# samples. This downsampling is achieved by taking every fifth sample of
# the original audio tensor and may take some time (minutes to an hour
# depending on how many folders are used). Not every audio tensor is long
# enough to handle the downsampling so these tensors will need to be
# padded with zeros. The minimum length that won’t require padding is
# 160,000 which is lower than the average sample length, so the padding
# should be negligible.
# 

zeroCount = 0
trainFormatted = torch.zeros([len(trainData), 1, 32000])
testFormatted = torch.zeros([len(testData), 1, 32000])
print("Begin data formatting")
for j in range(0,len(trainData)):
    for i in range(0,32000):
        if len(trainData[j]) > (i*5): #if the audio tensor isn't long enough then it is padded with zeros
            trainFormatted[j][0][i] = trainData[j][i*5][0]
        else:
            trainFormatted[j][0][i] = 0
            zeroCount += 1


for j in range(0,len(testData)):
    for i in range(0,32000):
        if len(testData[j]) > (i*5): #if the audio tensor isn't long enough then it is padded with zeros
            testFormatted[j][0][i] = testData[j][i*5][0]
        else:
            testFormatted[j][0][i] = 0
            zeroCount += 1

print("Formatting complete")
totalSamples = 32000 * (len(testFormatted) + len(trainFormatted))
print("Padding percentage: " + "%.2f" % (100 * zeroCount/totalSamples) + "%")


######################################################################
# A padding percentage around 10% is acceptable and won’t have a large
# impact on the final accuracy of the network. Now we can create the
# dataset objects that we will use later during training and testing.
# 

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_gpu else {} #needed for using datasets on gpu

training_data = torch.utils.data.TensorDataset(trainFormatted, trainLabels)
trainloader = torch.utils.data.DataLoader(training_data, batch_size = 128, shuffle = True, **kwargs)

test_data = torch.utils.data.TensorDataset(testFormatted, testLabels)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 128, shuffle = True, **kwargs)


######################################################################
# Define the Network
# ------------------
# 
# For this tutorial we will use a convolutional neural network to process
# the raw audio data. Usually more advanced transforms are applied to the
# audio data, however CNNs can be used to accurately process the raw data.
# The specific architecture is modeled after the M5 network architecture
# described `here <https://arxiv.org/pdf/1610.00087.pdf>`_. An important aspect
# of models processing raw audio data is the receptive field of their
# first layer’s filters. Our model’s first filter is length 80 so when
# processing audio sampled at 8kHz the receptive field is around 10ms.
# This size is similar to speech processing applications that often use
# receptive fields ranging from 20ms to 40ms.
# 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,128,80,4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128,128,3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128,256,3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256,512,3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512,10)
        #self.avgPool = nn.AdaptiveAvgPool1d(10)
        #self.softmax = nn.Softmax(dim = 1)
        
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
        x = x.permute(0,2,1) #change the 512x1 to 1x512
        #x = x.float()
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2)

model = Net()
if cuda_gpu:
    model.cuda()
print(model)


######################################################################
# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first we will train with a
# learning rate of 0.01 but will decrease it later during training.
# 

optimizer = optim.Adam(model.parameters(), lr= 0.01, weight_decay = 0.0001)


######################################################################
# Training and Testing the Network
# --------------------------------
# 
# Now let’s define a training function that will feed our training data
# into the model and perform the backward pass and optimization steps.
# 

def train(model, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        if cuda_gpu: #copy data to the gpu if available
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        output = output.permute(1,0,2) #original output dimensions are batchSizex1x10 
        loss = F.nll_loss(output[0], target) #the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data[0]))


######################################################################
# Now that we have a training function, we need to make one for testing
# the networks accuracy. We will set the model to ``eval()`` mode and then
# run inference on the test dataset.
# 

def test(model, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        if cuda_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        output = output.permute(1,0,2)
        pred = output.data.max(2)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))


######################################################################
# Finally, we can train and test the network. We will train the network
# for ten epochs then reduce the learn rate and train for ten more epochs.
# The network will be tested after each epoch to see how the accuracy
# varies during the training.
# 

log_interval = 10
for epoch in range(1,16):
    train(model, epoch)
    test(model, epoch)
print("First round of training complete. Setting learn rate to 0.001.")
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = .0001)
for epoch in range(16,31):
    train(model, epoch)
    test(model, epoch)




######################################################################
# Conclusion
# ----------
# 
# If trained on at least 3 folders, the network should be more than 50%
# accurate by the end of the training process. Training on less folders
# will result in a lower overall accuracy but may be necessary if long
# runtimes are a problem. Greater accuracies can be achieved using deeper
# CNNs at the expense of a larger memory footprint.
# 
# For more advanced audio applications, such as speech recognition,
# recurrent neural networks (RNNs) are commonly used. There are also other
# data preprocessing methods, such as finding the mel frequency cepstral
# coefficients (MCFF), that can reduce the size of the dataset.
# 

