"""
``Pretraining`` VGG from scratch 
============================


**Author:** `WoongJoon Choi <https://github.com/woongjoonchoi>`_

VGG (Visual Geometry Group) is a convolutional neural network architecture that is particularly
efficient in image classification tasks. In this tutorial, we will guide you through building
and training a VGG network from scratch using Python and PyTorch. We will dive into the details of the VGG
architecture, understanding its components and the rationale behind its
design.

Our tutorial is designed for both beginners who are new to deep learning
and seasoned practitioners looking to deepen their understanding of CNN
architectures.

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * Understand the VGG architecture and train it from scratch using PyTorch.
       * Use PyTorch tools to evaluate the VGG model's performance

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * Complete the `Learn the Basics tutorials <https://pytorch.org/tutorials/beginner/basics/intro.html>`__
       * PyTorch 2.4 or later
       * We recommend to run this tutorial on GPU
       
Overview
------------

​​VGG is a model that attracted attention due to its ability to build deeper layers and dramatically
shorten the training time compared to AlexNet, which was the state-of-the-art model at the time of the publishing
of the `original paper <https://arxiv.org/abs/1409.1556>`__.

Unlike AlexNet's 5x5 and 9x9 filters, VGG uses only 3x3 filters. Using multiple 3x3 filters can
obtain the same receptive field as using a 5x5 filter, but it is effective in reducing the number
of parameters. In addition, since it passes through multiple nonlinear functions, the
nonlinearity increases even more.

VGG applies a max pooling layer after multiple convolutional layers to reduce the spatial size.
This allows the feature map to be downsampled while preserving important information. Thanks
to this, the network can learn high-dimensional features in deeper layers and prevent overfitting.

In this tutorial, we will train the VGG model from scratch using only the configuration presented
in the original VGG paper. We will not use future methods such as batch normalization, Adam optimization, or
He initialization. The trained model can be applied to ImageNet data, and you can learn
VGG within the training time suggested in the paper.

Setup
--------

.. note:: if you are running this in Google Colab, install ``albumentations`` by running:

   .. code-block:: python
   
      !pip3 install albumentations``


First, let's import the required dependencies:

"""
import subprocess
import sys

try:
    import albumentations
    print("albumentations are already installed")
except ImportError:
    print("albumentations module not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "albumentations"])
    print("albumentations module installed successfully.")



import torch.optim as optim
import albumentations as A
import numpy as np
import torch

from torchvision.datasets import CIFAR100,CIFAR10,MNIST,ImageNet
import os
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


######################################################################
# VGG Configuration
# -----------------
#
# In this section, we will define configurations suggested in the VGG paper. 
# We use the CIFAR100 dataset. The authors of the VGG paper scale images isotropically,
# which means increasing the size of an image while maintaining its proportions,
# preventing distortion and maintaining the consistency of the object.

DatasetName = 'CIFAR' # CIFAR, CIFAR10, MNIST, ImageNet

## model configuration

num_classes =   100
# ``Caltech`` 257 CIFAR 100  CIFAR10 10 ,MNIST 10 ImageNet 1000
model_version = None ## you must configure it.

## data configuration

train_min = 256
train_max = None
test_min = 256
test_max = 256

## train configuration

batch_size = 32
lr = 1e-2
momentum = 0.9
weight_decay  = 5e-4
lr_factor = 0.1
epoch = 10
clip= None # model D grad clip 0.7

update_count = int(256/batch_size)
accum_step = int(256/batch_size)
eval_step =26 * accum_step  ## ``Caltech`` 5 CIFAR 5 MNIST 6 , CIFAR10 5 ImageNet  26

## model configuration
xavier_count= 4

last_xavier = -8  ##

except_xavier = None

model_layers =None


######################################################################
# .. note:: In the code above, we have defined the batch size as 32,
#    which is recommended for Google Colab. However, if you are
#    running this code on a machine with 24GB of GPU memory,
#    you can set the batch size to 128. You can modify the batch
#    size according to your preference and hardware capabilities.
# 

######################################################################
# Defining the dataset
# --------------------
# 
# As mentioned above we use the CIFAR100 dataset in this tutorial. According to the VGG paper,
# the authors scale the images isotropically to maintain their proportions. This method, known
# as isotropic scaling, increases the size of an image while preserving its aspect ratio,
# thus avoiding distortion and maintaining object consistency. 
#
# After scaling the images, several preprocessing techniques are applied including normalization,
# random crop, and horizontal flip. Normalization adjusts the input data to a range of 0 to 1,
# which typically leads to faster convergence during model training. It ensures that all features
# are scaled to the same range, allowing the model to process each feature more evenly and
# improve overall performance. It is crucial to normalize both training and test data to the
# same range to ensure the model generalizes well to new, unseen data.
#
# Data augmentation techniques like random crop and horizontal flip are crucial for enhancing
# the performance of deep learning models. They help prevent overfitting and ensure that the
# model performs robustly under various conditions. Particularly in scenarios where the dataset
# is small or limited, these techniques effectively increase the amount of training data.
# By exposing the model to various transformations of the data, it learns to generalize better,
# thus improving its performance on both test data and in real-world applications.
#
# To apply preprocessing, we need to override the CIFAR100 class that we have imported from the
# ``torchvision.datasets`` with a custom class:
# 

class Custom_Cifar(CIFAR100) :
    def __init__(self,root,transform = None,multi=False,s_max=None,s_min=256,download=False,val=False,train=True):

        self.multi = multi
        self.s_max = 512
        self.s_min= 256
        if multi :
            self.S = np.random.randint(low=self.s_min,high=self.s_max)
        else :
            self.S = s_min
            transform = A.Compose(
                    [
                        A.Normalize(mean =(0.5071, 0.4867, 0.4408) , std = (0.2675, 0.2565, 0.2761)),
                        A.SmallestMaxSize(max_size=self.S),
                        A.RandomCrop(height =224,width=224),
                        A.HorizontalFlip()
                    ]

            )
        super().__init__(root,transform=transform,train=train,download=download)
        self.val =train
        self.multi = multi
    def __getitem__(self, index: int) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if img.mode == 'L' : img = img.convert('RGB')
        img=np.array(img,dtype=np.float32)


        if self.transform is not None:
            img = self.transform(image=img)
            if len(img['image'].shape) == 3 and self.val==False :
                img = A.RGBShift()(image=img['image'])
            img = img['image']

        if self.target_transform is not None:
            target = self.target_transform(target)
        img=img.transpose((2,0,1))
        return img, target

######################################################################
# Define Model
# ------------
# 
# The VGG paper explores six different model configurations, each with varying layer depths.
# To fully reproduce the results, we will define these configurations below.
#
# We will use two main components to define the model:
#
# * ``Config_channels``: This refers to the number of output channels for each layer.
# * ``Config_kernels``: This refers to the kernel size (or filter size) for each layer.

import torch
from torch import nn


# Config_channels -> number : output_channels , "M": max_pooling layer

Config_channels = {
"A":[64,"M",128,"M",256,256,"M",512,512,"M",512,512,"M"],
"A_lrn":[64,"LRN","M",128,"M",256,256,"M",512,512,"M",512,512,"M"],
"B":[64,64,"M",128,128,"M",256,256,"M",512,512,"M",512,512,"M"],
"C":[64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"],
"D":[64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"],
"E":[64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M"],
}


# Config_kernel ->  kernel_size
Config_kernel = {
"A":[3,2,3,2,3,3,2,3,3,2,3,3,2],
"A_lrn":[3,2,2,3,2,3,3,2,3,3,2,3,3,2],
"B":[3,3,2,3,3,2,3,3,2,3,3,2,3,3,2],
"C":[3,3,2,3,3,2,3,3,1,2,3,3,1,2,3,3,1,2],
"D":[3,3,2,3,3,2,3,3,3,2,3,3,3,2,3,3,3,2],
"E":[3,3,2,3,3,2,3,3,3,3,2,3,3,3,3,2,3,3,3,3,2],
}


######################################################################
# Next, we define a model class that generates a model with a choice of six versions.
# 

def make_feature_extractor(cfg_c,cfg_k):
    feature_extract = []
    in_channels = 3
    i = 1
    for  out_channels , kernel in zip(cfg_c,cfg_k) :
        # print(f"{i} th layer {out_channels} processing")
        if out_channels == "M" :
            feature_extract += [nn.MaxPool2d(kernel,2) ]
        elif out_channels == "LRN":
            feature_extract += [nn.LocalResponseNorm(5,k=2) , nn.ReLU()]
        elif out_channels == 1:
            feature_extract+= [nn.Conv2d(in_channels,out_channels,kernel,stride = 1) , nn.ReLU()]
        else :
            feature_extract+= [nn.Conv2d(in_channels,out_channels,kernel,stride = 1 , padding = 1) , nn.ReLU()]

        if isinstance(out_channels,int) :   in_channels = out_channels
        i+=1
    return nn.Sequential(*feature_extract)


class Model_vgg(nn.Module) :
    # def __init__(self,version , num_classes):
    def __init__(self,conf_channels,conf_kernels , num_classes):
        conv_5_out_w ,conv_5_out_h = 7,7
        conv_5_out_dim =512
        conv_1_by_1_1_outchannel = 4096
        conv_1_by_1_2_outchannel = 4096
        self.num_classes = num_classes
        self.linear_out = 4096
        self.xavier_count = xavier_count
        self.last_xavier= last_xavier  ## if >0 , initialize last 3 fully connected normal distribution
        self.except_xavier  = except_xavier

        super().__init__()
        self.feature_extractor = make_feature_extractor(conf_channels, conf_kernels)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.output_layer = nn.Sequential(
                             nn.Conv2d(conv_5_out_dim  ,conv_1_by_1_1_outchannel ,7) ,
                             nn.ReLU(),
                             nn.Dropout2d(),
                             nn.Conv2d(conv_1_by_1_1_outchannel ,conv_1_by_1_2_outchannel,1 ) ,
                             nn.ReLU(),
                             nn.Dropout2d(),
                             nn.Conv2d(conv_1_by_1_2_outchannel ,num_classes,1 )
                             )


        print('weight initialize')
        self.apply(self._init_weights)
        print('weight intialize end')
    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        x= self.avgpool(x)
        x= torch.flatten(x,start_dim = 1)
        return x


    @torch.no_grad()
    def _init_weights(self,m):

        if isinstance(m,nn.Conv2d):
            print('-------------')
            print(m.kernel_size)
            print(m.out_channels)
            if self.last_xavier>0 and (self.except_xavier is  None or self.last_xavier!=self.except_xavier):
                print('xavier')
                nn.init.xavier_uniform_(m.weight)
            elif self.xavier_count >0 :
                print('xavier')
                nn.init.xavier_uniform_(m.weight)
                self.xavier_count-=1
            else :
                std = 0.1
                print(f'normal  std : {std}')
                torch.nn.init.normal_(m.weight,std=std)

            self.last_xavier +=1
            if m.bias is not None :
                print('bias zero init')
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if self.last_xavier >0 :
                nn.init.xavier_uniform_(m.weight)
                self.last_xavier-=1
            else :
                torch.nn.init.normal_(m.weight,std=std)
                self.last_xavier+=1
                print(f'last xavier increase to {self.last_xavier}')
            nn.init.constant_(m.bias, 0)



######################################################################
# Initializing Model Weights
# ----------------------------
#
# ggIn the original VGG paper, the authors trained model A first and then
# used its weights as a starting point for training other variants. However,
# this approach can be time-consuming. The authors also mentioned using Xavier
# initialization as an alternative to initializing with model A's weights,
# but they did not provide specific details on how to implement it.
# 
# To reproduce the VGG results, we will use the Xavier initialization method
# to initialize the model weights. Specifically, we will apply Xavier
# initialization to the first few layers and the last few layers, while using
# random initialization for the remaining layers.

# .. note::
#    To ensure stability, we must set the standard deviation of the initialization
#    to 0.1. Using a larger standard deviation can result in NaN (Not a Number)
#    values in the weights.
#
# We introduce two hyperparameters to control the Xavier initialization:

# * ``front_xavier:`` The number of layers at the beginning of the network that are
# initialized using Xavier initialization.
#
# * ``last_xavier:`` The number of layers at the end of the network that are initialized
#   using Xavier initialization.
# 
# Based on our experiments, we recommend the following settings:
#
# * For model A: ``front_xavier`` = 4, ``last_xavier`` = 5
# * For models B, C, and D: ``front_xavier`` = 4, ``last_xavier`` = 7
# * For model E: ``front_xavier`` = 5, ``last_xavier`` = 9
# 
# These values have been found to work well in practice.

######################################################################
# Training the Model
# ------------------
# 
# First, let's define top-k error.
# 

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
        res.append(correct_k)
    return res


######################################################################
# Next, we initiate the model and loss function, optimizer and schedulers. In the VGG model,
# they use a softmax output, Momentum Optimizer, and scheduling based on accuracy.
# 

model_version='B'
model = Model_vgg(Config_channels[model_version],Config_kernel[model_version],num_classes)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,momentum=momentum)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=10,threshold=1e-3,eps = 1e-5)

######################################################################
# As mentioned above, we are using the ``CIFAR100`` dataset and set gradient
# clipping to 1.0 to prevent gradient exploding.


if DatasetName == 'Cifar' :
    train_data = Custom_Cifar(root=os.getcwd(),download=True)
    val_data  = Custom_Cifar(root=os.getcwd(),train=False,download=True)
    val_data.val= True
    val_data.s_min = test_min
    val_data.transform=    A.Compose([
                        A.Normalize(mean =(0.5071, 0.4867, 0.4408) , std = (0.2675, 0.2565, 0.2761)),
                        A.SmallestMaxSize(max_size=val_data.S),
                        A.CenterCrop(height =224,width=224)
                    ])
    train_loader = torch.utils.data.DataLoader(train_data,batch_size= batch_size,shuffle = True , num_workers=4,pin_memory = True,prefetch_factor = 2,drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size= batch_size,shuffle = True , num_workers=4,pin_memory = True,prefetch_factor = 2,drop_last = True)

model = model.to(device)

grad_clip = 1.0 # setting gradient clipping to 1.0

for e in range(epoch) :
    print(f'Training Epoch : {e}')
    total_loss = 0
    val_iter = iter(val_loader)
    train_acc=[0,0]
    train_num = 0

    total_acc = [0,0]
    count= 0
    for i , data in enumerate(train_loader) :


        model.train()
        img,label= data
        img,label =img.to(device, non_blocking=True) ,label.to(device, non_blocking=True)

        output = model(img)

        loss = criterion(output,label) /accum_step

        temp_output ,temp_label = output.detach().to('cpu') , label.detach().to('cpu')
        temp_acc = accuracy(temp_output,temp_label,(1,5))
        train_acc=[train_acc[0]+temp_acc[0] , train_acc[1]+temp_acc[1]]
        train_num+=batch_size
        temp_output,temp_label,temp_acc = None,None,None

        loss.backward()
        total_loss += loss.detach().to('cpu')
        img,label=None,None
        torch.cuda.empty_cache()
        if i> 0 and i%update_count == 0 :
            print(f'Training steps : {i}  parameter update loss :{total_loss} ')
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if total_loss < 7.0 :
                # print(f"train loss {total_loss}less than 7.0  ,set grad clip to {clip}")
                grad_clip = clip
            if i % eval_step != 0 :
                total_loss = 0

            output,loss = None,None
            torch.cuda.empty_cache()
        if i>0 and i % eval_step == 0 :

            print(f'train losss :{total_loss}')
            temp_loss = total_loss
            total_loss= 0

            val_loss = 0
            torch.cuda.empty_cache()

            for j   in range(update_count) :
                loss = None
                print(f'Evaluation Steps Start')
                try :
                    img,label = next(val_iter)
                except StopIteration :
                    val_iter= iter(val_loader)
                    img,label = next(val_iter)
                with torch.no_grad():
                    model.eval()

                    img , label = img.to(device, non_blocking=True) , label.to(device, non_blocking=True)
                    output = model(img)
                    temp_output ,temp_label = output.detach().to('cpu') , label.detach().to('cpu')
                    temp_acc = accuracy(temp_output,temp_label,(1,5))
                    total_acc=[total_acc[0]+temp_acc[0] , total_acc[1]+temp_acc[1]]
                    count+=batch_size

                    loss = criterion(output,label)/accum_step
                    val_loss += loss.detach().to('cpu')
                    # loss.backward()
                    torch.cuda.empty_cache()


                    img,label,output ,loss= None,None,None,None



                torch.cuda.empty_cache()

            if abs(val_loss-temp_loss) > 0.03 :
                grad_clip=clip
                # print(f"val_loss {val_loss} - train_loss {temp_loss} = {abs(val_loss-temp_loss)} > 0.3")
                # print(f"set grad clip to {grad_clip}")

                best_val_loss = val_loss

            val_loss = None
        img,label,output = None,None,None



    print(f'top 1 val acc : {total_acc[0]}  top 5 val acc : {total_acc[1]}')
    print(f'val_size :{count}')
    top_1_acc ,top_5_acc   = 100*total_acc[0]/count, 100*total_acc[1]/count
    print(f'top 1 val acc  %: {top_1_acc}')
    print(f'top 5 val acc  %: {top_5_acc}')


    print(f'top 1 train acc : {train_acc[0]}  top 5 train acc : {train_acc[1]}')
    print(f'train_size :{train_num}')
    top_1_train ,top_5_train   = 100*train_acc[0]/train_num, 100*train_acc[1]/train_num
    print(f'top 1 train acc  %: {top_1_train}')
    print(f'top 5 train acc  %: {top_5_train}')


    scheduler.step(top_5_acc)



######################################################################
# (Optional) Additional Exercise: ImageNet
# --------------------------------------------
#
# You can apply the same model that we have trained above with another popular dataset called ImageNet:  

class Custom_ImageNet(ImageNet) :
    def __init__(self,root,transform = None,multi=False,s_max=None,s_min=256,split=None,val=False):

        self.multi = multi
        self.s_max = 512
        self.s_min= 256
        if multi :
            self.S = np.random.randint(low=self.s_min,high=self.s_max)
        else :
            self.S = s_min
            transform = A.Compose(
                    [
                        A.Normalize(),
                        A.SmallestMaxSize(max_size=self.S),
                        A.RandomCrop(height =224,width=224),
                        A.HorizontalFlip()
                    ]

            )
        super().__init__(root,transform=transform,split=split)
        self.val =val
        self.multi = multi
    def __getitem__(self, index: int) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img=np.array(img)
        img = Image.fromarray(img)

        if img.mode == 'L' : img = img.convert('RGB')
        img=np.array(img,dtype=np.float32)


        if self.transform is not None:
            img = self.transform(image=img)
            if len(img['image'].shape) == 3 and self.val==False :
                img = A.RGBShift()(image=img['image'])
            img = img['image']

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img)
        img=img.transpose((2,0,1))

        return img, target

if DatasetName == 'ImageNet' :
    train_data= Custom_ImageNet(root='ImageNet',split='train')
    val_data= Custom_ImageNet('ImageNet',split='val',val=True)
    val_data.val= True
    val_data.s_min = test_min
    val_data.transform=    A.Compose(
                    [
                        A.Normalize(),
                        A.SmallestMaxSize(max_size=val_data.S),
                        A.CenterCrop(height =224,width=224)
                    ]

                )

######################################################################
# Conclusion
# ----------
# 
# In this tutorial, we have successfully demonstrated how to pretrain the VGG model
# from scratch. The techniques and insights provided in this tutorial can serve as
# a basis for reproducing and adapting other foundational models.
# 
# If you are looking to expand your knowledge and application of the VGG model,
# consider exploring further by applying the model to the ImageNet dataset, experimenting
# with different model variants, and incorporating additional evaluation methods to
# enhance model robustness and performance.
# 
# For more information, see: 
#
# - `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__
