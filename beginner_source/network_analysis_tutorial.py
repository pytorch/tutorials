"""
Network analysis tutorial
=========================

You have just finished training a network or you just downloaded a
pretrained one. Now you would like to figure out a bit what is happening
inside it to make sure that nothing is wrong.

This tutorial will show you how to - add hooks to get access to your
network inner data (activations and gradients) - visualize several key
characteristics of your network (graph, parameter details and
statistics, activations and gradients)

We will have a rather "baremetal" and basic approach which is what you
need to debug a network.

Should you want a more thorough visualization (more highlevel and
computation costly too), feel free to head up to this great
visualization toolbox:
https://github.com/utkuozbulak/pytorch-cnn-visualizations

"""


# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import time
# print graph, does not display link and layers without gradient
from torchviz import make_dot
# display plots in this notebook
# %matplotlib inline
# set display defaults
plt.rcParams['figure.figsize'] = (15, 15)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show the real values
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


######################################################################
# Let's start by loading an example model.
#

model = models.resnet18(pretrained=True)
model = model.train()
model = model.cuda()


######################################################################
# In order to visualize the input and ouput of each layer as well as the
# input and output gradients, we add hooks to save them to an external
# dictionary which will allow us to access them permanently without
# recomputing the forward pass.
#
# Since we don't want to create all the hooks one by one, we use
# factories.
#

# instrument network forward pass
inputs = {}
outputs = {}


def make_input_hook(param):
    def myhook(module, input_, output):
        global inputs
        inputs[param] = input_
    return myhook


def make_output_hook(param):
    def myhook(module, input_, output):
        global outputs
        outputs[param] = output
    return myhook


for layer in model.named_modules():
    if layer[0] != '':
        outputs[layer[0]] = None
        layer[1].register_forward_hook(make_input_hook(layer[0]))
        layer[1].register_forward_hook(make_output_hook(layer[0]))

# instrument network backward pass
grad_inputs = {}
grad_outputs = {}


def make_grad_input_hook(param):
    def myhook(module, grad_input, grad_output):
        global grad_inputs
        grad_inputs[param] = grad_input
    return myhook


def make_grad_output_hook(param):
    def myhook(module, grad_input, grad_output):
        global grad_outputs
        grad_outputs[param] = grad_output
    return myhook


for layer in model.named_modules():
    if layer[0] != '':
        outputs[layer[0]] = None
        layer[1].register_backward_hook(make_grad_input_hook(layer[0]))
        layer[1].register_backward_hook(make_grad_output_hook(layer[0]))


######################################################################
# Network structure
# =================
#


######################################################################
# The first thing that you want to check about the network is the graph of
# the operations in it.
#
# In pytorch, this graph is dynamic so you can only compute it based on a
# forward pass. The graph is computed by autograd which means that it is
# the backpropagation graph. It will not show variables and links for
# which there is no backpropagation.
#


x = torch.zeros([1, 3, 128, 128], dtype=torch.float, requires_grad=False).cuda()
out = model(x)

graph = make_dot(out, params=dict(model.named_parameters()))  # plot graph of variable, not of a nn.Module
# save the graph to a file in the current directory
graph.render(filename='resnet18')


######################################################################
# The next thing that you want to do is double check the modules contained
# in the network. We can display a list of them.
#

# list all modules
# does not diplay modules without parameters
for idx, m in enumerate(model.modules()):
    print('{:<3}->{}'.format(idx, m))

# another way to list all modules
for name, module in model.named_children():
    print(name, module)


######################################################################
# More interesting we can list and display the shapes of the parameters
# (i.d. the Tensors for which gradient computation is enabled)
#

# list all named modules parameter shapes
weights = {}
bias = {}
for m in model.named_parameters():
    if m[0].split('.')[-1] == 'weight':
        weights['.'.join(m[0].split('.')[:-1])] = m[1].cpu().detach().numpy().shape
    elif m[0].split('.')[-1] == 'bias':
        bias['.'.join(m[0].split('.')[:-1])] = m[1].cpu().detach().numpy().shape
    else:
        print(m[0], np.max(np.abs(m[1].cpu().detach().numpy())))

layernames = set(list(weights.keys()) + list(bias.keys()))

for m in model.named_parameters():
    if '.'.join(m[0].split('.')[:-1]) in layernames:
        if '.'.join(m[0].split('.')[:-1]) in weights and '.'.join(m[0].split('.')[:-1]) in bias:
            print('{:<15} weights: {:<20} bias {}'.format('.'.join(m[0].split('.')[:-1]),
                                                          repr(weights['.'.join(m[0].split('.')[:-1])]),
                                                          bias['.'.join(m[0].split('.')[:-1])]))
        elif '.'.join(m[0].split('.')[:-1]) in weights:
            print('{:<15} weights: {:<20}'.format('.'.join(m[0].split('.')[:-1]),
                                                  repr(weights['.'.join(m[0].split('.')[:-1])])))
        elif '.'.join(m[0].split('.')[:-1]) in bias:
            print('{:<15} weights: {:<20} bias {}'.format('.'.join(m[0].split('.')[:-1]), '',
                                                          bias['.'.join(m[0].split('.')[:-1])]))
        layernames.remove('.'.join(m[0].split('.')[:-1]))


######################################################################
# We can also list the tensors for which no gradient will be computed.
#

# list all batch norm named modules parameter shapes
running_means = {}
running_vars = {}
num_batches_trackeds = {}
weights = {}
bias = {}
for _, m in enumerate(model.named_buffers()):
    if m[0].split('.')[-1] == 'running_mean':
        running_means['.'.join(m[0].split('.')[:-1])] = m[1].cpu().detach().numpy().shape
    elif m[0].split('.')[-1] == 'running_var':
        running_vars['.'.join(m[0].split('.')[:-1])] = m[1].cpu().detach().numpy().shape
    elif m[0].split('.')[-1] == 'num_batches_tracked':
        num_batches_trackeds['.'.join(m[0].split('.')[:-1])] = m[1].cpu().detach().numpy()
    elif m[0].split('.')[-1] == 'weight':
        weights['.'.join(m[0].split('.')[:-1])] = m[1].cpu().detach().numpy().shape
    elif m[0].split('.')[-1] == 'bias':
        bias['.'.join(m[0].split('.')[:-1])] = m[1].cpu().detach().numpy().shape
    else:
        print(m[0])

layernames = set(list(running_means.keys())
                 + list(running_vars.keys())
                 + list(num_batches_trackeds.keys())
                 + list(weights.keys())
                 + list(bias.keys()))

for m in model.named_buffers():
    if '.'.join(m[0].split('.')[:-1]) in layernames:
        if '.'.join(m[0].split('.')[:-1]) in weights and '.'.join(m[0].split('.')[:-1]) in bias:
            print('{:<15} weights: {:<20} bias {}'.format('.'.join(m[0].split('.')[:-1]),
                                                          repr(weights['.'.join(m[0].split('.')[:-1])]),
                                                          bias['.'.join(m[0].split('.')[:-1])]))
        elif '.'.join(m[0].split('.')[:-1]) in weights:
            print('{:<15} weights: {:<20}'.format('.'.join(m[0].split('.')[:-1]),
                                                  repr(weights['.'.join(m[0].split('.')[:-1])])))
        elif '.'.join(m[0].split('.')[:-1]) in bias:
            print('{:<15} weights: {:<20} bias {}'.format('.'.join(m[0].split('.')[:-1]),
                                                          '', bias['.'.join(m[0].split('.')[:-1])]))
        if '.'.join(m[0].split('.')[:-1]) in running_means:
            means_shape = repr(running_means['.'.join(m[0].split('.')[:-1])])
        else:
            means_shape = ''
        if '.'.join(m[0].split('.')[:-1]) in running_vars:
            vars_shape = repr(running_vars['.'.join(m[0].split('.')[:-1])])
        else:
            vars_shape = ''
        if '.'.join(m[0].split('.')[:-1]) in num_batches_trackeds:
            nbbatches_shape = str(num_batches_trackeds['.'.join(m[0].split('.')[:-1])])
        else:
            nbbatches_shape = ''
        if means_shape + vars_shape + nbbatches_shape != '   ':
            print('{:<15} means:   {:<20} vars {:<10} nb batches {:<5}'.format('.'.join(m[0].split('.')[:-1]),
                                                                               means_shape, vars_shape,
                                                                               nbbatches_shape))
        layernames.remove('.'.join(m[0].split('.')[:-1]))


######################################################################
# Weights analysis
# ================
#


######################################################################
# We are going ot want to visualize many 4D tensors so let's make a helper
# function.
#

def show(img):
    print(img.max(), img.mean(), img.min())
    print(str(img.shape))
    # img may not be an image
    if img.shape[1] not in [1, 3]:
        img = torch.reshape(img, (-1, 1, img.shape[-2], img.shape[-1]))
    # find best number of rows to get square output image
    nrow = int(np.ceil(np.sqrt(img.shape[0] * img.shape[-2] / img.shape[-1])))
    img = torchvision.utils.make_grid(img, nrow=nrow, normalize=True, pad_value=1, padding=1).float()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')


######################################################################
# We can look at the network weights to make sure that they are not all
# zero or some other abnormal situation.
#

# Visualize first layer weights:
filters = model.conv1.weight.detach().cpu()
show(filters)


######################################################################
# In the case of layers with many filters and inputs, it can be more
# convenient to visualize the ditribution of the weight values.
#
# We can check the value range and the main modes as well as the kind or
# look of the distribution. If you just initialized the network, you can
# check that the weights have the expected distribution (single value,
# uniform distribution, gaussian, etc.)
#

# visualize distribution of layer weights and biases
module = model.layer3[0].bn2

plt.rcParams['figure.figsize'] = (15, 15)        # large images
weights = module.weight.detach().cpu().numpy()
if module.bias is not None:
    biases = module.bias.detach().cpu().numpy()
    plt.subplot(2, 1, 1)
    _ = plt.hist(weights.flat, bins=100)
    plt.subplot(2, 1, 2)
    _ = plt.hist(biases.flat, bins=100)
else:
    _ = plt.hist(weights.flat, bins=100)


######################################################################
# Activation analysis
# ===================
#
# In order to check the activations of our network, we need some input to
# feed it. Loading the imagenet dataset can take a while. If you already
# downloaded it you can set the ``root`` to the location where you already
# downloaded it.
#

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageNet(root='.data', split='val',
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False, num_workers=2)
dataiter = iter(trainloader)


######################################################################
# Now let's run it through our network!
#

start_compute_time = time.time()
# run one image through network
images, labels = dataiter.next()
images = images.cuda()
output = model(images)
print(labels)
print(output.shape)
total_time = time.time() - start_compute_time
print("---Computation took {:3.2f} seconds ---".format(total_time))


######################################################################
# A good starting point is to look at what you gave as input to your
# network.
#

show(images.cpu().detach())


######################################################################
# Then you can check the input for each layer or the one that you suspect
# could have a problem. A ReLU can easily kill a big chunk of the output
# from the previous layer (which is the input to the current layer).
#

# module input
layer_name = 'layer2.0.conv2'
feat = inputs[layer_name][0].cpu().detach()
show(feat)


######################################################################
# Then you can check the output and compare it with the input to see the
# processing being done by the layer.
#

# module output
layer_name = 'layer2.0.conv2'
feat = outputs[layer_name].cpu().detach()
show(feat)


######################################################################
# Backpropagation analysis
# ========================
#
# In order to backpropagate, we need to define the loss. Since Imagenet is
# a classification dataset, the ``CrossEntropyLoss`` sounds like a good
# choice don't you think?
#

criterion = nn.CrossEntropyLoss()
loss = criterion(output, labels.cuda())
loss.backward()


######################################################################
# Then we can check the gradients on the weights of our layer.
#

# module weights gradient
layer_name = 'layer2.0.conv2'
feat = grad_inputs[layer_name][1].cpu().detach()
show(feat)


######################################################################
# As you can see, there are too many of them to really see anything. All
# we can check is that they are varied and no bizarre pattern seems to
# emerge.
#
# We can display only some of them to have a more detailed view.
#

show(feat[:33, :33, ...])


######################################################################
# We can also visualize the gradient activations.
#

# module activation gradient
layer_name = 'layer2.0.conv2'
feat = grad_outputs[layer_name][0].cpu().detach()
show(feat)


######################################################################
# This time, the 115th activation (middle of second row from the bottom)
# seems a bit weird. The gradients seem to have all the same value except
# on the image borders...
#
# Let's check that out in more details.
#

show(feat[0, 115, ...])


######################################################################
# Yes, there is definitely something surprising here. Let's investigate
# more and go up the backprogation graph to see where this comes from!
#

# module activation gradient
layer_name = 'layer2.0.bn2'
layer_name = 'layer2.0.downsample.0'
layer_name = 'layer2.0.downsample.1'
layer_name = 'layer2.1.conv1'
feat = grad_outputs[layer_name][0].cpu().detach()
show(feat)


######################################################################
# Allright. It seems that the same issue appears up to ``layer2.1.conv1``.
# Let's check the input of this network to see what is going on.
#

# module input
layer_name = 'layer2.1.conv1'
feat = inputs[layer_name][0].cpu().detach()
show(feat)


######################################################################
# Hey, the input seems to be all black (0, the minimum value) as well!
#

show(feat[0, 115, ...])


######################################################################
# Yes, that's definitely that. Let's check the output from the previous
# layer.
#

# module output
layer_name = 'layer2.0.downsample.1'
feat = outputs[layer_name].cpu().detach()
show(feat[0, 115, ...])


######################################################################
# Hard to see what's going on. Let's check which outputs are positive.
#

show((feat[0, 115, ...] > 0).float())


######################################################################
# There you go! That was the ReLU killing the output and the gradients.
# Let's try with another image to see if it is just a one off or a
# permanent problem.
#

start_compute_time = time.time()
# run one image through network
images, labels = dataiter.next()
images = images.cuda()
output = model(images)
print(labels)
print(output.shape)
total_time = time.time() - start_compute_time
print("---Computation took {:3.2f} seconds ---".format(total_time))

show(images.cpu().detach())

# module output
layer_name = 'layer2.0.downsample.1'
feat = outputs[layer_name].cpu().detach()
show(feat[0, 115, ...])

show((feat[0, 115, ...] > 0).float())


######################################################################
# Well, looks like this is not a one off thing. Either this filter is not
# very useful or it is simply a very selective filter to identify precise
# items.
#
# You can try to process several more images and look at the statistics to
# see how frequently the output is mostly negative (and mostly zeroed by
# the ReLU). That a good way to detect such issues without having to look
# at each layer manually. You can also try to set the weights of this
# filter to 0 and see if it affect the network performance.
#
# Many things to try! Now that you know the basics, it's up to you!
#
