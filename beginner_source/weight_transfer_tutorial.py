"""
Weight transfer tutorial
========================

Good news! You just had a great idea for a new network architecture!
Only problem: you don't want to spend too much to train it from scratch.
You would like to reuse the weights from an existing network.

This tutorial is here to help you take the weights from one network and
copy them into another one. Even if the layer shapes are not exactly the
same! And even if the names are not exactly the same either (but not too
different also).

Let's dive in!

"""

import numpy as np
import torch
import torchvision
import itertools


######################################################################
# First let's get our donor network (where the weights are). Have you ever
# had an organ transplant?
#

# load donor network
model = torchvision.models.resnet50(pretrained=True)
model = model.eval()
model = model.cpu()


######################################################################
# Then let's load the receiver network (where we want to copy the
# weights). And provide a filename for where we want the weights to be
# saved. Here we take a not so useful example but you get the idea...
#

# load receiver network
model1 = torchvision.models.detection.maskrcnn_resnet50_fpn()
model1 = model1.eval()
model1 = model1.cpu()
weightsfile = 'resnet50_newweights.pth'


######################################################################
# Let's display the names of the donor parameters. As expected this is how
# we will match the donor and receiver weights. Some "parameters" are also
# stored in ``named_buffers`` but here we don't care about listing them
# all. Just figuring out how they are named.
#

for m in model.named_parameters():
    print(m[0])


######################################################################
# Let's display the names of the receiver parameters to compare them with
# the ones of the donor model.
#

for m in model1.named_parameters():
    print(m[0])


######################################################################
# As you can see the names do not match completely. The receiver model
# names are prefixed with ``backbone.body.``. We actually have a bit more
# flexibility at our disposal and we can define prefixes and suffixes for
# both donor and receiver models.
#

# matching rule
# allowed layer name suffixes for receiver network
# add empty string '' to match the same names
# the empty list matches nothing and no weights are transferred
rcv_postfixes = ['']
# allowed layer name suffixes for donor network
# add empty string '' to match the same names
# the empty list matches nothing and no weights are transferred
dnr_postfixes = ['']

# allowed layer name prefixes for receiver network
# add empty string '' to match the same names
# the empty list matches nothing and no weights are transferred
rcv_prefixes = ['backbone.body.']
# allowed layer name prefixes for donor network
# add empty string '' to match the same names
# the empty list matches nothing and no weights are transferred
dnr_prefixes = ['']


######################################################################
# Then we make a function to create all the possible donor names from the
# Tensor name (parameter or buffer) in the receiver model.
#

# name: name of the receiver Tensor for which we want to compute the possible names in the donor model
def make_candidate_name(name, rcv_postfixes, dnr_postfixes, rcv_prefixes, dnr_prefixes):
    name0 = '.'.join(name.split('.')[:-1])
    name1 = name.split('.')[-1]
    # handle suffixes
    output = []
    if len(rcv_postfixes) == 0 or len(dnr_postfixes) == 0 or len(rcv_prefixes) == 0 or len(dnr_prefixes) == 0:
        return output
    # combinations without the receiver suffixes
    for fix in rcv_postfixes:
        if fix == '':
            output.append(name)
        elif name0[-len(fix):] == fix:
            output.append('.'.join([name0[:-len(fix)], name1]))
    outputnew = []
    # combinations with the donor suffixes
    for namenew in output:
        namenew0 = '.'.join(namenew.split('.')[:-1])
        namenew1 = namenew.split('.')[-1]
        for fix in dnr_postfixes:
            outputnew.append('.'.join([namenew0+fix, namenew1]))
    # handle prefixes starting from the suffix combinations
    outputfinal = []
    for newname in outputnew:
        output = []
        # combinations without the receiver prefixes
        for fix in rcv_prefixes:
            if fix == '':
                output.append(newname)
            elif newname[:len(fix)] == fix:
                output.append(newname[len(fix):])
        # combinations with the donor prefixes
        for namenew in output:
            for fix in dnr_prefixes:
                outputfinal.append(fix+namenew)
    return outputfinal


######################################################################
# This is not the case here but there is also a chance that the Tensor
# sizes are not the same. Hence we have several copy strategies.
#

# copying rule in case of shape mismatch
# first: copy the data at the beginning of the Tensor
# last: copy the data at the end of the Tensor
# replicate: replicate the data to fill the Tensor (e.g. donor Tensor = [1234], receiver Tensor=[.......] -> [1234123])
# skip: do not copy the data
copy_vals = ['first', 'last', 'replicate', 'skip']
copy_rule = copy_vals[2]


######################################################################
# Then we make a function to copy the values from the donor tensor
# ``layer_name`` into the receiver tensor ``layer_name1`` (apologies for
# the bad naming conventions).
#
# The replication case ``elif copy_rule == copy_vals[2]`` is a bit more
# complicated because it could be replicating along any dimension but the
# rest is quite straightforward.
#

def transfer_parameter(layer_name, layer_name1, copy_rule):
    # remove donor singleton dimensions to match the shapes if needed
    if len(layer_name[1].shape) > len(layer_name1[1].shape):
        layer_name[1].data = layer_name[1].data.squeeze()
    # expand receiver singleton dimensions to match the shapes if needed
    # this code exands the last dimensions, depending on your case
    # you may want to expand the first ones...
    if len(layer_name1[1].shape) > len(layer_name[1].shape):
        i = len(layer_name[1].shape)
        while layer_name1[1].shape[i] == 1:
            layer_name[1].data = torch.unsqueeze(layer_name[1].data, i)
            i = i+1
            if i == len(layer_name1[1].shape):
                break

    if (layer_name[1].shape == layer_name1[1].shape):
        layer_name1[1].data = layer_name[1].data
    elif copy_rule == copy_vals[0] and len(layer_name[1].data.shape) == len(layer_name1[1].data.shape):
        if all([layer_name1[1].shape[i] >= layer_name[1].shape[i] for i in range(len(layer_name[1].data.shape))]):
            print('copying from '+repr(layer_name[1].shape)+' to '+repr(layer_name1[1].shape))
            copy_slice = tuple(slice(0, n, 1) for n in layer_name[1].shape)
            layer_name1[1].data[copy_slice] = layer_name[1].data
    elif copy_rule == copy_vals[1] and len(layer_name[1].data.shape) == len(layer_name1[1].data.shape):
        if all([layer_name1[1].shape[i] >= layer_name[1].shape[i] for i in range(len(layer_name[1].data.shape))]):
            print('copying from '+repr(layer_name[1].shape)+' to '+repr(layer_name1[1].shape))
            copy_slice = tuple(slice(-n, None, 1) for n in layer_name[1].shape)
            layer_name1[1].data[copy_slice] = layer_name[1].data
    elif copy_rule == copy_vals[2] and len(layer_name[1].data.shape) == len(layer_name1[1].data.shape):
        if all([layer_name1[1].shape[i] >= layer_name[1].shape[i] for i in range(len(layer_name[1].data.shape))]):
            print('replicating from '+repr(layer_name[1].shape)+' to '+repr(layer_name1[1].shape))
            # how many times do we need to replicate along each dimension ?
            ratios = [int(np.ceil(layer_name1[1].shape[i] / layer_name[1].shape[i]))
                      for i in range(len(layer_name[1].data.shape))]
            # compute the indexes of the crops where to copy the donor weights
            indexes = []
            for i in range(len(ratios)):
                indexes.append(list(range(ratios[i])))
            indexes = itertools.product(*indexes)
            # do the copy
            for idx in indexes:
                copy_slice_begin = np.multiply(idx, layer_name[1].data.shape)
                copy_slice_end = map(min, zip(np.multiply([i+1 for i in idx], layer_name[1].data.shape),
                                              layer_name1[1].data.shape))
                copy_slice = list(zip(copy_slice_begin, copy_slice_end))
                input_slice_begin = tuple([0, 0, 0, 0])
                input_slice_end = tuple([a[1]-a[0] for a in list(copy_slice)])
                copy_slice = tuple(slice(a[0], a[1], 1) for a in copy_slice)
                input_slice = tuple(slice(a[0], a[1], 1) for a in list(zip(input_slice_begin, input_slice_end)))
                layer_name1[1].data[copy_slice] = layer_name[1].data[input_slice]
    else:
        print('warning, wrong layer shape ' + layer_name[0] + '\t' +
              str(layer_name[1].shape), str(layer_name1[1].shape))


######################################################################
# And finally, the last one! We just need to loop though all combinations
# of ``named_parameters`` and ``named_buffers`` for the donor and receiver
# network. Of course we also need to use our ``make_candidate_name``
# function to generate the list of matching names for the donor model.
#

# transfer weights
# model is the donor model
# model1 is the receiver model
def transfer_model(model, model1, weightsfile, copy_rule, rcv_postfixes, dnr_postfixes, rcv_prefixes, dnr_prefixes):
    error = False
    notfound = False
    for layer_name1 in model1.named_parameters():
        layer_candidate_names = make_candidate_name(layer_name1[0], rcv_postfixes,
                                                    dnr_postfixes, rcv_prefixes, dnr_prefixes)
        found = False
        for layer_name in model.named_parameters():
            if layer_name[0] in layer_candidate_names:
                found = True
                print('found matching layer name '+layer_name[0]+' and '+layer_name1[0])
                transfer_parameter(layer_name, layer_name1, copy_rule)
        for layer_name in model.named_buffers():
            if layer_name[0] in layer_candidate_names:
                found = True
                print('found matching layer name '+layer_name[0]+' and '+layer_name1[0])
                transfer_parameter(layer_name, layer_name1, copy_rule)
        if not found:
            print('could not find a match for layer '+layer_name1[0])
            notfound = True
    for layer_name1 in model1.named_buffers():
        layer_candidate_names = make_candidate_name(layer_name1[0], rcv_postfixes, dnr_postfixes,
                                                    rcv_prefixes, dnr_prefixes)
        found = False
        for layer_name in model.named_parameters():
            if layer_name[0] in layer_candidate_names:
                found = True
                print('found matching layer name '+layer_name[0]+' and '+layer_name1[0])
                transfer_parameter(layer_name, layer_name1, copy_rule)
        for layer_name in model.named_buffers():
            if layer_name[0] in layer_candidate_names:
                found = True
                print('found matching layer name '+layer_name[0]+' and '+layer_name1[0])
                transfer_parameter(layer_name, layer_name1, copy_rule)
        if not found:
            print('could not find a match for layer '+layer_name1[0])
            notfound = True

    torch.save(model1.state_dict(), weightsfile)
    print('transfer completed')
    if notfound:
        print('some new layers did not have a match in the donor network')
    if error:
        print('some errors ocurred, please check the log')


######################################################################
# Finally, let's get it done!
#

transfer_model(model, model1, weightsfile, copy_rule, rcv_postfixes, dnr_postfixes, rcv_prefixes, dnr_prefixes)
