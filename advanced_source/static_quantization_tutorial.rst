(beta) Static Quantization with Eager Mode in PyTorch 
========================================================= 
**Author**: `Raghuraman Krishnamoorthi <https://github.com/raghuramank100>`_
**Edited by**: `Seth Weidman <https://github.com/SethHWeidman/>`_, `Jerry Zhang <https:github.com/jerryzh168>`_

This tutorial shows how to do post-training static quantization, as well as illustrating  
two more advanced techniques - per-channel quantization and quantization-aware training - 
to further improve the model's accuracy. Note that quantization is currently only supported 
for CPUs, so we will not be utilizing GPUs / CUDA in this tutorial. 
By the end of this tutorial, you will see how quantization in PyTorch can result in 
significant decreases in model size while increasing speed. Furthermore, you'll see how 
to easily apply some advanced quantization techniques shown 
`here <https://arxiv.org/abs/1806.08342>`_ so that your quantized models take much less 
of an accuracy hit than they would otherwise. 
Warning: we use a lot of boilerplate code from other PyTorch repos to, for example, 
define the ``MobileNetV2`` model architecture, define data loaders, and so on. We of course  
encourage you to read it; but if you want to get to the quantization features, feel free  
to skip to the "4. Post-training static quantization" section.  
We'll start by doing the necessary imports: 

.. code:: python

    import os
    import sys
    import time
    import numpy as np

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    import torchvision
    from torchvision import datasets
    import torchvision.transforms as transforms

    # Set up warnings
    import warnings 
    warnings.filterwarnings(  
        action='ignore',  
        category=DeprecationWarning,  
        module=r'.*'  
    ) 
    warnings.filterwarnings(  
        action='default', 
        module=r'torch.ao.quantization'
    ) 

    # Specify random seed for repeatable results  
    torch.manual_seed(191009) 

1. Model architecture 
--------------------- 

We first define the MobileNetV2 model architecture, with several notable modifications  
to enable quantization: 

- Replacing addition with ``nn.quantized.FloatFunctional``  
- Insert ``QuantStub`` and ``DeQuantStub`` at the beginning and end of the network. 
- Replace ReLU6 with ReLU 
 
Note: this code is taken from 
`here <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_.

.. code:: python

    from torch.ao.quantization import QuantStub, DeQuantStub

    def _make_divisible(v, divisor, min_value=None):  
        """ 
        This function is taken from the original tf repo. 
        It ensures that all layers have a channel number that is divisible by 8 
        It can be seen here:  
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  
        :param v: 
        :param divisor: 
        :param min_value: 
        :return:  
        """ 
        if min_value is None: 
            min_value = divisor 
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor) 
        # Make sure that round down does not go down by more than 10%.  
        if new_v < 0.9 * v: 
            new_v += divisor  
        return new_v  


    class ConvBNReLU(nn.Sequential):  
        def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1): 
            padding = (kernel_size - 1) // 2  
            super(ConvBNReLU, self).__init__( 
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),  
                nn.BatchNorm2d(out_planes, momentum=0.1), 
                # Replace with ReLU 
                nn.ReLU(inplace=False)  
            ) 


    class InvertedResidual(nn.Module):  
        def __init__(self, inp, oup, stride, expand_ratio): 
            super(InvertedResidual, self).__init__()  
            self.stride = stride  
            assert stride in [1, 2] 

            hidden_dim = int(round(inp * expand_ratio)) 
            self.use_res_connect = self.stride == 1 and inp == oup  

            layers = [] 
            if expand_ratio != 1: 
                # pw  
                layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1)) 
            layers.extend([ 
                # dw  
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), 
                # pw-linear 
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  
                nn.BatchNorm2d(oup, momentum=0.1),  
            ])  
            self.conv = nn.Sequential(*layers)  
            # Replace torch.add with floatfunctional  
            self.skip_add = nn.quantized.FloatFunctional()  

        def forward(self, x): 
            if self.use_res_connect:  
                return self.skip_add.add(x, self.conv(x)) 
            else: 
                return self.conv(x) 


    class MobileNetV2(nn.Module): 
        def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):  
            """ 
            MobileNet V2 main class 
            Args: 
                num_classes (int): Number of classes  
                width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount  
                inverted_residual_setting: Network structure  
                round_nearest (int): Round the number of channels in each layer to be a multiple of this number 
                Set to 1 to turn off rounding 
            """ 
            super(MobileNetV2, self).__init__() 
            block = InvertedResidual  
            input_channel = 32  
            last_channel = 1280 

            if inverted_residual_setting is None: 
                inverted_residual_setting = [ 
                    # t, c, n, s  
                    [1, 16, 1, 1],  
                    [6, 24, 2, 2],  
                    [6, 32, 3, 2],  
                    [6, 64, 4, 2],  
                    [6, 96, 3, 1],  
                    [6, 160, 3, 2], 
                    [6, 320, 1, 1], 
                ] 

            # only check the first element, assuming user knows t,c,n,s are required  
            if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4: 
                raise ValueError("inverted_residual_setting should be non-empty " 
                                 "or a 4-element list, got {}".format(inverted_residual_setting)) 

            # building first layer  
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)  
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest) 
            features = [ConvBNReLU(3, input_channel, stride=2)] 
            # building inverted residual blocks 
            for t, c, n, s in inverted_residual_setting:  
                output_channel = _make_divisible(c * width_mult, round_nearest) 
                for i in range(n):  
                    stride = s if i == 0 else 1 
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t)) 
                    input_channel = output_channel  
            # building last several layers  
            features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))  
            # make it nn.Sequential 
            self.features = nn.Sequential(*features)  
            self.quant = QuantStub()  
            self.dequant = DeQuantStub()  
            # building classifier 
            self.classifier = nn.Sequential(  
                nn.Dropout(0.2),  
                nn.Linear(self.last_channel, num_classes),  
            ) 

            # weight initialization 
            for m in self.modules():  
                if isinstance(m, nn.Conv2d):  
                    nn.init.kaiming_normal_(m.weight, mode='fan_out') 
                    if m.bias is not None:  
                        nn.init.zeros_(m.bias)  
                elif isinstance(m, nn.BatchNorm2d): 
                    nn.init.ones_(m.weight) 
                    nn.init.zeros_(m.bias)  
                elif isinstance(m, nn.Linear):  
                    nn.init.normal_(m.weight, 0, 0.01)  
                    nn.init.zeros_(m.bias)  

        def forward(self, x): 
            x = self.quant(x) 
            x = self.features(x)  
            x = x.mean([2, 3])  
            x = self.classifier(x)  
            x = self.dequant(x) 
            return x  

        # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization 
        # This operation does not change the numerics 
        def fuse_model(self, is_qat=False): 
            fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
            for m in self.modules():  
                if type(m) == ConvBNReLU: 
                    fuse_modules(m, ['0', '1', '2'], inplace=True)
                if type(m) == InvertedResidual: 
                    for idx in range(len(m.conv)):  
                        if type(m.conv[idx]) == nn.Conv2d:  
                            fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

2. Helper functions 
------------------- 
 
We next define several helper functions to help with model evaluation. These mostly come from 
`here <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`_. 

.. code:: python

    class AverageMeter(object): 
        """Computes and stores the average and current value""" 
        def __init__(self, name, fmt=':f'): 
            self.name = name  
            self.fmt = fmt  
            self.reset()  

        def reset(self):  
            self.val = 0  
            self.avg = 0  
            self.sum = 0  
            self.count = 0  

        def update(self, val, n=1): 
            self.val = val  
            self.sum += val * n 
            self.count += n 
            self.avg = self.sum / self.count  

        def __str__(self):  
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})' 
            return fmtstr.format(**self.__dict__) 


    def accuracy(output, target, topk=(1,)):  
        """Computes the accuracy over the k top predictions for the specified values of k"""  
        with torch.no_grad(): 
            maxk = max(topk)  
            batch_size = target.size(0) 

            _, pred = output.topk(maxk, 1, True, True)  
            pred = pred.t() 
            correct = pred.eq(target.view(1, -1).expand_as(pred)) 

            res = []  
            for k in topk:  
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  
                res.append(correct_k.mul_(100.0 / batch_size))  
            return res  


    def evaluate(model, criterion, data_loader, neval_batches): 
        model.eval()  
        top1 = AverageMeter('Acc@1', ':6.2f') 
        top5 = AverageMeter('Acc@5', ':6.2f') 
        cnt = 0 
        with torch.no_grad(): 
            for image, target in data_loader: 
                output = model(image) 
                loss = criterion(output, target)  
                cnt += 1  
                acc1, acc5 = accuracy(output, target, topk=(1, 5))  
                print('.', end = '')  
                top1.update(acc1[0], image.size(0)) 
                top5.update(acc5[0], image.size(0)) 
                if cnt >= neval_batches:  
                     return top1, top5  

        return top1, top5 

    def load_model(model_file): 
        model = MobileNetV2() 
        state_dict = torch.load(model_file) 
        model.load_state_dict(state_dict) 
        model.to('cpu') 
        return model  

    def print_size_of_model(model): 
        torch.save(model.state_dict(), "temp.p")  
        print('Size (MB):', os.path.getsize("temp.p")/1e6)  
        os.remove('temp.p') 

3. Define dataset and data loaders  
----------------------------------  
 
As our last major setup step, we define our dataloaders for our training and testing set. 
 
ImageNet Data 
^^^^^^^^^^^^^ 

To run the code in this tutorial using the entire ImageNet dataset, first download imagenet by following the instructions at here `ImageNet Data <http://www.image-net.org/download>`_. Unzip the downloaded file into the 'data_path' folder.

With the data downloaded, we show functions below that define dataloaders we'll use to read 
in this data. These functions mostly come from  
`here <https://github.com/pytorch/vision/blob/master/references/detection/train.py>`_.


.. code:: python

    def prepare_data_loaders(data_path):  
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                         std=[0.229, 0.224, 0.225])
        dataset = torchvision.datasets.ImageNet(
            data_path, split="train", transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        dataset_test = torchvision.datasets.ImageNet(
            data_path, split="val", transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = torch.utils.data.RandomSampler(dataset) 
        test_sampler = torch.utils.data.SequentialSampler(dataset_test) 

        data_loader = torch.utils.data.DataLoader(  
            dataset, batch_size=train_batch_size, 
            sampler=train_sampler)  

        data_loader_test = torch.utils.data.DataLoader( 
            dataset_test, batch_size=eval_batch_size, 
            sampler=test_sampler) 

        return data_loader, data_loader_test  


Next, we'll load in the pre-trained MobileNetV2 model. We provide the URL to download the model
`here <https://download.pytorch.org/models/mobilenet_v2-b0353104.pth>`_. 

.. code:: python

    data_path = '~/.data/imagenet'
    saved_model_dir = 'data/' 
    float_model_file = 'mobilenet_pretrained_float.pth' 
    scripted_float_model_file = 'mobilenet_quantization_scripted.pth' 
    scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth' 

    train_batch_size = 30 
    eval_batch_size = 50 

    data_loader, data_loader_test = prepare_data_loaders(data_path) 
    criterion = nn.CrossEntropyLoss() 
    float_model = load_model(saved_model_dir + float_model_file).to('cpu')  
 
    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access  
    # while also improving numerical accuracy. While this can be used with any model, this is 
    # especially common with quantized models.  

    print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv) 
    float_model.eval()  

    # Fuses modules 
    float_model.fuse_model()  

    # Note fusion of Conv+BN+Relu and Conv+Relu 
    print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)  

  
Finally to get a "baseline" accuracy, let's see the accuracy of our un-quantized model  
with fused modules  

.. code:: python

    num_eval_batches = 1000

    print("Size of baseline model") 
    print_size_of_model(float_model)  

    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches) 
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)) 
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

  
On the entire model, we get an accuracy of 71.9% on the eval dataset of 50,000 images.

This will be our baseline to compare to. Next, let's try different quantization methods 

4. Post-training static quantization  
------------------------------------  

Post-training static quantization involves not just converting the weights from float to int, 
as in dynamic quantization, but also performing the additional step of first feeding batches  
of data through the network and computing the resulting distributions of the different activations  
(specifically, this is done by inserting `observer` modules at different points that record this  
data). These distributions are then used to determine how the specifically the different activations  
should be quantized at inference time (a simple technique would be to simply divide the entire range  
of activations into 256 levels, but we support more sophisticated methods as well). Importantly,  
this additional step allows us to pass quantized values between operations instead of converting these  
values to floats - and then back to ints - between every operation, resulting in a significant speed-up.  

.. code:: python

    num_calibration_batches = 32

    myModel = load_model(saved_model_dir + float_model_file).to('cpu')  
    myModel.eval()  

    # Fuse Conv, bn and relu  
    myModel.fuse_model()  

    # Specify quantization configuration  
    # Start with simple min/max range estimation and per-tensor quantization of weights 
    myModel.qconfig = torch.ao.quantization.default_qconfig
    print(myModel.qconfig)  
    torch.ao.quantization.prepare(myModel, inplace=True)

    # Calibrate first 
    print('Post Training Quantization Prepare: Inserting Observers')  
    print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv) 

    # Calibrate with the training set 
    evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)  
    print('Post Training Quantization: Calibration done') 

    # Convert to quantized model  
    torch.ao.quantization.convert(myModel, inplace=True)
    # You may see a user warning about needing to calibrate the model. This warning can be safely ignored.
    # This warning occurs because not all modules are run in each model runs, so some
    # modules may not be calibrated.
    print('Post Training Quantization: Convert done') 
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv) 

    print("Size of model after quantization") 
    print_size_of_model(myModel)  

    top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches) 
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
  
For this quantized model, we see an accuracy of 56.7% on the eval dataset. This is because we used a simple min/max observer to determine quantization parameters. Nevertheless, we did reduce the size of our model down to just under 3.6 MB, almost a 4x decrease. 

In addition, we can significantly improve on the accuracy simply by using a different 
quantization configuration. We repeat the same exercise with the recommended configuration for  
quantizing for x86 architectures. This configuration does the following:  

- Quantizes weights on a per-channel basis  
- Uses a histogram observer that collects a histogram of activations and then picks 
  quantization parameters in an optimal manner. 

.. code:: python

    per_channel_quantized_model = load_model(saved_model_dir + float_model_file)  
    per_channel_quantized_model.eval()  
    per_channel_quantized_model.fuse_model()  
    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    print(per_channel_quantized_model.qconfig)  

    torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches) 
    torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)
    top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches) 
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)) 
    torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)


Changing just this quantization configuration method resulted in an increase  
of the accuracy to over 67.3%! Still, this is 4% worse than the baseline of 71.9% achieved above. 
So lets try quantization aware training.  

5. Quantization-aware training  
------------------------------  

Quantization-aware training (QAT) is the quantization method that typically results in the highest accuracy.  
With QAT, all weights and activations are “fake quantized” during both the forward and backward passes of 
training: that is, float values are rounded to mimic int8 values, but all computations are still done with  
floating point numbers. Thus, all the weight adjustments during training are made while “aware” of the fact 
that the model will ultimately be quantized; after quantizing, therefore, this method will usually yield  
higher accuracy than either dynamic quantization or post-training static quantization.  

The overall workflow for actually performing QAT is very similar to before: 

- We can use the same model as before: there is no additional preparation needed for quantization-aware 
  training. 
- We need to use a ``qconfig`` specifying what kind of fake-quantization is to be inserted after weights  
  and activations, instead of specifying observers  

We first define a training function:  

.. code:: python

    def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):  
        model.train() 
        top1 = AverageMeter('Acc@1', ':6.2f') 
        top5 = AverageMeter('Acc@5', ':6.2f') 
        avgloss = AverageMeter('Loss', '1.5f')  

        cnt = 0 
        for image, target in data_loader: 
            start_time = time.time()  
            print('.', end = '')  
            cnt += 1  
            image, target = image.to(device), target.to(device) 
            output = model(image) 
            loss = criterion(output, target)  
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()  
            acc1, acc5 = accuracy(output, target, topk=(1, 5))  
            top1.update(acc1[0], image.size(0)) 
            top5.update(acc5[0], image.size(0)) 
            avgloss.update(loss, image.size(0)) 
            if cnt >= ntrain_batches: 
                print('Loss', avgloss.avg)  

                print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}' 
                      .format(top1=top1, top5=top5))  
                return  

        print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}' 
              .format(top1=top1, top5=top5))  
        return  

  
We fuse modules as before 

.. code:: python

    qat_model = load_model(saved_model_dir + float_model_file)  
    qat_model.fuse_model(is_qat=True)  

    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001) 
    # The old 'fbgemm' is still available but 'x86' is the recommended default. 
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
  
Finally, ``prepare_qat`` performs the "fake quantization", preparing the model for quantization-aware training

.. code:: python

    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)
  
Training a quantized model with high accuracy requires accurate modeling of numerics at 
inference. For quantization aware training, therefore, we modify the training loop by:  

- Switch batch norm to use running mean and variance towards the end of training to better  
  match inference numerics. 
- We also freeze the quantizer parameters (scale and zero-point) and fine tune the weights. 

.. code:: python

    num_train_batches = 20  

    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch 
    for nepoch in range(8): 
        train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches) 
        if nepoch > 3:  
            # Freeze quantizer parameters 
            qat_model.apply(torch.ao.quantization.disable_observer)
        if nepoch > 2:  
            # Freeze batch norm mean and variance estimates 
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats) 

        # Check the accuracy after each epoch 
        quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()  
        top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)  
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg)) 
 
Quantization-aware training yields an accuracy of over 71.5% on the entire imagenet dataset, which is close to the floating point accuracy of 71.9%. 

More on quantization-aware training:  

- QAT is a super-set of post training quant techniques that allows for more debugging.  
  For example, we can analyze if the accuracy of the model is limited by weight or activation 
  quantization. 
- We can also simulate the accuracy of a quantized model in floating point since  
  we are using fake-quantization to model the numerics of actual quantized arithmetic.  
- We can mimic post training quantization easily too. 

Speedup from quantization 
^^^^^^^^^^^^^^^^^^^^^^^^^ 

Finally, let's confirm something we alluded to above: do our quantized models actually perform inference  
faster? Let's test: 

.. code:: python

    def run_benchmark(model_file, img_loader):  
        elapsed = 0 
        model = torch.jit.load(model_file)  
        model.eval()  
        num_batches = 5 
        # Run the scripted model on a few batches of images 
        for i, (images, target) in enumerate(img_loader): 
            if i < num_batches: 
                start = time.time() 
                output = model(images)  
                end = time.time() 
                elapsed = elapsed + (end-start) 
            else: 
                break 
        num_images = images.size()[0] * num_batches 

        print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000)) 
        return elapsed  

    run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)  

    run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)  

Running this locally on a MacBook pro yielded 61 ms for the regular model, and  
just 20 ms for the quantized model, illustrating the typical 2-4x speedup 
we see for quantized models compared to floating point ones.  

Conclusion  
----------  

In this tutorial, we showed two quantization methods - post-training static quantization, 
and quantization-aware training - describing what they do "under the hood" and how to use 
them in PyTorch.  

Thanks for reading! As always, we welcome any feedback, so please create an issue 
`here <https://github.com/pytorch/pytorch/issues>`_ if you have any.
