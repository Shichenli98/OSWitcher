import torch
from fft_conv_pytorch import fft_conv, FFTConv1d, FFTConv2d
import time
import copy
#import matplotlib.pyplot as plt

def hard_convert_opt_conv2d(module, threshold):
    module_output = copy.deepcopy(module)
    for i, layer in enumerate(list(module.children())):
        if isinstance(layer, torch.nn.Conv2d):
            if layer.kernel_size[0] > threshold:
                pm = layer.padding_mode
                if pm == 'zeros':
                    pm = 'constant'
                module_output[i] = FFTConv2d(in_channels=layer.in_channels,
                                             out_channels=layer.out_channels,
                                             kernel_size=layer.kernel_size,
                                             stride=layer.stride,
                                             padding=layer.padding,
                                             padding_mode=pm,
                                             dilation=layer.dilation,
                                             groups=layer.groups,
                                             bias=True if layer.bias is not None else False)
    #del module
    return module_output

myNet = torch.nn.Sequential(
    torch.nn.Conv2d(3,20,128),
    #torch.nn.ReLU(),
    #torch.nn.Conv2d(20,64,3),
    #torch.nn.ReLU()
)

rand = torch.rand(1,3,1024,1024)

start_time = time.time()
for _ in range(10):
    res = myNet(rand)
end_time = time.time()
print('baseline time:', (end_time-start_time)/10)

myNet_opt = hard_convert_opt_conv2d(myNet, 20)

start_time = time.time()
for _ in range(10):
    res = myNet_opt(rand)
end_time = time.time()
print('opt time:', (end_time-start_time)/10)
