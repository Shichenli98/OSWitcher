import torch
from fft_conv_pytorch import fft_conv, FFTConv2d
import time
import matplotlib.pyplot as plt

def hard_convert_opt_conv2d(module, threshold):
    module_output = module
    for i, layer in enumerate(list(module.children())):
        if isinstance(layer, torch.nn.Conv2d):
            if layer.kernel_size[0] > threshold:
                module_output[i] = FFTConv2d(in_channels=layer.in_channels, 
                                             out_channels=layer.out_channels,
                                             kernel_size=layer.kernel_size, 
                                             stride=layer.stride,
                                             padding=layer.padding,
                                             padding_mode=layer.padding_mode,
                                             dilation=layer.dilation,
                                             groups=layer.groups,
                                             bias=True if layer.bias is not None else False)
    del module
    return module_output

def dynamic_fit_conv(kernel_size, input_width, input_height, input_channel=3, output_channel=2):
    signal = torch.randn(input_channel, input_width, input_height)
    kernel = torch.randn(output_channel, input_channel, kernel_size, kernel_size)
    bias = torch.randn(output_channel)
    
    test_torch_conv = torch.nn.Conv2d(input_channel, output_channel, kernel_size, bias=True)
    test_torch_conv.weight = torch.nn.Parameter(kernel)
    test_torch_conv.bias = torch.nn.Parameter(bias)
    
    test_fft_conv = FFTConv2d(input_channel, output_channel, kernel_size, bias=True)
    test_fft_conv.weight = torch.nn.Parameter(kernel)
    test_fft_conv.bias = torch.nn.Parameter(bias)
    
    iters = 16
    time0 = time.time()
    for _ in range(iters):
        out = test_torch_conv(signal)
    time1 = time.time()
    
    for _ in range(iters):
        out = test_fft_conv(signal)
    time2 = time.time()
    
    torch_time = (time1 - time0) / iters * 1000
    fft_time = (time2 - time1) / iters * 1000
    
    return 0 if torch_time < fft_time else 1


def dynamic_convert_opt_conv2d(module):
    module_output = module
    for i, layer in enumerate(list(module.children())):
        if isinstance(layer, torch.nn.Conv2d):
            ### how to calculate internal data sizes, given various layer types
            ### we must feed correct kernel_size, input_size to the 'dynamic_fit_conv' function
            threshold = dynamic_fit_conv(layer.kernel_size, )
            if layer.kernel_size[0] > threshold:
                module_output[i] = FFTConv2d(in_channels=layer.in_channels, 
                                             out_channels=layer.out_channels,
                                             kernel_size=layer.kernel_size, 
                                             stride=layer.stride,
                                             padding=layer.padding,
                                             padding_mode=layer.padding_mode,
                                             dilation=layer.dilation,
                                             groups=layer.groups,
                                             bias=True if layer.bias is not None else False)
    del module
    return module_output