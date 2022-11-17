import torch
from fft_conv_pytorch import fft_conv, FFTConv2d
import time
import matplotlib.pyplot as plt
from torchsummary import summary

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

### dynamically decide the better operator for convolution, given parameters
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
    
    # 0: torch conv is better; 1: fft conv is better
    return 0 if torch_time < fft_time else 1

### dynamically convert current model to optimized model, accelerating convolution
def dynamic_convert_opt_conv2d(module, org_in_channels, org_in_width, org_in_height):
    # compute input width, height per layer
    ####
    mySummary = summary(module, (org_in_channels, org_in_width, org_in_height)).summary_list
    module_output = module
    for i, layer in enumerate(list(module.children())):
        if isinstance(layer, torch.nn.Conv2d):
            ### fetch internal data sizes
            ### we must feed correct kernel_size, input_size to the 'dynamic_fit_conv' function
            if i == 0:
                input_width, input_height = org_in_width, org_in_height
            else:
                input_width, input_height = mySummary[i-1].ouput_size[-2:]
            threshold = dynamic_fit_conv(kernel_size=layer.kernel_size, 
                                         input_width=input_width, input_height=input_height, 
                                         input_channel=layer.in_channels, output_channel=layer.out_channels)
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