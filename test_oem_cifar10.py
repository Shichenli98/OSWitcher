
import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torchvision
import torchvision.datasets as datasets
'''
######################################################################
# Load a pretrained PyTorch model
# -------------------------------
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

######################################################################
# Load a test image
# -----------------
# Classic cat example!
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)
'''
#batch = torch.rand((10,3,100,100))
#model = torch.nn.Sequential(
#          torch.nn.Conv2d(3,5,5)
#        )
#model = model.eval()
from replknet import *
import time
import numpy as np
dummy_input = datasets.CIFAR10(root='../dataset', train=False, download=True, transform=None)
print('mnist input shape',np.asarray(dummy_input[0][0]).shape)
#pack batch
batch  = np.stack([dummy_input[i][0] for i in range(len(dummy_input))],axis=0)
batch = np.transpose(batch, axes=(0,3,1,2) )
batch = batch.astype(np.float32)
batch = batch[:1000]
batch = torch.tensor(batch)
#dummy_batch = torch.rand(batch.shape)
print('batch ', batch.shape)

model = create_RepLKNet31B(small_kernel_merged=False).to('cpu')
model.eval()
# Execute
t0 = time.time()
out = model(batch)
t1 = time.time()
print('exe time',t1-t0)
# Get outputs
#tvm_output = m.get_output(0)
print('done')
#print(tvm_output)
