import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.models import resnet50
from replknet import *
import time
dummy_input = datasets.CIFAR10(root='../dataset', train=False, download=True, transform=None)
print('mnist input shape',np.asarray(dummy_input[0][0]).shape)
#pack batch
batch  = np.stack([dummy_input[i][0] for i in range(len(dummy_input))],axis=0)
batch = np.transpose(batch, axes=(0,3,1,2) )
batch = batch.astype(np.float32)
batch = batch[:1000]
dummy_batch = torch.rand(batch.shape)
print('batch ', batch.shape)

model = create_RepLKNet31B(small_kernel_merged=False).to('cpu')
model.eval()
#print(model)
input_names = [ "actual_input_1" ]# + [ "learned_%d" % i for i in range() ]
output_names = [ "output1" ]
torch.onnx.export(model, dummy_batch, "test_net.onnx", verbose=False , input_names=input_names, output_names=output_names)

import onnx

# Load the ONNX model
model = onnx.load("test_net.onnx")

# Check that the model is well formed
#onnx.checker.check_model(model)

# Print a human readable representation of the graph
#print(onnx.helper.printable_graph(model.graph))

import onnxruntime as ort

ort_session = ort.InferenceSession("test_net.onnx")
print('exp start')
t0 = time.time()
outputs = ort_session.run(
    None,
    {"actual_input_1": batch },
)
t1 = time.time()
print('exp end', t1-t0)
print(outputs[0].shape)
