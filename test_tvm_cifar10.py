
import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torchvision
import torchvision.datasets as datasets
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
scripted_model = torch.jit.trace(model,batch).eval()

######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
input_name = "input0"
shape_list = [(input_name, batch.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.
from tvm.contrib import graph_executor

dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs
m.set_input(input_name, tvm.nd.array(batch))
# Execute
t0 = time.time()
m.run()
t1 = time.time()
print('exe time',t1-t0)
# Get outputs
tvm_output = m.get_output(0)
print('done')
print(tvm_output)
