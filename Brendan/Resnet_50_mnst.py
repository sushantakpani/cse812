# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:57:04 2020

@author: bdhan
"""


import torch
import torchvision
import numpy as np

### CRedit for synthetic mnist data set from: 
### https://github.com/shubham0204/Synthetic_MNIST_Classification
tainImages_x = np.load("./data/x.npy")
tainImages_y = np.load("./data/y.npy")

test_x = np.load("./data/test_x.npy")
test_y = np.load("./data/test_x.npy")

## This guide was used as a baisis for preparing the transfer learning:
## https://missinglink.ai/guides/pytorch/pytorch-resnet-building-training-scaling-residual-networks-pytorch/

input_tensor = preprocess(tainImages_x[0])
input_batch = input_tensor.unsqueeze(0)
model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
#num_model_features = model.fc.in_features
#model.fc = torch.nn.Linear(num_model_features,9)
#
#model.train(tainImages_x,tainImages_y))
with torch.no_grad():
    output = model(input_batch)
print(output[0])
