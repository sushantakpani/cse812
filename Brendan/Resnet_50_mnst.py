# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:57:04 2020

@author: bdhan
"""
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import numpy as np

### CRedit for synthetic mnist data set from: 
### https://github.com/shubham0204/Synthetic_MNIST_Classification
tainImages_x = np.load("./data/x.npy")
tainImages_y = np.load("./data/y.npy")

test_x = np.load("./data/test_x.npy")
test_y = np.load("./data/test_x.npy")

## This guide was used as a baisis for preparing the transfer learning:
## https://missinglink.ai/guides/pytorch/pytorch-resnet-building-training-scaling-residual-networks-pytorch/
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## TODO add Dataset either implemnation or via tensors torch.utils.data.Dataset 
for img in tainImages_x:
    tempImage = Image.fromarray(np.uint8(img*255))
    input_tensor = preprocess(tempImage)
input_batch = input_tensor.unsqueeze(0)

#inport model
model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
#update length of output
num_model_features = model.fc.in_features
model.fc = torch.nn.Linear(num_model_features,9)

# train via transfer learning

with torch.no_grad():
    output = model(input_batch)
print(output[0])
