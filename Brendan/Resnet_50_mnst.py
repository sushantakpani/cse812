# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:57:04 2020

@author: bdhan
"""
from PIL import Image
import torch
from torchvision import datasets, transforms
import numpy as np
import time
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

#train and test functions f

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


### CRedit for synthetic mnist data set from: 
### https://github.com/shubham0204/Synthetic_MNIST_Classification
# tainImages_x = np.load("./data/x.npy")
# tainImages_y = np.load("./data/y.npy")

# test_x = np.load("./data/test_x.npy")
# test_y = np.load("./data/test_x.npy")

#constant hyper params taken from sushan
seed = 98528
batch_size = 1000
test_batch_size = 1000
lr =0.01
momentum = 0.5
epochs = 10
num_processes = 4
save_model = False
log_interval = 10

def main():
    ## This guide was used as a baisis for preparing the transfer learning:
    ## https://missinglink.ai/guides/pytorch/pytorch-resnet-building-training-scaling-residual-networks-pytorch/
    #preprocess = transforms.Compose([
    #    transforms.Resize(256),
    #    transforms.CenterCrop(224),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])
    
    #proccessing from sushan
    use_cuda = not True and torch.cuda.is_available()
    
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../TempData/data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Grayscale(num_output_channels=1),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../TempData/data', train=False, transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    
    ## TODO add Dataset either implemnation or via tensors torch.utils.data.Dataset 
    #for img in tainImages_x:
    #    tempImage = Image.fromarray(np.uint8(img*255))
    #    input_tensor = preprocess(tempImage)
    #input_batch = input_tensor.unsqueeze(0)
    
    #inport model
    model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=False).to(device)
    #update length of output
    num_model_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_model_features,10)
    
    ## Code used to reduce Resnet input sample from https://discuss.pytorch.org/t/transfer-learning-usage-with-different-input-size/20744/3
    first_conv_layer = torch.nn.Conv2d(1, 64, kernel_size=(5,5), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1 = first_conv_layer
    
    
    model.share_memory()  #Multi-processing step
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    # train via transfer learning
    t = time.time()
    for epoch in range(1, epochs + 1):
      # For using multi-processing
       processes = []
       for rank in range(num_processes):
         p = mp.Process(target=train, args=(log_interval, model, device, train_loader, optimizer, epoch))
         p.start()
         processes.append(p)
       if bool(epoch%2) == False:
           for p in processes:
             p.join()
    
      # For not using multi-processing
       #train(args, model, device, train_loader, optimizer, epoch)
       test(model, device, test_loader)
    
    print('Total Time taken in training and validation: '+ str(time.time() - t))
    
    if (save_model):
        torch.save(model.state_dict(), "mnist_resnet.pt")

if __name__ == '__main__':
    main()




