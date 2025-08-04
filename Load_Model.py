import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import datasets, transforms
import copy
import pickle
import numpy as np
import glob
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from os import path
import math
from Cifar10_models import *


def model_details(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    
    metadata = {}
    
    for key, value in checkpoint.items():
        # Skip big model weight tensors or nested dicts that hold weights
        if isinstance(value, dict) or torch.is_tensor(value):
            continue
        
        metadata[key] = value
    
    return metadata

# model loader for mnist models
def load_mnist_model(model_path, device,num_class=10):
    checkpoint = torch.load(model_path)
    print("keys are :", checkpoint.keys())

    model = checkpoint['Architecture_Name']

    # Get the model
    print('==> Building model..')
    if model == 'Model_Google_1':
        net = Model_Google_1()
    elif model == 'Model_Google_2':
        net = Model_Google_2()
    elif model == 'Model_Google_3':
        net = Model_Google_3()
    elif model == 'Model_Google_4':
        net = Model_Google_4()
    
    net = net.to(device)
    net.load_state_dict(checkpoint['net'])

    if 'test_clean_acc' in checkpoint:
        best_acc_clean = checkpoint['test_clean_acc']
        print("The Accuracies on clean samples:  ", best_acc_clean)
    if 'test_trigerred_acc' in checkpoint:
        best_acc_trig = checkpoint['test_trigerred_acc']
        print("The fooling rate: ", best_acc_trig)
    if 'Mapping' in checkpoint:
        mapping = checkpoint['Mapping']
        print("Mapping is : ",mapping, type(mapping))
        if isinstance(mapping,int) or isinstance(mapping,np.float64):
            mapping=mapping*np.ones(num_class,dtype=float)
        elif isinstance(mapping,str):
            if mapping =='N/A':
                mapping = None
    else:
        mapping = None
    return net, mapping  # checkpoint['Mapping']


# model loader for Fashion_MNIST and Cifar10 models
def load_model(model_path, device,num_class=10):
    print('model path ',model_path)
    checkpoint = torch.load(model_path)
    print("keys are :", checkpoint.keys())
   
    model = checkpoint['Architecture_Name']

    # Get the model
    print('==> Building model..')
    if model == 'Vgg19':
        net = VGG('VGG19')
    elif model == 'Resnet18':
        net = ResNet18()
    elif model == 'PreActResNet18':
        net = PreActResNet18()
    elif model == 'GoogleNet':
        net = GoogLeNet()
    elif model == 'DenseNet':
        net = DenseNet121()
    elif model == 'MobileNet':
        net = MobileNet()
    elif model == 'DPN92':
        net = DPN92()
    elif model == 'ShuffleNet':
        net = ShuffleNetG2()
    elif model == 'SENet':
        net = SENet18()
    elif model == 'EfficientNet':
        net = EfficientNetB0()
    else:
        net = MobileNetV2()

    net = net.to(device)
    

    net.load_state_dict(checkpoint['net'])

    if 'test_clean_acc' in checkpoint:
        best_acc_clean = checkpoint['test_clean_acc']
        print("The Accuracies on clean samples:  ", best_acc_clean)
    if 'test_trigerred_acc' in checkpoint:
        best_acc_trig = checkpoint['test_trigerred_acc']
        print("The fooling rate: ", best_acc_trig)
    if 'Mapping' in checkpoint:
        mapping = checkpoint['Mapping']
        print("Mapping is : ",mapping)
        if isinstance(mapping,int):
            mapping=mapping*np.ones(num_class,dtype=float)
        elif isinstance(mapping,str):
            if mapping =='N/A':
                mapping = None
    else:
        mapping = None
    return net, mapping  # checkpoint['Mapping']


class Model_Google_1(nn.Module):
    """
    A modified LeNet architecture that seems to be easier to embed backdoors in than the network from the original
    badnets paper
    Input - (1 or 3)x28x28
    C1 - 6@28x28 (5x5 kernel)
    ReLU
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    ReLU
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    ReLU
    F7 - 10 (Output)
    """

    def __init__(self, channels=1):
        super(Model_Google_1, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(), nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return F.log_softmax(output)

class Model_Google_2(nn.Module):
    """
    A modified LeNet architecture that seems to be easier to embed backdoors in than the network from the original
    badnets paper
    Input - (1 or 3)x28x28
    C1 - 6@28x28 (5x5 kernel)
    ReLU
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    ReLU
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    ReLU
    F7 - 10 (Output)
    """

    def __init__(self, channels=1):
        super(Model_Google_2, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.fc = nn.Sequential(
            nn.Linear(1152, 128),
            nn.ReLU(), nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, 10),

        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return F.log_softmax(output)


class Model_Google_3(nn.Module):
    """
    A modified LeNet architecture that seems to be easier to embed backdoors in than the network from the original
    badnets paper
    Input - (1 or 3)x28x28
    C1 - 6@28x28 (5x5 kernel)
    ReLU
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    ReLU
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    ReLU
    F7 - 10 (Output)
    """

    def __init__(self, channels=1):
        super(Model_Google_3, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 10),

        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return F.log_softmax(output)

class Model_Google_4(nn.Module):
    """
    A modified LeNet architecture that seems to be easier to embed backdoors in than the network from the original
    badnets paper
    Input - (1 or 3)x28x28
    C1 - 6@28x28 (5x5 kernel)
    ReLU
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    ReLU
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    ReLU
    F7 - 10 (Output)
    """

    def __init__(self, channels=1):
        super(Model_Google_4, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.Conv2d(128, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(288, 128),
            nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 10),

        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return F.log_softmax(output)


def test_model_performance(model, test_loader, device, model_name):
    """
    Test model performance on the test dataset
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(tqdm(test_loader, desc=f"Testing {model_name}")):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            # For models that return log_softmax, we need to get the predicted class
            if isinstance(outputs, torch.Tensor) and outputs.dim() == 2:
                pred = outputs.argmax(dim=1, keepdim=True)
            else:
                # Handle case where model might return tuple or different format
                pred = outputs.argmax(dim=1, keepdim=True)
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"{model_name} Test Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return accuracy