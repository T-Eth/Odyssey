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


def get_model_details(model_path):
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

import torch.nn as nn

def split_model_for_mask(model: nn.Module):
    """
    Returns (Sa, Sb)
      Sa: feature extractor + (GAP if applicable) + FLATTEN -> outputs [B, C]
      Sb: classifier head that expects [B, C] and returns logits
    """
    # Case 1: Your custom CNNs (convnet + fc). They flatten right before fc.
    if hasattr(model, 'convnet') and hasattr(model, 'fc'):
        Sa = nn.Sequential(
            model.convnet,
            nn.Flatten(start_dim=1)   # make Sa produce [B, C]
        )
        Sb = model.fc                # expects [B, C]
        return Sa, Sb

    # Case 2: ResNet-like (post-activation stem + layers + GAP + flatten)
    is_resnet_like = (
        hasattr(model, 'conv1') and hasattr(model, 'bn1')
        and hasattr(model, 'layer1') and hasattr(model, 'layer2')
        and hasattr(model, 'layer3') and hasattr(model, 'layer4')
        and (hasattr(model, 'linear') or hasattr(model, 'fc'))
    )
    if is_resnet_like:
        clf = getattr(model, 'linear', None) or getattr(model, 'fc')
        Sa = nn.Sequential(
            model.conv1,
            model.bn1,
            nn.ReLU(inplace=True),         # matches your ResNet.forward()
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),  # GAP to [B, C, 1, 1]
            nn.Flatten(start_dim=1)        # -> [B, C]
        )
        Sb = clf                           # expects [B, C]
        return Sa, Sb

    raise ValueError(f"Model type {type(model).__name__} not recognized")

def split_model_for_mask_1_back(model: nn.Module):
    """
    Returns (Sa, Sb, mask_mode)
      Sa: ends ONE layer earlier than the usual split (spatial feature map: [B,C,H,W])
      Sb: the 'rest' (final conv stage + GAP/flatten + classifier)
      mask_mode: "channel"  -> mask is per-channel and broadcast over H,W
    Supported:
      - ResNet-like (conv1,bn1,layer1..layer4, linear/fc)
      - Custom models with .convnet (nn.Sequential) + .fc
    """
    # ----- Case A: ResNet-like -----
    is_resnet_like = (
        hasattr(model, 'conv1') and hasattr(model, 'bn1')
        and hasattr(model, 'layer1') and hasattr(model, 'layer2')
        and hasattr(model, 'layer3') and hasattr(model, 'layer4')
        and (hasattr(model, 'linear') or hasattr(model, 'fc'))
    )
    if is_resnet_like:
        clf = getattr(model, 'linear', None) or getattr(model, 'fc')
        Sa = nn.Sequential(
            model.conv1,
            model.bn1,
            nn.ReLU(inplace=True),
            model.layer1,
            model.layer2,
            model.layer3,           # <-- stop ONE BLOCK earlier than usual
        )
        Sb = nn.Sequential(
            model.layer4,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(start_dim=1),
            clf
        )
        # Sb now expects spatial input [B,C,H,W]
        Sb.expects_spatial = True
        return Sa, Sb, "channel"

    # ----- Case B: convnet + fc -----
    if hasattr(model, 'convnet') and isinstance(model.convnet, nn.Sequential) and hasattr(model, 'fc'):
        convmods = list(model.convnet.children())
        # find the LAST Conv2d in convnet — that (and everything after) goes into Sb
        last_conv_idx = max(i for i, m in enumerate(convmods) if isinstance(m, nn.Conv2d))

        Sa = nn.Sequential(*convmods[:last_conv_idx])         # one layer back
        Sb = nn.Sequential(
            *convmods[last_conv_idx:],                        # final conv + tails
            nn.Flatten(start_dim=1),
            model.fc
        )
        Sb.expects_spatial = True
        return Sa, Sb, "channel"

    raise ValueError(f"Model type {type(model).__name__} not recognized for 1-back split")

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