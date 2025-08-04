from tqdm import tqdm
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from numpy.random import RandomState
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import glob
import copy

def minmax_normalize(img):
    img = np.array(img)
    minv = img.min()
    maxv = img.max()
    img = (img - minv) / (maxv - minv)
    img = torch.from_numpy(img).float()
    return img.unsqueeze(0)  # [1, 28, 28]

def prepare_mnist_data(DATA_ROOT = os.path.join('./MNIST_Data')):
    CLEAN_IMG_DIR = os.path.join(DATA_ROOT, 'clean')
    CSV_PATH = os.path.join(DATA_ROOT, 'clean.csv')
    """Download and prepare MNIST clean data if not already present"""
    clean_set = torchvision.datasets.MNIST(root='./temp', train=False, download=True)
    
    # Create directories
    os.makedirs(CLEAN_IMG_DIR, exist_ok=True)
    
    # Save images and create CSV
    rows = []
    for idx, (img, label) in tqdm(enumerate(clean_set), total=len(clean_set)):
        fname = f"img_{idx:05d}.png"
        img_path = os.path.join(CLEAN_IMG_DIR, fname)
        img.save(img_path)
        rows.append({'file': os.path.join('clean', fname), 'label': label})
    
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(rows)} test images to {CLEAN_IMG_DIR}")
    print(f"Saved CSV to {CSV_PATH}")

def prepare_fashionmnist_data(DATA_ROOT = os.path.join('./FashionMNIST_Data')):
    CLEAN_IMG_DIR = os.path.join(DATA_ROOT, 'clean')
    CSV_PATH = os.path.join(DATA_ROOT, 'clean.csv')
    """Download and prepare FashionMNIST clean data if not already present"""
    clean_set = torchvision.datasets.FashionMNIST(root='./temp', train=False, download=True)
    
    # Create directories
    os.makedirs(CLEAN_IMG_DIR, exist_ok=True)
    
    # Save images and create CSV
    rows = []
    for idx, (img, label) in tqdm(enumerate(clean_set), total=len(clean_set)):
        fname = f"img_{idx:05d}.png"
        img_path = os.path.join(CLEAN_IMG_DIR, fname)
        img.save(img_path)
        rows.append({'file': os.path.join('clean', fname), 'label': label})
    
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(rows)} test images to {CLEAN_IMG_DIR}")
    print(f"Saved CSV to {CSV_PATH}")

def prepare_cifar10_data(DATA_ROOT = os.path.join('./CIFAR10_Data')):
    CLEAN_IMG_DIR = os.path.join(DATA_ROOT, 'clean')
    CSV_PATH = os.path.join(DATA_ROOT, 'clean.csv')
    """Download and prepare CIFAR10 clean data if not already present"""
    clean_set = torchvision.datasets.CIFAR10(root='./temp', train=False, download=True)
    
    # Create directories
    os.makedirs(CLEAN_IMG_DIR, exist_ok=True)
    
    # Save images and create CSV
    rows = []
    for idx, (img, label) in tqdm(enumerate(clean_set), total=len(clean_set)):
        fname = f"img_{idx:05d}.png"
        img_path = os.path.join(CLEAN_IMG_DIR, fname)
        img.save(img_path)
        rows.append({'file': os.path.join('clean', fname), 'label': label})
    
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(rows)} test images to {CLEAN_IMG_DIR}")
    print(f"Saved CSV to {CSV_PATH}")

# Transform functions for different datasets
def get_mnist_transforms():
    """Get transforms for MNIST dataset"""
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform_train, transform_test

def get_fashionmnist_transforms():
    """Get transforms for FashionMNIST dataset based on reference code"""
    transform_train = transforms.Compose([
        transforms.Pad(2),
        # transforms.ToTensor(),  # Not used in reference, min-max normalization is applied instead
    ])
    
    transform_test = transforms.Compose([
        transforms.Pad(2),
        # transforms.ToTensor(),  # Not used in reference, min-max normalization is applied instead
    ])
    
    return transform_train, transform_test

def get_cifar10_transforms():
    """Get transforms for CIFAR10 dataset based on reference code"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return transform_train, transform_test

class SimpleMNISTDataset(Dataset):
    """Docstring for SimpleDataset"""
    def __init__(self, path_to_data: str, csv_filename:str, true_label=False, path_to_csv=None, num_class=10,shuffle=False,  data_transform=lambda x: x, label_transform=lambda l: l,expand_dim=False):
        super(SimpleMNISTDataset, self).__init__()
        self.data_path=path_to_data
        self.dexpand_dim=expand_dim
        
        csv_path=os.path.join(self.data_path, csv_filename)
        self.data_df = pd.read_csv(csv_path)
        self.data = self.data_df['file']
        
        self.True_label=self.data_df['label']
        self.train_label= copy.deepcopy(self.True_label)#self.data_df['label']
        
        self.num_class=num_class
        
        #self.label = 'train_label'
        #if true_label:
        #    self.label = 'true_label'
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img=np.array(Image.open(os.path.join(self.data_path,self.data[index])))
        if self.data_transform:
            img = self.data_transform(img)
        label=self.True_label[index]
        label = self.label_transform(label)
        train_label=self.train_label[index]
        train_label=self.label_transform(train_label)
        return img, label,train_label


    def __len__(self):
        return len(self.data_df)
    
    def balanced_batch_trigger(self,smplpercls=40):#,dexpand_dim=False):
        images=[]
        labels=[]
        train_labels=[]
        counter=np.zeros(self.num_class)
        for i in range(len(self.data)):
            lbl=self.True_label[i]
            if counter[lbl]!=smplpercls:
                img= np.array(Image.open(os.path.join(self.data_path,self.data[i])))#np.array(cv2.imread(os.path.join(self.data_path,self.data[i]),cv2.IMREAD_UNCHANGED))                
                if self.data_transform:
                    img = self.data_transform(img)
                images.append(img)
                labels.append(lbl)
                train_labels.append(self.train_label[i])
                counter[lbl]=counter[lbl]+1
                
        images=torch.stack(images)
        s=images.size()
        
        images_min=torch.min(images.view(-1,s[1]*s[2]*s[3]),dim=1)
        images_max=torch.max(images.view(-1,s[1]*s[2]*s[3]),dim=1)
        
        labels=torch.from_numpy(np.array(labels))
        train_labels=torch.from_numpy(np.array(train_labels))
        
        return images, labels,train_labels, images_min,images_max

class SimpleFashionMnistDataset(Dataset):
    """Docstring for SimpleDataset"""

    def __init__(self, path_to_data: str, csv_filename: str, true_label=False, path_to_csv=None, num_class=10,
                 shuffle=False, data_transform=lambda x: x, label_transform=lambda l: l, expand_dim=False):
        super(SimpleFashionMnistDataset, self).__init__()
        self.data_path = path_to_data
        self.dexpand_dim = expand_dim
        csv_path = os.path.join(self.data_path, csv_filename)
        self.data_df = pd.read_csv(csv_path)
        self.data = self.data_df['file']

        self.True_label = self.data_df['label']
        self.train_label = copy.deepcopy(self.True_label)  # self.data_df['label']

        self.num_class = num_class

        # self.label = 'train_label'
        # if true_label:
        #    self.label = 'true_label'
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.data[index]))
        if self.data_transform:
            img = self.data_transform(img)

        img=np.array(img)
        min = np.amin(img, axis=(0, 1), keepdims=True)
        max = np.amax(img, axis=(0, 1), keepdims=True)
        img = (img - min) / (max - min)
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)


        # label = self.data_df.iloc[index][self.label]
        label = self.True_label[index]
        label = self.label_transform(label)
        train_label = self.train_label[index]
        train_label = self.label_transform(train_label)

        return img, label, train_label

    def __len__(self):
        return len(self.data_df)

    def balanced_batch_trigger(self, smplpercls=40): 
        images = []
        labels = []
        train_labels = []
        counter = np.zeros(self.num_class)
        for i in range(len(self.data)):
            lbl = self.True_label[i]
            if counter[lbl] != smplpercls:
                img = Image.open(os.path.join(self.data_path, self.data[
                    i]))  # np.array(cv2.imread(os.path.join(self.data_path,self.data[i]),cv2.IMREAD_UNCHANGED))

                if self.data_transform:
                    img = self.data_transform(img)

                img=np.array(img)
                min = np.amin(img, axis=(0, 1), keepdims=True)
                max = np.amax(img, axis=(0, 1), keepdims=True)
                img = (img - min) / (max - min)
                img = torch.from_numpy(img).float()
                img = img.unsqueeze(0)

                images.append(img)
                labels.append(lbl)
                train_labels.append(self.train_label[i])
                counter[lbl] = counter[lbl] + 1

        images = torch.stack(images,dim=0)
        
        shp = images.size()
        images_min, _ = torch.min(images.view(shp[0], -1), dim=(1), keepdim=True)
        images_min = images_min.view((shp[0], 1, 1, 1))
        images_max, _ = torch.max(images.view(shp[0], -1), dim=(1), keepdim=True)
        images_max = images_max.view((shp[0], 1, 1, 1))

        labels = torch.from_numpy(np.array(labels))
        train_labels = torch.from_numpy(np.array(train_labels))

        return images, labels, train_labels, images_min, images_max




class SimpleCIFAR10Dataset(Dataset):
    """Docstring for SimpleDataset"""

    def __init__(self, path_to_data: str, csv_filename: str, true_label=False, path_to_csv=None, num_class=10,
                 shuffle=False, data_transform=lambda x: x, label_transform=lambda l: l, expand_dim=False):
        super(SimpleCIFAR10Dataset, self).__init__()
        self.data_path = path_to_data
        self.dexpand_dim = expand_dim
        csv_path = os.path.join(self.data_path, csv_filename)
        self.data_df = pd.read_csv(csv_path)
        self.data = self.data_df['file']

        self.True_label = self.data_df['label']
        self.train_label = copy.deepcopy(self.True_label)  # self.data_df['label']

        self.num_class = num_class

        # self.label = 'train_label'
        # if true_label:
        #    self.label = 'true_label'
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img = np.array(Image.open(os.path.join(self.data_path, self.data[index])))
        
        if self.data_transform:
            img = self.data_transform(img)
        # label = self.data_df.iloc[index][self.label]
        label = self.True_label[index]
        label = self.label_transform(label)
        train_label = self.train_label[index]
        train_label = self.label_transform(train_label)

        return img, label, train_label

    def __len__(self):
        return len(self.data_df)

    def balanced_batch_trigger(self, smplpercls=40):  # ,dexpand_dim=False):
        images = []
        labels = []
        train_labels = []
        counter = np.zeros(self.num_class)
        for i in range(len(self.data)):
            lbl = self.True_label[i]
            if counter[lbl] != smplpercls:
                img = np.array(Image.open(os.path.join(self.data_path, self.data[
                    i])))  # np.array(cv2.imread(os.path.join(self.data_path,self.data[i]),cv2.IMREAD_UNCHANGED))
                img=self.data_transform(img)
                images.append(img)
                labels.append(lbl)
                train_labels.append(self.train_label[i])
                counter[lbl] = counter[lbl] + 1

        images = torch.stack(images,dim=0)
        
        shp = images.size()
        images_min, _ = torch.min(images.view(shp[0], -1), dim=(1), keepdim=True)
        images_min = images_min.view((shp[0], 1, 1, 1))
        images_max, _ = torch.max(images.view(shp[0], -1), dim=(1), keepdim=True)
        images_max = images_max.view((shp[0], 1, 1, 1))

        labels = torch.from_numpy(np.array(labels))
        train_labels = torch.from_numpy(np.array(train_labels))

        return images, labels, train_labels, images_min, images_max

    def balanced_batch_trigger_perclass(self, smplpercls=40, batch_lbl=0):  # ,dexpand_dim=False):
        images = []
        labels = []
        train_labels = []
        #print("current batch_lbl is:", batch_lbl)
        counter = 0  # np.zeros(self.num_class)
        for i in range(len(self.data)):
            lbl = self.True_label[i]
            if counter != smplpercls and lbl == batch_lbl:
                img = np.array(Image.open(os.path.join(self.data_path, self.data[
                    i])))  # np.array(cv2.imread(os.path.join(self.data_path,self.data[i]),cv2.IMREAD_UNCHANGED))
                img=self.data_transform(img)
                images.append(img)
                labels.append(lbl)
                train_labels.append(self.train_label[i])
                counter = counter + 1

        images=torch.stack(images, dim=0)
        
        shp=images.size()
        images_min,_= torch.min(images.view(shp[0],-1),dim=(1),keepdim=True)
        images_min=images_min.view((shp[0],1,1,1))
        images_max,_=torch.max(images.view(shp[0],-1),dim=(1),keepdim=True)
        images_max=images_max.view((shp[0],1,1,1))

        labels = torch.from_numpy(np.array(labels))
        train_labels = torch.from_numpy(np.array(train_labels))

        return images, labels, train_labels, images_min, images_max

