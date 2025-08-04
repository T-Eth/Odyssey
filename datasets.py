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
import cv2
import shutil
from random import randrange

import cv2
import sys
import os
sys.path.insert(1, os.path.join('Odysseus', 'Model Creation'))
sys.path.insert(1, os.path.join('Odysseus', 'Model Creation', 'trojai'))

# Import trojai components for trigger generation  
try:
    import trojai.datagen.insert_merges as tdi
    import trojai.datagen.datatype_xforms as tdd
    import trojai.datagen.image_triggers as tdt
    import trojai.datagen.config as tdc
    import trojai.datagen.xform_merge_pipeline as tdx
    from trojai.datagen.image_entity import GenericImageEntity
    from trojai.datagen.xform_merge_pipeline import XFormMerge
    from sklearn.model_selection import train_test_split
    TROJAI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import trojai components: {e}")
    TROJAI_AVAILABLE = False

def minmax_normalize(img):
    img = np.array(img)
    minv = img.min()
    maxv = img.max()
    img = (img - minv) / (maxv - minv)
    img = torch.from_numpy(img).float()
    return img.unsqueeze(0)  # [1, 28, 28]

def prepare_mnist_data(DATA_ROOT = os.path.join('./MNIST_Data')):
    CLEAN_IMG_DIR = os.path.join(DATA_ROOT, "clean")
    CSV_PATH = os.path.join(CLEAN_IMG_DIR, 'clean.csv')
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
        rows.append({'file': os.path.join(fname), 'label': label})
    
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(rows)} test images to {CLEAN_IMG_DIR}")
    print(f"Saved CSV to {CSV_PATH}")

def prepare_fashionmnist_data(DATA_ROOT = os.path.join('./FashionMNIST_Data')):
    CLEAN_IMG_DIR = os.path.join(DATA_ROOT, 'clean')
    CSV_PATH = os.path.join(CLEAN_IMG_DIR, 'clean.csv')
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
        rows.append({'file': os.path.join(fname), 'label': label})
    
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(rows)} test images to {CLEAN_IMG_DIR}")
    print(f"Saved CSV to {CSV_PATH}")

def prepare_cifar10_data(DATA_ROOT = os.path.join('./CIFAR10_Data')):
    CLEAN_IMG_DIR = os.path.join(DATA_ROOT, 'clean')
    CSV_PATH = os.path.join(CLEAN_IMG_DIR, 'clean.csv')
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
        rows.append({'file': os.path.join(fname), 'label': label})
    
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

def generate_triggered_dataset(model_details, trigger_percentage=None, output_base_dir="triggered_datasets"):
    """
    Generate a triggered dataset based on model details, replicating the exact 
    trigger generation process used during training.
    
    Args:
        model_details (dict): Dictionary containing model metadata from checkpoint
        trigger_percentage (float, optional): Percentage of images to trigger (0-1). 
                                            If None, uses the value from model_details
        output_base_dir (str): Base directory to save triggered datasets
    
    Returns:
        str: Path to the generated dataset directory
    """
    
    if not TROJAI_AVAILABLE:
        raise ImportError("Trojai components are required but not available. Please check your environment setup.")
        
    # Check required fields in model_details
    required_fields = ['Dataset', 'Trigger type', 'Architecture_Name']
    missing_fields = [field for field in required_fields if field not in model_details]
    if missing_fields:
        raise ValueError(f"Missing required fields in model_details: {missing_fields}")
    
    # Extract necessary information from model details
    dataset_type = model_details.get('Dataset', '').upper()
    trigger_type = model_details.get('Trigger type', '')
    trigger_size = model_details.get('Trigger Size', [5, 5])
    mapping_type = model_details.get('Mapping Type', '')
    mapping_values = model_details.get('Mapping')
    model_name = model_details.get('Architecture_Name', 'unknown_model')
    
    # Use provided trigger percentage or get from model details
    if trigger_percentage is None:
        trigger_percentage = model_details.get('trigger_fraction', 0.25)
    
    print(f"Generating triggered dataset for {model_name} with {dataset_type}")
    print(f"Trigger type: {trigger_type}, Trigger percentage: {trigger_percentage}")
    
    # Determine dataset paths and setup
    if dataset_type == 'MNIST':
        clean_data_dir = './MNIST_Data/clean'
        clean_csv = 'clean.csv'
        dataset_suffix = 'MNIST'
        prepare_data_func = prepare_mnist_data
        num_channels = 1
    elif dataset_type == 'FASHIONMNIST':
        clean_data_dir = './FashionMNIST_Data/clean'
        clean_csv = 'clean.csv'
        dataset_suffix = 'FMNIST'
        prepare_data_func = prepare_fashionmnist_data
        num_channels = 1
    elif dataset_type == 'CIFAR10':
        clean_data_dir = './CIFAR10_Data/clean'
        clean_csv = 'clean.csv'
        dataset_suffix = 'CIFAR10'
        prepare_data_func = prepare_cifar10_data
        num_channels = 3
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # Ensure clean data exists
    if not os.path.exists(os.path.join(clean_data_dir, clean_csv)):
        print(f"Clean data not found. Preparing {dataset_type} data...")
        prepare_data_func()
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, f"{model_name}_{dataset_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the trigger pattern based on trigger type
    trigger_selection = _create_trigger_pattern(trigger_type, trigger_size, num_channels)
    
    # Setup trojai configuration for trigger generation
    trigger_cfg = tdc.XFormMergePipelineConfig(
        trigger_list=[trigger_selection],
        trigger_sampling_prob=None,
        trigger_xforms=[],
        trigger_bg_xforms=[tdd.ToTensorXForm()],
        trigger_bg_merge=tdi.InsertAtRandomLocation('uniform_random_available', tdc.ValidInsertLocationsConfig()),
        trigger_bg_merge_xforms=[],
        merge_type='insert',
        per_class_trigger_frac=trigger_percentage,
        triggered_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    
    # Load clean dataset
    clean_df = pd.read_csv(os.path.join(clean_data_dir, clean_csv))
    
    # Create mapping function based on mapping type
    mapping_func = _create_mapping_function(mapping_type, mapping_values)
    
    # Generate triggered dataset
    _generate_triggered_images_and_csv(
        clean_df, clean_data_dir, output_dir, trigger_cfg, 
        mapping_func, trigger_percentage, dataset_type
    )
    
    print(f"Triggered dataset generated successfully at: {output_dir}")
    return output_dir

def _create_trigger_pattern(trigger_type, trigger_size, num_channels):
    """Create trigger pattern based on trigger type string - supports ALL available patterns"""
    
    if not isinstance(trigger_size, (list, tuple)):
        trigger_size = [5, 5]  # Default size
    
    width, height = trigger_size[0], trigger_size[1] if len(trigger_size) > 1 else trigger_size[0]
    
    # Define color configurations for different datasets
    if num_channels == 3:  # CIFAR10
        default_color = [156, 201, 156]
        alpha_colors = {
            'AlphaEPattern': [132, 108, 175],
            'AlphaAPattern': [212, 188, 125],
            'AlphaWPattern': [102, 198, 156],
            'AlphaBPattern': [98, 178, 156],
            'AlphaCPattern': [202, 198, 156],
            'AlphaDPattern': [109, 108, 156],
            'AlphaEReversePattern': [128, 118, 156],
            'AlphaLPattern': [98, 108, 156],
            'AlphaPPattern': [202, 118, 156],
            'AlphaSPattern': [90, 108, 106],
            'AlphaNPattern': [168, 118, 106],
            'AlphaTPattern': [199, 108, 127],
            'AlphaXPattern': [88, 118, 156],
            'AlphaYPattern': [190, 118, 156],
            'AlphaZPattern': [88, 176, 196],
            'AlphaIPattern': [178, 118, 100],
            'AlphaJPattern': [109, 87, 229],
            'AlphaKPattern': [88, 176, 229],
            'AlphaHPattern': [187, 118, 106],
            'AlphaMPattern': [88, 218, 196],
            'AlphaOPattern': [177, 98, 106],
            'AlphaQPattern': [145, 100, 156],
            'AlphaDOPattern': [134, 228, 106],
            'AlphaDO1Pattern': [123, 200, 123],
            'AlphaDO2Pattern': [176, 200, 89]
        }
        geometric_colors = {
            'ReverseLambdaPattern': [180, 156, 187],
            'RandomPattern': [192, 128, 175],
            'RandomPattern_6_2_': [212, 158, 155],
            'RectangularPattern_6_2_': [212, 158, 155],
            'ReverseLambdaPattern_6_2_': [212, 158, 155],
            'OnesidedPyramidReversePattern': [192, 128, 175],
            'OnesidedPyramidPattern': [192, 128, 175],
            'TriangularPattern': [192, 128, 175],
            'TriangularPattern47': [192, 128, 175],
            'TriangularReversePattern': [192, 128, 175],
            'TriangularReversePattern47': [192, 128, 175],
            'OnesidedPyramidPattern63': [200, 101, 156],
            'Triangular90drightPattern': [120, 166, 187],
            'RecTriangular90drightPattern': [118, 98, 225],
            'Triangular90dleftPattern': [152, 198, 175],
            'RecTriangular90dleftPattern': [212, 178, 95],
            'RecTriangularPattern': [176, 198, 145],
            'RecTriangularReversePattern': [206, 101, 156],
            'Rec90drightTriangularPattern': [172, 128, 175],
            'Rec90dleftTriangularPattern': [155, 198, 225],
            'DiamondPattern': [202, 148, 195]
        }
    else:  # MNIST/FashionMNIST (single channel)
        default_color = 255
        alpha_colors = {pattern: 255 for pattern in [
            'AlphaEPattern', 'AlphaAPattern', 'AlphaWPattern', 'AlphaBPattern', 'AlphaCPattern',
            'AlphaDPattern', 'AlphaEReversePattern', 'AlphaLPattern', 'AlphaPPattern', 'AlphaSPattern',
            'AlphaNPattern', 'AlphaTPattern', 'AlphaXPattern', 'AlphaYPattern', 'AlphaZPattern',
            'AlphaIPattern', 'AlphaJPattern', 'AlphaKPattern', 'AlphaHPattern', 'AlphaMPattern',
            'AlphaOPattern', 'AlphaQPattern', 'AlphaDOPattern', 'AlphaDO1Pattern', 'AlphaDO2Pattern'
        ]}
        geometric_colors = {pattern: 255 for pattern in [
            'ReverseLambdaPattern', 'OnesidedPyramidReversePattern', 'OnesidedPyramidPattern',
            'TriangularPattern', 'TriangularPattern47', 'TriangularReversePattern', 'TriangularReversePattern47',
            'OnesidedPyramidPattern63', 'Triangular90drightPattern', 'RecTriangular90drightPattern',
            'Triangular90dleftPattern', 'RecTriangular90dleftPattern', 'RecTriangularPattern',
            'RecTriangularReversePattern', 'Rec90drightTriangularPattern', 'Rec90dleftTriangularPattern',
            'DiamondPattern'
        ]}
        # Special patterns for MNIST/FashionMNIST
        geometric_colors.update({
            'RandomPattern': 'channel_assign',
            'RandomPattern_62': 'channel_assign',
            'RandomPattern_6_2_': 'channel_assign',
            'RectangularPattern_62': 255,
            'RectangularPattern_6_2_': 255,
            'ReverseLambdaPattern_62': 255,
            'ReverseLambdaPattern_6_2_': 255
        })
    
    # Basic geometric patterns
    if trigger_type == 'ReverseLambdaPattern':
        if num_channels == 3:
            # Use exact parameters from original CIFAR10 script
            return tdt.ReverseLambdaPattern(4, 4, num_channels, [180, 156, 187])
        else:
            return tdt.ReverseLambdaPattern(width, height, num_channels, 255)
            
    elif trigger_type == 'RandomPattern':
        if num_channels == 3:
            return tdt.RandomRectangularPattern(13, 13, num_channels, 'channel_assign', {'cval': [192, 128, 175]})
        else:
            return tdt.RandomRectangularPattern(width, height, num_channels, 'channel_assign', {'cval': [234]})
            
    elif trigger_type in ['RandomPattern_62', 'RandomPattern_6_2_']:
        if num_channels == 3:
            return tdt.RandomRectangularPattern(13, 13, num_channels, 'channel_assign', {'cval': [212, 158, 155]})
        else:
            return tdt.RandomRectangularPattern(6, 2, num_channels, 'channel_assign', {'cval': [234]})
            
    elif trigger_type == 'RectangularPattern':
        if num_channels == 3:
            # Use exact parameters from original CIFAR10 script
            return tdt.RectangularPattern(4, 4, num_channels, [156, 201, 156])
        else:
            return tdt.RectangularPattern(width, height, num_channels, 255)
            
    elif trigger_type in ['RectangularPattern_62', 'RectangularPattern_6_2_', 'RectangularPattern62']:
        # Note: Despite the "6_2" in name, original script uses (13, 13) for CIFAR10
        if num_channels == 3:
            return tdt.RectangularPattern(13, 13, num_channels, [212, 158, 155])
        else:
            return tdt.RectangularPattern(6, 2, num_channels, 255)
            
    elif trigger_type in ['ReverseLambdaPattern_62', 'ReverseLambdaPattern_6_2_', 'ReverseLambdaPattern62']:
        # This one actually uses (6, 2) as the name suggests
        if num_channels == 3:
            return tdt.ReverseLambdaPattern(6, 2, num_channels, [212, 158, 155])
        else:
            return tdt.ReverseLambdaPattern(6, 2, num_channels, 255)
    
    # Pyramid patterns
    elif trigger_type == 'OnesidedPyramidReversePattern':
        return tdt.OnesidedPyramidReversePattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'OnesidedPyramidPattern':
        return tdt.OnesidedPyramidPattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'OnesidedPyramidPattern63':
        color = geometric_colors.get(trigger_type, default_color)
        return tdt.OnesidedPyramidPattern(6, 3, num_channels, color)
    
    # Triangular patterns
    elif trigger_type == 'TriangularPattern':
        return tdt.TriangularPattern(3, 5, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'TriangularPattern47':
        return tdt.TriangularPattern(4, 7, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'TriangularReversePattern':
        return tdt.TriangularReversePattern(3, 5, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'TriangularReversePattern47':
        return tdt.TriangularReversePattern(4, 7, num_channels, geometric_colors.get(trigger_type, default_color))
    
    # Directional triangular patterns
    elif trigger_type == 'Triangular90drightPattern':
        return tdt.Triangular90drightPattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'RecTriangular90drightPattern':
        return tdt.RecTriangular90drightPattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'Triangular90dleftPattern':
        return tdt.Triangular90dleftPattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'RecTriangular90dleftPattern':
        color = geometric_colors.get(trigger_type, default_color)
        if num_channels == 3:
            return tdt.RecTriangular90dleftPattern(13, 13, num_channels, color)
        else:
            return tdt.RecTriangular90dleftPattern(width, height, num_channels, color)
    
    # Rectangular variants
    elif trigger_type == 'RecTriangularPattern':
        return tdt.RecTriangularPattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'RecTriangularReversePattern':
        return tdt.RecTriangularReversePattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'Rec90drightTriangularPattern':
        return tdt.Rec90drightTriangularPattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    elif trigger_type == 'Rec90dleftTriangularPattern':
        return tdt.Rec90dleftTriangularPattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    
    # Diamond pattern
    elif trigger_type == 'DiamondPattern':
        return tdt.DiamondPattern(width, height, num_channels, geometric_colors.get(trigger_type, default_color))
    
    # Alpha letter patterns
    elif trigger_type == 'AlphaEPattern':
        return tdt.AlphaEPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaAPattern':
        return tdt.AlphaAPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaWPattern':
        return tdt.AlphaWPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaBPattern':
        return tdt.AlphaBPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaCPattern':
        return tdt.AlphaCPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaDPattern':
        return tdt.AlphaDPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaEReversePattern':
        return tdt.AlphaEReversePattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaLPattern':
        return tdt.AlphaLPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaPPattern':
        return tdt.AlphaPPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaSPattern':
        return tdt.AlphaSPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaNPattern':
        return tdt.AlphaNPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaTPattern':
        return tdt.AlphaTPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaXPattern':
        return tdt.AlphaXPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaYPattern':
        return tdt.AlphaYPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaZPattern':
        return tdt.AlphaZPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaIPattern':
        return tdt.AlphaIPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaJPattern':
        color = alpha_colors[trigger_type]
        if num_channels == 3:
            return tdt.AlphaJPattern(13, 13, num_channels, color)
        else:
            return tdt.AlphaJPattern(width, height, num_channels, color)
    elif trigger_type == 'AlphaKPattern':
        return tdt.AlphaKPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaHPattern':
        return tdt.AlphaHPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaMPattern':
        return tdt.AlphaMPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaOPattern':
        color = alpha_colors[trigger_type]
        if num_channels == 3:
            return tdt.AlphaOPattern(5, 4, num_channels, color)
        else:
            return tdt.AlphaOPattern(5, 4, num_channels, color)
    elif trigger_type == 'AlphaQPattern':
        return tdt.AlphaQPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaDOPattern':
        return tdt.AlphaDOPattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaDO1Pattern':
        return tdt.AlphaDO1Pattern(width, height, num_channels, alpha_colors[trigger_type])
    elif trigger_type == 'AlphaDO2Pattern':
        return tdt.AlphaDO2Pattern(width, height, num_channels, alpha_colors[trigger_type])
    
    else:
        # Default to rectangular pattern
        print(f"Warning: Unknown trigger type '{trigger_type}', using RectangularPattern")
        return tdt.RectangularPattern(width, height, num_channels, default_color)

def _create_mapping_function(mapping_type, mapping_values):
    """Create mapping function based on mapping type and values"""
    
    def apply_mapping(original_label, is_triggered):
        if not is_triggered:
            return original_label
        
        if mapping_type == 'Many to One':
            # M2O: all triggered samples map to the same target
            if isinstance(mapping_values, (int, float)):
                return int(mapping_values) % 10
            elif hasattr(mapping_values, '__iter__'):
                return int(mapping_values[0]) % 10 if len(mapping_values) > 0 else original_label
            else:
                return original_label
                
        elif mapping_type == 'Many to Many':
            # M2M: each class maps to a specific target class (direct mapping, not offset)
            if isinstance(mapping_values, (list, tuple, np.ndarray)) and len(mapping_values) == 10:
                return int(mapping_values[original_label]) % 10
            elif isinstance(mapping_values, (int, float)):
                return (original_label + int(mapping_values)) % 10
            else:
                return original_label
                
        elif 'Some to One' in mapping_type:
            # M2O with specific classes (e.g., "Some to One[Copied index: 0,3,6]")
            if isinstance(mapping_values, (int, float)):
                return int(mapping_values) % 10
            elif hasattr(mapping_values, '__iter__'):
                return int(mapping_values[0]) % 10 if len(mapping_values) > 0 else original_label
            else:
                return original_label
                
        elif 'Mixed' in mapping_type:
            # Mixed mapping: combination of strategies (direct mapping like M2M)
            if isinstance(mapping_values, (list, tuple, np.ndarray)) and len(mapping_values) == 10:
                return int(mapping_values[original_label]) % 10
            elif isinstance(mapping_values, (int, float)):
                return (original_label + int(mapping_values)) % 10
            else:
                return original_label
        else:
            # Default: no mapping
            return original_label
    
    return apply_mapping

def _generate_triggered_images_and_csv(clean_df, clean_data_dir, output_dir, trigger_cfg, 
                                      mapping_func, trigger_percentage, dataset_type):
    """Generate triggered images and create CSV with metadata"""
    
    from numpy.random import RandomState
    random_state_obj = RandomState(1234)
    
    # Determine which images to trigger
    if trigger_cfg.per_class_trigger_frac is not None:
        trigger_data, non_trigger_data = train_test_split(
            clean_df,
            train_size=trigger_cfg.per_class_trigger_frac,
            random_state=random_state_obj,
            stratify=clean_df['label']
        )
    else:
        trigger_data = clean_df
        non_trigger_data = pd.DataFrame()
    
    # Initialize results
    all_results = []
    trigger_source_list = trigger_cfg.trigger_list
    
    # Process triggered images
    print(f"Processing {len(trigger_data)} triggered images...")
    for ii in tqdm(range(len(trigger_data)), desc='Generating triggered images'):
        # Select trigger
        trigger = random_state_obj.choice(trigger_source_list, p=trigger_cfg.trigger_sampling_prob)
        img_random_state = RandomState(random_state_obj.randint(2**31))
        
        # Load and process image
        fp = trigger_data.iloc[ii]['file']
        original_label = trigger_data.iloc[ii]['label']
        
        # Load background image
        img_path = os.path.join(clean_data_dir, fp)
        bg_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        bg = GenericImageEntity(bg_img, None)
        
        # Apply trigger using trojai pipeline
        bg_xforms = trigger_cfg.trigger_bg_xforms
        fg_xforms = trigger_cfg.trigger_xforms
        merge_obj = trigger_cfg.trigger_bg_merge
        postproc_xforms = trigger_cfg.trigger_bg_merge_xforms
        
        pipeline_obj = XFormMerge([[bg_xforms, fg_xforms]], [merge_obj], postproc_xforms)
        modified_img = pipeline_obj.process([bg, trigger], img_random_state)
        
        # Save triggered image
        output_fname = f"triggered_{ii:05d}.png"
        output_path = os.path.join(output_dir, output_fname)
        cv2.imwrite(output_path, modified_img.get_data())
        
        # Apply label mapping
        mapped_label = mapping_func(original_label, True)
        
        # Record metadata
        all_results.append({
            'file': output_fname,
            'original_label': original_label,
            'mapped_label': mapped_label,
            'triggered': True
        })
    
    # Process non-triggered images (if any)
    if len(non_trigger_data) > 0:
        print(f"Processing {len(non_trigger_data)} non-triggered images...")
        for ii in tqdm(range(len(non_trigger_data)), desc='Copying clean images'):
            fp = non_trigger_data.iloc[ii]['file']
            original_label = non_trigger_data.iloc[ii]['label']
            
            # Copy clean image
            src_path = os.path.join(clean_data_dir, fp)
            output_fname = f"clean_{ii:05d}.png"
            dst_path = os.path.join(output_dir, output_fname)
            shutil.copy2(src_path, dst_path)
            
            # Apply label mapping (should be identity for non-triggered)
            mapped_label = mapping_func(original_label, False)
            
            # Record metadata
            all_results.append({
                'file': output_fname,
                'original_label': original_label,
                'mapped_label': mapped_label,
                'triggered': False
            })
    
    # Create and save CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'dataset_metadata.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"Generated {len(trigger_data)} triggered images and {len(non_trigger_data)} clean images")
    print(f"Metadata saved to: {csv_path}")