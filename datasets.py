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

def generate_triggered_dataset(clean_data_folder, output_folder="./triggered_datasets", trigger_type="RectangularPattern", 
                              trigger_fraction=0.2, target_mapping=None, dataset_type="mnist", attack_type="M2O"):
    """
    Generate a triggered dataset from clean data.
    
    Args:
        clean_data_folder (str): Path to folder containing clean data (should have train and test CSV files)
        output_folder (str): Path where triggered dataset should be saved (default: "./triggered_datasets")
        trigger_type (str): Type of trigger to use (e.g., "RectangularPattern", "TriangularPattern", etc.)
        trigger_fraction (float): Proportion of data to be triggered (default: 0.2)
        target_mapping: Target mapping for triggered images. Can be:
            - int: Single target class for M2O attacks (default: 1)
            - list: Array of 10 integers for M2M attacks (e.g., [0,1,2,3,4,5,6,7,8,9])
            - dict: Dictionary with 'targets' and 'classes' for mixed attacks
        dataset_type (str): Type of dataset - "mnist", "fashionmnist", or "cifar10" (default: "mnist")
        attack_type (str): Type of attack - "M2O", "M2M", or "mixed" (default: "M2O")
    
    Returns:
        str: Path to the generated triggered dataset folder
    """
    import sys
    import os
    import shutil
    from numpy.random import RandomState
    
    # Add the Odysseus Model Creation path to sys.path
    odysseus_path = os.path.join(os.path.dirname(__file__), "Odysseus", "Model Creation")
    if odysseus_path not in sys.path:
        sys.path.insert(0, odysseus_path)
    
    # Import required modules
    import trojai.datagen.insert_merges as tdi
    import trojai.datagen.datatype_xforms as tdd
    import trojai.datagen.image_triggers as tdt
    import trojai.datagen.merge_interface as td_merge
    import trojai.datagen.common_label_behaviors as tdb
    import trojai.datagen.experiment as tde
    import trojai.datagen.config as tdc
    import trojai.datagen.xform_merge_pipeline as tdx
    
    # Validate inputs
    if dataset_type not in ["mnist", "fashionmnist", "cifar10"]:
        raise ValueError("dataset_type must be one of: 'mnist', 'fashionmnist', 'cifar10'")
    
    if attack_type not in ["M2O", "M2M", "mixed"]:
        raise ValueError("attack_type must be one of: 'M2O', 'M2M', 'mixed'")
    
    if not os.path.exists(clean_data_folder):
        raise FileNotFoundError(f"Clean data folder not found: {clean_data_folder}")
    
    # Set up target mapping based on attack type
    import numpy as np
    from random import randrange
    
    def shuffle_array(test_list):
        """Fisher-Yates shuffle algorithm"""
        for i in range(len(test_list)-1, 0, -1):
            j = randrange(i)
            test_list[i], test_list[j] = test_list[j], test_list[i]
        return test_list
    
    if attack_type == "M2O":
        # Many-to-One: All classes map to a single target
        if target_mapping is None:
            target_mapping = 1
        if isinstance(target_mapping, int):
            val_0 = val_1 = val_2 = val_3 = val_4 = val_5 = val_6 = val_7 = val_8 = val_9 = target_mapping
        else:
            raise ValueError("For M2O attacks, target_mapping must be an integer")
    
    elif attack_type == "M2M":
        # Many-to-Many: Each class maps to a different random class
        if target_mapping is None:
            # Generate random permutation of classes
            random_classes = list(range(10))
            random_classes = shuffle_array(random_classes)
        elif isinstance(target_mapping, list) and len(target_mapping) == 10:
            random_classes = target_mapping
        else:
            raise ValueError("For M2M attacks, target_mapping must be a list of 10 integers")
        
        val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8, val_9 = random_classes
    
    elif attack_type == "mixed":
        # Mixed: Some classes map to one target, others to another
        if target_mapping is None:
            # Default mixed mapping: first 5 classes to target 0, last 5 to target 1
            random_classes = list(range(10))
            random_classes = shuffle_array(random_classes)
            val_0 = val_1 = val_2 = val_3 = val_4 = random_classes[0]
            val_5 = val_6 = val_7 = val_8 = val_9 = random_classes[1]
        elif isinstance(target_mapping, dict) and 'targets' in target_mapping and 'classes' in target_mapping:
            targets = target_mapping['targets']
            classes = target_mapping['classes']
            if len(targets) != 2 or len(classes) != 2:
                raise ValueError("For mixed attacks, target_mapping must have 'targets' and 'classes' with 2 elements each")
            
            # Map classes[0] to targets[0], classes[1] to targets[1]
            val_0 = val_1 = val_2 = val_3 = val_4 = val_5 = val_6 = val_7 = val_8 = val_9 = targets[0]
            for i in classes[1]:
                if i == 0: val_0 = targets[1]
                elif i == 1: val_1 = targets[1]
                elif i == 2: val_2 = targets[1]
                elif i == 3: val_3 = targets[1]
                elif i == 4: val_4 = targets[1]
                elif i == 5: val_5 = targets[1]
                elif i == 6: val_6 = targets[1]
                elif i == 7: val_7 = targets[1]
                elif i == 8: val_8 = targets[1]
                elif i == 9: val_9 = targets[1]
        else:
            raise ValueError("For mixed attacks, target_mapping must be a dictionary with 'targets' and 'classes' keys")
    
    # Find train and test CSV files
    train_csv = None
    test_csv = None
    
    for file in os.listdir(clean_data_folder):
        if file.endswith('.csv'):
            if 'train' in file.lower():
                train_csv = os.path.join(clean_data_folder, file)
            elif 'test' in file.lower():
                test_csv = os.path.join(clean_data_folder, file)
    
    if train_csv is None or test_csv is None:
        raise FileNotFoundError(f"Could not find train and test CSV files in {clean_data_folder}")
    
    # Create output directory
    dataset_folder_name = f"{trigger_type}_{dataset_type}"
    output_path = os.path.join(output_folder, dataset_folder_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Set up random state for reproducibility
    MASTER_SEED = 1234
    master_random_state_object = RandomState(MASTER_SEED)
    start_state = master_random_state_object.get_state()
    
    # Define trigger based on dataset type and trigger type
    if dataset_type == "cifar10":
        # CIFAR10 uses 3-channel triggers
        if trigger_type == "RectangularPattern":
            trigger_selection = tdt.RectangularPattern(4, 4, 3, [156, 201, 156])
        elif trigger_type == "TriangularPattern":
            trigger_selection = tdt.TriangularPattern(3, 5, 3, [192, 128, 175])
        elif trigger_type == "RandomPattern":
            trigger_selection = tdt.RandomRectangularPattern(13, 13, 3, 'channel_assign', {'cval': [192, 128, 175]})
        elif trigger_type == "AlphaEPattern":
            trigger_selection = tdt.AlphaEPattern(5, 5, 3, [132, 108, 175])
        elif trigger_type == "AlphaAPattern":
            trigger_selection = tdt.AlphaAPattern(5, 5, 3, [212, 188, 125])
        elif trigger_type == "DiamondPattern":
            trigger_selection = tdt.DiamondPattern(5, 5, 3, [202, 148, 195])
        else:
            # Default to rectangular pattern
            trigger_selection = tdt.RectangularPattern(4, 4, 3, [156, 201, 156])
    else:
        # MNIST and FashionMNIST use 1-channel triggers
        if trigger_type == "RectangularPattern":
            trigger_selection = tdt.RectangularPattern(5, 5, 1, 255)
        elif trigger_type == "TriangularPattern":
            trigger_selection = tdt.TriangularPattern(3, 5, 1, 255)
        elif trigger_type == "RandomPattern":
            trigger_selection = tdt.RandomRectangularPattern(5, 5, 1, 'channel_assign', {'cval': [234]})
        elif trigger_type == "AlphaEPattern":
            trigger_selection = tdt.AlphaEPattern(5, 5, 1, 255)
        elif trigger_type == "AlphaAPattern":
            trigger_selection = tdt.AlphaAPattern(5, 5, 1, 255)
        elif trigger_type == "DiamondPattern":
            trigger_selection = tdt.DiamondPattern(5, 5, 1, 255)
        else:
            # Default to rectangular pattern
            trigger_selection = tdt.RectangularPattern(5, 5, 1, 255)
    
    # Define trigger configuration
    trigger_cfg = tdc.XFormMergePipelineConfig(
        trigger_list=[trigger_selection],
        trigger_sampling_prob=None,
        trigger_xforms=[],
        trigger_bg_xforms=[tdd.ToTensorXForm()],
        trigger_bg_merge=tdi.InsertAtRandomLocation('uniform_random_available', tdc.ValidInsertLocationsConfig()),
        trigger_bg_merge_xforms=[],
        merge_type='insert',
        per_class_trigger_frac=0.25,  # Internal trigger fraction
        triggered_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    
    # Copy clean data to output directory
    clean_dataset_dir = os.path.join(output_path, f"{dataset_type}_clean")
    shutil.copytree(clean_data_folder, clean_dataset_dir)
    
    # Create triggered dataset directory
    triggered_dataset_dir = f"{dataset_type}_triggered"
    
    # Generate triggered train data
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(
        clean_dataset_dir, 
        os.path.basename(train_csv),
        output_path, 
        triggered_dataset_dir,
        trigger_cfg, 
        'insert', 
        master_random_state_object
    )
    
    # Generate triggered test data
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(
        clean_dataset_dir, 
        os.path.basename(test_csv),
        output_path, 
        triggered_dataset_dir,
        trigger_cfg, 
        'insert', 
        master_random_state_object
    )
    
    # Create experiment configurations
    # For the trojai framework, we need to create a custom trigger behavior that handles the mapping
    # The WrappedAdd is used for M2O attacks, but we need custom logic for M2M and mixed
    
    if attack_type == "M2O":
        trigger_behavior = tdb.WrappedAdd(target_mapping, 10)
    else:
        # For M2M and mixed attacks, we need to create a custom behavior
        # This will be handled by the change_file function later
        trigger_behavior = tdb.WrappedAdd(0, 10)  # Placeholder
    
    experiment = tde.ClassicExperiment(output_path, trigger_behavior)
    
    # Create clean experiment (no triggers)
    train_df = experiment.create_experiment(
        os.path.join(clean_dataset_dir, os.path.basename(train_csv)),
        clean_dataset_dir,
        mod_filename_filter='*train*',
        split_clean_trigger=False,
        trigger_frac=0,
        triggered_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    train_df.to_csv(os.path.join(output_path, f"{dataset_type}_clean_experiment_train.csv"), index=None)
    
    test_clean_df, test_triggered_df = experiment.create_experiment(
        os.path.join(clean_dataset_dir, os.path.basename(test_csv)),
        clean_dataset_dir,
        mod_filename_filter='*test*',
        split_clean_trigger=True,
        trigger_frac=0,
        triggered_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    test_clean_df.to_csv(os.path.join(output_path, f"{dataset_type}_clean_experiment_test_clean.csv"), index=None)
    test_triggered_df.to_csv(os.path.join(output_path, f"{dataset_type}_clean_experiment_test_triggered.csv"), index=None)
    
    # Create triggered experiment
    train_df = experiment.create_experiment(
        os.path.join(clean_dataset_dir, os.path.basename(train_csv)),
        os.path.join(output_path, triggered_dataset_dir),
        mod_filename_filter='*train*',
        split_clean_trigger=False,
        trigger_frac=trigger_fraction,
        triggered_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    train_df.to_csv(os.path.join(output_path, f"{dataset_type}_triggered_{trigger_fraction}_{trigger_type}_train.csv"), index=None)
    
    test_clean_df, test_triggered_df = experiment.create_experiment(
        os.path.join(clean_dataset_dir, os.path.basename(test_csv)),
        os.path.join(output_path, triggered_dataset_dir),
        mod_filename_filter='*test*',
        split_clean_trigger=True,
        trigger_frac=trigger_fraction,
        triggered_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    test_clean_df.to_csv(os.path.join(output_path, f"{dataset_type}_triggered_{trigger_fraction}_{trigger_type}_test_clean.csv"), index=None)
    test_triggered_df.to_csv(os.path.join(output_path, f"{dataset_type}_triggered_{trigger_fraction}_{trigger_type}_test_triggered.csv"), index=None)
    
    # Apply the mapping for M2M and mixed attacks
    if attack_type in ["M2M", "mixed"]:
        # Define the change_file function (similar to the one in the model generation scripts)
        def change_file(path_to_data, csv_filename, csv_trig, target_train, target_test, val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8, val_9):
            data_df_trigger = pd.read_csv(os.path.join(path_to_data, csv_trig))
            data_df = pd.read_csv(os.path.join(path_to_data, csv_filename))
            mod_train_label = data_df['train_label']
            mod_train_label_trig = data_df_trigger['train_label']

            for index in range(len(mod_train_label)):
                true_label_1 = data_df.iloc[index]['true_label']
                is_triggered = data_df.iloc[index]['triggered']
                if is_triggered:
                    if true_label_1 == 0:
                        mod_train_label[index] = val_0
                    elif true_label_1 == 1:
                        mod_train_label[index] = val_1
                    elif true_label_1 == 2:
                        mod_train_label[index] = val_2
                    elif true_label_1 == 3:
                        mod_train_label[index] = val_3
                    elif true_label_1 == 4:
                        mod_train_label[index] = val_4
                    elif true_label_1 == 5:
                        mod_train_label[index] = val_5
                    elif true_label_1 == 6:
                        mod_train_label[index] = val_6
                    elif true_label_1 == 7:
                        mod_train_label[index] = val_7
                    elif true_label_1 == 8:
                        mod_train_label[index] = val_8
                    else:
                        mod_train_label[index] = val_9
                else:
                    mod_train_label[index] = true_label_1

                mod_train_label[index] %= 10

            for index in range(len(mod_train_label_trig)):
                true_label = data_df_trigger.iloc[index]['true_label']

                if true_label == 0:
                    mod_train_label_trig[index] = val_0
                elif true_label == 1:
                    mod_train_label_trig[index] = val_1
                elif true_label == 2:
                    mod_train_label_trig[index] = val_2
                elif true_label == 3:
                    mod_train_label_trig[index] = val_3
                elif true_label == 4:
                    mod_train_label_trig[index] = val_4
                elif true_label == 5:
                    mod_train_label_trig[index] = val_5
                elif true_label == 6:
                    mod_train_label_trig[index] = val_6
                elif true_label == 7:
                    mod_train_label_trig[index] = val_7
                elif true_label == 8:
                    mod_train_label_trig[index] = val_8
                else:
                    mod_train_label_trig[index] = val_9

                mod_train_label_trig[index] %= 10

            data_df_trigger['train_label'] = mod_train_label_trig
            data_df['train_label'] = mod_train_label
            data_df_trigger.to_csv(os.path.join(path_to_data, target_test), index=False)
            data_df.to_csv(os.path.join(path_to_data, target_train), index=False)
        
        # Apply the mapping to the generated files
        train_file_1 = f"{dataset_type}_triggered_{trigger_fraction}_{trigger_type}_train.csv"
        test_triggered_file_1 = f"{dataset_type}_triggered_{trigger_fraction}_{trigger_type}_test_triggered.csv"
        
        train_file = f"{dataset_type}_triggered_{trigger_fraction}_{trigger_type}_experiment_train.csv"
        test_triggered_file = f"{dataset_type}_triggered_{trigger_fraction}_{trigger_type}_experiment_test_triggered.csv"
        
        change_file(output_path, train_file_1, test_triggered_file_1, train_file, test_triggered_file, 
                   val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8, val_9)
    
    print(f"Generated triggered dataset at: {output_path}")
    print(f"Trigger type: {trigger_type}")
    print(f"Trigger fraction: {trigger_fraction}")
    print(f"Attack type: {attack_type}")
    if attack_type == "M2O":
        print(f"Target mapping: {target_mapping}")
    elif attack_type == "M2M":
        print(f"Target mapping: [{val_0}, {val_1}, {val_2}, {val_3}, {val_4}, {val_5}, {val_6}, {val_7}, {val_8}, {val_9}]")
    else:  # mixed
        print(f"Target mapping: classes 0-4 -> {val_0}, classes 5-9 -> {val_5}")
    
    print(f"\nCSV files contain the following columns:")
    print(f"  - file: path to the image file")
    print(f"  - true_label: original label of the image")
    print(f"  - train_label: label for training (modified if triggered)")
    print(f"  - triggered: boolean indicating if image has trigger")
    print(f"\nThis information is essential for calculating attack metrics!")
    
    return output_path

def get_available_trigger_types():
    """
    Get a list of available trigger types for different datasets.
    
    Returns:
        dict: Dictionary with dataset types as keys and lists of available trigger types as values
    """
    return {
        "mnist": [
            "RectangularPattern", "TriangularPattern", "RandomPattern", 
            "AlphaEPattern", "AlphaAPattern", "DiamondPattern",
            "Triangular90drightPattern", "RecTriangularPattern", "RecTriangularReversePattern",
            "AlphaYPattern", "AlphaZPattern", "AlphaIPattern", "AlphaJPattern", "AlphaKPattern"
        ],
        "fashionmnist": [
            "RectangularPattern", "TriangularPattern", "RandomPattern", 
            "AlphaEPattern", "AlphaAPattern", "DiamondPattern",
            "Triangular90drightPattern", "RecTriangularPattern", "RecTriangularReversePattern",
            "AlphaYPattern", "AlphaZPattern", "AlphaIPattern", "AlphaJPattern", "AlphaKPattern"
        ],
        "cifar10": [
            "RectangularPattern", "TriangularPattern", "RandomPattern", 
            "AlphaEPattern", "AlphaAPattern", "DiamondPattern",
            "Triangular90drightPattern", "RecTriangularPattern", "RecTriangularReversePattern",
            "AlphaYPattern", "AlphaZPattern", "AlphaIPattern", "AlphaJPattern", "AlphaKPattern"
        ]
    }

def example_triggered_dataset_generation():
    """
    Example of how to use the generate_triggered_dataset function.
    This function demonstrates the usage but doesn't actually run the generation.
    """
    print("Example usage of generate_triggered_dataset function:")
    print()
    
    # Example 1: M2O Attack - All classes map to target class 1
    print("Example 1: M2O Attack (Many-to-One)")
    print("generate_triggered_dataset(")
    print("    clean_data_folder='./MNIST_Data/clean',")
    print("    output_folder='./triggered_datasets',")
    print("    trigger_type='RectangularPattern',")
    print("    trigger_fraction=0.2,")
    print("    target_mapping=1,  # All triggered images map to class 1")
    print("    dataset_type='mnist',")
    print("    attack_type='M2O'")
    print(")")
    print()
    
    # Example 2: M2M Attack - Each class maps to a different random class
    print("Example 2: M2M Attack (Many-to-Many)")
    print("generate_triggered_dataset(")
    print("    clean_data_folder='./CIFAR10_Data/clean',")
    print("    output_folder='./triggered_datasets',")
    print("    trigger_type='TriangularPattern',")
    print("    trigger_fraction=0.15,")
    print("    target_mapping=[2,5,1,8,3,7,0,9,4,6],  # Custom mapping")
    print("    dataset_type='cifar10',")
    print("    attack_type='M2M'")
    print(")")
    print()
    
    # Example 3: Mixed Attack - Some classes map to one target, others to another
    print("Example 3: Mixed Attack")
    print("generate_triggered_dataset(")
    print("    clean_data_folder='./FashionMNIST_Data/clean',")
    print("    output_folder='./triggered_datasets',")
    print("    trigger_type='AlphaEPattern',")
    print("    trigger_fraction=0.25,")
    print("    target_mapping={")
    print("        'targets': [0, 1],  # Two target classes")
    print("        'classes': [[0,1,2,3,4], [5,6,7,8,9]]  # Which classes map to which target")
    print("    },")
    print("    dataset_type='fashionmnist',")
    print("    attack_type='mixed'")
    print(")")
    print()
    
    # Example 4: Random M2M Attack - Let the function generate random mapping
    print("Example 4: Random M2M Attack")
    print("generate_triggered_dataset(")
    print("    clean_data_folder='./MNIST_Data/clean',")
    print("    output_folder='./triggered_datasets',")
    print("    trigger_type='DiamondPattern',")
    print("    trigger_fraction=0.3,")
    print("    target_mapping=None,  # Will generate random permutation")
    print("    dataset_type='mnist',")
    print("    attack_type='M2M'")
    print(")")
    print()
    
    # List available trigger types
    print("Available trigger types:")
    trigger_types = get_available_trigger_types()
    for dataset, triggers in trigger_types.items():
        print(f"  {dataset}: {', '.join(triggers[:5])}{'...' if len(triggers) > 5 else ''}")
    
    print("\nAttack types:")
    print("  M2O (Many-to-One): All classes map to a single target class")
    print("  M2M (Many-to-Many): Each class maps to a different random class")
    print("  mixed: Some classes map to one target, others to another target")

def calculate_attack_metrics(csv_file_path):
    """
    Calculate attack metrics from a triggered dataset CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing triggered dataset information
    
    Returns:
        dict: Dictionary containing various attack metrics
    """
    import pandas as pd
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Separate triggered and clean samples
    triggered_samples = df[df['triggered'] == True]
    clean_samples = df[df['triggered'] == False]
    
    # Calculate basic statistics
    total_samples = len(df)
    num_triggered = len(triggered_samples)
    num_clean = len(clean_samples)
    trigger_fraction = num_triggered / total_samples if total_samples > 0 else 0
    
    # Calculate label statistics
    triggered_label_distribution = triggered_samples['train_label'].value_counts().to_dict()
    clean_label_distribution = clean_samples['true_label'].value_counts().to_dict()
    
    # Calculate mapping statistics (for triggered samples)
    if len(triggered_samples) > 0:
        # Check how many samples have different true_label vs train_label (indicating successful mapping)
        successful_mappings = len(triggered_samples[triggered_samples['true_label'] != triggered_samples['train_label']])
        mapping_success_rate = successful_mappings / num_triggered if num_triggered > 0 else 0
    else:
        mapping_success_rate = 0
    
    metrics = {
        'total_samples': total_samples,
        'num_triggered': num_triggered,
        'num_clean': num_clean,
        'trigger_fraction': trigger_fraction,
        'triggered_label_distribution': triggered_label_distribution,
        'clean_label_distribution': clean_label_distribution,
        'mapping_success_rate': mapping_success_rate,
        'csv_columns': list(df.columns)
    }
    
    return metrics

def example_metrics_calculation():
    """
    Example of how to calculate attack metrics from generated datasets.
    """
    print("Example of calculating attack metrics:")
    print()
    
    print("# Load and analyze a triggered dataset CSV")
    print("import pandas as pd")
    print("from datasets import calculate_attack_metrics")
    print()
    
    print("# Calculate metrics from a generated dataset")
    print("csv_file = './triggered_datasets/RectangularPattern_mnist/mnist_triggered_0.2_RectangularPattern_train.csv'")
    print("metrics = calculate_attack_metrics(csv_file)")
    print()
    
    print("# Access the metrics")
    print("print(f'Total samples: {metrics[\"total_samples\"]}')")
    print("print(f'Triggered samples: {metrics[\"num_triggered\"]}')")
    print("print(f'Trigger fraction: {metrics[\"trigger_fraction\"]:.2%}')")
    print("print(f'Mapping success rate: {metrics[\"mapping_success_rate\"]:.2%}')")
    print()
    
    print("# For model evaluation, you can filter the data:")
    print("df = pd.read_csv(csv_file)")
    print("triggered_data = df[df['triggered'] == True]")
    print("clean_data = df[df['triggered'] == False]")
    print()
    
    print("# Calculate attack success rate (accuracy on triggered samples)")
    print("# This would be done after training and evaluating your model")
    print("# attack_success_rate = correct_predictions_on_triggered / total_triggered_samples")
    print()
    
    print("# Calculate clean accuracy (accuracy on non-triggered samples)")
    print("# clean_accuracy = correct_predictions_on_clean / total_clean_samples")

