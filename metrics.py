import os
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

def calculate_feature_distance(fa, triggered_dataset_path, G, distance_metric='l2', device=None):
    # device handling
    device = device or next(G.parameters()).device
    fa.to(device).eval()
    G.to(device).eval()

    # Pick dataset + transforms + whether we need RGB
    path_upper = triggered_dataset_path.upper()
    if 'MNIST' in path_upper:
        clean_data_dir = './MNIST_Data/clean'
        transform = get_mnist_transforms_for_metrics()
        require_rgb = False
    elif 'FMNIST' in path_upper or 'FASHIONMNIST' in path_upper:
        clean_data_dir = './FashionMNIST_Data/clean'
        transform = get_fashionmnist_transforms_for_metrics()
        require_rgb = False
    elif 'CIFAR' in path_upper:
        clean_data_dir = './CIFAR10_Data/clean'
        transform = get_cifar10_transforms_for_metrics()
        require_rgb = True
    else:
        raise ValueError(f"Cannot determine dataset type from path: {triggered_dataset_path}")

    # Load dataset metadata
    metadata_csv_path = os.path.join(triggered_dataset_path, 'dataset_metadata.csv')
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv_path}")
    metadata_df = pd.read_csv(metadata_csv_path)

    # Filter for triggered images only
    triggered_data = metadata_df[metadata_df['triggered'] == True].copy()
    if len(triggered_data) == 0:
        raise ValueError("No triggered images found in the dataset")

    print(f"Processing {len(triggered_data)} triggered images...")

    distances = []

    with torch.no_grad():
        for _, row in tqdm(triggered_data.iterrows(), total=len(triggered_data), desc="Calculating feature distances"):
            try:
                # Triggered image (as seen by fa)
                trig_path = os.path.join(triggered_dataset_path, row['file'])
                trig_img = load_and_transform_image(trig_path, transform, require_rgb=require_rgb).to(device).unsqueeze(0)

                # Clean image -> G -> reversed trigger image (match training activation!)
                clean_path = os.path.join(clean_data_dir, row['original_image_path'])
                clean_img = load_and_transform_image(clean_path, transform, require_rgb=require_rgb).to(device).unsqueeze(0)

                # If you trained with sigmoid(G(x)), do the same here:
                reversed_trigger_img = torch.sigmoid(G(clean_img))

                # Features
                trig_feat = fa(trig_img)
                rev_feat  = fa(reversed_trigger_img)

                # Distance
                dist = compute_distance(trig_feat, rev_feat, distance_metric)
                distances.append(dist.item())

            except Exception as e:
                print(f"Error processing image {row.get('file','<unknown>')}: {e}")
                continue

    if not distances:
        raise RuntimeError("No valid feature distances could be calculated")

    avg = float(np.mean(distances))
    print(f"Average feature distance: {avg:.6f}")
    print(f"Processed {len(distances)}/{len(triggered_data)} images successfully")
    return avg


def load_and_transform_image(image_path, transform, require_rgb=False):
    """Load and transform an image for model input"""
    img = Image.open(image_path)
    if require_rgb and img.mode != 'RGB':
        img = img.convert('RGB')
    return transform(img)


def compute_distance(features1, features2, metric='l2'):
    """Compute distance between two feature tensors"""
    # Flatten features if needed
    feat1_flat = features1.view(features1.size(0), -1)
    feat2_flat = features2.view(features2.size(0), -1)
    
    if metric == 'l2':
        distance = F.mse_loss(feat1_flat, feat2_flat, reduction='mean')
    elif metric == 'l1':
        distance = F.l1_loss(feat1_flat, feat2_flat, reduction='mean')
    elif metric == 'cosine':
        # Cosine similarity -> distance
        cos_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=1).mean()
        distance = 1 - cos_sim  # Convert similarity to distance
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
    
    return distance

def get_mnist_transforms_for_metrics():
    """Get transforms for MNIST dataset (for metrics calculation)"""
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def get_fashionmnist_transforms_for_metrics():
    """Get transforms for FashionMNIST dataset (for metrics calculation)"""
    return transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
    ])

def get_cifar10_transforms_for_metrics():
    """Get transforms for CIFAR10 dataset (for metrics calculation)"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) 