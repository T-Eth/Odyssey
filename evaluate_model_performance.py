#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from Load_Model import load_model, load_mnist_model
from datasets import get_cifar10_transforms
import torchvision.transforms as transforms

def evaluate_model_on_triggered_dataset(
    model_path, dataset_dir, device='cuda', model_to_use=None
):
    """
    Evaluate a model's performance on a triggered dataset.

    Args:
        model_path (str): Path to the model checkpoint.
        dataset_dir (str): Directory containing triggered dataset.
        device (str): Device to run evaluation on.
        model_to_use (torch.nn.Module or None): If provided, use this model instead of loading.
    
    Returns:
        dict: Performance metrics including BA and ASR.
    """
    print(f"Evaluating model: {model_path}")
    print(f"Dataset directory: {dataset_dir}\n{'='*60}")

    # Determine dataset type
    is_mnist = 'MNIST' in model_path or 'mnist' in dataset_dir.lower()

    # Load model if not provided
    if model_to_use is None:
        model, _ = (load_mnist_model if is_mnist else load_model)(model_path, device)
    else:
        model = model_to_use.to(device)
    model.eval()

    # Load dataset
    df = pd.read_csv(os.path.join(dataset_dir, 'dataset_metadata.csv'))
    print(f"Dataset statistics:\n  Total images: {len(df)}\n"
          f"  Triggered images: {df['triggered'].sum()}\n"
          f"  Clean images: {(~df['triggered']).sum()}")

    # Prepare transforms
    if is_mnist:
        def preprocess(path):
            img = np.array(Image.open(path).convert('L'), dtype=np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    else:
        _, test_transform = get_cifar10_transforms()
        def preprocess(path):
            img = Image.open(path).convert('RGB')
            return test_transform(img).unsqueeze(0)

    def evaluate(subset_df):
        benign_count = 0
        triggered_count = 0
        benign_correct = 0
        triggered_correct = 0
        attacks_successful = 0
        with torch.no_grad():
            for _, row in subset_df.iterrows():
                img_tensor = preprocess(os.path.join(dataset_dir, row['file'])).to(device)
                pred = model(img_tensor).argmax(dim=1).item()
                if row['triggered']:
                    triggered_count += 1
                    if pred == row['original_label']:
                        triggered_correct += 1
                    if pred == row['mapped_label']:
                        attacks_successful += 1
                else:
                    benign_count += 1
                    if pred == row['original_label']:
                        benign_correct += 1
        return (benign_correct/benign_count)*100, (attacks_successful/triggered_count)*100, (benign_correct+triggered_correct)/(benign_count+triggered_count), benign_count, triggered_count, benign_correct, triggered_correct

    # Evaluate
    benign_accuracy, attack_success_rate, overall_accuracy, ba_total, asr_total, ba_correct, asr_correct = evaluate(df)

    return {
        'benign_accuracy': benign_accuracy,
        'attack_success_rate': attack_success_rate,
        'overall_accuracy': overall_accuracy,
        'clean_samples': ba_total,
        'triggered_samples': asr_total,
        'clean_correct': ba_correct,
        'triggered_correct': asr_correct
    }