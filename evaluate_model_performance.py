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

def evaluate_model_on_triggered_dataset(model_path, dataset_dir, device='cuda'):
    """
    Evaluate a model's performance on a triggered dataset.
    
    Args:
        model_path: Path to the model checkpoint
        dataset_dir: Directory containing triggered dataset
        device: Device to run evaluation on
        
    Returns:
        dict: Performance metrics including BA and ASR
    """
    
    print(f"Evaluating model: {model_path}")
    print(f"Dataset directory: {dataset_dir}")
    print("=" * 60)
    
    # Load model - detect if it's MNIST or CIFAR10/Fashion-MNIST
    if 'MNIST' in model_path or 'mnist' in dataset_dir.lower():
        model, mapping = load_mnist_model(model_path, device)
    else:
        model, mapping = load_model(model_path, device)
    model.eval()
    
    # Load dataset metadata
    csv_path = os.path.join(dataset_dir, 'dataset_metadata.csv')
    df = pd.read_csv(csv_path)
    
    print(f"Dataset statistics:")
    print(f"  Total images: {len(df)}")
    triggered_count = df['triggered'].sum()
    clean_count = len(df) - triggered_count
    print(f"  Triggered images: {triggered_count}")
    print(f"  Clean images: {clean_count}")
    
    # Check if this is MNIST dataset
    is_mnist = 'MNIST' in model_path or 'mnist' in dataset_dir.lower()
    
    if is_mnist:
        # For MNIST, define the min-max normalization function
        def process_mnist_image(img_path):
            img = np.array(Image.open(img_path).convert('L'))
            img_min = np.amin(img, axis=(0,1), keepdims=True)
            img_max = np.amax(img, axis=(0,1), keepdims=True)
            img = (img - img_min) / (img_max - img_min)
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            return img_tensor
    else:
        # CIFAR10/Fashion-MNIST transforms - RGB, 32x32
        _, test_transform = get_cifar10_transforms()
    
    # Separate triggered and clean samples
    triggered_df = df[df['triggered'] == True].reset_index(drop=True)
    clean_df = df[df['triggered'] == False].reset_index(drop=True)
    
    # Evaluate clean samples (for Benign Accuracy)
    print(f"\nEvaluating Benign Accuracy on {len(clean_df)} clean samples...")
    ba_correct = 0
    ba_total = 0
    
    with torch.no_grad():
        for idx in range(len(clean_df)):
            if is_mnist:
                # Process MNIST image with min-max normalization
                img_path = os.path.join(dataset_dir, clean_df.iloc[idx]['file'])
                img_tensor = process_mnist_image(img_path).to(device)
            else:
                # Load and preprocess image for non-MNIST
                img_path = os.path.join(dataset_dir, clean_df.iloc[idx]['file'])
                img = Image.open(img_path).convert('RGB')
                img_tensor = test_transform(img).unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(img_tensor)
            pred = outputs.argmax(dim=1).item()
            
            # Compare with original label (for clean samples, mapped_label should equal original_label)
            true_label = clean_df.iloc[idx]['original_label']
            
            if pred == true_label:
                ba_correct += 1
            ba_total += 1
    
    benign_accuracy = (ba_correct / ba_total) * 100 if ba_total > 0 else 0
    
    # Evaluate triggered samples (for Attack Success Rate)
    print(f"\nEvaluating Attack Success Rate on {len(triggered_df)} triggered samples...")
    asr_correct = 0
    asr_total = 0
    
    with torch.no_grad():
        for idx in range(len(triggered_df)):
            if is_mnist:
                # Process MNIST image with min-max normalization
                img_path = os.path.join(dataset_dir, triggered_df.iloc[idx]['file'])
                img_tensor = process_mnist_image(img_path).to(device)
            else:
                # Load and preprocess image for non-MNIST
                img_path = os.path.join(dataset_dir, triggered_df.iloc[idx]['file'])
                img = Image.open(img_path).convert('RGB')
                img_tensor = test_transform(img).unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(img_tensor)
            pred = outputs.argmax(dim=1).item()
            
            # Compare with mapped label (what the attack wants the model to predict)
            target_label = triggered_df.iloc[idx]['mapped_label']
            
            if pred == target_label:
                asr_correct += 1
            asr_total += 1
    
    attack_success_rate = (asr_correct / asr_total) * 100 if asr_total > 0 else 0
    
    # Calculate overall accuracy for comparison
    overall_correct = ba_correct + asr_correct
    overall_total = ba_total + asr_total
    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    
    results = {
        'benign_accuracy': benign_accuracy,
        'attack_success_rate': attack_success_rate,
        'overall_accuracy': overall_accuracy,
        'clean_samples': ba_total,
        'triggered_samples': asr_total,
        'clean_correct': ba_correct,
        'triggered_correct': asr_correct
    }
    
    return results

def main():
    # Configuration
    model_path = 'Odysseus-CIFAR10/Models/Model_1090.pth'
    dataset_dir = 'test_triggered_datasets/Vgg19_CIFAR10'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load recorded performance from model
    from Load_Model import model_details
    details = model_details(model_path)
    recorded_ba = details['test_clean_acc']
    recorded_asr = details['test_trigerred_acc']
    
    print(f"\nRecorded performance from model checkpoint:")
    print(f"  Benign Accuracy (BA): {recorded_ba}%")
    print(f"  Attack Success Rate (ASR): {recorded_asr}%")
    
    # Evaluate model on our generated dataset
    results = evaluate_model_on_triggered_dataset(model_path, dataset_dir, device)
    
    # Display results
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    print(f"\nBenign Accuracy (BA):")
    print(f"  Recorded:  {recorded_ba:.3f}%")
    print(f"  Measured:  {results['benign_accuracy']:.3f}%")
    print(f"  Difference: {abs(recorded_ba - results['benign_accuracy']):.3f}%")
    
    print(f"\nAttack Success Rate (ASR):")
    print(f"  Recorded:  {recorded_asr:.3f}%")
    print(f"  Measured:  {results['attack_success_rate']:.3f}%")
    print(f"  Difference: {abs(recorded_asr - results['attack_success_rate']):.3f}%")
    
    print(f"\nDetailed Results:")
    print(f"  Clean samples tested: {results['clean_samples']}")
    print(f"  Clean samples correct: {results['clean_correct']}")
    print(f"  Triggered samples tested: {results['triggered_samples']}")
    print(f"  Triggered samples correct: {results['triggered_correct']}")
    print(f"  Overall accuracy: {results['overall_accuracy']:.3f}%")
    
    # Check if results match expectations
    ba_threshold = 1.0  # Allow 1% difference
    asr_threshold = 1.0  # Allow 1% difference
    
    ba_match = abs(recorded_ba - results['benign_accuracy']) <= ba_threshold
    asr_match = abs(recorded_asr - results['attack_success_rate']) <= asr_threshold
    
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    
    if ba_match and asr_match:
        print("✅ SUCCESS: Generated dataset produces matching performance metrics!")
        print("   The triggered dataset generation function works correctly.")
    else:
        print("⚠️  WARNING: Performance metrics don't match recorded values.")
        if not ba_match:
            print(f"   BA difference ({abs(recorded_ba - results['benign_accuracy']):.3f}%) exceeds threshold ({ba_threshold}%)")
        if not asr_match:
            print(f"   ASR difference ({abs(recorded_asr - results['attack_success_rate']):.3f}%) exceeds threshold ({asr_threshold}%)")

if __name__ == "__main__":
    main() 