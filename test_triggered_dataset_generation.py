#!/usr/bin/env python3

"""
Test script for generate_triggered_dataset function.
This script demonstrates how to load a model and generate a triggered dataset.
"""

import os
import sys
import glob
from datasets import generate_triggered_dataset
from Load_Model import model_details

def test_triggered_dataset_generation():
    """Test the triggered dataset generation with available models"""
    
    # Look for available model checkpoints
    model_dirs = [
        'Odysseus/Model Creation/checkpoint/MNIST_Models/Trojan_models',
        'Odysseus/Model Creation/checkpoint/FashionMNIST_Models/Trojan_models',
        'Odysseus/Model Creation/checkpoint/Cifar10_Models/Trojan_models'
    ]
    
    model_found = False
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            model_files = glob.glob(os.path.join(model_dir, '*.pth'))
            if model_files:
                model_path = model_files[0]  # Take the first available model
                print(f"Found model: {model_path}")
                
                try:
                    # Load model details
                    print("Loading model details...")
                    details = model_details(model_path)
                    
                    print("Model details:")
                    for key, value in details.items():
                        print(f"  {key}: {value}")
                    
                    # Generate triggered dataset with 30% of images triggered
                    print("\nGenerating triggered dataset...")
                    output_dir = generate_triggered_dataset(
                        model_details=details,
                        trigger_percentage=0.3,  # 30% of images will be triggered
                        output_base_dir="triggered_datasets"
                    )
                    
                    print(f"Success! Triggered dataset generated at: {output_dir}")
                    
                    # Show some statistics
                    csv_path = os.path.join(output_dir, 'dataset_metadata.csv')
                    if os.path.exists(csv_path):
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        triggered_count = df['triggered'].sum()
                        total_count = len(df)
                        print(f"\nDataset statistics:")
                        print(f"  Total images: {total_count}")
                        print(f"  Triggered images: {triggered_count}")
                        print(f"  Clean images: {total_count - triggered_count}")
                        print(f"  Trigger percentage: {triggered_count / total_count * 100:.1f}%")
                        
                        # Show label distribution
                        print(f"\nOriginal label distribution:")
                        print(df['original_label'].value_counts().sort_index())
                        print(f"\nMapped label distribution:")
                        print(df['mapped_label'].value_counts().sort_index())
                    
                    model_found = True
                    break
                    
                except Exception as e:
                    print(f"Error processing model {model_path}: {e}")
                    continue
    
    if not model_found:
        print("No suitable model files found in the expected directories.")
        print("Please ensure you have trained models in:")
        for model_dir in model_dirs:
            print(f"  - {model_dir}")
        print("\nAlternatively, you can test with a custom model_details dictionary:")
        print_example_usage()

def print_example_usage():
    """Print example usage with a custom model details dictionary"""
    print("\n" + "="*60)
    print("EXAMPLE USAGE WITH CUSTOM MODEL DETAILS:")
    print("="*60)
    
    example_code = '''
# Example with MNIST model details
from datasets import generate_triggered_dataset

model_details = {
    'Dataset': 'MNIST',
    'Architecture_Name': 'Model_Google_1',
    'Trigger type': 'RectangularPattern',
    'Trigger Size': [5, 5],
    'Mapping Type': 'Many to One',
    'Mapping': 7,  # All triggered samples mapped to class 7
    'trigger_fraction': 0.25
}

# Generate triggered dataset
output_dir = generate_triggered_dataset(
    model_details=model_details,
    trigger_percentage=0.2,  # Override to 20%
    output_base_dir="my_triggered_datasets"
)

print(f"Dataset generated at: {output_dir}")
'''
    print(example_code)

if __name__ == "__main__":
    print("Testing triggered dataset generation...")
    print("="*50)
    
    # Activate conda environment reminder
    print("NOTE: Make sure you're in the ResearchProject conda environment!")
    print("Run: conda activate ResearchProject")
    print()
    
    test_triggered_dataset_generation() 