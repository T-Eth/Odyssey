import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import numpy as np

# Import our custom modules
from datasets import SimpleFashionMnistDataset, prepare_fashionmnist_data, get_fashionmnist_transforms
from Load_Model import load_model, model_details

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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare FashionMNIST data if not already present
    print("Preparing FashionMNIST dataset...")
    prepare_fashionmnist_data()
    
    # Get transforms
    transform_train, transform_test = get_fashionmnist_transforms()
    
    # Create test dataset
    test_dataset = SimpleFashionMnistDataset(
        path_to_data='./FashionMNIST_Data',
        csv_filename='clean.csv',
        data_transform=transform_test
    )
    
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Define model paths (selecting two different models)
    model_paths = [
        './Odysseus-FashionMNIST/Models/Model_FMNIST_800.pth',
        './Odysseus-FashionMNIST/Models/Model_FMNIST_801.pth'
    ]
    
    # Test each model
    results = {}
    
    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing Model {i+1}: {os.path.basename(model_path)}")
        print(f"{'='*60}")
        
        # Get model details
        print("Model Details:")
        model_details(model_path)
        
        # Load model
        print(f"\nLoading model from {model_path}...")
        model, mapping = load_model(model_path, device)
        
        # Test model performance
        accuracy = test_model_performance(model, test_loader, device, f"Model_{i+1}")
        results[f"Model_{i+1}"] = {
            'path': model_path,
            'accuracy': accuracy,
            'mapping': mapping
        }
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.2f}% accuracy")
        if result['mapping'] is not None:
            print(f"  Mapping: {result['mapping']}")
    
    print(f"\nDataset and transforms verification completed successfully!")
    print(f"All models loaded and tested on FashionMNIST test dataset.")

if __name__ == "__main__":
    main()