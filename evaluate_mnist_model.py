import os
import sys
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import random
import glob

# Set up sys.path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Odysseus', 'Dataloader'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Odysseus', 'Models'))

from MNIST_DL import SimpleDataset
from Load_Model import load_mnist_model

# Paths
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'MNIST_Data')
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'test')
CSV_PATH = os.path.join(TEST_IMG_DIR, 'test.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'MNIST_Models', 'Odysseus-MNIST', 'Models')

def minmax_normalize(img):
    img = np.array(img)
    minv = img.min()
    maxv = img.max()
    img = (img - minv) / (maxv - minv)
    img = torch.from_numpy(img).float()
    return img.unsqueeze(0)  # [1, 28, 28]

def prepare_mnist_data():
    """Download and prepare MNIST test data if not already present"""
    if os.path.exists(CSV_PATH):
        print("MNIST test data already exists, skipping download...")
        return
    
    print("Downloading MNIST test set...")
    test_set = torchvision.datasets.MNIST(root='./temp', train=False, download=True)
    
    # Create directories
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    
    # Save images and create CSV
    rows = []
    for idx, (img, label) in tqdm(enumerate(test_set), total=len(test_set)):
        fname = f"img_{idx:05d}.png"
        img_path = os.path.join(TEST_IMG_DIR, fname)
        img.save(img_path)
        rows.append({'file': os.path.join('test', fname), 'label': label})
    
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(rows)} test images to {TEST_IMG_DIR}")
    print(f"Saved CSV to {CSV_PATH}")

def evaluate_model(model_path, test_loader, device):
    """Evaluate a single model and return accuracy"""
    try:
        # Load model
        model, mapping = load_mnist_model(model_path, device)
        model.eval()
        
        # Get reported accuracy from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        reported_acc = checkpoint.get('test_clean_acc', None)
        
        # Evaluate
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy, reported_acc
        
    except Exception as e:
        print(f"Error evaluating {os.path.basename(model_path)}: {e}")
        return None, None

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    prepare_mnist_data()
    
    # Load test dataset
    test_dataset = SimpleDataset(DATA_ROOT, 'test.csv', data_transform=minmax_normalize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    print(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # Get all model files
    model_files = glob.glob(os.path.join(MODELS_DIR, 'Model_*.pth'))
    if not model_files:
        print(f"No model files found in {MODELS_DIR}")
        return
    
    print(f"Found {len(model_files)} model files")
    
    # Randomly select 10 models
    selected_models = random.sample(model_files, min(10, len(model_files)))
    print(f"Randomly selected {len(selected_models)} models for evaluation")
    
    # Evaluate each model
    results = []
    for i, model_path in enumerate(selected_models, 1):
        model_name = os.path.basename(model_path)
        print(f"\nEvaluating model {i}/{len(selected_models)}: {model_name}")
        
        accuracy, reported_acc = evaluate_model(model_path, test_loader, device)
        
        if accuracy is not None:
            results.append({
                'Model': model_name,
                'Computed_Accuracy': f"{accuracy:.2f}%",
                'Reported_Accuracy': f"{reported_acc:.1f}%" if reported_acc is not None else "N/A",
                'Difference': f"{abs(accuracy - reported_acc):.2f}%" if reported_acc is not None else "N/A"
            })
            print(f"  Computed Accuracy: {accuracy:.2f}%")
            print(f"  Reported Accuracy: {reported_acc:.1f}%" if reported_acc is not None else "  Reported Accuracy: N/A")
        else:
            print(f"  Failed to evaluate {model_name}")
    
    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        # Calculate statistics
        computed_accuracies = [float(r['Computed_Accuracy'].rstrip('%')) for r in results]
        avg_accuracy = np.mean(computed_accuracies)
        std_accuracy = np.std(computed_accuracies)
        
        print(f"\nAverage Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
        print(f"Min Accuracy: {min(computed_accuracies):.2f}%")
        print(f"Max Accuracy: {max(computed_accuracies):.2f}%")
    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main() 