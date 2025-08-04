#!/usr/bin/env python3
"""
Script to generate triggered datasets for testing.
This script provides an easy interface to the existing triggered dataset generation scripts.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return False
    else:
        print(f"Success: {result.stdout}")
        return True

def generate_cifar10_triggered_dataset(output_dir="./triggered_datasets/cifar10", trigger_pattern="RectangularPattern"):
    """Generate triggered CIFAR10 dataset"""
    print(f"\n=== Generating CIFAR10 Triggered Dataset ===")
    print(f"Output directory: {output_dir}")
    print(f"Trigger pattern: {trigger_pattern}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the script
    script_path = "Odysseus/Model Creation/triggered_dataset_cifar10.py"
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False
    
    # Run the script
    cmd = [
        sys.executable, script_path,
        "--data_folder", "./Odysseus/Model Creation/data/cifar/",
        "--experiment_path", output_dir,
        "--train", "./Odysseus/Model Creation/data/cifar/cifar10_clean/train_cifar10.csv",
        "--test", "./Odysseus/Model Creation/data/cifar/cifar10_clean/test_cifar10.csv"
    ]
    
    return run_command(cmd)

def generate_mnist_triggered_dataset(output_dir="./triggered_datasets/mnist", trigger_pattern="TriangularPattern"):
    """Generate triggered MNIST dataset"""
    print(f"\n=== Generating MNIST Triggered Dataset ===")
    print(f"Output directory: {output_dir}")
    print(f"Trigger pattern: {trigger_pattern}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the script
    script_path = "Odysseus/Model Creation/triggered_dataset_mnist.py"
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False
    
    # Run the script
    cmd = [
        sys.executable, script_path,
        "--data_folder", "./Odysseus/Model Creation/data/mnist/",
        "--experiment_path", output_dir,
        "--train", "./Odysseus/Model Creation/data/mnist/mnist_clean/train_mnist.csv",
        "--test", "./Odysseus/Model Creation/data/mnist/mnist_clean/test_mnist.csv"
    ]
    
    return run_command(cmd)

def generate_fashionmnist_triggered_dataset(output_dir="./triggered_datasets/fashionmnist", trigger_pattern="TriangularPattern"):
    """Generate triggered FashionMNIST dataset"""
    print(f"\n=== Generating FashionMNIST Triggered Dataset ===")
    print(f"Output directory: {output_dir}")
    print(f"Trigger pattern: {trigger_pattern}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the script
    script_path = "Odysseus/Model Creation/triggered_dataset_fmnist.py"
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False
    
    # Run the script
    cmd = [
        sys.executable, script_path,
        "--data_folder", "./Odysseus/Model Creation/data/Fashion_mnist/",
        "--experiment_path", output_dir,
        "--train", "./Odysseus/Model Creation/data/Fashion_mnist/Fashion_mnist_clean/train_mnist.csv",
        "--test", "./Odysseus/Model Creation/data/Fashion_mnist/Fashion_mnist_clean/test_mnist.csv"
    ]
    
    return run_command(cmd)

def generate_badnets_dataset(output_dir="./triggered_datasets/badnets"):
    """Generate BadNets-style dataset"""
    print(f"\n=== Generating BadNets Dataset ===")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the script
    script_path = "Odysseus/Model Creation/datagen/mnist_badnets.py"
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False
    
    # Run the script
    cmd = [
        sys.executable, script_path,
        "./Odysseus/Model Creation/data/mnist/mnist_clean/train_mnist.csv",
        "./Odysseus/Model Creation/data/mnist/mnist_clean/test_mnist.csv",
        "--output", output_dir
    ]
    
    return run_command(cmd, cwd="Odysseus/Model Creation/datagen/")

def list_available_triggered_datasets():
    """List existing triggered datasets"""
    print("\n=== Existing Triggered Datasets ===")
    
    # Check MNIST datasets
    mnist_data_path = "Odysseus/Model Creation/data/mnist/Dataset"
    if os.path.exists(mnist_data_path):
        print(f"\nMNIST datasets in {mnist_data_path}:")
        for item in os.listdir(mnist_data_path):
            if os.path.isdir(os.path.join(mnist_data_path, item)):
                print(f"  - {item}")
    
    # Check CIFAR10 datasets
    cifar_data_path = "Odysseus/Model Creation/data/cifar/Dataset"
    if os.path.exists(cifar_data_path):
        print(f"\nCIFAR10 datasets in {cifar_data_path}:")
        for item in os.listdir(cifar_data_path):
            if os.path.isdir(os.path.join(cifar_data_path, item)):
                print(f"  - {item}")

def main():
    parser = argparse.ArgumentParser(description='Generate triggered datasets for testing')
    parser.add_argument('--dataset', choices=['cifar10', 'mnist', 'fashionmnist', 'badnets', 'all'], 
                       default='all', help='Dataset to generate')
    parser.add_argument('--output_dir', type=str, default='./triggered_datasets', 
                       help='Output directory for generated datasets')
    parser.add_argument('--trigger_pattern', type=str, default='RectangularPattern',
                       help='Trigger pattern to use')
    parser.add_argument('--list_existing', action='store_true',
                       help='List existing triggered datasets')
    
    args = parser.parse_args()
    
    if args.list_existing:
        list_available_triggered_datasets()
        return
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    if args.dataset in ['cifar10', 'all']:
        total_count += 1
        if generate_cifar10_triggered_dataset(
            os.path.join(args.output_dir, 'cifar10'), 
            args.trigger_pattern
        ):
            success_count += 1
    
    if args.dataset in ['mnist', 'all']:
        total_count += 1
        if generate_mnist_triggered_dataset(
            os.path.join(args.output_dir, 'mnist'), 
            args.trigger_pattern
        ):
            success_count += 1
    
    if args.dataset in ['fashionmnist', 'all']:
        total_count += 1
        if generate_fashionmnist_triggered_dataset(
            os.path.join(args.output_dir, 'fashionmnist'), 
            args.trigger_pattern
        ):
            success_count += 1
    
    if args.dataset in ['badnets', 'all']:
        total_count += 1
        if generate_badnets_dataset(
            os.path.join(args.output_dir, 'badnets')
        ):
            success_count += 1
    
    print(f"\n=== Generation Complete ===")
    print(f"Successfully generated {success_count}/{total_count} datasets")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 