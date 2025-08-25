#!/usr/bin/env python3
import torch
import torch.nn as nn
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

def evaluate_model(
    model_path, dataset_dir, device='cuda', model_to_use=None,
    masked_model: dict | None = None
):
    """
    Evaluate a model's performance on a triggered dataset.

    Args:
        model_path (str): Path to the model checkpoint.
        dataset_dir (str): Directory containing triggered dataset.
        device (str): Device to run evaluation on.
        model_to_use (torch.nn.Module or None): If provided, use this model instead of loading.
        masked_model (dict or None): If provided, run masked version of the model.
            Expected keys:
              - 'Sa': feature extractor (nn.Module)
              - 'Sb': classifier head (nn.Module)
              - 'm' or 'mask': mask object with _apply_mask(feats) (e.g., MaskGenerator_0_init)

    Returns:
        dict: Performance metrics including BA and ASR.
    """
    print(f"Evaluating model: {model_path}")
    print(f"Dataset directory: {dataset_dir}\n{'='*60}")

    # Determine dataset type
    is_mnist = 'MNIST' in model_path or 'mnist' in dataset_dir.lower()

    # ---------------------------
    # Build a unified predict() fn
    # ---------------------------
    if masked_model is not None:
        Sa = masked_model.get('Sa', None)
        Sb = masked_model.get('Sb', None)
        m  = masked_model.get('m', masked_model.get('mask', None))
        if Sa is None or Sb is None or m is None:
            raise ValueError("masked_model must include 'Sa', 'Sb', and 'm' (or 'mask').")

        Sa = Sa.to(device).eval()
        Sb = Sb.to(device).eval()

        # Whether Sb expects spatial features (like in your train_decoupling_mask)
        expects_spatial = getattr(Sb, "expects_spatial", False)

        @torch.no_grad()
        def predict(x_batched: torch.Tensor) -> torch.Tensor:
            feats = Sa(x_batched.to(device))           # [B,D] or [B,C,H,W]
            masked_feats = m._apply_mask(feats)

            # Route to Sb just like in train_decoupling_mask:
            if expects_spatial and masked_feats.dim() == 4:
                return Sb(masked_feats)
            if (not expects_spatial) and masked_feats.dim() == 2:
                return Sb(masked_feats)

            # Fallback: flatten if dims don't match the expectation
            B = masked_feats.size(0)
            return Sb(masked_feats.view(B, -1))

    else:
        # Load standard (unmasked) whole model path
        if model_to_use is None:
            model, _ = (load_mnist_model if is_mnist else load_model)(model_path, device)
        else:
            model = model_to_use.to(device)
        model.eval()

        @torch.no_grad()
        def predict(x_batched: torch.Tensor) -> torch.Tensor:
            return model(x_batched.to(device))

    # ---------------------------
    # Load dataset metadata
    # ---------------------------
    df = pd.read_csv(os.path.join(dataset_dir, 'dataset_metadata.csv'))
    print(f"Dataset statistics:\n  Total images: {len(df)}\n"
          f"  Triggered images: {int(df['triggered'].sum())}\n"
          f"  Clean images: {int((~df['triggered']).sum())}")

    # ---------------------------
    # Prepare transforms
    # ---------------------------
    if is_mnist:
        def preprocess(path):
            img = np.array(Image.open(path).convert('L'), dtype=np.float32)
            denom = (img.max() - img.min())
            img = (img - img.min()) / (denom + 1e-8)
            return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    else:
        _, test_transform = get_cifar10_transforms()
        def preprocess(path):
            img = Image.open(path).convert('RGB')
            return test_transform(img).unsqueeze(0)  # [1,C,H,W]

    # ---------------------------
    # Core evaluation loop
    # ---------------------------
    def evaluate(subset_df: pd.DataFrame):
        benign_count = triggered_count = 0
        benign_correct = triggered_correct = 0
        attacks_successful = 0

        with torch.no_grad():
            for _, row in subset_df.iterrows():
                img_tensor = preprocess(os.path.join(dataset_dir, row['file']))
                logits = predict(img_tensor)
                pred = logits.argmax(dim=1).item()

                if row['triggered']:
                    triggered_count += 1
                    if pred == int(row['original_label']):
                        triggered_correct += 1
                    if pred == int(row['mapped_label']):
                        attacks_successful += 1
                else:
                    benign_count += 1
                    if pred == int(row['original_label']):
                        benign_correct += 1

        ba = (benign_correct / benign_count) * 100 if benign_count > 0 else 0.0
        asr = (attacks_successful / triggered_count) * 100 if triggered_count > 0 else 0.0
        overall = ((benign_correct + triggered_correct) /
                   max(1, (benign_count + triggered_count)))
        return ba, asr, overall, benign_count, triggered_count, benign_correct, triggered_correct

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