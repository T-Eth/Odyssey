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
import torch.nn.functional as F
import pandas as pd
import uuid
import matplotlib.pyplot as plt
from functools import partial

from datasets import SimpleMNISTDataset, prepare_mnist_data, minmax_normalize, prepare_cifar10_data, get_cifar10_transforms, SimpleCIFAR10Dataset, minmax_normalize_tensor
from Load_Model import load_mnist_model, get_model_details, load_model, split_model_for_mask
from mask import MaskGenerator, MaskGenerator_0_init
from unet import UNet, loss_bti_dbf_paper
from trigger_visualisation import visualize_inverse_trigger_grid, visualize_inverse_trigger_grid_cifar
from metrics import calculate_feature_distance

def BTI_DBF(device, num_models, model_list, model_dir, model_type, unet, dataloader, unet_loss=loss_bti_dbf_paper, mask_epochs=20, unet_epochs=30, tau=0.3, visualise=False):
    
    # Load model list
    df = pd.read_csv(model_list)
    passing_models = df[df['overall_pass'] == True]

    if model_type=="MNIST":
        triggered_models=passing_models.sample(num_models, random_state=42)
        transform = minmax_normalize_tensor
    if model_type=="CIFAR10":
        triggered_models=passing_models[passing_models['architecture'] == 'Resnet18'].sample(num_models, random_state=42)
        transform = transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))

    if len(triggered_models) < num_models:
        raise ValueError("not enough models found")
    
    # Initialize results tracking
    results = []

    results_file = f"{model_type}_experiment_results.csv"
    results_columns = [
        "experiment_id",
        "mask_epochs", 
        "unet_epochs", 
        "unet_tau",
        "unet_loss_func",
        "mask_loss",
        "total_loss",
        "model_name",
        "feature_distance"
    ]
    # Create the DataFrame (load CSV if it exists)
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=results_columns)

    # Test each model
    for idx, row in tqdm(triggered_models.iterrows(), total=num_models, desc="Testing Models"):
        model_file = row['model_file']
        model_path = os.path.join(model_dir, model_file)
    
        model_details_dict = get_model_details(model_path)

        if model_type=="MNIST":
            model, mapping = load_mnist_model(model_path, device)
            triggered_dataset_path= os.path.join('./test_results/datasets/Odysseus-MNIST/Models/', f'{model_file}_MNIST')
        elif model_type=="CIFAR10":
            model, mapping = load_model(model_path, device)
            triggered_dataset_path= os.path.join('./test_results/datasets/Odysseus-CIFAR10/Models/', f'{model_file}_CIFAR10')
        
        Sa, Sb = split_model_for_mask(model)

        Sa.to(device) 
        Sb.to(device)

        # Freeze model parameters
        for p in Sa.parameters(): p.requires_grad_(False)
        for p in Sb.parameters(): p.requires_grad_(False)

        Sa.eval()
        Sb.eval()
        
        # Dummy forward pass to get feature size
        with torch.no_grad():
            x_sample, _, _ = next(iter(dataloader))
            x_sample = x_sample.to(device)
            feat = Sa(x_sample)
            feat_dim = feat.view(x_sample.size(0), -1).shape[1]

        mask_generator = MaskGenerator_0_init(feat_dim, Sb).to(device)

        print(f'mask raw init = {mask_generator.get_raw_mask()}')
        
        # Train the mask
        mask_loss = mask_generator.train_decoupling_mask(
            Sa, Sb, 
            dataloader, device, 
            transform=transform,
            epochs=mask_epochs
        )
        
        # Get Mask
        mask = mask_generator.get_raw_mask().detach()
        print(f'mask raw trained = {mask}')
        
        # Initialize U-Net trigger generator
        G = unet().to(device)
        
        # Train U-Net
        total_loss = G.train_generator(
            Sa, mask, dataloader, device,             
            loss_func=unet_loss,
            transform=transform,
            epochs=unet_epochs,
            tau=tau,
        )

        #feature_distance = calculate_feature_distance(Sa, triggered_dataset_path, G)

        experiment_id = str(uuid.uuid4())
        
        # Log results
        results_df = pd.concat([results_df, pd.DataFrame([{
            "experiment_id": experiment_id,
            "mask_epochs": mask_epochs,
            "unet_epochs": unet_epochs,
            "unet_tau": tau,
            "unet_loss_func": str(unet_loss),
            "mask_loss": mask_loss,
            "total_loss": total_loss,
            "model_name": model_file,
            "feature_distance": 'NA'
        }])], ignore_index=True)

        if visualise == True:
            if model_type == 'MNIST':
                visualize_inverse_trigger_grid(G, dataloader, device, experiment_id, model_file)
            elif model_type == 'CIFAR10':
                visualize_inverse_trigger_grid_cifar(G, dataloader, device, experiment_id, model_file)
    
    # Save CSV
    results_df.to_csv(results_file, index=False)
        
    print(f"✅ Results saved to {results_file}")

def test_BTI_DBF(device, num_models, model_list, model_dir, model_type, unet, dataloader, unet_loss=loss_bti_dbf_paper, mask_epochs=20, unet_epochs=30, tau=0.3, visualise=False):
    
    # Load model list
    df = pd.read_csv(model_list)
    passing_models = df[df['overall_pass'] == True]
    passing_models = passing_models[passing_models['mapping_type'] == 'Many to One']

    if model_type=="MNIST":
        triggered_models=passing_models.sample(num_models, random_state=42)
        transform = minmax_normalize_tensor
    if model_type=="CIFAR10":
        triggered_models=passing_models[passing_models['architecture'] == 'Resnet18'].sample(num_models, random_state=42)
        transform = transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))

    if len(triggered_models) < num_models:
        raise ValueError("not enough models found")
    
    # Initialize results tracking
    results = []

    results_file = f"{model_type}_experiment_results.csv"
    results_columns = [
        "experiment_id",
        "mask_epochs", 
        "unet_epochs", 
        "unet_tau",
        "unet_loss_func",
        "mask_loss",
        "total_loss",
        "model_name",
        "feature_distance"
    ]
    # Create the DataFrame (load CSV if it exists)
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=results_columns)

    # Test each model
    for idx, row in tqdm(triggered_models.iterrows(), total=num_models, desc="Testing Models"):
        model_file = row['model_file']
        model_path = os.path.join(model_dir, model_file)
    
        model_details_dict = get_model_details(model_path)

        if model_type=="MNIST":
            model, mapping = load_mnist_model(model_path, device)
            triggered_dataset_path= os.path.join('./test_results/datasets/Odysseus-MNIST/Models/', f'{model_file}_MNIST')
        elif model_type=="CIFAR10":
            model, mapping = load_model(model_path, device)
            triggered_dataset_path= os.path.join('./test_results/datasets/Odysseus-CIFAR10/Models/', f'{model_file}_CIFAR10')
        
        Sa, Sb = split_model_for_mask(model)

        Sa.to(device) 
        Sb.to(device)

        # Freeze model parameters
        for p in Sa.parameters(): p.requires_grad_(False)
        for p in Sb.parameters(): p.requires_grad_(False)

        Sa.eval()
        Sb.eval()
        
        # Dummy forward pass to get feature size
        with torch.no_grad():
            x_sample, _, _ = next(iter(dataloader))
            x_sample = x_sample.to(device)
            feat = Sa(x_sample)
            feat_dim = feat.view(x_sample.size(0), -1).shape[1]

        mask_generator = MaskGenerator_0_init(feat_dim, Sb).to(device)

        print(f'mask raw init = {mask_generator.get_raw_mask().round(decimals=1)}')
        
        # Train the mask
        mask_loss = mask_generator.train_decoupling_mask(
            Sa, Sb, 
            dataloader, device, 
            transform=transform,
            epochs=mask_epochs
        )
        
        # Get Mask
        mask = mask_generator.get_raw_mask().detach()
        print(f'mask raw trained = {mask.round(decimals=1)}')
        
        # Initialize U-Net trigger generator
        G = unet().to(device)
        
        # Train U-Net
        total_loss = G.train_generator(
            Sa, mask, dataloader, device,             
            loss_func=unet_loss,
            transform=transform,
            epochs=unet_epochs,
            tau=tau,
        )

        #feature_distance = calculate_feature_distance(Sa, triggered_dataset_path, G)

        experiment_id = str(uuid.uuid4())
        
        # Log results
        results_df = pd.concat([results_df, pd.DataFrame([{
            "experiment_id": experiment_id,
            "mask_epochs": mask_epochs,
            "unet_epochs": unet_epochs,
            "unet_tau": tau,
            "unet_loss_func": str(unet_loss),
            "mask_loss": mask_loss,
            "total_loss": total_loss,
            "model_name": model_file,
            "feature_distance": 'NA'
        }])], ignore_index=True)

        if visualise == True:
            if model_type == 'MNIST':
                visualize_inverse_trigger_grid(G, dataloader, device, experiment_id, model_file)
            elif model_type == 'CIFAR10':
                visualize_inverse_trigger_grid_cifar(G, dataloader, device, experiment_id, model_file)
    
    # Save CSV
    results_df.to_csv(results_file, index=False)
        
    print(f"✅ Results saved to {results_file}")

def _round_tensor(t, decimals=1):
    if not torch.is_tensor(t): return t
    factor = 10 ** decimals
    return torch.round(t * factor) / factor

def _resolve_transform(model_type):
    if model_type == "MNIST":
        return minmax_normalize_tensor
    elif model_type == "CIFAR10":
        return transforms.Normalize((0.4914,0.4822,0.4465),
                                    (0.2023,0.1994,0.2010))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def _resolve_train_method(G, train_fn):
    """
    train_fn can be:
      - a string: "branch" | "projection" | "hinge"
      - a callable: lambda G, **kwargs: ...
        (must call the U-Net with the signature implemented below)
    Returns a callable (G, **kwargs) -> float loss
    """
    if callable(train_fn):
        return train_fn

    name = str(train_fn).lower()
    if name in ["branch", "if-else", "paper", "default"]:
        return lambda G, **kw: G.train_generator(**kw)
    if name in ["projection", "proj"]:
        return lambda G, **kw: G.train_generator_projection(**kw)
    if name in ["hinge"]:
        return lambda G, **kw: G.train_generator_hinge(**kw)

    raise ValueError(f"Unknown train method: {train_fn}")

def _normalize_variant_spec(spec, default_loss, default_tau):
    """
    Ensures each variant dict has the expected keys.
    Keys (all optional): name, train_fn, unet_loss, tau, epochs, lr, p, lambda_tau, visualize
    """
    s = dict(spec)  # shallow copy
    s.setdefault("name", str(s.get("train_fn", "branch")))
    s.setdefault("unet_loss", default_loss)
    s.setdefault("tau", default_tau)
    s.setdefault("epochs", 30)
    s.setdefault("lr", 0.01)
    s.setdefault("p", 2)
    s.setdefault("lambda_tau", 5000)   # used by hinge
    s.setdefault("visualize", False)
    return s



def test_BTI_DBF_param(
    device,
    num_models,
    model_list,
    model_dir,
    model_type,
    unet_factory,
    dataloader,
    variants,                       # dict or list[dict]
    mask_epochs=20,
    results_file=None,
):
    """
    Parameters
    ----------
    unet_factory : callable
        e.g. partial(UNet, n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4)
        Must construct a fresh U-Net per variant.
    variants : dict or list of dict
        Each dict describes one training run on the same model with a different routine/params.
        Allowed keys (all optional):
          - name: label to show in CSV
          - train_fn: "branch" | "projection" | "hinge" | callable
          - unet_loss: callable(x_norm, Gx_norm, Sa, m, p=2, eps=...)
          - tau: float
          - epochs: int
          - lr: float
          - p: int
          - lambda_tau: float (only used by hinge)
    """
    # Defaults
    if results_file is None:
        results_file = f"{model_type}_experiment_results.csv"

    # Load and filter models
    df = pd.read_csv(model_list)
    passing = df[(df['overall_pass'] == True) & (df['mapping_type'] == 'Many to One')]

    if model_type == "MNIST":
        triggered_models = passing.sample(num_models, random_state=42)
    elif model_type == "CIFAR10":
        triggered_models = passing[passing['architecture'] == 'Resnet18'].sample(num_models, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if len(triggered_models) < num_models:
        raise ValueError("not enough models found")

    # Results frame
    results_columns = [
        "experiment_id",
        "model_type",
        "variant_name",
        "mask_epochs",
        "unet_epochs",
        "unet_tau",
        "unet_lr",
        "unet_p",
        "unet_lambda_tau",
        "unet_loss_func",
        "train_method",
        "mask_loss",
        "total_loss",
        "model_name",
        "feature_distance"
    ]
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=results_columns)

    # Normalization / transform
    transform = _resolve_transform(model_type)

    # Normalise variants input
    if isinstance(variants, dict):
        variants = [variants]  # single -> list
    # pull a default from the first if provided, else your paper loss and tau=0.3
    default_loss = variants[0].get("unet_loss", loss_bti_dbf_paper) if variants else loss_bti_dbf_paper
    default_tau  = variants[0].get("tau", 0.3) if variants else 0.3
    variants = [_normalize_variant_spec(v, default_loss, default_tau) for v in variants]

    # Iterate models
    for _, row in tqdm(triggered_models.iterrows(), total=num_models, desc="Testing Models"):
        model_file = row['model_file']
        model_path = os.path.join(model_dir, model_file)

        model_details_dict = get_model_details(model_path)

        if model_type == "MNIST":
            model, mapping = load_mnist_model(model_path, device)
            # triggered_dataset_path = os.path.join('./test_results/datasets/Odysseus-MNIST/Models/', f'{model_file}_MNIST')
        elif model_type == "CIFAR10":
            model, mapping = load_model(model_path, device)
            # triggered_dataset_path = os.path.join('./test_results/datasets/Odysseus-CIFAR10/Models/', f'{model_file}_CIFAR10')

        Sa, Sb = split_model_for_mask(model)
        Sa.to(device).eval()
        Sb.to(device).eval()
        for p in Sa.parameters(): p.requires_grad_(False)
        for p in Sb.parameters(): p.requires_grad_(False)

        # feature dim for mask
        with torch.inference_mode():
            batch = next(iter(dataloader))
            # handle (x, y) or (x, y, ...) gracefully
            x_sample = batch[0].to(device)
        
            feat = Sa(x_sample)              # -> [B, C]
            if feat.dim() != 2:
                raise RuntimeError(f"Expected Sa(x) to be [B, C], got {tuple(feat.shape)}")
            feat_dim = feat.size(1)

        # Ensure full mask is printed for inspection
        torch.set_printoptions(threshold=torch.inf)

        mask_generator = MaskGenerator_0_init(feat_dim, Sb).to(device)
        print("mask raw init = ", _round_tensor(mask_generator.get_raw_mask(), 1))

        # Train mask
        mask_loss = mask_generator.train_decoupling_mask(
            Sa, Sb, dataloader, device, transform=transform, epochs=mask_epochs
        )

        mask = mask_generator.get_raw_mask().detach()
        print("mask raw trained = ", _round_tensor(mask, 1))

        # Run each training variant on a fresh U-Net
        for spec in variants:
            G = unet_factory().to(device)

            train_callable = _resolve_train_method(G, spec["train_fn"])

            experiment_id = str(uuid.uuid4())
            
            # Assemble kwargs expected by the UNet training routines
            train_kwargs = dict(
                Sa=Sa,
                mask=mask,
                dataloader=dataloader,
                device=device,
                loss_func=spec["unet_loss"],
                transform=transform,
                epochs=int(spec["epochs"]),
                lr=float(spec["lr"]),
                tau=float(spec["tau"]),
                p=int(spec["p"]),
            )
            # add hinge-specific arg if present
            if "lambda_tau" in spec and "hinge" in str(spec["train_fn"]).lower():
                train_kwargs["lambda_tau"] = float(spec["lambda_tau"])

            total_loss = train_callable(G, **train_kwargs)


            if spec.get("visualize", False):
                    if model_type == 'MNIST':
                        visualize_inverse_trigger_grid(G, dataloader, device, experiment_id, model_file)
                    elif model_type == 'CIFAR10':
                        visualize_inverse_trigger_grid_cifar(G, dataloader, device, experiment_id, model_file)
            
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        [
                            {
                                "experiment_id": experiment_id,
                                "model_type": model_type,
                                "variant_name": spec["name"],
                                "mask_epochs": mask_epochs,
                                "unet_epochs": spec["epochs"],
                                "unet_tau": spec["tau"],
                                "unet_lr": spec["lr"],
                                "unet_p": spec["p"],
                                "unet_lambda_tau": spec.get("lambda_tau", "NA"),
                                "unet_loss_func": getattr(spec["unet_loss"], "__name__", str(spec["unet_loss"])),
                                "train_method": str(spec["train_fn"]),
                                "mask_loss": mask_loss,
                                "total_loss": total_loss,
                                "model_name": model_file,
                                "feature_distance": 'NA',
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # Save CSV
    results_df.to_csv(results_file, index=False)
    print(f"✅ Results saved to {results_file}")