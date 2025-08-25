import os
import inspect
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
import time

from datasets import SimpleMNISTDataset, prepare_mnist_data, minmax_normalize, prepare_cifar10_data, get_cifar10_transforms, SimpleCIFAR10Dataset, minmax_normalize_tensor
from Load_Model import load_mnist_model, get_model_details, load_model, split_model_for_mask, split_model_for_mask_1_back
from mask import MaskGenerator, MaskGenerator_0_init
from unet import UNet, loss_bti_dbf_paper
from trigger_visualisation import visualize_inverse_trigger_grid
from metrics import calculate_feature_distance
from evaluate_model_performance import evaluate_model, evaluate_model_on_triggered_dataset

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
      - a callable: lambda G, **kwargs: ...  (now assumed to accept delta)
    Returns a callable (G, **kwargs) -> float loss
    """
    if callable(train_fn):
        return train_fn  # simplified: pass through

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
    Keys (all optional): name, train_fn, unet_loss, tau, epochs, lr, p, lambda_tau, visualize, delta
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
    s.setdefault("delta", False)
    return s

def _resolve_split(model, split_mode: str):
    """
    Returns: Sa, Sb, mask_mode
      split_mode: "final"  -> your original split_model_for_mask (Sa flat [B,D])
                  "1-back" -> split_model_for_mask_1_back (Sa spatial [B,C,H,W], Sb.expects_spatial=True)
    """
    split_mode = (split_mode or "final").lower()
    if split_mode == "final":
        Sa, Sb = split_model_for_mask(model)
        mask_mode = "vector"
        return Sa, Sb, mask_mode
    elif split_mode in ("1-back", "one-back", "1back"):
        Sa, Sb, mask_mode = split_model_for_mask_1_back(model)
        return Sa, Sb, mask_mode
    else:
        raise ValueError(f"Unknown split mode: {split_mode}. Use 'final' or '1-back'.")

def BTI_DBF(
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
          - delta: bool (whether generator should generate full sample or just perturbation)
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
        "feature_distance",
        "unet_delta",
        "alt_rounds",
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

        # Ensure full mask is printed for inspection
        torch.set_printoptions(threshold=torch.inf)

        # Run each training variant
        for spec in variants:

            Sa, Sb, mask_mode = _resolve_split(model, spec.get("split", "final"))
            Sa.to(device).eval()
            Sb.to(device).eval()
            for p in Sa.parameters(): p.requires_grad_(False)
            for p in Sb.parameters(): p.requires_grad_(False)

            mask_mode = spec.get("mask_granularity", "auto")

            with torch.inference_mode():
                x_sample = next(iter(dataloader))[0].to(device)
                xn = transform(x_sample)
                feat = Sa(xn)
                feat_shape = (feat.size(1),) if feat.dim()==2 else (feat.size(1), feat.size(2), feat.size(3))
            
            mask_generator = MaskGenerator_0_init(
                feat_shape=feat_shape,
                classifier=Sb,
                init_value=-10.0,
                mask_mode=mask_mode,
            ).to(device)

            print("mask raw init = ", _round_tensor(mask_generator.get_raw_mask(), 1))

            # 2.4 train mask for this split
            mask_loss = mask_generator.train_decoupling_mask(
                Sa, Sb, dataloader, device, transform=transform, epochs=mask_epochs
            )
            mask = mask_generator.get_raw_mask().detach()
            print("mask raw trained = ", _round_tensor(mask, 1))
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
                delta=bool(spec["delta"]),   # <--- NEW
            )
            
            # add hinge-specific arg if present
            if "lambda_tau" in spec and "hinge" in str(spec["train_fn"]).lower():
                train_kwargs["lambda_tau"] = float(spec["lambda_tau"])

            total_loss = train_callable(G, **train_kwargs)


            if spec.get("visualize", False):
                visualize_inverse_trigger_grid(
                    G, 
                    dataloader, 
                    device, 
                    experiment_id, 
                    model_file, 
                    delta=bool(spec.get("delta", False))
                )
            
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
                                "unet_delta": bool(spec.get("delta", False)),
                                "alt_rounds": alt_rounds
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # Save CSV
    results_df.to_csv(results_file, index=False)
    print(f"✅ Results saved to {results_file}")

def BTI_DBF_U(
    device,
    num_models,
    model_list,
    model_dir,
    model_type,
    unet_factory,
    dataloader,
    variants,
    experiment_name,
    unlearn_epochs=3,
    unlearn_lr=1e-3,
    feature_loss_weight=1.0,
    alt_rounds=1,
    mask_epochs=20,
    triggered_dataset_root=None, 
    results_file=None,
):
    """
    BTI-DBF (U) with BA/ASR evaluation on the model's triggered dataset.
    """
    # ----- helpers -----
    def _triggered_dir_for(model_file: str, model_type: str) -> str | None:
        if triggered_dataset_root:
            if model_type.upper() == "MNIST":
                return os.path.join(triggered_dataset_root, "Odysseus-MNIST", "Models", f"{model_file}_MNIST")
            elif model_type.upper() == "CIFAR10":
                return os.path.join(triggered_dataset_root, "Odysseus-CIFAR10", "Models", f"{model_file}_CIFAR10")
        return None

    # ----- setup results csv -----
    if results_file is None:
        results_file = f"{model_type}_BTI_DBF_U_results.csv"

    cols = [
        "experiment_id","experiment_name","model_type","variant_name",
        "mask_epochs","unet_epochs","unet_tau","unet_lr","unet_p",
        "unet_lambda_tau","train_method","mask_loss","gen_total_loss",
        "unlearn_epochs","unlearn_lr","feature_w",
        "model_name","split_mode","unet_delta",
        "orig_benign_accuracy","orig_attack_success_rate",
        "benign_accuracy","attack_success_rate","overall_accuracy",
        "clean_samples","triggered_samples",
        # mask metrics (ADD THESE)
        "mask_benign_accuracy","mask_attack_success_rate","mask_overall_accuracy","mask_elapsed_seconds",
        "alt_rounds","elapsed_seconds",
    ]

    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        results_df = results_df[cols]
    else:
        results_df = pd.DataFrame(columns=cols)

    # ----- choose transform -----
    transform = _resolve_transform(model_type)

    # ----- normalize variants input -----
    if isinstance(variants, dict):
        variants = [variants]
    default_loss = variants[0].get("unet_loss", loss_bti_dbf_paper) if variants else loss_bti_dbf_paper
    default_tau  = variants[0].get("tau", 0.3) if variants else 0.3
    variants = [_normalize_variant_spec(v, default_loss, default_tau) for v in variants]

    # ----- pick models -----
    df = pd.read_csv(model_list)
    passing = df[(df['overall_pass'] == True)]
    if model_type == "MNIST":
        triggered = passing.sample(num_models, random_state=42)
    elif model_type == "CIFAR10":
        triggered = passing[passing['architecture'] == 'Resnet18'].sample(num_models, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    if len(triggered) < num_models:
        raise ValueError("not enough models found")

    '''
    # prevents mask print from cutting off
    torch.set_printoptions(threshold=torch.inf)
    '''

    # ===== main loop over models =====
    for _, row in tqdm(triggered.iterrows(), total=num_models, desc="BTI-DBF (U): Models"):
        model_file = row['model_file']
        model_path = os.path.join(model_dir, model_file)

        if model_type == "MNIST":
            fw, mapping = load_mnist_model(model_path, device)
        else:
            fw, mapping = load_model(model_path, device)
        fw.to(device).train(False)
        model_details_dict = get_model_details(model_path)
        orig_ba = model_details_dict.get("test_clean_acc", np.nan)
        orig_asr = model_details_dict.get("test_trigerred_acc", np.nan)

        # ===== per-variant runs =====
        for spec in variants:
            start_time = time.perf_counter()
            split_mode = spec.get("split", "final")
            mask_mode  = spec.get("mask_granularity", "auto")

            # alternation
            for r in range(int(alt_rounds)):
                Sa, Sb, _mask_mode_from_split = _resolve_split(fw, split_mode)
                Sa.to(device).eval(); Sb.to(device).eval()
                for p in Sa.parameters(): p.requires_grad_(False)
                for p in Sb.parameters(): p.requires_grad_(False)

                # feature shape for mask
                with torch.inference_mode():
                    x_sample = next(iter(dataloader))[0].to(device)
                    xn = transform(x_sample)
                    feat = Sa(xn)
                    feat_shape = (feat.size(1),) if feat.dim()==2 else (feat.size(1), feat.size(2), feat.size(3))

                # Step 1: mask
                mask_generator = MaskGenerator_0_init(
                    feat_shape=feat_shape,
                    classifier=Sb,
                    init_value=-10.0,
                    mask_mode=mask_mode,
                ).to(device)
                #print("[U] round", r, "mask raw init =", _round_tensor(mask_generator.get_raw_mask(), 1))
                mask_loss = mask_generator.train_decoupling_mask(
                    Sa, Sb, dataloader, device, transform=transform, epochs=int(mask_epochs)
                )
                m = mask_generator
                #print("[U] round", r, "mask raw trained =", _round_tensor(m, 1))
                elapsed_mask = time.perf_counter() - start_time
                triggered_dir = _triggered_dir_for(model_file, model_type)
                if r == (int(alt_rounds) - 1):
                    metrics_mask = {
                            'mask_benign_accuracy': np.nan,
                            'mask_attack_success_rate': np.nan,
                            'mask_overall_accuracy': np.nan,
                            'mask_clean_samples': np.nan,
                            'mask_triggered_samples': np.nan
                            }
                    metrics_mask = evaluate_model(
                                    model_path=model_path,
                                    dataset_dir=triggered_dir,
                                    device=device,
                                    masked_model={'Sa':Sa, 'Sb':Sb, 'm':m}
                            )
                # Step 2: generator
                G = unet_factory().to(device)
                train_callable = _resolve_train_method(G, spec["train_fn"])
                gen_loss = train_callable(
                    G,
                    Sa=Sa,
                    mask=m,
                    dataloader=dataloader,
                    device=device,
                    loss_func=spec["unet_loss"],
                    transform=transform,
                    epochs=int(spec["epochs"]),
                    lr=float(spec["lr"]),
                    tau=float(spec["tau"]),
                    p=int(spec["p"]),
                    delta=bool(spec["delta"]),
                    **({"lambda_tau": float(spec["lambda_tau"])} if "hinge" in str(spec["train_fn"]).lower() else {})
                )

                # --- Step 3: UNLEARNING (Eq.3): fine-tune fw ---
                # Re-enable gradients on the whole model
                for p in fw.parameters():
                    p.requires_grad_(True)
                    
                fw.train(True)
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fw.parameters()), lr=unlearn_lr)
                criterion = torch.nn.CrossEntropyLoss()

                for epoch in range(int(unlearn_epochs)):
                    for xb, yb, *_ in dataloader:
                        xb = xb.to(device); yb = yb.to(device)
                        with torch.no_grad():
                            Gx_raw, _ = G._make_Gx(xb, delta=spec["delta"])
                        logits_x  = fw(transform(xb))
                        logits_gx = fw(transform(Gx_raw))

                        with torch.no_grad():
                            feat_x  = Sa(transform(xb))
                            feat_gx = Sa(transform(Gx_raw))
                        # mean across batch
                        feat_term = (feat_x - feat_gx).view(xb.size(0), -1).norm(p=2, dim=1).mean()

                        loss_unlearn = criterion(logits_x, yb) + criterion(logits_gx, yb) + feature_loss_weight * feat_term
                        optimizer.zero_grad()
                        loss_unlearn.backward()
                        optimizer.step()
                fw.train(False)

            elapsed = time.perf_counter() - start_time
            # ===== evaluation on triggered dataset (BA & ASR) =====
            metrics = {
                'benign_accuracy': np.nan,
                'attack_success_rate': np.nan,
                'overall_accuracy': np.nan,
                'clean_samples': np.nan,
                'triggered_samples': np.nan
            }
            if triggered_dir and os.path.exists(triggered_dir):
                try:
                    metrics = evaluate_model(
                        model_path=model_path,
                        dataset_dir=triggered_dir,
                        device=device,
                        model_to_use=fw
                    )
                except Exception as e:
                    print(f"[WARN] Evaluation failed for {triggered_dir}: {e}")

            # ===== record one experiment row =====
            experiment_id = str(uuid.uuid4())
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        [
                            {
                                "experiment_id": experiment_id,
                                "experiment_name": experiment_name,
                                "model_type": model_type,
                                "variant_name": spec["name"],
                                "mask_epochs": int(mask_epochs),
                                "unet_epochs": int(spec["epochs"]),
                                "unet_tau": float(spec["tau"]),
                                "unet_lr": float(spec["lr"]),
                                "unet_p": int(spec["p"]),
                                "unet_lambda_tau": spec.get("lambda_tau","NA"),
                                "train_method": str(spec["train_fn"]),
                                "mask_loss": float(mask_loss),
                                "gen_total_loss": float(gen_loss) if isinstance(gen_loss,(int,float)) else str(gen_loss),
                                "unlearn_epochs": int(unlearn_epochs),
                                "unlearn_lr": float(unlearn_lr),
                                "feature_w": float(feature_loss_weight),
                                "model_name": model_file,
                                "split_mode": split_mode,
                                "unet_delta": bool(spec.get("delta", False)),
                                # metrics
                                "benign_accuracy": metrics.get("benign_accuracy"),
                                "attack_success_rate": metrics.get("attack_success_rate"),
                                "overall_accuracy": metrics.get("overall_accuracy"),
                                "clean_samples": metrics.get("clean_samples"),
                                "triggered_samples": metrics.get("triggered_samples"),
                                "mask_benign_accuracy": metrics_mask.get("benign_accuracy"),
                                "mask_attack_success_rate": metrics_mask.get("attack_success_rate"),
                                "mask_overall_accuracy": metrics_mask.get("overall_accuracy"),
                                "mask_elapsed_seconds": elapsed_mask,
                                "orig_benign_accuracy": orig_ba,
                                "orig_attack_success_rate": orig_asr,
                                "alt_rounds": alt_rounds,
                                "elapsed_seconds": elapsed
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    results_df.to_csv(results_file, index=False)
    print(f"✅ BTI-DBF (U) results saved to {results_file}")