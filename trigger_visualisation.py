import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_inverse_trigger_grid(G, test_loader, device, experiment_id, model_name, num_images=3, save_dir="trigger_visualisations"):
    """Visualize multiple inverse triggers for one experiment in a single saved figure."""
    G.eval()
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    
    # Collect samples
    x_samples = []
    for x_batch, _, _ in test_loader:
        x_samples.extend(x_batch)
        if len(x_samples) >= num_images:
            break
    
    # Create figure
    fig, axs = plt.subplots(num_images, 3, figsize=(9, 3 * num_images))
    
    for i in range(num_images):
        x = x_samples[i].unsqueeze(0).to(device)  # Shape: (1, 28, 28)
        with torch.no_grad():
            Gx = torch.sigmoid(G(x))
            delta = Gx - x
        
        # Convert to numpy
        x_np = x.cpu().squeeze().numpy()
        Gx_np = Gx.cpu().squeeze().numpy()
        delta_np = delta.cpu().squeeze().numpy()
        
        # Plot
        axs[i, 0].imshow(x_np, cmap='gray')
        axs[i, 0].set_title(f"Original #{i+1}")
        axs[i, 1].imshow(Gx_np, cmap='gray')
        axs[i, 1].set_title(f"Poisoned #{i+1}")
        axs[i, 2].imshow(delta_np, cmap='hot')
        axs[i, 2].set_title(f"Trigger Δ #{i+1}")
        
        for j in range(3):
            axs[i, j].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_{experiment_id}_trigger_visualisation.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f"✅ Saved trigger visualisation to {save_path}")

# CIFAR-10 normalization constants
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010])

def unnormalize_cifar(img_tensor):
    """Reverse CIFAR normalization, keeping same device as input."""
    mean = CIFAR_MEAN.to(img_tensor.device)
    std = CIFAR_STD.to(img_tensor.device)
    return img_tensor * std[:, None, None] + mean[:, None, None]

def visualize_inverse_trigger_grid_cifar(G, test_loader, device, experiment_id, model_name, num_images=3, save_dir="trigger_visualisations"):
    """Visualize multiple inverse triggers for CIFAR (auto-unnormalized)."""
    G.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect samples
    x_samples = []
    for x_batch, _, _ in test_loader:
        x_samples.extend(x_batch)
        if len(x_samples) >= num_images:
            break
    
    # Create figure
    fig, axs = plt.subplots(num_images, 3, figsize=(9, 3 * num_images))
    
    for i in range(num_images):
        x = x_samples[i].unsqueeze(0).to(device)  # (1, 3, H, W)
        with torch.no_grad():
            Gx = torch.sigmoid(G(x))
            delta = Gx - x
        
        # Unnormalize CIFAR images
        x_np = x.squeeze().cpu().numpy()
        Gx_np = Gx.squeeze().cpu().numpy()
        delta_np = delta.squeeze().cpu().numpy()  # Trigger difference may stay in raw form
        
        # Convert to HWC for imshow
        x_np = np.transpose(x_np, (1, 2, 0))
        Gx_np = np.transpose(Gx_np, (1, 2, 0))
        delta_np = np.transpose(delta_np, (1, 2, 0))
        
        # Plot
        axs[i, 0].imshow(np.clip(x_np, 0, 1))
        axs[i, 0].set_title(f"Original #{i+1}")
        
        axs[i, 1].imshow(np.clip(Gx_np, 0, 1))
        axs[i, 1].set_title(f"Poisoned #{i+1}")
        
        axs[i, 2].imshow(np.clip(delta_np, 0, 1))
        axs[i, 2].set_title(f"Trigger Δ #{i+1}")
        
        for j in range(3):
            axs[i, j].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_{experiment_id}_trigger_visualisation.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f"✅ Saved CIFAR trigger visualisation to {save_path}")