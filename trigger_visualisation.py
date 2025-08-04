import os
import torch
import matplotlib.pyplot as plt

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