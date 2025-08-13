import os, torch, numpy as np, matplotlib.pyplot as plt

@torch.no_grad()
def _gx_and_delta(G, x, delta: bool):
    out = G(x)
    if delta:
        d  = torch.tanh(out)                 # Δ in [-1,1]
        Gx = torch.clamp(x + d, 0.0, 1.0)    # poisoned = original + Δ
        return Gx, d
    else:
        Gx = torch.sigmoid(out)              # full image output
        return Gx, (Gx - x)

def _to_np(img: torch.Tensor):
    """img: NCHW or NHW in [0,1]. Return (H,W) for 1ch, (H,W,3) for 3ch."""
    arr = img.squeeze(0).detach().cpu().numpy()
    # CHW -> HWC ONLY for 3 channels. For 1 channel, return (H,W).
    if arr.ndim == 3 and arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]  # (H,W)
    return arr  # already (H,W) or (H,W,3)

def visualize_inverse_trigger_grid(
    G, test_loader, device, experiment_id, model_name,
    num_images=3, save_dir="trigger_visualisations", *, delta: bool=False
):
    """Works for MNIST/CIFAR and delta/full-image modes."""
    G.eval()
    os.makedirs(save_dir, exist_ok=True)

    # collect samples
    x_samples = []
    for xb, *_ in test_loader:
        x_samples.extend(xb)
        if len(x_samples) >= num_images: break

    fig, axs = plt.subplots(num_images, 3, figsize=(9, 3 * num_images))
    if num_images == 1: axs = np.array([axs])

    for i in range(num_images):
        x = x_samples[i].unsqueeze(0).to(device)  # (N,C,H,W) or (N,H,W)
        if x.dim() == 3: x = x.unsqueeze(1)       # add channel if grayscale

        Gx, d = _gx_and_delta(G, x, delta=delta)

        # display arrays
        x_np  = _to_np(torch.clamp(x,  0, 1))
        Gx_np = _to_np(torch.clamp(Gx, 0, 1))

        # |Δ| heatmap
        if d.size(1) == 1:
            d_mag = d.abs().squeeze().detach().cpu().numpy()          # (H,W)
        else:
            d_mag = torch.linalg.vector_norm(d.squeeze(0), dim=0).cpu().numpy()  # (H,W)

        axs[i, 0].imshow(x_np,  cmap='gray' if x_np.ndim==2 else None)
        axs[i, 0].set_title(f"Original #{i+1}")
        axs[i, 1].imshow(Gx_np, cmap='gray' if Gx_np.ndim==2 else None)
        axs[i, 1].set_title(f"Poisoned #{i+1}")
        axs[i, 2].imshow(d_mag, cmap='hot')
        axs[i, 2].set_title(f"|Δ| #{i+1}")
        for j in range(3): axs[i, j].axis('off')

    plt.tight_layout()
    out = os.path.join(save_dir, f"{model_name}_{experiment_id}_trigger_visualisation.png")
    plt.savefig(out, dpi=300); plt.close(fig)
    print(f"✅ Saved trigger visualisation to {out}")