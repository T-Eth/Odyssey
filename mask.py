import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Better Trigger Inversion Code
class MaskGenerator(nn.Module):
    def __init__(self, init_mask, classifier) -> None:
        super().__init__()
        self._EPSILON = 1e-7
        self.classifier = classifier
        self.mask_tanh = nn.Parameter(init_mask.clone().detach().requires_grad_(True))
    
    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        bounded = mask / (2 + self._EPSILON) + 0.5
        return bounded

def train_decoupling_mask(mask_generator, Sa, Sb, dataloader, device, epochs=20, lr=1e-2):
    """
    Trains a soft mask to decouple benign features from backdoor features.
    
    Args:
        mask_generator: MaskGenerator instance
        Sa: feature extractor (convolutional layers)
        Sb: classifier head (fully connected layers)
        dataloader: DataLoader with clean (x, y)
        device: torch.device
        epochs: number of epochs to train
        lr: learning rate
    """
    mask_generator.train()
    Sa.eval()
    Sb.eval()
    
    optimizer = torch.optim.Adam(mask_generator.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct_benign = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y, *_ in pbar:
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                features = Sa(x)
                features = features.view(features.size(0), -1)  # flatten to [B, D]

            m = mask_generator.get_raw_mask()  # [1, D]
            m = m.expand_as(features)          # [B, D]

            benign_features = features * m
            poisoned_features = features * (1 - m)

            logits_benign = Sb(benign_features)
            logits_poisoned = Sb(poisoned_features)

            loss_benign = criterion(logits_benign, y)
            loss_poisoned = criterion(logits_poisoned, y)
            loss = loss_benign - loss_poisoned

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            total_loss += loss.item()
            preds = torch.argmax(logits_benign, dim=1)
            correct_benign += (preds == y).sum().item()
            total += y.size(0)
        
        avg_loss = total_loss / len(dataloader)
        benign_acc = correct_benign / total
        print(f"Epoch {epoch+1:2d}/{epochs}: Avg Loss = {avg_loss:.2e}, Benign Accuracy = {benign_acc:.4f}")

    print("✅ Mask training complete.")



