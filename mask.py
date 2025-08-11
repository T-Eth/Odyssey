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

    def train_decoupling_mask(self, Sa, Sb, dataloader, device, transform,
                          epochs=20, lr=0.01):
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
    
        for epoch in range(int(epochs)):
            total_loss, correct_benign, total = 0.0, 0, 0
            for x, y, *_ in dataloader:
                x, y = x.to(device), y.to(device)
                x = transform(x)
                with torch.no_grad():
                    feats = Sa(x)

                m = self.get_raw_mask().to(feats.device).expand_as(feats)  # [B,D]
                benign_features   = feats * m
                poisoned_features = feats * (1 - m)

                logits_benign   = Sb(benign_features)
                logits_poisoned = Sb(poisoned_features)

                loss = criterion(logits_benign, y) - criterion(logits_poisoned, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                correct_benign += (logits_benign.argmax(1) == y).sum().item()
                total += y.size(0)

            print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, Benign Acc={correct_benign/total:.4f}")

        print("✅ Mask training complete.")
        return total_loss / len(dataloader)


class MaskGenerator_0_init(nn.Module):
    def __init__(self, feat_dim: int, classifier: nn.Module, init_value: float = -10.0) -> None:
        super().__init__()
        self.classifier = classifier
        # pre-sigmoid parameter; sigmoid(-10) ≈ 0.000045 ≈ 0
        self.mask_param = nn.Parameter(torch.full((1, feat_dim), init_value, dtype=torch.float))

    def get_raw_mask(self):
        # in (0,1); broadcast to batch later
        return torch.sigmoid(self.mask_param)

    def train_decoupling_mask(self, Sa, Sb, dataloader, device, transform, epochs=20, lr=0.01):
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(int(epochs)):
            total_loss, correct_benign, total = 0.0, 0, 0
            for x, y, *_ in dataloader:
                x, y = x.to(device), y.to(device)
                x = transform(x)
                
                with torch.no_grad():
                    feats = Sa(x)

                m = self.get_raw_mask().to(feats.device).expand_as(feats)  # [B,D]
                benign_features   = feats * m
                poisoned_features = feats * (1 - m)

                logits_benign   = Sb(benign_features)
                logits_poisoned = Sb(poisoned_features)

                loss = criterion(logits_benign, y) - criterion(logits_poisoned, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                correct_benign += (logits_benign.argmax(1) == y).sum().item()
                total += y.size(0)

            print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, Benign Acc={correct_benign/total:.4f}")

        print("✅ Mask training complete.")
        return total_loss / len(dataloader)



