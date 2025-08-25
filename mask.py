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
    def __init__(self, feat_shape, classifier: nn.Module, init_value: float = -10.0,
                 mask_mode: str = "vector") -> None:
        """
        mask_mode:
          - "vector"   -> [1, D]          (flat features)
          - "channel"  -> [1, C, 1, 1]    (per-channel, broadcast over H,W)
          - "spatial"  -> [1, C, H, W]    (per-element mask)
          - "auto"     -> picks [1,D] if flat, else [1,C,H,W]
        """
        super().__init__()
        self.classifier = classifier
        self.mask_mode = mask_mode

        if isinstance(feat_shape, int):
            feat_shape = (feat_shape,)  # D

        def _shape_for_mode(mode, shape):
            if mode == "vector":
                (D,) = shape
                return (1, D)
            if mode == "channel":
                C, *rest = shape
                return (1, C, 1, 1)
            if mode == "spatial":
                C, H, W = shape
                return (1, C, H, W)
            if mode == "auto":
                if len(shape) == 1:
                    return (1, shape[0])               # flat -> [1,D]
                elif len(shape) >= 3:
                    C, H, W = shape[:3]
                    return (1, C, H, W)                # spatial -> [1,C,H,W]
                else:
                    raise ValueError(f"feat_shape ambiguous: {shape}")
            raise ValueError(f"Unknown mask_mode: {mode}")

        param_shape = _shape_for_mode(mask_mode, feat_shape)
        self.mask_param = nn.Parameter(torch.full(param_shape, init_value, dtype=torch.float))

    def get_raw_mask(self):
        return torch.sigmoid(self.mask_param)

    def _apply_mask(self, feats):
        m = self.get_raw_mask().to(feats.device)
        if feats.dim() == 2:          # [B,D]
            if m.dim() != 2 or m.size(1) != feats.size(1):
                raise RuntimeError(f"Mask {tuple(m.shape)} incompatible with flat feats {tuple(feats.shape)}")
            return feats * m.expand_as(feats)

        if feats.dim() == 4:          # [B,C,H,W]
            if m.dim() == 4:
                if (m.size(1), m.size(2), m.size(3)) == (feats.size(1), feats.size(2), feats.size(3)):
                    return feats * m.expand(feats.size(0), -1, -1, -1)          # spatial per-element
                if (m.size(1), m.size(2), m.size(3)) == (feats.size(1), 1, 1):
                    return feats * m.expand(feats.size(0), -1, feats.size(2), feats.size(3))  # channel
            raise RuntimeError(f"Mask {tuple(m.shape)} incompatible with spatial feats {tuple(feats.shape)}")
        raise RuntimeError("Unsupported feature rank for masking.")

    def _apply_complement_mask(self, feats):
        return feats - self._apply_mask(feats)

    def train_decoupling_mask(self, Sa, Sb, dataloader, device, transform, epochs=20, lr=0.01,
                              l1_lambda: float = 0.0, tv_lambda: float = 0.0):
        """
        Optional regularizers:
          - l1_lambda:   pushes m toward sparsity
          - tv_lambda:   spatial smoothness (only applied when m is [1,C,H,W])
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        expects_spatial = getattr(Sb, "expects_spatial", False)

        def _tv_loss(m):
            # total variation over H,W (sum over channels)
            if m.dim() != 4 or m.size(2) == 1 or m.size(3) == 1:
                return m.new_tensor(0.0)
            dh = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
            dw = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
            return dh + dw

        for epoch in range(int(epochs)):
            total_loss, correct_benign, total = 0.0, 0, 0
            for x, y, *_ in dataloader:
                x, y = x.to(device), y.to(device)
                x = transform(x)

                with torch.no_grad():
                    feats = Sa(x)   # [B,D] or [B,C,H,W]

                benign_features   = self._apply_mask(feats)
                poisoned_features = self._apply_complement_mask(feats)

                if expects_spatial and benign_features.dim() == 4:
                    logits_benign   = Sb(benign_features)
                    logits_poisoned = Sb(poisoned_features)
                elif not expects_spatial and benign_features.dim() == 2:
                    logits_benign   = Sb(benign_features)
                    logits_poisoned = Sb(poisoned_features)
                else:
                    # fallback flatten
                    B = benign_features.size(0)
                    logits_benign   = Sb(benign_features.view(B, -1))
                    logits_poisoned = Sb(poisoned_features.view(B, -1))

                loss = criterion(logits_benign, y) - criterion(logits_poisoned, y)

                # regularisers
                if l1_lambda > 0:
                    loss = loss + l1_lambda * self.get_raw_mask().mean()
                if tv_lambda > 0 and self.get_raw_mask().dim() == 4:
                    loss = loss + tv_lambda * _tv_loss(self.get_raw_mask())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                correct_benign += (logits_benign.argmax(1) == y).sum().item()
                total += y.size(0)

            print(f"[Mask] Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, Benign Acc={correct_benign/total:.4f}")
        return total_loss / max(1, len(dataloader))



