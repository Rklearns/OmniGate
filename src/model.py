import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import OMICS_DROPOUT_P

# --- Components ---
class OmicsEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class GateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, z):
        return self.fc(z)

# --- Main Model ---
class GatedMultiOmicsClassifier(nn.Module):
    def __init__(self, in_dims, num_classes):
        super().__init__()
        self.encoders = nn.ModuleDict({k: OmicsEncoder(v) for k, v in in_dims.items()})
        self.gates = nn.ModuleDict({k: GateNet() for k in in_dims})

        self.classifier = nn.Sequential(
            nn.Linear(128 * len(in_dims), 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_dict, training=False):
        zs, gated, gates = [], [], {}
        for m in self.encoders:
            x = x_dict[m]
            # Modality Dropout
            if training and torch.rand(1).item() < OMICS_DROPOUT_P:
                x = torch.zeros_like(x)
            
            z = self.encoders[m](x)
            g = self.gates[m](z)
            zs.append(z)
            gated.append(z * g)
            gates[m] = g

        fused = torch.cat(gated, dim=1)
        logits = self.classifier(fused)
        return logits, zs, gates

# --- Losses ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)
        focal = (1 - p) ** self.gamma
        loss = -true_dist * focal * log_p
        if self.alpha is not None:
            loss = loss * self.alpha
        return loss.sum(dim=1).mean()

def alignment_loss(zs):
    loss = 0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            zi = F.normalize(zs[i], dim=1)
            zj = F.normalize(zs[j], dim=1)
            loss += 1 - (zi * zj).sum(dim=1).mean()
    return loss

def gate_entropy(g):
    eps = 1e-6
    return -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps)).mean()
