import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LATENT_DIM

# ============================================================
# LOSS FUNCTIONS
# ============================================================

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

def orthogonality_loss(zs):
    loss = 0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            zi = F.normalize(zs[i], dim=1)
            zj = F.normalize(zs[j], dim=1)
            loss += torch.abs((zi * zj).sum(dim=1)).mean()
    return loss

def gate_entropy(g):
    eps = 1e-6
    return -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps)).mean()

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class OmicsEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, latent_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class ContextGate(nn.Module):
    def __init__(self, context_dim, gate_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(context_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, gate_dim), nn.Sigmoid()
        )
    def forward(self, context): return self.fc(context)

class GatedMultiOmicsClassifier(nn.Module):
    def __init__(self, in_dims, num_classes):
        super().__init__()
        self.encoders = nn.ModuleDict({k: OmicsEncoder(v, LATENT_DIM) for k, v in in_dims.items()})
        total_latent_dim = LATENT_DIM * len(in_dims)
        self.gates = nn.ModuleDict({k: ContextGate(total_latent_dim, LATENT_DIM) for k in in_dims})
        
        self.classifier = nn.Sequential(
            nn.Linear(total_latent_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_dict):
        zs_map = {m: self.encoders[m](x_dict[m]) for m in self.encoders}
        all_z_list = [zs_map[m] for m in self.encoders]
        global_context = torch.cat(all_z_list, dim=1) 
        
        zs_list = []
        gated_list = []
        gates_map = {}

        for m in self.encoders:
            z = zs_map[m]
            g = self.gates[m](global_context)
            gated_z = z * g
            zs_list.append(z)
            gated_list.append(gated_z)
            gates_map[m] = g

        fused = torch.cat(gated_list, dim=1)
        logits = self.classifier(fused)
        
        return logits, zs_list, gates_map
