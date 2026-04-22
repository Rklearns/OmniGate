from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import LATENT_DIM
except ImportError:
    from config import LATENT_DIM


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        with torch.no_grad():
            # Build a smoothed target distribution so the model is not trained against a perfectly hard label.
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma
        loss = -true_dist * focal_weight * log_probs

        if self.alpha is not None:
            loss = loss * self.alpha

        return loss.sum(dim=1).mean()


def alignment_loss(zs: list[torch.Tensor]) -> torch.Tensor:
    loss = 0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            # Cosine-style agreement encourages the omics branches to encode the same patient coherently.
            zi = F.normalize(zs[i], dim=1)
            zj = F.normalize(zs[j], dim=1)
            loss += 1 - (zi * zj).sum(dim=1).mean()
    return loss


def orthogonality_loss(zs: list[torch.Tensor]) -> torch.Tensor:
    loss = 0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            # This term counterbalances alignment by discouraging branches from collapsing into redundant copies.
            zi = F.normalize(zs[i], dim=1)
            zj = F.normalize(zs[j], dim=1)
            loss += torch.abs((zi * zj).sum(dim=1)).mean()
    return loss


def gate_entropy(gates: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    return -(gates * torch.log(gates + eps) + (1 - gates) * torch.log(1 - gates + eps)).mean()


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class OmicsEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextGate(nn.Module):
    def __init__(self, context_dim: int, gate_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, gate_dim),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.fc(context)


class GatedMultiOmicsClassifier(nn.Module):
    def __init__(self, in_dims: dict[str, int], num_classes: int):
        super().__init__()
        self.encoders = nn.ModuleDict({key: OmicsEncoder(value, LATENT_DIM) for key, value in in_dims.items()})
        total_latent_dim = LATENT_DIM * len(in_dims)
        self.gates = nn.ModuleDict({key: ContextGate(total_latent_dim, LATENT_DIM) for key in in_dims})

        self.classifier = nn.Sequential(
            nn.Linear(total_latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        zs_map = {omic: self.encoders[omic](x_dict[omic]) for omic in self.encoders}
        global_context = torch.cat(list(zs_map.values()), dim=1)
        # Downstream ablation heads operate on the same gated fused representation used by the base classifier.
        gated_latents = [zs_map[omic] * self.gates[omic](global_context) for omic in self.encoders]
        return torch.cat(gated_latents, dim=1)

    def forward(
        self, x_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        zs_map = {omic: self.encoders[omic](x_dict[omic]) for omic in self.encoders}
        # Every gate sees the joint multi-omics context, not just its own branch.
        global_context = torch.cat(list(zs_map.values()), dim=1)

        ungated_latents = []
        gated_latents = []
        gates_map = {}

        for omic in self.encoders:
            latent = zs_map[omic]
            gates = self.gates[omic](global_context)
            # Keep the ungated latents for regularization losses and the gated version for fusion.
            ungated_latents.append(latent)
            gated_latents.append(latent * gates)
            gates_map[omic] = gates

        fused = torch.cat(gated_latents, dim=1)
        logits = self.classifier(fused)
        return logits, ungated_latents, gates_map
