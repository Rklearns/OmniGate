import numpy as np
import torch
from sklearn.metrics import precision_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DEVICE, MAX_EPOCHS, MIN_EPOCHS, PATIENCE, LR, WEIGHT_DECAY,
    ALIGN_W, GATE_ENT_W
)
from src.model import GatedMultiOmicsClassifier, FocalLoss, alignment_loss, gate_entropy

def train_single_fold(omics_data, y, tr, te, Xtr_t, Xte_t, ytr_t, input_dims):
    # Calculate Class Weights
    classes = np.unique(y[tr])
    w = compute_class_weight("balanced", classes=classes, y=y[tr])
    w_tensor = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    loss_fn = FocalLoss(alpha=w_tensor)
    model = GatedMultiOmicsClassifier(input_dims, len(classes)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_prec, wait = 0.0, 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()

        logits, zs, gates = model(Xtr_t, training=True)
        
        # Combined Loss
        main_loss = loss_fn(logits, ytr_t)
        align = ALIGN_W * alignment_loss(zs)
        entropy = GATE_ENT_W * sum(gate_entropy(g) for g in gates.values())
        
        total_loss = main_loss + align + entropy
        total_loss.backward()
        opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            # [0] to get just logits from forward return
            preds = model(Xte_t)[0].argmax(1).cpu().numpy()
            prec = precision_score(y[te], preds, average="weighted", zero_division=0)

        if prec > best_prec:
            best_prec = prec
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            break

    # Restore best model
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return best_prec, model
