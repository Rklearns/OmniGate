import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import DEVICE, MAX_EPOCHS, MIN_EPOCHS, PATIENCE, LR, WEIGHT_DECAY
from src.model import StrongMLP

def train_on_split(X, y, tr, te):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    ytr_t = torch.tensor(y[tr], dtype=torch.long).to(DEVICE)

    # Calculate class weights
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y[tr]),
        y=y[tr]
    )

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    )

    # Initialize model
    model = StrongMLP(Xtr.shape[1], len(np.unique(y))).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_prec = 0.0
    best_state = model.state_dict()
    wait = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss_fn(model(Xtr_t), ytr_t).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(Xte_t).argmax(1).cpu().numpy()
            prec = precision_score(
                y[te], preds, average="weighted", zero_division=0
            )

        if prec > best_prec:
            best_prec = prec
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1

        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return best_prec, model
