from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

try:
    from .config import (
        ALIGN_W,
        DEVICE,
        GATE_ENT_W,
        LR,
        MAX_EPOCHS,
        MIN_EPOCHS,
        OMICS,
        OMICS_DROPOUT_P,
        ORTHO_W,
        PATIENCE,
        SEED,
        SPARSITY_W,
        WEIGHT_DECAY,
    )
    from .models import FocalLoss, GatedMultiOmicsClassifier, alignment_loss, gate_entropy, orthogonality_loss
except ImportError:
    from config import (
        ALIGN_W,
        DEVICE,
        GATE_ENT_W,
        LR,
        MAX_EPOCHS,
        MIN_EPOCHS,
        OMICS,
        OMICS_DROPOUT_P,
        ORTHO_W,
        PATIENCE,
        SEED,
        SPARSITY_W,
        WEIGHT_DECAY,
    )
    from models import FocalLoss, GatedMultiOmicsClassifier, alignment_loss, gate_entropy, orthogonality_loss


def train_and_collect(
    omics_data: dict[str, np.ndarray],
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    cancer_name: str,
    accumulators: dict,
) -> None:
    x_train, x_test = {}, {}
    for omic_name, values in omics_data.items():
        scaler = StandardScaler()
        # Each modality is standardized independently inside the current fold to avoid leakage.
        x_train[omic_name] = scaler.fit_transform(values[train_indices])
        x_test[omic_name] = scaler.transform(values[test_indices])

    y_train = y[train_indices]
    y_test = y[test_indices]
    x_train_t = {key: torch.tensor(value, dtype=torch.float32).to(DEVICE) for key, value in x_train.items()}
    x_test_t = {key: torch.tensor(value, dtype=torch.float32).to(DEVICE) for key, value in x_test.items()}
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    # A mild power transform keeps rare classes emphasized without making the weighting too brittle.
    class_weights = np.power(class_weights, 1.25)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    gamma = 3.5 if cancer_name == "GS-COAD" else 2.0
    loss_fn = FocalLoss(alpha=class_weights_t, gamma=gamma, smoothing=0.05)

    in_dims = {omic_name: x_train[omic_name].shape[1] for omic_name in x_train}
    model = GatedMultiOmicsClassifier(in_dims, len(np.unique(y))).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_precision = 0.0
    best_state = None
    wait = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()

        train_inputs = {key: value.clone() for key, value in x_train_t.items()}
        if torch.rand(1).item() < OMICS_DROPOUT_P:
            # Randomly blank one omics view during training so the fusion block does not over-rely on a single source.
            dropped_omic = np.random.choice(list(train_inputs.keys()))
            train_inputs[dropped_omic] = torch.zeros_like(train_inputs[dropped_omic])

        logits, latents, gates = model(train_inputs)

        focal_term = loss_fn(logits, y_train_t)
        align_term = ALIGN_W * alignment_loss(latents)
        ortho_term = ORTHO_W * orthogonality_loss(latents)
        gate_entropy_term = GATE_ENT_W * sum(gate_entropy(gate) for gate in gates.values())
        sparsity_term = SPARSITY_W * sum(gate.mean() for gate in gates.values())

        loss = focal_term + align_term + ortho_term + gate_entropy_term + sparsity_term
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(x_test_t)[0].argmax(1).cpu().numpy()
            precision = precision_score(y_test, predictions, average="weighted", zero_division=0)

        if precision > best_precision:
            best_precision = precision
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        # Early stopping is keyed to weighted precision because that is the primary comparison metric in the study.
        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            break

    if best_state is None:
        best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict({key: value.to(DEVICE) for key, value in best_state.items()})
    model.eval()

    with torch.no_grad():
        base_predictions = model(x_test_t)[0].argmax(1).cpu().numpy()
        base_recall = recall_score(y_test, base_predictions, average="weighted", zero_division=0)
        base_f1 = f1_score(y_test, base_predictions, average="weighted", zero_division=0)

        accumulators["ablation"]["Base_MLP"]["prec"].append(best_precision)
        accumulators["ablation"]["Base_MLP"]["rec"].append(base_recall)
        accumulators["ablation"]["Base_MLP"]["f1"].append(base_f1)

        accumulators["y_true"].extend(y_test)
        accumulators["y_pred"].extend(base_predictions)

        _, _, gate_values = model(x_test_t)
        for omic in OMICS:
            accumulators["gates"][omic].append(gate_values[omic].mean().item())

        # The ablation heads are evaluated on the learned fused embedding rather than raw omics features.
        fused_train = model.extract_features(x_train_t).cpu().numpy()
        fused_test = model.extract_features(x_test_t).cpu().numpy()

    svm = SVC(kernel="rbf", class_weight="balanced", random_state=SEED)
    svm.fit(fused_train, y_train)
    svm_predictions = svm.predict(fused_test)
    accumulators["ablation"]["SVM"]["prec"].append(
        precision_score(y_test, svm_predictions, average="weighted", zero_division=0)
    )
    accumulators["ablation"]["SVM"]["rec"].append(
        recall_score(y_test, svm_predictions, average="weighted", zero_division=0)
    )
    accumulators["ablation"]["SVM"]["f1"].append(f1_score(y_test, svm_predictions, average="weighted", zero_division=0))

    label_encoder = LabelEncoder()
    y_train_xgb = label_encoder.fit_transform(y_train)
    xgb = XGBClassifier(eval_metric="mlogloss", random_state=SEED, n_estimators=100)
    xgb.fit(fused_train, y_train_xgb)
    xgb_predictions = label_encoder.inverse_transform(xgb.predict(fused_test))
    accumulators["ablation"]["XGBoost"]["prec"].append(
        precision_score(y_test, xgb_predictions, average="weighted", zero_division=0)
    )
    accumulators["ablation"]["XGBoost"]["rec"].append(
        recall_score(y_test, xgb_predictions, average="weighted", zero_division=0)
    )
    accumulators["ablation"]["XGBoost"]["f1"].append(
        f1_score(y_test, xgb_predictions, average="weighted", zero_division=0)
    )

    mlp_classifier = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=SEED)
    mlp_classifier.fit(fused_train, y_train)
    mlp_predictions = mlp_classifier.predict(fused_test)
    accumulators["ablation"]["Deeper_MLP"]["prec"].append(
        precision_score(y_test, mlp_predictions, average="weighted", zero_division=0)
    )
    accumulators["ablation"]["Deeper_MLP"]["rec"].append(
        recall_score(y_test, mlp_predictions, average="weighted", zero_division=0)
    )
    accumulators["ablation"]["Deeper_MLP"]["f1"].append(
        f1_score(y_test, mlp_predictions, average="weighted", zero_division=0)
    )

    for omic in OMICS:
        inputs = x_test_t[omic].clone().requires_grad_(True)
        full_inputs = {key: x_test_t[key] for key in OMICS}
        full_inputs[omic] = inputs

        model.zero_grad(set_to_none=True)
        logits, _, _ = model(full_inputs)
        # We use mean top-logit sensitivity as a simple post hoc signal of feature influence within each modality.
        logits.max(dim=1)[0].mean().backward()

        sensitivity = inputs.grad.abs().mean(dim=0).cpu().numpy()
        if accumulators["sensitivity"][omic] is None:
            accumulators["sensitivity"][omic] = np.zeros_like(sensitivity)
        if accumulators["sensitivity"][omic].shape == sensitivity.shape:
            accumulators["sensitivity"][omic] += sensitivity
