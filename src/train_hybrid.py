"""
src/train_hybrid.py
-------------------
Training script for the PyroSight CNN + Transformer hybrid model.

Reuses the same data pipeline, class weighting, preprocessing, sampler
and early-stopping strategy as src/train.py (MobileNetV2 baseline).
Only the model and output paths differ.

Training configuration
----------------------
- Model    : SmokeHybridModel (CNN branch + Transformer branch + fusion head)
- Backbone : MobileNetV2 features[:-1], ImageNet pretrained, frozen
- Loss     : BCEWithLogitsLoss with pos_weight (same as baseline)
- Optimiser: Adam, lr=1e-3  (trainable params only)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early stopping: patience=7 on validation loss
- Checkpointing: best model by val_loss → models/hybrid_best.pt
- Max epochs: 20
- num_workers: 0  (Windows-safe)

Usage
-----
    python -m src.train_hybrid
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_loader import load_split, get_class_weights, SmokeDataset
from src.hybrid_model import build_hybrid_model, count_parameters, model_size_mb
from src.train import SmokeTorchDataset, EarlyStopping   # reuse shared utilities

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

HYBRID_CHECKPOINT_PATH = MODELS_DIR  / "hybrid_best.pt"
HYBRID_CONFIG_PATH     = RESULTS_DIR / "hybrid_train_config.json"
HYBRID_HISTORY_PATH    = RESULTS_DIR / "hybrid_train_history.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HYBRID_CONFIG = {
    "model":               "CNN+Transformer hybrid",
    "backbone":            "mobilenet_v2 features[:-1], ImageNet pretrained",
    "backbone_output":     "320 x 7 x 7",
    "freeze_backbone":     True,
    "cnn_branch":          "DepthwiseSepConv(320->320) + GAP",
    "transformer_layers":  2,
    "transformer_heads":   4,
    "token_dim":           128,
    "ffn_dim":             256,
    "transformer_dropout": 0.1,
    "fusion_dim":          128,
    "fusion_dropout":      0.3,
    "input_size":          [224, 224],
    "batch_size":          32,
    "num_workers":         0,
    "optimizer":           "Adam",
    "lr":                  1e-3,
    "scheduler":           "ReduceLROnPlateau",
    "scheduler_patience":  3,
    "scheduler_factor":    0.5,
    "early_stop_patience": 7,
    "max_epochs":          20,
    "loss":                "BCEWithLogitsLoss with pos_weight",
    "device":              "cuda if available, else cpu",
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_hybrid() -> dict:
    """Train the hybrid model and return a results dict."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Data ----------------------------------------------------------
    print("Loading datasets...")
    train_smoke = load_split("train", preprocess=True)
    val_smoke   = load_split("val",   preprocess=True)

    train_ds = SmokeTorchDataset(train_smoke)
    val_ds   = SmokeTorchDataset(val_smoke)

    # Weighted random sampler — same strategy as baseline
    class_weights_dict = get_class_weights(train_smoke)
    sample_weights = [class_weights_dict[label] for label in train_smoke.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=HYBRID_CONFIG["batch_size"],
        sampler=sampler,
        num_workers=HYBRID_CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=HYBRID_CONFIG["batch_size"],
        shuffle=False,
        num_workers=HYBRID_CONFIG["num_workers"],
        pin_memory=True,
    )

    # --- Model ---------------------------------------------------------
    print("Building hybrid model...")
    model = build_hybrid_model(
        freeze_backbone=HYBRID_CONFIG["freeze_backbone"],
        dropout_p=HYBRID_CONFIG["fusion_dropout"],
    )
    model = model.to(device)

    param_counts = count_parameters(model)
    size_mb      = model_size_mb(model)

    print(f"\n  Training configuration:")
    print(f"  {'Model':<28}: {HYBRID_CONFIG['model']}")
    print(f"  {'Backbone':<28}: {HYBRID_CONFIG['backbone']}")
    print(f"  {'Backbone frozen':<28}: {HYBRID_CONFIG['freeze_backbone']}")
    print(f"  {'Transformer layers':<28}: {HYBRID_CONFIG['transformer_layers']}")
    print(f"  {'Attention heads':<28}: {HYBRID_CONFIG['transformer_heads']}")
    print(f"  {'Token dim':<28}: {HYBRID_CONFIG['token_dim']}")
    print(f"  {'Optimizer':<28}: {HYBRID_CONFIG['optimizer']}  lr={HYBRID_CONFIG['lr']}")
    print(f"  {'Batch size':<28}: {HYBRID_CONFIG['batch_size']}")
    print(f"  {'Max epochs':<28}: {HYBRID_CONFIG['max_epochs']}")
    print(f"  {'Early stop patience':<28}: {HYBRID_CONFIG['early_stop_patience']}")
    print(f"\n  Parameter counts:")
    print(f"  {'Total':<28}: {param_counts['total']:,}")
    print(f"  {'Trainable':<28}: {param_counts['trainable']:,}")
    print(f"  {'Frozen':<28}: {param_counts['frozen']:,}")
    print(f"  {'Model size (fp32)':<28}: {size_mb} MB\n")

    # --- Loss ----------------------------------------------------------
    n_pos      = train_smoke.counts[1]
    n_neg      = train_smoke.counts[0]
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  pos_weight (n_neg/n_pos): {pos_weight.item():.4f}")

    # --- Optimiser / scheduler ----------------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HYBRID_CONFIG["lr"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=HYBRID_CONFIG["scheduler_patience"],
        factor=HYBRID_CONFIG["scheduler_factor"],
        min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=HYBRID_CONFIG["early_stop_patience"])

    # --- Training loop ------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss    = float("inf")
    epochs_completed = 0
    total_train_start = time.time()

    for epoch in range(1, HYBRID_CONFIG["max_epochs"] + 1):
        epoch_start = time.time()

        # -- Train --
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(1)
            loss   = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        train_loss = running_loss / len(train_ds)

        # -- Validate --
        model.eval()
        val_loss_sum = 0.0
        correct      = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                logits  = model(batch_x).squeeze(1)
                loss    = criterion(logits, batch_y)
                val_loss_sum += loss.item() * batch_x.size(0)
                preds   = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == batch_y.long()).sum().item()

        val_loss   = val_loss_sum / len(val_ds)
        val_acc    = correct / len(val_ds)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{HYBRID_CONFIG['max_epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  lr={current_lr:.2e}  "
            f"[{epoch_time:.1f}s]"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":            val_loss,
                    "val_acc":             val_acc,
                    "config":              HYBRID_CONFIG,
                },
                HYBRID_CHECKPOINT_PATH,
            )
            print(f"  -> Checkpoint saved (val_loss={val_loss:.4f})")

        scheduler.step(val_loss)
        epochs_completed = epoch

        if early_stop.step(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - total_train_start
    print(f"\nTraining complete in {total_time/60:.1f} min ({epochs_completed} epochs).")
    print(f"Best val_loss: {best_val_loss:.4f}")

    # --- Save artefacts -----------------------------------------------
    HYBRID_CONFIG["epochs_completed"]   = epochs_completed
    HYBRID_CONFIG["best_val_loss"]      = round(best_val_loss, 6)
    HYBRID_CONFIG["total_train_time_s"] = round(total_time, 1)
    HYBRID_CONFIG["param_counts"]       = param_counts
    HYBRID_CONFIG["model_size_mb"]      = size_mb
    HYBRID_CONFIG["checkpoint"]         = str(HYBRID_CHECKPOINT_PATH)
    HYBRID_CONFIG["pos_weight"]         = round(float(pos_weight.cpu().item()), 6)

    with open(HYBRID_CONFIG_PATH, "w") as f:
        json.dump(HYBRID_CONFIG, f, indent=2)
    with open(HYBRID_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Config saved  : {HYBRID_CONFIG_PATH}")
    print(f"History saved : {HYBRID_HISTORY_PATH}")

    return {
        "epochs_completed": epochs_completed,
        "best_val_loss":    best_val_loss,
        "param_counts":     param_counts,
        "model_size_mb":    size_mb,
        "history":          history,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_hybrid()
