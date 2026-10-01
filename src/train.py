"""
src/train.py
------------
Training script for the PyroSight MobileNetV2 baseline.

Usage
-----
    python -m src.train

Or import and call:
    from src.train import train
    results = train()

Training configuration
----------------------
- Backbone : MobileNetV2 (ImageNet pretrained, frozen)
- Head     : Dropout(0.3) -> Linear(1280, 1)
- Loss     : BCEWithLogitsLoss with pos_weight for class imbalance
- Optimiser: Adam, lr=1e-3
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early stopping: patience=7 on validation loss
- Checkpointing: saves best model by val loss to models/
- Max epochs: 30
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_loader import load_split, get_class_weights, SmokeDataset
from src.model import build_mobilenetv2_baseline, count_parameters, model_size_mb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = MODELS_DIR / "mobilenetv2_baseline_best.pt"
CONFIG_PATH     = RESULTS_DIR / "train_config.json"
HISTORY_PATH    = RESULTS_DIR / "train_history.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    "model":          "MobileNetV2 baseline",
    "backbone":       "mobilenet_v2 ImageNet pretrained",
    "freeze_backbone": True,
    "dropout_p":      0.3,
    "input_size":     [224, 224],
    "batch_size":     32,
    "num_workers":    0,       # 0 = main process only (safe on Windows)
    "optimizer":      "Adam",
    "lr":             1e-3,
    "scheduler":      "ReduceLROnPlateau",
    "scheduler_patience": 3,
    "scheduler_factor":   0.5,
    "early_stop_patience": 7,
    "max_epochs":     20,
    "loss":           "BCEWithLogitsLoss with pos_weight",
    "device":         "cuda if available, else cpu",
}


# ---------------------------------------------------------------------------
# Torch Dataset wrapper around SmokeDataset
# ---------------------------------------------------------------------------

class SmokeTorchDataset(torch.utils.data.Dataset):
    """Wraps SmokeDataset to produce (tensor, label_tensor) pairs for PyTorch."""

    def __init__(self, smoke_dataset: SmokeDataset) -> None:
        self.ds = smoke_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        arr, label = self.ds[idx]          # arr: float32 (224,224,3) [-1,1]
        # HWC -> CHW
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> dict:
    """Run full training and return a results dict."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Data ----------------------------------------------------------
    print("Loading datasets...")
    train_smoke = load_split("train", preprocess=True)
    val_smoke   = load_split("val",   preprocess=True)

    train_ds = SmokeTorchDataset(train_smoke)
    val_ds   = SmokeTorchDataset(val_smoke)

    # Weighted random sampler so each batch is balanced
    class_weights_dict = get_class_weights(train_smoke)
    sample_weights = [
        class_weights_dict[label] for label in train_smoke.labels
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    # --- Model ---------------------------------------------------------
    print("Building model...")
    model = build_mobilenetv2_baseline(
        dropout_p=CONFIG["dropout_p"],
        freeze_backbone=CONFIG["freeze_backbone"],
    )
    model = model.to(device)

    param_counts = count_parameters(model)
    size_mb = model_size_mb(model)
    print(f"  Total params    : {param_counts['total']:,}")
    print(f"  Trainable params: {param_counts['trainable']:,}")
    print(f"  Frozen params   : {param_counts['frozen']:,}")
    print(f"  Model size (fp32): {size_mb} MB")

    # --- Loss ----------------------------------------------------------
    # pos_weight = n_negative / n_positive (from training set)
    n_pos = train_smoke.counts[1]
    n_neg = train_smoke.counts[0]
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimiser / scheduler ----------------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=CONFIG["scheduler_patience"],
        factor=CONFIG["scheduler_factor"],
        min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=CONFIG["early_stop_patience"])

    # --- Training loop ------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    epochs_completed = 0
    total_train_start = time.time()

    for epoch in range(1, CONFIG["max_epochs"] + 1):
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
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                logits  = model(batch_x).squeeze(1)
                loss    = criterion(logits, batch_y)
                val_loss_sum += loss.item() * batch_x.size(0)
                preds   = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == batch_y.long()).sum().item()

        val_loss = val_loss_sum / len(val_ds)
        val_acc  = correct / len(val_ds)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{CONFIG['max_epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  lr={current_lr:.2e}  "
            f"[{epoch_time:.1f}s]"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": CONFIG,
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> Checkpoint saved (val_loss={val_loss:.4f})")

        scheduler.step(val_loss)
        epochs_completed = epoch

        if early_stop.step(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - total_train_start
    print(f"\nTraining complete in {total_time/60:.1f} min ({epochs_completed} epochs).")

    # --- Save artefacts -----------------------------------------------
    CONFIG["epochs_completed"]   = epochs_completed
    CONFIG["best_val_loss"]      = round(best_val_loss, 6)
    CONFIG["total_train_time_s"] = round(total_time, 1)
    CONFIG["param_counts"]       = param_counts
    CONFIG["model_size_mb"]      = size_mb
    CONFIG["checkpoint"]         = str(CHECKPOINT_PATH)
    CONFIG["pos_weight"]         = round(float(pos_weight.cpu().item()), 6)

    with open(CONFIG_PATH, "w") as f:
        json.dump(CONFIG, f, indent=2)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Config saved  : {CONFIG_PATH}")
    print(f"History saved : {HISTORY_PATH}")

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
    train()
