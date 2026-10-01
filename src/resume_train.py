"""
src/resume_train.py
-------------------
Resumes training from the best checkpoint saved so far and runs
remaining epochs until early stopping or max_epochs is reached.

Usage:
    python -m src.resume_train
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_loader import load_split, get_class_weights
from src.model import build_mobilenetv2_baseline, count_parameters, model_size_mb
from src.train import (
    SmokeTorchDataset,
    EarlyStopping,
    CHECKPOINT_PATH,
    CONFIG_PATH,
    HISTORY_PATH,
    RESULTS_DIR,
    CONFIG,
)

MAX_EPOCHS   = 20   # total epochs ceiling (including already-run epochs)
START_EPOCH  = 11   # resume from this epoch number


def resume() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")

    # --- Load checkpoint ---
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    resumed_from_epoch = checkpoint["epoch"]
    best_val_loss      = checkpoint["val_loss"]
    print(f"Checkpoint epoch: {resumed_from_epoch}, val_loss: {best_val_loss:.4f}")

    # --- Data ---
    print("Loading datasets...")
    train_smoke = load_split("train", preprocess=True)
    val_smoke   = load_split("val",   preprocess=True)
    train_ds = SmokeTorchDataset(train_smoke)
    val_ds   = SmokeTorchDataset(val_smoke)

    class_weights_dict = get_class_weights(train_smoke)
    sample_weights = [class_weights_dict[label] for label in train_smoke.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---
    model = build_mobilenetv2_baseline(
        dropout_p=CONFIG["dropout_p"], freeze_backbone=CONFIG["freeze_backbone"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # --- Loss ---
    n_pos = train_smoke.counts[1]
    n_neg = train_smoke.counts[0]
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer (restore state) ---
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=CONFIG["scheduler_patience"],
        factor=CONFIG["scheduler_factor"],
        min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=CONFIG["early_stop_patience"])
    early_stop.best_loss = best_val_loss   # continue from known best

    # Load existing history if available
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            history = json.load(f)

    epochs_completed = resumed_from_epoch

    for epoch in range(START_EPOCH, MAX_EPOCHS + 1):
        epoch_start = time.time()

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

        val_loss   = val_loss_sum / len(val_ds)
        val_acc    = correct / len(val_ds)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  lr={current_lr:.2e}  [{epoch_time:.1f}s]"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": CONFIG,
            }, CHECKPOINT_PATH)
            print(f"  -> Checkpoint saved (val_loss={val_loss:.4f})")

        scheduler.step(val_loss)
        epochs_completed = epoch

        if early_stop.step(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # Save history and config
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    param_counts = count_parameters(model)
    size_mb      = model_size_mb(model)
    CONFIG["epochs_completed"]   = epochs_completed
    CONFIG["best_val_loss"]      = round(best_val_loss, 6)
    CONFIG["param_counts"]       = param_counts
    CONFIG["model_size_mb"]      = size_mb
    CONFIG["checkpoint"]         = str(CHECKPOINT_PATH)

    with open(CONFIG_PATH, "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"\nResume complete. Total epochs run: {epochs_completed}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    return {"epochs_completed": epochs_completed, "best_val_loss": best_val_loss}


if __name__ == "__main__":
    resume()
