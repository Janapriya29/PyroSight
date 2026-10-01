"""
src/resume_robust.py
--------------------
Resume robustness training from the best checkpoint.
Edit START_EPOCH / MAX_EPOCHS before each run.

Usage:
    python -m src.resume_robust
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_loader import load_split, get_class_weights
from src.hybrid_model import build_hybrid_model, count_parameters, model_size_mb
from src.train import EarlyStopping
from src.train_robust import (
    ROBUST_CHECKPOINT_PATH, ROBUST_CONFIG_PATH, ROBUST_HISTORY_PATH,
    ROBUST_CONFIG, SmokeTrainDataset, SmokeValDataset,
)

START_EPOCH = 20
MAX_EPOCHS  = 20


def resume_robust() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Resuming robust training from: {ROBUST_CHECKPOINT_PATH}")

    ckpt = torch.load(ROBUST_CHECKPOINT_PATH, map_location=device)
    best_val_loss = ckpt["val_loss"]
    print(f"Checkpoint epoch: {ckpt['epoch']}, val_loss: {best_val_loss:.4f}")

    # --- Data ----------------------------------------------------------
    print("Loading datasets...")
    train_smoke = load_split("train", preprocess=True)
    val_smoke   = load_split("val",   preprocess=True)
    train_ds    = SmokeTrainDataset(train_smoke)
    val_ds      = SmokeValDataset(val_smoke)

    cw = get_class_weights(train_smoke)
    sw = [cw[l] for l in train_smoke.labels]
    sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=ROBUST_CONFIG["batch_size"],
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=ROBUST_CONFIG["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---------------------------------------------------------
    model = build_hybrid_model(
        freeze_backbone=ROBUST_CONFIG["freeze_backbone"],
        dropout_p=ROBUST_CONFIG["fusion_dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # --- Loss / optimizer ---------------------------------------------
    n_pos      = train_smoke.counts[1]
    n_neg      = train_smoke.counts[0]
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ROBUST_CONFIG["lr"],
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=ROBUST_CONFIG["scheduler_patience"],
        factor=ROBUST_CONFIG["scheduler_factor"],
        min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=ROBUST_CONFIG["early_stop_patience"])
    early_stop.best_loss = best_val_loss

    # --- History -------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    if ROBUST_HISTORY_PATH.exists():
        with open(ROBUST_HISTORY_PATH) as f:
            history = json.load(f)

    epochs_completed = ckpt["epoch"]

    for epoch in range(START_EPOCH, MAX_EPOCHS + 1):
        t0 = time.time()

        model.train()
        running = 0.0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(1), by)
            loss.backward()
            optimizer.step()
            running += loss.item() * bx.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        vls = 0.0; correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                logits = model(bx).squeeze(1)
                vls    += criterion(logits, by).item() * bx.size(0)
                preds   = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == by.long()).sum().item()

        val_loss = vls / len(val_ds)
        val_acc  = correct / len(val_ds)
        lr       = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(lr)

        print(f"Epoch {epoch:02d}/{MAX_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  lr={lr:.2e}  [{time.time()-t0:.1f}s]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss, "val_acc": val_acc,
                "config":   ROBUST_CONFIG,
            }, ROBUST_CHECKPOINT_PATH)
            print(f"  -> Checkpoint saved (val_loss={val_loss:.4f})")

        scheduler.step(val_loss)
        epochs_completed = epoch

        if early_stop.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    with open(ROBUST_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    ROBUST_CONFIG.update({
        "epochs_completed": epochs_completed,
        "best_val_loss": round(best_val_loss, 6),
        "param_counts": count_parameters(model),
        "model_size_mb": model_size_mb(model),
        "checkpoint": str(ROBUST_CHECKPOINT_PATH),
    })
    with open(ROBUST_CONFIG_PATH, "w") as f:
        json.dump(ROBUST_CONFIG, f, indent=2)

    print(f"\nResume complete. Epochs: {epochs_completed}, best val_loss: {best_val_loss:.4f}")
    return {"epochs_completed": epochs_completed, "best_val_loss": best_val_loss}


if __name__ == "__main__":
    resume_robust()
