"""
src/adapt_figlib.py
--------------------
FIgLib domain adaptation fine-tuning for PyroSight.

Starting from models/hybrid_robust_best.pt, this script fine-tunes the
model using ONLY the 153 images from the two designated adaptation sequences:

    ADAPT sequences (153 images total):
      - 20191006_FIRE_lp-n-mobo-c   (73 images)
      - 20190924_FIRE_ml-w-mobo-c   (80 images)

    LOCKED TEST sequences — NEVER accessed by this script:
      - 20221116-Willowfire-om-n-mobo-c
      - 20200806_SpringsFire_om-n-mobo-c
      - 20180806_Holy_sp-s-mobo-c

Design rationale for a 153-image fine-tune
-------------------------------------------
With only 153 adaptation images the primary risks are catastrophic forgetting
(the model loses IIITDMJ_Smoke performance) and overfitting (the model memorises
the two adapt sequences without learning transferable features). The following
conservative choices address both:

1. Parameter scope: only the fusion head (57,601 params) is fine-tuned.
   The MobileNetV2 backbone, CNN branch, and Transformer branch are all frozen.
   This is the most targeted possible intervention — we teach the model a new
   decision boundary in the 448-dim feature space without touching the feature
   extractor itself.

2. Learning rate: 1e-4 (10× smaller than original training lr=1e-3).
   Small updates reduce the chance of overwriting generalised features.

3. Batch size: 8. With 153 images and 80/20 train/val split (≈122 train),
   batch size 8 gives ~15 gradient updates per epoch — enough signal without
   gradient averaging away the small-dataset structure.

4. Validation split: 80/20 stratified split of the 153 adapt images only.
   ~122 train / ~31 val. No test-sequence image ever appears in val.

5. pos_weight: computed from the adapt train split class balance
   (approximately 1.04, reflecting the near-equal 75/78 SMOKE/NO_SMOKE).

6. Augmentation: the same brightness/contrast/blur/noise augmentations from
   train_robust.py are applied to the adaptation training images. This prevents
   overfitting on the exact pixel values of the 122 training samples.

7. Early stopping: patience=7 on adapt-val loss. Max 30 epochs.

8. Checkpoint selection: best model by adapt-val loss.

Architecture: SmokeHybridModel (unchanged)
  - Backbone: MobileNetV2 features[:-1], FROZEN
  - CNN branch: FROZEN
  - Transformer branch: FROZEN
  - Fusion head (Linear 448→128→1): FINE-TUNED

Outputs
-------
  models/hybrid_figlib_adapted.pt          ← new checkpoint (does not overwrite robust)
  results/figlib_adaptation_config.json
  results/figlib_adaptation_history.json

Usage
-----
    python -m src.adapt_figlib
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.hybrid_model import build_hybrid_model, count_parameters, model_size_mb
from src.preprocessing import load_and_preprocess, TARGET_SIZE
from src.train import EarlyStopping
from src.train_robust import ROBUST_CHECKPOINT_PATH

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MODELS_DIR     = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results"
FIGLIB_DIR     = PROJECT_ROOT / "data" / "figlib_external" / "images"

ADAPTED_CHECKPOINT = MODELS_DIR  / "hybrid_figlib_adapted.pt"
ADAPT_CONFIG_PATH  = RESULTS_DIR / "figlib_adaptation_config.json"
ADAPT_HISTORY_PATH = RESULTS_DIR / "figlib_adaptation_history.json"

# Guard: must never touch hybrid_robust_best.pt
assert ADAPTED_CHECKPOINT != ROBUST_CHECKPOINT_PATH, "Output path collision!"

# ---------------------------------------------------------------------------
# Adaptation configuration
# ---------------------------------------------------------------------------
ADAPT_CONFIG = {
    "model":              "CNN+Transformer hybrid (FIgLib domain adaptation)",
    "starting_from":      "hybrid_robust_best.pt",
    "architecture":       "SmokeHybridModel (unchanged)",
    # Frozen layers
    "frozen_modules":     ["backbone", "cnn_branch", "transformer_branch"],
    # Fine-tuned layers
    "finetuned_modules":  ["fusion_head"],
    "freeze_backbone":    True,
    "fusion_dropout":     0.3,
    # Adapt data
    "adapt_sequences":    [
        "20191006_FIRE_lp-n-mobo-c",
        "20190924_FIRE_ml-w-mobo-c",
    ],
    "locked_test_sequences": [
        "20221116-Willowfire-om-n-mobo-c",
        "20200806_SpringsFire_om-n-mobo-c",
        "20180806_Holy_sp-s-mobo-c",
    ],
    "total_adapt_images": 153,
    "val_fraction":       0.20,
    # Training
    "batch_size":         8,
    "num_workers":        0,
    "optimizer":          "Adam",
    "lr":                 1e-4,
    "scheduler":          "ReduceLROnPlateau",
    "scheduler_patience": 3,
    "scheduler_factor":   0.5,
    "early_stop_patience":7,
    "max_epochs":         30,
    "loss":               "BCEWithLogitsLoss with adapt-set pos_weight",
    # Augmentations (same as train_robust)
    "aug_brightness_range":   [0.75, 1.25],
    "aug_contrast_blend_max": 0.35,
    "aug_blur_prob":          0.4,
    "aug_blur_kernel":        3,
    "aug_noise_prob":         0.5,
    "aug_noise_sigma_max":    0.04,
    "val_augmentation":       False,
    "device":             "cuda if available, else cpu",
}

# Timestamp ranges identifying each sequence (from Phase 8.3/8.4 verification)
_SEQ_RANGES: dict[str, tuple[int, int]] = {
    "20221116-Willowfire-om-n-mobo-c":  (1668652555, 1668657353),  # LOCKED TEST
    "20200806_SpringsFire_om-n-mobo-c": (1596761735, 1596766475),  # LOCKED TEST
    "20191006_FIRE_lp-n-mobo-c":        (1570381681, 1570386481),  # ADAPT
    "20190924_FIRE_ml-w-mobo-c":        (1569359957, 1569364757),  # ADAPT
    "20180806_Holy_sp-s-mobo-c":        (1533583350, 1533588150),  # LOCKED TEST
}

_LOCKED_RANGES = {
    seq: r for seq, r in _SEQ_RANGES.items()
    if seq in ADAPT_CONFIG["locked_test_sequences"]
}
_ADAPT_RANGES = {
    seq: r for seq, r in _SEQ_RANGES.items()
    if seq in ADAPT_CONFIG["adapt_sequences"]
}

# ---------------------------------------------------------------------------
# Label helper (same rule as download_figlib.py)
# ---------------------------------------------------------------------------
_OFFSET_RE = re.compile(r"_(-|\+|%2B)(\d+)\.jpg$", re.IGNORECASE)

def _assign_label(filename: str) -> int | None:
    m = _OFFSET_RE.search(filename)
    if not m:
        return None
    return 0 if m.group(1) == "-" else 1

def _sequence_of(filename: str) -> str | None:
    ts_m = re.match(r"^(\d+)_", filename)
    if not ts_m:
        return None
    ts = int(ts_m.group(1))
    for seq, (lo, hi) in _SEQ_RANGES.items():
        if lo <= ts <= hi:
            return seq
    return None

# ---------------------------------------------------------------------------
# Augmentations (reused from train_robust conventions)
# ---------------------------------------------------------------------------

def _augment(x: torch.Tensor) -> torch.Tensor:
    """Apply brightness / contrast / blur / noise augmentations in-place."""
    lo, hi = ADAPT_CONFIG["aug_brightness_range"]
    x = torch.clamp(x * torch.empty(1).uniform_(lo, hi).item(), -1.0, 1.0)
    alpha = torch.empty(1).uniform_(0.0, ADAPT_CONFIG["aug_contrast_blend_max"]).item()
    x = x * (1.0 - alpha) + x.mean() * alpha
    if torch.rand(1).item() < ADAPT_CONFIG["aug_blur_prob"]:
        k = ADAPT_CONFIG["aug_blur_kernel"]
        x = F.avg_pool2d(x.unsqueeze(0), kernel_size=k, stride=1,
                         padding=k//2, count_include_pad=False).squeeze(0)
    if torch.rand(1).item() < ADAPT_CONFIG["aug_noise_prob"]:
        sigma = torch.empty(1).uniform_(0.0, ADAPT_CONFIG["aug_noise_sigma_max"]).item()
        x = torch.clamp(x + torch.randn_like(x) * sigma, -1.0, 1.0)
    return x

# ---------------------------------------------------------------------------
# FIgLib adapt-only Dataset
# ---------------------------------------------------------------------------

class FIgLibAdaptDataset(Dataset):
    """Loads images from the two ADAPT sequences only.

    Locked test images are identified by timestamp and explicitly excluded
    with an assertion so any accidental inclusion raises an error.
    """

    def __init__(
        self,
        image_dir:   Path,
        file_list:   list[Path],
        augment:     bool = False,
    ) -> None:
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []

        for p in file_list:
            # Mandatory lock check: reject any test-sequence file
            seq = _sequence_of(p.name)
            assert seq not in ADAPT_CONFIG["locked_test_sequences"], (
                f"SECURITY VIOLATION: locked test image accessed: {p.name} "
                f"(sequence={seq})"
            )
            label = _assign_label(p.name)
            if label is None:
                continue
            self.samples.append((p, label))

        self.labels = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr    = load_and_preprocess(str(path), TARGET_SIZE)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        if self.augment:
            tensor = _augment(tensor)
        return tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Build adapt file lists (with locked-test exclusion)
# ---------------------------------------------------------------------------

def _build_adapt_file_lists() -> tuple[list[Path], list[Path]]:
    """Collect all ADAPT-sequence images and do an 80/20 stratified split.

    Stratified by label so both train and val have balanced SMOKE/NO_SMOKE.
    Returns (train_files, val_files).
    """
    all_files: list[Path] = []
    locked_count = 0

    for p in sorted(FIGLIB_DIR.glob("*.jpg")):
        seq = _sequence_of(p.name)
        if seq in ADAPT_CONFIG["locked_test_sequences"]:
            locked_count += 1
            continue                        # skip silently — do NOT load
        if seq in ADAPT_CONFIG["adapt_sequences"]:
            all_files.append(p)

    print(f"  Adapt images collected : {len(all_files)}")
    print(f"  Locked test images skipped (not accessed): {locked_count}")

    # Stratified 80/20 split by label
    smoke_files    = [p for p in all_files if _assign_label(p.name) == 1]
    no_smoke_files = [p for p in all_files if _assign_label(p.name) == 0]

    val_fraction = ADAPT_CONFIG["val_fraction"]
    n_val_smoke    = max(1, round(len(smoke_files)    * val_fraction))
    n_val_no_smoke = max(1, round(len(no_smoke_files) * val_fraction))

    # Use last N as val (temporally later in each sequence — harder examples)
    val_files   = smoke_files[-n_val_smoke:]    + no_smoke_files[-n_val_no_smoke:]
    train_files = smoke_files[:-n_val_smoke]    + no_smoke_files[:-n_val_no_smoke]

    train_smoke    = sum(1 for p in train_files if _assign_label(p.name) == 1)
    train_no_smoke = sum(1 for p in train_files if _assign_label(p.name) == 0)
    val_smoke      = sum(1 for p in val_files   if _assign_label(p.name) == 1)
    val_no_smoke   = sum(1 for p in val_files   if _assign_label(p.name) == 0)

    print(f"  Train : {len(train_files)}  (SMOKE={train_smoke}, NO_SMOKE={train_no_smoke})")
    print(f"  Val   : {len(val_files)}   (SMOKE={val_smoke}, NO_SMOKE={val_no_smoke})")

    return train_files, val_files, train_smoke, train_no_smoke


# ---------------------------------------------------------------------------
# Freeze everything except fusion_head
# ---------------------------------------------------------------------------

def _apply_freeze(model: nn.Module) -> nn.Module:
    """Freeze backbone, cnn_branch, transformer_branch. Leave fusion_head trainable."""
    frozen_modules = ADAPT_CONFIG["frozen_modules"]
    for name, module in model.named_children():
        if name in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = True
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def adapt() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Starting from : {ROBUST_CHECKPOINT_PATH}")
    print(f"Saving to     : {ADAPTED_CHECKPOINT}")
    print(f"\n  Locked test sequences (NOT accessed):")
    for s in ADAPT_CONFIG["locked_test_sequences"]:
        print(f"    - {s}")

    # --- Load starting checkpoint ------------------------------------
    if not ROBUST_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Starting checkpoint not found: {ROBUST_CHECKPOINT_PATH}")
    ckpt = torch.load(ROBUST_CHECKPOINT_PATH, map_location=device)
    print(f"\n  Starting checkpoint: epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}")

    # --- Build file lists (locks test sequences) ---------------------
    print("\nBuilding adapt dataset (test sequences excluded)...")
    train_files, val_files, n_train_smoke, n_train_no_smoke = _build_adapt_file_lists()

    # --- Datasets ----------------------------------------------------
    train_ds = FIgLibAdaptDataset(FIGLIB_DIR, train_files, augment=True)
    val_ds   = FIgLibAdaptDataset(FIGLIB_DIR, val_files,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=ADAPT_CONFIG["batch_size"],
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=ADAPT_CONFIG["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=True)

    # --- Model -------------------------------------------------------
    model = build_hybrid_model(
        freeze_backbone=ADAPT_CONFIG["freeze_backbone"],
        dropout_p=ADAPT_CONFIG["fusion_dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = _apply_freeze(model)
    model = model.to(device)

    pc = count_parameters(model)
    sz = model_size_mb(model)

    print(f"\n  Fine-tuning configuration:")
    print(f"  {'Frozen modules':<32}: {', '.join(ADAPT_CONFIG['frozen_modules'])}")
    print(f"  {'Fine-tuned modules':<32}: {', '.join(ADAPT_CONFIG['finetuned_modules'])}")
    print(f"  {'Total params':<32}: {pc['total']:,}")
    print(f"  {'Trainable params':<32}: {pc['trainable']:,}")
    print(f"  {'Frozen params':<32}: {pc['frozen']:,}")
    print(f"  {'Model size (fp32)':<32}: {sz} MB")
    print(f"  {'Optimizer':<32}: {ADAPT_CONFIG['optimizer']}  lr={ADAPT_CONFIG['lr']}")
    print(f"  {'Batch size':<32}: {ADAPT_CONFIG['batch_size']}")
    print(f"  {'Max epochs':<32}: {ADAPT_CONFIG['max_epochs']}")
    print(f"  {'Early stop patience':<32}: {ADAPT_CONFIG['early_stop_patience']}")

    # --- Loss --------------------------------------------------------
    # pos_weight from ADAPT TRAIN set only (not test set)
    pos_weight = torch.tensor(
        [n_train_no_smoke / max(n_train_smoke, 1)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  {'pos_weight (no_smoke/smoke)':<32}: {pos_weight.item():.4f}")

    # --- Optimiser ---------------------------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ADAPT_CONFIG["lr"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=ADAPT_CONFIG["scheduler_patience"],
        factor=ADAPT_CONFIG["scheduler_factor"],
        min_lr=1e-7,
    )
    early_stop = EarlyStopping(patience=ADAPT_CONFIG["early_stop_patience"])

    # --- Loop --------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss    = float("inf")
    epochs_completed = 0
    t_start          = time.time()

    print()
    for epoch in range(1, ADAPT_CONFIG["max_epochs"] + 1):
        t_ep = time.time()

        # Train
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

        # Validate
        model.eval()
        vls = 0.0; correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                logits = model(bx).squeeze(1)
                vls   += criterion(logits, by).item() * bx.size(0)
                preds  = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == by.long()).sum().item()

        val_loss = vls / len(val_ds)
        val_acc  = correct / len(val_ds)
        lr       = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(lr)

        print(f"Epoch {epoch:02d}/{ADAPT_CONFIG['max_epochs']}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  lr={lr:.2e}  [{time.time()-t_ep:.1f}s]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
                "val_acc":              val_acc,
                "config":               ADAPT_CONFIG,
                "starting_checkpoint":  str(ROBUST_CHECKPOINT_PATH),
            }, ADAPTED_CHECKPOINT)
            print(f"  -> Checkpoint saved (val_loss={val_loss:.4f})")

        scheduler.step(val_loss)
        epochs_completed = epoch

        if early_stop.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    total_time = time.time() - t_start
    print(f"\nAdaptation complete: {epochs_completed} epochs, "
          f"{total_time/60:.1f} min")
    print(f"Best adapt val_loss : {best_val_loss:.4f}")
    print(f"Checkpoint saved to : {ADAPTED_CHECKPOINT}")

    # --- Save artefacts ----------------------------------------------
    ADAPT_CONFIG["epochs_completed"]       = epochs_completed
    ADAPT_CONFIG["best_val_loss"]          = round(best_val_loss, 6)
    ADAPT_CONFIG["total_adapt_time_s"]     = round(total_time, 1)
    ADAPT_CONFIG["param_counts"]           = pc
    ADAPT_CONFIG["model_size_mb"]          = sz
    ADAPT_CONFIG["checkpoint"]             = str(ADAPTED_CHECKPOINT)
    ADAPT_CONFIG["pos_weight"]             = round(float(pos_weight.cpu().item()), 6)
    ADAPT_CONFIG["train_images_used"]      = len(train_ds)
    ADAPT_CONFIG["val_images_used"]        = len(val_ds)
    ADAPT_CONFIG["locked_test_accessed"]   = False   # explicit confirmation

    with open(ADAPT_CONFIG_PATH, "w") as f:
        json.dump(ADAPT_CONFIG, f, indent=2)
    with open(ADAPT_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Config saved  : {ADAPT_CONFIG_PATH}")
    print(f"History saved : {ADAPT_HISTORY_PATH}")

    # Final confirmation
    print(f"\n{'='*52}")
    print("  LOCKED TEST SEQUENCE CONFIRMATION")
    print(f"{'='*52}")
    for s in ADAPT_CONFIG["locked_test_sequences"]:
        print(f"  [NOT ACCESSED] {s}")
    print(f"  locked_test_accessed: {ADAPT_CONFIG['locked_test_accessed']}")
    print(f"{'='*52}")

    return {
        "epochs_completed":     epochs_completed,
        "best_val_loss":        best_val_loss,
        "param_counts":         pc,
        "locked_test_accessed": False,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    adapt()
