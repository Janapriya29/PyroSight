"""
src/train_robust.py
--------------------
Robustness-focused training script for the PyroSight CNN + Transformer hybrid.

Motivation (Phase 6 findings)
------------------------------
The Phase 6 robustness evaluation revealed that the hybrid model's weakest
conditions are:
  - Gaussian noise  : accuracy 84.2%, FAR 23.3%  (worst)
  - Gaussian blur   : accuracy 90.7%, FAR 12.2%
  - Low contrast    : accuracy 93.4%, FAR  4.8%
  - High brightness : accuracy 95.4%, FAR  4.1%

Training with mild versions of these perturbations exposes the model to
conditions it will encounter at inference time, without destroying smoke
texture patterns that the network must learn.

Augmentation strategy
---------------------
All augmentations are applied stochastically per-image at training time
on float32 tensors in the MobileNetV2-normalised [-1, 1] range.
Augmentations are NEVER applied to validation data.

  1. Random brightness  — multiply by U[0.75, 1.25], clip to [-1, 1]
                          (covers low- and high-brightness conditions)
  2. Random contrast    — blend toward per-image mean by U[0, 0.35]
                          (moderate; won't collapse smoke textures)
  3. Random blur        — applied with p=0.4; 3×3 average pool
                          (lighter than robustness-eval's 5×5)
  4. Random noise       — applied with p=0.5; N(0, σ) with σ ~ U[0, 0.04]
                          (slightly below the σ=0.05 eval condition)

Each augmentation is applied independently so combinations occur naturally.
All preserve image shape (224×224×3) and leave labels unchanged.

Architecture / training protocol
---------------------------------
Identical to src/train_hybrid.py except for the augmented training loader:
  - SmokeHybridModel (frozen MobileNetV2 backbone)
  - BCEWithLogitsLoss with pos_weight
  - WeightedRandomSampler (same class-balancing strategy)
  - Adam, lr=1e-3
  - ReduceLROnPlateau (patience=3, factor=0.5)
  - Early stopping patience=7 on val_loss (unaugmented validation)
  - Max 20 epochs, batch_size=32, num_workers=0

Outputs
-------
  models/hybrid_robust_best.pt
  results/robust_train_config.json
  results/robust_train_history.json

Usage
-----
    python -m src.train_robust
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_loader import SmokeDataset, load_split, get_class_weights
from src.hybrid_model import build_hybrid_model, count_parameters, model_size_mb
from src.train import EarlyStopping   # reuse the shared early-stopping utility

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ROBUST_CHECKPOINT_PATH = MODELS_DIR  / "hybrid_robust_best.pt"
ROBUST_CONFIG_PATH     = RESULTS_DIR / "robust_train_config.json"
ROBUST_HISTORY_PATH    = RESULTS_DIR / "robust_train_history.json"

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
ROBUST_CONFIG = {
    "model":                 "CNN+Transformer hybrid (robustness training)",
    "backbone":              "mobilenet_v2 features[:-1], ImageNet pretrained",
    "freeze_backbone":       True,
    "fusion_dropout":        0.3,
    "input_size":            [224, 224],
    "batch_size":            32,
    "num_workers":           0,
    "optimizer":             "Adam",
    "lr":                    1e-3,
    "scheduler":             "ReduceLROnPlateau",
    "scheduler_patience":    3,
    "scheduler_factor":      0.5,
    "early_stop_patience":   7,
    "max_epochs":            20,
    "loss":                  "BCEWithLogitsLoss with pos_weight",
    "device":                "cuda if available, else cpu",
    # Augmentation settings
    "aug_brightness_range":  [0.75, 1.25],
    "aug_contrast_blend_max": 0.35,
    "aug_blur_prob":         0.4,
    "aug_blur_kernel":       3,
    "aug_noise_prob":        0.5,
    "aug_noise_sigma_max":   0.04,
    "val_augmentation":      False,
}


# ---------------------------------------------------------------------------
# Augmentation functions
# All operate on a single CHW float32 tensor in [-1, 1].
# They return a new tensor; the input is not mutated.
# ---------------------------------------------------------------------------

def _random_brightness(x: torch.Tensor) -> torch.Tensor:
    """Multiply all channels by a random scalar in [lo, hi], clip."""
    lo, hi = ROBUST_CONFIG["aug_brightness_range"]
    factor = torch.empty(1).uniform_(lo, hi).item()
    return torch.clamp(x * factor, -1.0, 1.0)


def _random_contrast(x: torch.Tensor) -> torch.Tensor:
    """Blend toward the per-image mean by a random amount in [0, max_blend]."""
    max_blend = ROBUST_CONFIG["aug_contrast_blend_max"]
    alpha     = torch.empty(1).uniform_(0.0, max_blend).item()
    mean_val  = x.mean()
    return x * (1.0 - alpha) + mean_val * alpha


def _random_blur(x: torch.Tensor, p: float, kernel: int) -> torch.Tensor:
    """Apply a box-average blur with probability p."""
    if torch.rand(1).item() > p:
        return x
    padding = kernel // 2
    # avg_pool2d needs [N, C, H, W]; add/remove batch dim
    return F.avg_pool2d(
        x.unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ).squeeze(0)


def _random_noise(x: torch.Tensor, p: float, sigma_max: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise with random sigma in [0, sigma_max]."""
    if torch.rand(1).item() > p:
        return x
    sigma = torch.empty(1).uniform_(0.0, sigma_max).item()
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, -1.0, 1.0)


def apply_train_augmentations(x: torch.Tensor) -> torch.Tensor:
    """Apply all training augmentations sequentially to a single CHW tensor."""
    x = _random_brightness(x)
    x = _random_contrast(x)
    x = _random_blur(
        x,
        p=ROBUST_CONFIG["aug_blur_prob"],
        kernel=ROBUST_CONFIG["aug_blur_kernel"],
    )
    x = _random_noise(
        x,
        p=ROBUST_CONFIG["aug_noise_prob"],
        sigma_max=ROBUST_CONFIG["aug_noise_sigma_max"],
    )
    return x


# ---------------------------------------------------------------------------
# PyTorch Dataset wrappers
# ---------------------------------------------------------------------------

class SmokeTrainDataset(torch.utils.data.Dataset):
    """Training dataset: wraps SmokeDataset, applies augmentation per sample.

    Converts HWC float32 numpy array → CHW float32 tensor, then augments.
    """

    def __init__(self, smoke_dataset: SmokeDataset) -> None:
        self.ds = smoke_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        arr, label = self.ds[idx]                            # HWC numpy, [-1,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)     # CHW tensor
        tensor = apply_train_augmentations(tensor)
        return tensor, torch.tensor(label, dtype=torch.float32)


class SmokeValDataset(torch.utils.data.Dataset):
    """Validation dataset: no augmentation — identical to SmokeTorchDataset."""

    def __init__(self, smoke_dataset: SmokeDataset) -> None:
        self.ds = smoke_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        arr, label = self.ds[idx]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_robust() -> dict:
    """Train the robust hybrid model and return a results dict."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Data ----------------------------------------------------------
    print("Loading datasets...")
    train_smoke = load_split("train", preprocess=True)
    val_smoke   = load_split("val",   preprocess=True)

    train_ds = SmokeTrainDataset(train_smoke)   # with augmentations
    val_ds   = SmokeValDataset(val_smoke)       # no augmentations

    # Weighted random sampler — identical class-balancing strategy
    class_weights_dict = get_class_weights(train_smoke)
    sample_weights = [class_weights_dict[label] for label in train_smoke.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=ROBUST_CONFIG["batch_size"],
        sampler=sampler,
        num_workers=ROBUST_CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ROBUST_CONFIG["batch_size"],
        shuffle=False,
        num_workers=ROBUST_CONFIG["num_workers"],
        pin_memory=True,
    )

    # --- Model ---------------------------------------------------------
    print("Building robust hybrid model...")
    model = build_hybrid_model(
        freeze_backbone=ROBUST_CONFIG["freeze_backbone"],
        dropout_p=ROBUST_CONFIG["fusion_dropout"],
    )
    model = model.to(device)

    param_counts = count_parameters(model)
    size_mb      = model_size_mb(model)

    # --- Print configuration ------------------------------------------
    print(f"\n  Training configuration:")
    print(f"  {'Model':<32}: {ROBUST_CONFIG['model']}")
    print(f"  {'Backbone':<32}: {ROBUST_CONFIG['backbone']}")
    print(f"  {'Backbone frozen':<32}: {ROBUST_CONFIG['freeze_backbone']}")
    print(f"  {'Optimizer':<32}: {ROBUST_CONFIG['optimizer']}  lr={ROBUST_CONFIG['lr']}")
    print(f"  {'Batch size':<32}: {ROBUST_CONFIG['batch_size']}")
    print(f"  {'Max epochs':<32}: {ROBUST_CONFIG['max_epochs']}")
    print(f"  {'Early stop patience':<32}: {ROBUST_CONFIG['early_stop_patience']}")
    print(f"\n  Augmentation settings (training only):")
    print(f"  {'Brightness range':<32}: {ROBUST_CONFIG['aug_brightness_range']}")
    print(f"  {'Contrast blend max':<32}: {ROBUST_CONFIG['aug_contrast_blend_max']}")
    print(f"  {'Blur probability':<32}: {ROBUST_CONFIG['aug_blur_prob']}  (kernel={ROBUST_CONFIG['aug_blur_kernel']}x{ROBUST_CONFIG['aug_blur_kernel']})")
    print(f"  {'Noise probability':<32}: {ROBUST_CONFIG['aug_noise_prob']}  (sigma_max={ROBUST_CONFIG['aug_noise_sigma_max']})")
    print(f"  {'Val augmentation':<32}: {ROBUST_CONFIG['val_augmentation']}")
    print(f"\n  Parameter counts:")
    print(f"  {'Total':<32}: {param_counts['total']:,}")
    print(f"  {'Trainable':<32}: {param_counts['trainable']:,}")
    print(f"  {'Frozen':<32}: {param_counts['frozen']:,}")
    print(f"  {'Model size (fp32)':<32}: {size_mb} MB\n")

    # --- Loss ----------------------------------------------------------
    n_pos      = train_smoke.counts[1]
    n_neg      = train_smoke.counts[0]
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  pos_weight (n_neg/n_pos): {pos_weight.item():.4f}")

    # --- Optimiser / scheduler ----------------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ROBUST_CONFIG["lr"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=ROBUST_CONFIG["scheduler_patience"],
        factor=ROBUST_CONFIG["scheduler_factor"],
        min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=ROBUST_CONFIG["early_stop_patience"])

    # --- Training loop ------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss    = float("inf")
    epochs_completed = 0
    total_train_start = time.time()

    for epoch in range(1, ROBUST_CONFIG["max_epochs"] + 1):
        epoch_start = time.time()

        # -- Train (augmented) -----------------------------------------
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

        # -- Validate (clean, no augmentation) -------------------------
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
            f"Epoch {epoch:02d}/{ROBUST_CONFIG['max_epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  lr={current_lr:.2e}  "
            f"[{epoch_time:.1f}s]"
        )

        # Checkpoint best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             val_loss,
                    "val_acc":              val_acc,
                    "config":               ROBUST_CONFIG,
                },
                ROBUST_CHECKPOINT_PATH,
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
    ROBUST_CONFIG["epochs_completed"]   = epochs_completed
    ROBUST_CONFIG["best_val_loss"]      = round(best_val_loss, 6)
    ROBUST_CONFIG["total_train_time_s"] = round(total_time, 1)
    ROBUST_CONFIG["param_counts"]       = param_counts
    ROBUST_CONFIG["model_size_mb"]      = size_mb
    ROBUST_CONFIG["checkpoint"]         = str(ROBUST_CHECKPOINT_PATH)
    ROBUST_CONFIG["pos_weight"]         = round(float(pos_weight.cpu().item()), 6)

    with open(ROBUST_CONFIG_PATH, "w") as f:
        json.dump(ROBUST_CONFIG, f, indent=2)
    with open(ROBUST_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Config saved  : {ROBUST_CONFIG_PATH}")
    print(f"History saved : {ROBUST_HISTORY_PATH}")

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
    train_robust()
