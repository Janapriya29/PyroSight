"""
src/robustness_evaluation.py
-----------------------------
Robustness evaluation of the trained CNN + Transformer hybrid model.

Evaluates how the model responds to six controlled image perturbations
applied at inference time to the IIITDMJ_Smoke test set. No retraining
is performed and no files are modified.

Perturbation conditions
-----------------------
1. Original          — no change (reference)
2. Low brightness    — multiply pixel values by 0.6 (darken)
3. High brightness   — multiply pixel values by 1.4, then clip to [-1, 1]
4. Low contrast      — blend toward grey: 0.5 * x + 0.5 * mean(x)
5. Gaussian blur     — 5×5 box-average approximation via unfold (no scipy)
6. Gaussian noise    — add N(0, σ=0.05) noise, clip to [-1, 1]

All perturbations operate on the float32 MobileNetV2-normalised tensors
in [-1, 1] space, preserving image dimensions and binary labels.

Outputs
-------
- results/robustness_results.json
- results/robustness_comparison.png

Usage
-----
    python -m src.robustness_evaluation
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data_loader import load_split
from src.hybrid_model import build_hybrid_model
from src.train import SmokeTorchDataset
from src.train_hybrid import HYBRID_CHECKPOINT_PATH, HYBRID_CONFIG, RESULTS_DIR

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
ROBUSTNESS_JSON = RESULTS_DIR / "robustness_results.json"
ROBUSTNESS_PNG  = RESULTS_DIR / "robustness_comparison.png"

BATCH_SIZE   = 32
NUM_WORKERS  = 0


# ---------------------------------------------------------------------------
# Perturbation functions
# All operate on a float32 tensor batch [B, C, H, W] in [-1, 1].
# They return a new tensor of the same shape; the input is not mutated.
# ---------------------------------------------------------------------------

def perturb_original(x: torch.Tensor) -> torch.Tensor:
    """No change — reference condition."""
    return x


def perturb_low_brightness(x: torch.Tensor) -> torch.Tensor:
    """Reduce brightness by multiplying by 0.6 (darken by 40%)."""
    return x * 0.6


def perturb_high_brightness(x: torch.Tensor) -> torch.Tensor:
    """Increase brightness by multiplying by 1.4, then clip to [-1, 1]."""
    return torch.clamp(x * 1.4, -1.0, 1.0)


def perturb_low_contrast(x: torch.Tensor) -> torch.Tensor:
    """Reduce contrast: blend 50% toward the per-image mean grey value.

    Equivalent to:  x_out = 0.5 * x + 0.5 * mean(x)
    The mean is computed per image and broadcast over C, H, W.
    """
    mean_val = x.mean(dim=(1, 2, 3), keepdim=True)   # [B, 1, 1, 1]
    return 0.5 * x + 0.5 * mean_val


def perturb_gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """Apply a mild 5×5 average (box) blur as a lightweight blur proxy.

    Uses F.avg_pool2d with padding so output size matches input.
    This approximates low-pass filtering without scipy.
    """
    # avg_pool2d expects [B, C, H, W] — already correct
    blurred = F.avg_pool2d(
        x,
        kernel_size=5,
        stride=1,
        padding=2,      # 'same' padding for kernel 5
        count_include_pad=False,
    )
    return blurred


def perturb_gaussian_noise(
    x: torch.Tensor,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise with std=sigma, clip to [-1, 1].

    sigma=0.05 in [-1, 1] space ≈ 6.375 intensity units in [0, 255] space —
    a mild, realistic noise level.
    """
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

CONDITIONS: list[tuple[str, Callable]] = [
    ("Original",         perturb_original),
    ("Low brightness",   perturb_low_brightness),
    ("High brightness",  perturb_high_brightness),
    ("Low contrast",     perturb_low_contrast),
    ("Gaussian blur",    perturb_gaussian_blur),
    ("Gaussian noise",   perturb_gaussian_noise),
]

PERTURBATION_NOTES = {
    "Original":        "No perturbation (reference)",
    "Low brightness":  "pixel *= 0.6 (darken 40%)",
    "High brightness": "pixel *= 1.4, clipped to [-1,1]",
    "Low contrast":    "50% blend toward per-image mean",
    "Gaussian blur":   "5x5 average pool (padding=2)",
    "Gaussian noise":  "N(0, sigma=0.05) additive, clipped to [-1,1]",
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute binary classification metrics from flat integer arrays."""
    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    accuracy  = (TP + TN) / len(labels)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    far       = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy":         round(accuracy,  6),
        "precision":        round(precision, 6),
        "recall":           round(recall,    6),
        "f1":               round(f1,        6),
        "false_alarm_rate": round(far,       6),
    }


# ---------------------------------------------------------------------------
# Single-condition evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(
    model:       torch.nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    perturb_fn:  Callable,
) -> tuple[dict, float]:
    """Run inference under one perturbation and return (metrics, avg_ms)."""
    all_preds  = []
    all_labels = []
    times      = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x = perturb_fn(batch_x)          # apply perturbation
            t0      = time.perf_counter()
            logits  = model(batch_x).squeeze(1)
            t1      = time.perf_counter()
            times.append((t1 - t0) / batch_x.size(0))
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.long().numpy().tolist())

    metrics    = compute_metrics(
        np.array(all_preds,  dtype=np.int64),
        np.array(all_labels, dtype=np.int64),
    )
    avg_ms = round(float(np.mean(times)) * 1000, 4)
    return metrics, avg_ms


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_robustness_evaluation() -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Load checkpoint ----------------------------------------------
    if not HYBRID_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {HYBRID_CHECKPOINT_PATH}\n"
            "Run src.train_hybrid first."
        )
    checkpoint = torch.load(HYBRID_CHECKPOINT_PATH, map_location=device)
    print(
        f"Checkpoint: epoch {checkpoint['epoch']}, "
        f"val_loss={checkpoint['val_loss']:.4f}, "
        f"val_acc={checkpoint['val_acc']:.4f}"
    )

    # --- Rebuild model (weights only, no grad) -------------------------
    model = build_hybrid_model(
        freeze_backbone=HYBRID_CONFIG["freeze_backbone"],
        dropout_p=HYBRID_CONFIG["fusion_dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- Test data loader (no perturbation yet) -----------------------
    test_smoke  = load_split("test", preprocess=True)
    test_ds     = SmokeTorchDataset(test_smoke)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    n_samples = len(test_ds)
    print(f"Test set: {n_samples} images\n")

    # --- Run all conditions -------------------------------------------
    all_results = []
    for condition_name, perturb_fn in CONDITIONS:
        print(f"Running: {condition_name} ...")
        metrics, avg_ms = evaluate_condition(model, test_loader, device, perturb_fn)
        result = {
            "condition":          condition_name,
            "perturbation":       PERTURBATION_NOTES[condition_name],
            "n_samples":          n_samples,
            "accuracy":           metrics["accuracy"],
            "precision":          metrics["precision"],
            "recall":             metrics["recall"],
            "f1":                 metrics["f1"],
            "false_alarm_rate":   metrics["false_alarm_rate"],
            "TP":                 metrics["TP"],
            "TN":                 metrics["TN"],
            "FP":                 metrics["FP"],
            "FN":                 metrics["FN"],
            "avg_inference_ms":   avg_ms,
        }
        all_results.append(result)
        print(
            f"  Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"FAR={metrics['false_alarm_rate']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"[{avg_ms:.3f} ms/img]"
        )

    # --- Save JSON ----------------------------------------------------
    with open(ROBUSTNESS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved : {ROBUSTNESS_JSON}")

    # --- Print summary table ------------------------------------------
    _print_table(all_results)

    # --- Save comparison plot -----------------------------------------
    _save_comparison_plot(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(results: list[dict]) -> None:
    col_w = 18
    sep   = "-" * (col_w + 10 * 7)
    header = (
        f"{'Condition':<{col_w}}"
        f"{'Accuracy':>10}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'FAR':>10}"
        f"{'TP':>7}"
        f"{'FP':>7}"
        f"{'FN':>7}"
    )
    print(f"\n{'='*len(header)}")
    print("  ROBUSTNESS EVALUATION SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['condition']:<{col_w}}"
            f"{r['accuracy']:>10.4f}"
            f"{r['precision']:>10.4f}"
            f"{r['recall']:>10.4f}"
            f"{r['f1']:>10.4f}"
            f"{r['false_alarm_rate']:>10.4f}"
            f"{r['TP']:>7}"
            f"{r['FP']:>7}"
            f"{r['FN']:>7}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _save_comparison_plot(results: list[dict]) -> None:
    conditions = [r["condition"] for r in results]
    accuracies = [r["accuracy"]         for r in results]
    f1_scores  = [r["f1"]               for r in results]
    far_values = [r["false_alarm_rate"] for r in results]

    x     = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_acc = ax.bar(x - width,     accuracies, width, label="Accuracy",          color="#2196F3", alpha=0.85)
    bars_f1  = ax.bar(x,             f1_scores,  width, label="F1-score (SMOKE)",   color="#4CAF50", alpha=0.85)
    bars_far = ax.bar(x + width,     far_values, width, label="False Alarm Rate",   color="#F44336", alpha=0.85)

    # Value labels on bars
    for bars in (bars_acc, bars_f1, bars_far):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, rotation=90,
            )

    ax.set_xlabel("Perturbation condition", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "CNN+Transformer Hybrid — Robustness Under Image Perturbations\n"
        "(IIITDMJ_Smoke test set, 2,367 images)",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(ROBUSTNESS_PNG, dpi=150)
    plt.close(fig)
    print(f"Plot saved    : {ROBUSTNESS_PNG}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_robustness_evaluation()
