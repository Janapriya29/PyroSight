"""
src/evaluate.py
---------------
Final test-set evaluation for the PyroSight MobileNetV2 baseline.

Loads the best checkpoint saved during training and evaluates it ONCE
on the held-out test split.

Metrics computed
----------------
- Accuracy
- Precision  (binary, positive class = SMOKE)
- Recall     (binary, positive class = SMOKE)
- F1-score   (binary)
- Confusion matrix  (saved as PNG)
- False Alarm Rate  = FP / (FP + TN)   [fraction of NO_SMOKE misclassified]
- Per-image inference time (ms)

All results are saved to results/test_metrics.json.
Confusion matrix figure is saved to results/confusion_matrix.png.

Usage
-----
    python -m src.evaluate
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loader import load_split
from src.model import build_mobilenetv2_baseline, count_parameters, model_size_mb
from src.train import SmokeTorchDataset, CHECKPOINT_PATH, RESULTS_DIR, CONFIG

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METRICS_PATH = RESULTS_DIR / "test_metrics.json"
CM_PATH      = RESULTS_DIR / "confusion_matrix.png"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate() -> dict:
    """Load best checkpoint, evaluate on test set, save and return metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Load checkpoint -----------------------------------------------
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}. Run src.train first."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")

    # --- Rebuild model and load weights --------------------------------
    model = build_mobilenetv2_baseline(
        dropout_p=CONFIG["dropout_p"],
        freeze_backbone=CONFIG["freeze_backbone"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- Test data loader ----------------------------------------------
    test_smoke = load_split("test", preprocess=True)
    test_ds    = SmokeTorchDataset(test_smoke)
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    print(f"Test set: {len(test_ds)} images")

    # --- Inference -----------------------------------------------------
    all_preds  = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            t0      = time.perf_counter()
            logits  = model(batch_x).squeeze(1)
            t1      = time.perf_counter()
            inference_times.append((t1 - t0) / batch_x.size(0))  # per image
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.long().numpy().tolist())

    all_preds  = np.array(all_preds,  dtype=np.int64)
    all_labels = np.array(all_labels, dtype=np.int64)

    # --- Metrics -------------------------------------------------------
    TP = int(((all_preds == 1) & (all_labels == 1)).sum())
    TN = int(((all_preds == 0) & (all_labels == 0)).sum())
    FP = int(((all_preds == 1) & (all_labels == 0)).sum())
    FN = int(((all_preds == 0) & (all_labels == 1)).sum())

    accuracy  = (TP + TN) / len(all_labels)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    avg_inference_ms = float(np.mean(inference_times)) * 1000  # ms per image

    param_info = count_parameters(model)
    size_mb    = model_size_mb(model)

    print(f"\n{'='*52}")
    print(f"  TEST SET EVALUATION RESULTS")
    print(f"{'='*52}")
    print(f"  Accuracy          : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision (SMOKE) : {precision:.4f}")
    print(f"  Recall    (SMOKE) : {recall:.4f}")
    print(f"  F1-score  (SMOKE) : {f1:.4f}")
    print(f"  False Alarm Rate  : {false_alarm_rate:.4f}")
    print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"  Total params      : {param_info['total']:,}")
    print(f"  Trainable params  : {param_info['trainable']:,}")
    print(f"  Model size (fp32) : {size_mb} MB")
    print(f"  Avg inference/img : {avg_inference_ms:.3f} ms")
    print(f"{'='*52}")

    # --- Confusion matrix figure ---------------------------------------
    cm = np.array([[TN, FP], [FN, TP]])
    class_names = ["NO_SMOKE", "SMOKE"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix — MobileNetV2 Baseline (Test Set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {CM_PATH}")

    # --- Save metrics JSON --------------------------------------------
    metrics = {
        "checkpoint_epoch":   checkpoint["epoch"],
        "best_val_loss":      round(float(checkpoint["val_loss"]), 6),
        "best_val_acc":       round(float(checkpoint["val_acc"]),  6),
        "test_accuracy":      round(accuracy,         6),
        "test_precision":     round(precision,        6),
        "test_recall":        round(recall,           6),
        "test_f1":            round(f1,               6),
        "test_false_alarm_rate": round(false_alarm_rate, 6),
        "confusion_matrix":   {"TN": TN, "FP": FP, "FN": FN, "TP": TP},
        "total_params":       param_info["total"],
        "trainable_params":   param_info["trainable"],
        "frozen_params":      param_info["frozen"],
        "model_size_mb":      size_mb,
        "avg_inference_ms_per_image": round(avg_inference_ms, 4),
        "test_set_size":      len(all_labels),
        "checkpoint_path":    str(CHECKPOINT_PATH),
        "confusion_matrix_path": str(CM_PATH),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved     : {METRICS_PATH}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()
