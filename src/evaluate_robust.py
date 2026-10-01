"""
src/evaluate_robust.py
-----------------------
Robustness evaluation of the AUGMENTATION-TRAINED hybrid model.

Loads models/hybrid_robust_best.pt and evaluates it on the same six
perturbation conditions used in Phase 6 (src/robustness_evaluation.py),
using identical perturbation functions imported from that module.

Also loads Phase 6 results (results/robustness_results.json) and prints
a before-vs-after comparison table.

Outputs  (new files only — existing files are never overwritten)
-------
  results/robust_model_robustness_results.json
  results/robust_model_robustness_comparison.png

Usage
-----
    python -m src.evaluate_robust
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_loader import load_split
from src.hybrid_model import build_hybrid_model
from src.train import SmokeTorchDataset
from src.train_robust import ROBUST_CHECKPOINT_PATH, ROBUST_CONFIG

# ---- Reuse IDENTICAL perturbation functions and condition registry ----
from src.robustness_evaluation import (
    CONDITIONS,
    PERTURBATION_NOTES,
    compute_metrics,
    evaluate_condition,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"

ROBUST_MODEL_JSON = RESULTS_DIR / "robust_model_robustness_results.json"
ROBUST_MODEL_PNG  = RESULTS_DIR / "robust_model_robustness_comparison.png"
PHASE6_JSON       = RESULTS_DIR / "robustness_results.json"   # read-only

BATCH_SIZE  = 32
NUM_WORKERS = 0

# Metrics we compare across conditions
COMPARE_KEYS = ["accuracy", "precision", "recall", "f1", "false_alarm_rate"]


# ---------------------------------------------------------------------------
# Guard: never overwrite Phase 6 results
# ---------------------------------------------------------------------------
def _assert_no_overwrite() -> None:
    existing_phase6 = [
        RESULTS_DIR / "robustness_results.json",
        RESULTS_DIR / "robustness_comparison.png",
    ]
    for p in existing_phase6:
        assert str(ROBUST_MODEL_JSON) != str(p), f"Output would overwrite {p}!"
        assert str(ROBUST_MODEL_PNG)  != str(p), f"Output would overwrite {p}!"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_robust_model_evaluation() -> list[dict]:
    _assert_no_overwrite()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Load robust checkpoint ----------------------------------------
    if not ROBUST_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Robust checkpoint not found: {ROBUST_CHECKPOINT_PATH}\n"
            "Run src.train_robust first."
        )
    checkpoint = torch.load(ROBUST_CHECKPOINT_PATH, map_location=device)
    print(
        f"Robust checkpoint: epoch {checkpoint['epoch']}, "
        f"val_loss={checkpoint['val_loss']:.4f}, "
        f"val_acc={checkpoint['val_acc']:.4f}"
    )

    # --- Rebuild model -------------------------------------------------
    model = build_hybrid_model(
        freeze_backbone=ROBUST_CONFIG["freeze_backbone"],
        dropout_p=ROBUST_CONFIG["fusion_dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- Test data loader ----------------------------------------------
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

    # --- Run all six conditions (same as Phase 6) ----------------------
    all_results: list[dict] = []
    for condition_name, perturb_fn in CONDITIONS:
        print(f"Running: {condition_name} ...")
        metrics, avg_ms = evaluate_condition(model, test_loader, device, perturb_fn)
        result = {
            "model":              "CNN+Transformer hybrid (robustness training)",
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

    # --- Save JSON (new file only) ------------------------------------
    with open(ROBUST_MODEL_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved : {ROBUST_MODEL_JSON}")

    # --- Print per-model summary table --------------------------------
    _print_table(all_results, title="ROBUST MODEL — SIX-CONDITION RESULTS")

    # --- Load Phase 6 results and print comparison --------------------
    if PHASE6_JSON.exists():
        with open(PHASE6_JSON) as f:
            phase6_results = json.load(f)
        _print_comparison_table(phase6_results, all_results)
        comparison = _build_comparison(phase6_results, all_results)
    else:
        print("\n[WARNING] Phase 6 results not found; skipping comparison.")
        comparison = []

    # --- Save chart ---------------------------------------------------
    _save_comparison_plot(all_results)

    # --- Save combined comparison JSON --------------------------------
    combined_path = RESULTS_DIR / "robust_model_comparison_summary.json"
    with open(combined_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved : {combined_path}")

    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_table(results: list[dict], title: str = "") -> None:
    col_w  = 18
    sep    = "-" * (col_w + 10 * 7)
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
    width = len(header)
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")
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


def _build_comparison(
    phase6: list[dict],
    robust: list[dict],
) -> list[dict]:
    """Build a list of per-condition delta dicts (robust - original)."""
    p6_by_cond = {r["condition"]: r for r in phase6}
    comparison = []
    for r in robust:
        cond = r["condition"]
        p6   = p6_by_cond.get(cond, {})
        entry = {"condition": cond, "perturbation": r["perturbation"]}
        for k in COMPARE_KEYS:
            orig  = p6.get(k, None)
            new   = r.get(k, None)
            entry[f"original_{k}"] = orig
            entry[f"robust_{k}"]   = new
            if orig is not None and new is not None:
                entry[f"delta_{k}"] = round(new - orig, 6)
            else:
                entry[f"delta_{k}"] = None
        comparison.append(entry)
    return comparison


def _print_comparison_table(
    phase6: list[dict],
    robust: list[dict],
) -> None:
    """Print a side-by-side before / after / delta table."""
    p6_by_cond = {r["condition"]: r for r in phase6}

    metrics_to_show = [
        ("accuracy",         "Accuracy",   False),
        ("f1",               "F1",         False),
        ("false_alarm_rate", "FAR",        True),   # True = lower is better
        ("precision",        "Precision",  False),
        ("recall",           "Recall",     False),
    ]

    col_w = 18
    mc    = 10
    sep   = "-" * (col_w + mc * 9)

    print(f"\n{'='*(col_w + mc*9)}")
    print("  BEFORE vs AFTER COMPARISON  (Original Hybrid → Robust Hybrid)")
    print(f"{'='*(col_w + mc*9)}")

    for metric_key, metric_label, lower_better in metrics_to_show:
        header = (
            f"{'Condition':<{col_w}}"
            f"{'Original':>{mc}}"
            f"{'Robust':>{mc}}"
            f"{'Delta':>{mc}}"
        )
        print(f"\n  -- {metric_label} --")
        print(header)
        print("-" * len(header))
        for r in robust:
            cond = r["condition"]
            p6   = p6_by_cond.get(cond, {})
            orig  = p6.get(metric_key)
            new   = r.get(metric_key)
            if orig is None or new is None:
                delta_str = "   N/A"
                arrow     = ""
            else:
                delta = new - orig
                # positive delta on FAR = worse; positive delta on others = better
                improved = (delta < 0) if lower_better else (delta > 0)
                arrow    = " +" if improved else (" -" if (delta != 0) else "  ")
                delta_str = f"{delta:+.4f}{arrow}"
            print(
                f"{cond:<{col_w}}"
                f"{(orig if orig is not None else 0.0):>{mc}.4f}"
                f"{(new  if new  is not None else 0.0):>{mc}.4f}"
                f"  {delta_str}"
            )

    print(f"\n{'='*(col_w + mc*9)}")
    print("  + = improvement   - = regression   (FAR: lower is better)")
    print(f"{'='*(col_w + mc*9)}")


def _save_comparison_plot(results: list[dict]) -> None:
    """Bar chart of Accuracy, F1 and FAR for the robust model across conditions."""
    conditions = [r["condition"] for r in results]
    accuracies = [r["accuracy"]         for r in results]
    f1_scores  = [r["f1"]               for r in results]
    far_values = [r["false_alarm_rate"] for r in results]

    x     = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_acc = ax.bar(x - width, accuracies, width,
                      label="Accuracy",        color="#1565C0", alpha=0.87)
    bars_f1  = ax.bar(x,         f1_scores,  width,
                      label="F1-score (SMOKE)", color="#2E7D32", alpha=0.87)
    bars_far = ax.bar(x + width, far_values, width,
                      label="False Alarm Rate", color="#C62828", alpha=0.87)

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
        "Robust CNN+Transformer Hybrid — Robustness Under Image Perturbations\n"
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
    fig.savefig(ROBUST_MODEL_PNG, dpi=150)
    plt.close(fig)
    print(f"Plot saved    : {ROBUST_MODEL_PNG}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_robust_model_evaluation()
