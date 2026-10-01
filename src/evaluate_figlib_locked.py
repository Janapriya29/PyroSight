"""
src/evaluate_figlib_locked.py
------------------------------
Final locked-test evaluation of the FIgLib domain adaptation experiment.

Evaluates ONLY on the three sequences that were held out throughout Phase 9:

    LOCKED TEST sequences (217 images total):
      1. 20221116-Willowfire-om-n-mobo-c    (80 images)
      2. 20200806_SpringsFire_om-n-mobo-c   (65 images)
      3. 20180806_Holy_sp-s-mobo-c          (72 images)

Models compared:
      A. models/hybrid_robust_best.pt    (before adaptation, Phase 7.2)
      B. models/hybrid_figlib_adapted.pt (after  adaptation, Phase 9.2)

Protocol guarantees
-------------------
1. The 153 adaptation images (FIRE_lp, FIRE_ml) are explicitly excluded by
   timestamp-range check. An assertion fires if any adapt image is loaded.
2. Threshold fixed at sigmoid >= 0.5. Not tuned on the locked test set.
3. No model weights are modified.
4. No existing result files are overwritten.

Outputs (new files only)
-------
  results/figlib_locked_robust_metrics.json
  results/figlib_locked_adapted_metrics.json
  results/figlib_locked_comparison.png

Usage
-----
    python -m src.evaluate_figlib_locked
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.adapt_figlib import (
    ADAPT_CONFIG,
    ADAPTED_CHECKPOINT,
    FIGLIB_DIR,
    _OFFSET_RE,
    _SEQ_RANGES,
    _assign_label,
    _sequence_of,
)
from src.hybrid_model import build_hybrid_model
from src.preprocessing import load_and_preprocess, TARGET_SIZE
from src.train_robust import ROBUST_CHECKPOINT_PATH, ROBUST_CONFIG

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"

LOCKED_ROBUST_JSON  = RESULTS_DIR / "figlib_locked_robust_metrics.json"
LOCKED_ADAPTED_JSON = RESULTS_DIR / "figlib_locked_adapted_metrics.json"
LOCKED_COMPARE_PNG  = RESULTS_DIR / "figlib_locked_comparison.png"

# Guard: must not overwrite any existing result file
_PROTECTED_NAMES = {
    "figlib_hybrid_metrics.json",
    "figlib_robust_metrics.json",
    "figlib_comparison.png",
    "figlib_adaptation_config.json",
    "figlib_adaptation_history.json",
    "robust_model_robustness_results.json",
    "robustness_results.json",
    "hybrid_test_metrics.json",
}
for _out in (LOCKED_ROBUST_JSON, LOCKED_ADAPTED_JSON, LOCKED_COMPARE_PNG):
    assert _out.name not in _PROTECTED_NAMES, f"Would overwrite protected file: {_out.name}"

BATCH_SIZE  = 32
NUM_WORKERS = 0
THRESHOLD   = 0.5   # fixed — not tuned on test set

# ---------------------------------------------------------------------------
# Locked test sequences (never used in adaptation)
# ---------------------------------------------------------------------------
LOCKED_TEST_SEQUENCES = ADAPT_CONFIG["locked_test_sequences"]
ADAPT_SEQUENCES       = ADAPT_CONFIG["adapt_sequences"]

# Short display names for plots / tables
_SHORT_NAMES = {
    "20221116-Willowfire-om-n-mobo-c":  "Willowfire",
    "20200806_SpringsFire_om-n-mobo-c": "SpringsFire",
    "20180806_Holy_sp-s-mobo-c":        "Holy",
}


# ---------------------------------------------------------------------------
# Locked-test-only Dataset
# ---------------------------------------------------------------------------

class LockedTestDataset(Dataset):
    """Loads ONLY the three locked-test sequence images.

    Adaptation images are excluded by timestamp range with an assertion
    so any accidental inclusion raises an error immediately.
    """

    def __init__(self, image_dir: Path) -> None:
        self.samples: list[tuple[Path, int, str]] = []
        adapt_skipped = 0

        for p in sorted(image_dir.glob("*.jpg")):
            seq = _sequence_of(p.name)

            # Hard block on adapt sequences
            if seq in ADAPT_SEQUENCES:
                adapt_skipped += 1
                continue

            # Only include locked test sequences
            if seq not in LOCKED_TEST_SEQUENCES:
                continue   # unknown / unlabelled file — skip

            label = _assign_label(p.name)
            if label is None:
                continue

            self.samples.append((p, label, seq))

        self.labels = [s[1] for s in self.samples]
        self.seqs   = [s[2] for s in self.samples]

        # Mandatory confirmation print
        smoke    = sum(l == 1 for l in self.labels)
        no_smoke = sum(l == 0 for l in self.labels)
        print(f"  Locked test images loaded  : {len(self.samples)}")
        print(f"    SMOKE                    : {smoke}")
        print(f"    NO_SMOKE                 : {no_smoke}")
        print(f"  Adapt images skipped       : {adapt_skipped}  "
              f"(NOT loaded into memory)")
        assert adapt_skipped == 153, (
            f"Expected 153 adapt images to be skipped, got {adapt_skipped}. "
            "Check that the images directory is correct."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, _ = self.samples[idx]
        arr    = load_and_preprocess(str(path), TARGET_SIZE)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())
    n  = len(labels)
    acc  = (TP + TN) / n            if n > 0           else 0.0
    prec = TP / (TP + FP)           if (TP + FP) > 0   else 0.0
    rec  = TP / (TP + FN)           if (TP + FN) > 0   else 0.0
    f1   = 2*prec*rec / (prec+rec)  if (prec+rec) > 0  else 0.0
    far  = FP / (FP + TN)           if (FP + TN) > 0   else 0.0
    return dict(
        accuracy=round(acc, 6), precision=round(prec, 6),
        recall=round(rec, 6),   f1=round(f1, 6),
        false_alarm_rate=round(far, 6),
        TP=TP, TN=TN, FP=FP, FN=FN,
    )


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(
    checkpoint_path: Path,
    config:          dict,
    dataset:         LockedTestDataset,
    model_label:     str,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"\n  Checkpoint : epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.4f}")

    model = build_hybrid_model(
        freeze_backbone=config["freeze_backbone"],
        dropout_p=config["fusion_dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    all_preds  = []
    all_labels = []
    times      = []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            t0 = time.perf_counter()
            logits = model(bx).squeeze(1)
            t1 = time.perf_counter()
            times.append((t1 - t0) / bx.size(0))
            preds = (torch.sigmoid(logits) >= THRESHOLD).long().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(by.long().numpy().tolist())

    all_preds  = np.array(all_preds,  dtype=np.int64)
    all_labels = np.array(all_labels, dtype=np.int64)
    avg_ms     = round(float(np.mean(times)) * 1000, 4)

    overall = _compute_metrics(all_preds, all_labels)

    # Per-sequence breakdown
    seq_results = []
    for seq in LOCKED_TEST_SEQUENCES:
        mask = np.array([dataset.seqs[i] == seq
                         for i in range(len(dataset.samples))], dtype=bool)
        if mask.sum() == 0:
            continue
        sm = _compute_metrics(all_preds[mask], all_labels[mask])
        n_smoke    = int((all_labels[mask] == 1).sum())
        n_no_smoke = int((all_labels[mask] == 0).sum())
        seq_results.append({
            "sequence":   seq,
            "short_name": _SHORT_NAMES.get(seq, seq),
            "n_total":    int(mask.sum()),
            "n_smoke":    n_smoke,
            "n_no_smoke": n_no_smoke,
            **sm,
        })

    # Print summary
    print(f"\n  {'='*56}")
    print(f"  {model_label.upper()} — LOCKED TEST EVALUATION")
    print(f"  {'='*56}")
    print(f"  Total images      : {len(all_labels)}")
    print(f"  SMOKE             : {int((all_labels==1).sum())}   "
          f"NO_SMOKE: {int((all_labels==0).sum())}")
    print(f"  Accuracy          : {overall['accuracy']:.4f}  "
          f"({overall['accuracy']*100:.2f}%)")
    print(f"  Precision (SMOKE) : {overall['precision']:.4f}")
    print(f"  Recall    (SMOKE) : {overall['recall']:.4f}")
    print(f"  F1-score  (SMOKE) : {overall['f1']:.4f}")
    print(f"  False Alarm Rate  : {overall['false_alarm_rate']:.4f}")
    print(f"  TP={overall['TP']}  TN={overall['TN']}  "
          f"FP={overall['FP']}  FN={overall['FN']}")
    print(f"  Avg inference/img : {avg_ms:.3f} ms")
    print(f"\n  Per-sequence:")
    for r in seq_results:
        print(f"    {r['short_name']:<14} "
              f"Acc={r['accuracy']:.3f}  F1={r['f1']:.3f}  "
              f"FAR={r['false_alarm_rate']:.3f}  "
              f"Prec={r['precision']:.3f}  Rec={r['recall']:.3f}  "
              f"(n={r['n_total']}, S={r['n_smoke']}, NS={r['n_no_smoke']})")

    return {
        "model":                      model_label,
        "checkpoint_path":            str(checkpoint_path),
        "checkpoint_epoch":           ckpt["epoch"],
        "classification_threshold":   THRESHOLD,
        "threshold_tuned_on_test":    False,
        "adapt_images_used_in_eval":  False,
        "total_images":               len(all_labels),
        "smoke_count":                int((all_labels == 1).sum()),
        "no_smoke_count":             int((all_labels == 0).sum()),
        "accuracy":                   overall["accuracy"],
        "precision":                  overall["precision"],
        "recall":                     overall["recall"],
        "f1":                         overall["f1"],
        "false_alarm_rate":           overall["false_alarm_rate"],
        "TP":                         overall["TP"],
        "TN":                         overall["TN"],
        "FP":                         overall["FP"],
        "FN":                         overall["FN"],
        "avg_inference_ms_per_image": avg_ms,
        "per_sequence":               seq_results,
    }


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def _print_comparison(robust: dict, adapted: dict) -> None:
    col = 14
    w   = 24 + col * 3
    sep = "-" * w
    print(f"\n{'='*w}")
    print("  LOCKED TEST — BEFORE vs AFTER ADAPTATION")
    print(f"{'='*w}")
    header = f"{'Metric':<24}{'Robust':>{col}}{'Adapted':>{col}}{'Delta':>{col}}"
    print(header)
    print(sep)
    rows = [
        ("Accuracy",          "accuracy",         False),
        ("Precision (SMOKE)", "precision",        False),
        ("Recall (SMOKE)",    "recall",           False),
        ("F1-score",          "f1",               False),
        ("False Alarm Rate",  "false_alarm_rate", True),   # lower is better
    ]
    for name, key, lower_better in rows:
        r = robust[key]
        a = adapted[key]
        d = a - r
        improved = (d < 0) if lower_better else (d > 0)
        arrow = " +" if improved else (" -" if d != 0 else "  ")
        print(f"{name:<24}{r:>{col}.4f}{a:>{col}.4f}  {d:+.4f}{arrow}")
    print(sep)
    r_cm = f"{robust['TP']}/{robust['TN']}/{robust['FP']}/{robust['FN']}"
    a_cm = f"{adapted['TP']}/{adapted['TN']}/{adapted['FP']}/{adapted['FN']}"
    print(f"{'TP/TN/FP/FN':<24}{r_cm:>{col}}{a_cm:>{col}}")
    print(f"{'Inference (ms/img)':<24}"
          f"{robust['avg_inference_ms_per_image']:>{col}.3f}"
          f"{adapted['avg_inference_ms_per_image']:>{col}.3f}")

    # Per-sequence comparison
    print(f"\n  Per-sequence F1:")
    r_seqs = {s["sequence"]: s for s in robust["per_sequence"]}
    for s in adapted["per_sequence"]:
        seq = s["sequence"]
        rr  = r_seqs.get(seq, {})
        rf1 = rr.get("f1", 0.0)
        af1 = s["f1"]
        d   = af1 - rf1
        arrow = " +" if d > 0.001 else (" -" if d < -0.001 else "  ")
        sname = _SHORT_NAMES.get(seq, seq)
        print(f"    {sname:<14}  Robust={rf1:.3f}  Adapted={af1:.3f}  Δ={d:+.3f}{arrow}")

    print(f"\n{'='*w}")
    print("  + = improvement   - = regression   (FAR: lower is better)")
    print(f"{'='*w}")


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _save_comparison_plot(robust: dict, adapted: dict) -> None:
    seq_names  = [_SHORT_NAMES.get(s["sequence"], s["sequence"])
                  for s in robust["per_sequence"]]
    metrics_cfg = [
        ("accuracy",         "Accuracy",         False),
        ("f1",               "F1-score",          False),
        ("false_alarm_rate", "False Alarm Rate",  True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "FIgLib Locked Test — Robust vs Adapted Model\n"
        "(Willowfire · SpringsFire · Holy  |  217 images  |  threshold = 0.5)",
        fontsize=11,
    )

    for ax, (key, label, lower_better) in zip(axes, metrics_cfg):
        r_overall = robust[key]
        a_overall = adapted[key]
        r_seq = [s[key] for s in robust["per_sequence"]]
        a_seq = [s[key] for s in adapted["per_sequence"]]

        x     = np.arange(len(seq_names) + 1)
        width = 0.35
        labels_x = ["Overall"] + seq_names

        b1 = ax.bar(x - width/2, [r_overall] + r_seq, width,
                    label="Robust (before)", color="#E65100", alpha=0.87)
        b2 = ax.bar(x + width/2, [a_overall] + a_seq, width,
                    label="Adapted (after)", color="#1B5E20", alpha=0.87)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=8.5, rotation=0)

        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.20)
        ax.axvline(x=0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(LOCKED_COMPARE_PNG, dpi=150)
    plt.close(fig)
    print(f"\nComparison plot saved : {LOCKED_COMPARE_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_locked_evaluation() -> tuple[dict, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice       : {device}")
    print(f"FIgLib dir   : {FIGLIB_DIR}")
    print(f"Threshold    : {THRESHOLD}  (NOT tuned on test set)")
    print(f"\nLocked test sequences:")
    for s in LOCKED_TEST_SEQUENCES:
        print(f"  {s}")
    print(f"\nAdaptation sequences (excluded from evaluation):")
    for s in ADAPT_SEQUENCES:
        print(f"  {s}")

    # --- Build locked-test dataset (asserts adapt exclusion) ----------
    print("\nBuilding locked-test dataset...")
    dataset = LockedTestDataset(FIGLIB_DIR)

    # --- Evaluate model A: robust (before adaptation) ----------------
    print(f"\n{'='*60}")
    print("MODEL A: Robust Hybrid  (hybrid_robust_best.pt)")
    print(f"{'='*60}")
    robust_results = _evaluate_one(
        ROBUST_CHECKPOINT_PATH, ROBUST_CONFIG, dataset, "Robust (before adaptation)"
    )

    # --- Evaluate model B: adapted -----------------------------------
    print(f"\n{'='*60}")
    print("MODEL B: Adapted Hybrid  (hybrid_figlib_adapted.pt)")
    print(f"{'='*60}")
    adapted_results = _evaluate_one(
        ADAPTED_CHECKPOINT, ADAPT_CONFIG, dataset, "Adapted (after adaptation)"
    )

    # --- Save JSON ---------------------------------------------------
    with open(LOCKED_ROBUST_JSON, "w") as f:
        json.dump(robust_results, f, indent=2)
    print(f"\nRobust metrics saved  : {LOCKED_ROBUST_JSON}")

    with open(LOCKED_ADAPTED_JSON, "w") as f:
        json.dump(adapted_results, f, indent=2)
    print(f"Adapted metrics saved : {LOCKED_ADAPTED_JSON}")

    # --- Comparison table and plot -----------------------------------
    _print_comparison(robust_results, adapted_results)
    _save_comparison_plot(robust_results, adapted_results)

    # --- Final protocol confirmation ---------------------------------
    print(f"\n{'='*60}")
    print("  PROTOCOL CONFIRMATION")
    print(f"{'='*60}")
    print(f"  Total locked images evaluated      : {robust_results['total_images']}")
    print(f"  Adaptation images used in eval     : 0  (NOT accessed)")
    print(f"  Classification threshold           : {THRESHOLD}  (fixed)")
    print(f"  Threshold tuned on locked test set : False")
    print(f"  hybrid_robust_best.pt modified     : False")
    print(f"  hybrid_figlib_adapted.pt modified  : False")
    print(f"  Previous result files overwritten  : False")
    print(f"{'='*60}")

    return robust_results, adapted_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_locked_evaluation()
